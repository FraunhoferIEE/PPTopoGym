"""A lightsim2grid power-flow backend that understands double-busbar substations.

Why this exists: lightsim2grid solves these grids ~47x faster than pandapower, but it
**ignores pandapower switches**. Handed the expanded net directly it reads case30's 93 buses as
93 separate, mostly injection-free nodes and returns an all-NaN answer in 0.3 ms -- a fast wrong
result (see ``CLAUDE.md``). The blocker is the switch layer, not the library.

So this re-expresses the same topology the way grid2op does, as a **per-element bus assignment**
over the busbars of each substation:

- The *mirror net* is the grid with the auxiliary busbar buses and all substation switches
  removed. Every element terminal attaches directly to one of its substation's busbar buses.
  That net has no switches at all, so lightsim2grid consumes it natively.
- Each power flow, the live switch state is read back into an assignment -- element *i* sits on
  busbar *k* if ``b{k}_switches[i]`` is closed -- and applied to the ``GridModel`` with
  ``change_bus_*``. A closed bus coupler means the busbars are one electrical node, so every
  element of that substation collapses onto busbar 0.

Results are **not** bit-identical to the pandapower path, but the reason is narrower than "a
different solver": with ``use_ls2g="auto"`` pandapower already hands its assembled ppc to
lightsim2grid's Newton, so both paths run the *same* numerical core. What differs is only how
the model is assembled -- pandapower's ``_pd2ppc`` rebuild versus this mirror net plus per-solve
bus assignments. That is why they agree to ~1e-9 on the grids measured, and why the risk that
needs testing is the topology translation rather than the numerics. The backend is opt-in
(``env_config["backend"] = "lightsim"``) and gated on tolerance parity rather than byte parity.

N-1 rides on the same machinery: :meth:`LightsimBackend.solve_nminus1` switches one element out
of service at a time and re-solves, aggregating the worst loadings exactly as
``pandapower.contingency.run_contingency`` does. It is selected by the same
``env_config["backend"] = "lightsim"`` as the N-0 path, so an N-1 environment gets the fast
solver without a second switch to set.

Known limits, all silent rather than loud: transformer **tap positions** are not pushed to the
model (a PST action would be ignored), ``impedance`` / ``ward`` results are not written back and
``res_bus.p_mw`` / ``q_mvar`` are left NaN. ``trafo3w`` is the one loud limit -- the N-1 sweep
raises on a net that has one, rather than quietly skipping those contingencies.

:raises ImportError: if lightsim2grid is not installed (only when this backend is selected).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandapower as pp
import pandas as pd

from pandapower_env.toolbox.utils import run_powerflow, select_topk_line_contingencies

if TYPE_CHECKING:
    from pandapower.auxiliary import pandapowerNet

logger = logging.getLogger(__name__)

# Element types whose outage is evaluated as a contingency and whose loading is monitored during
# the N-1 sweep -- the same set pandapower's run_contingency covers, minus trafo3w (see
# :meth:`LightsimBackend.solve_nminus1`).
_MONITORED_ELEMENTS = ("line", "trafo")

# The loading a contingency must push an element past to count as "causing overloading", when the
# net does not state one. Matches the placeholder ``run_nminus1_powerflow`` fills in.
_DEFAULT_LOADING_LIMIT_PERCENT = 100.0

# How an element type's terminals are named, and which GridModel setter moves them: each
# entry is a table paired with its (bus column, GridModel setter) terminals.
_ELEMENT_TERMINALS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("line", (("from_bus", "change_bus_powerline_or"), ("to_bus", "change_bus_powerline_ex"))),
    ("trafo", (("hv_bus", "change_bus_trafo_hv"), ("lv_bus", "change_bus_trafo_lv"))),
    ("load", (("bus", "change_bus_load"),)),
    ("sgen", (("bus", "change_bus_sgen"),)),
    ("gen", (("bus", "change_bus_gen"),)),
    ("shunt", (("bus", "change_bus_shunt"),)),
)

# The column layout of each result table this backend writes, in pandapower's own order. Named
# here because :func:`_result_frame` builds every table from one values matrix, so the column
# names live apart from the arithmetic that fills them.
_RES_BUS_COLUMNS = ("vm_pu", "va_degree", "p_mw", "q_mvar")
_RES_LINE_COLUMNS = (
    "p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "pl_mw", "ql_mvar",
    "i_from_ka", "i_to_ka", "i_ka", "loading_percent",
)
_RES_TRAFO_COLUMNS = (
    "p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar", "pl_mw", "ql_mvar",
    "i_hv_ka", "i_lv_ka", "loading_percent",
)
_RES_INJECTION_COLUMNS = ("p_mw", "q_mvar")
_RES_GEN_COLUMNS = ("p_mw", "q_mvar", "vm_pu")


# Prototype for the result table of an element type the net does not have. ``pd.DataFrame()``
# takes ~132 us -- more than twice a real 41x10 frame -- because the no-argument constructor goes
# down a general path that has nothing to do here; copying this prototype is ~8.5 us and yields an
# equal frame. See :func:`_empty_result_frame`.
_EMPTY_RESULT_FRAME = pd.DataFrame()


def _empty_result_frame() -> pd.DataFrame:
    """Return the result table for an element type this net has none of.

    A *fresh* object every call, exactly as ``pd.DataFrame()`` was: the frames this backend
    publishes must not be shared between solves, or a caller holding last step's table would
    see it change under them. Copying the module prototype is 15x cheaper than constructing one
    (~8.5 us against ~132 us) and compares equal to it -- same shape, index, columns and dtypes.
    On a net without transformers or static generators that is two of these per solve, which was
    ~17% of the whole solve.

    :return: an empty result table.
    :rtype: pd.DataFrame
    """
    return _EMPTY_RESULT_FRAME.copy()


def _result_frame(
    values: np.ndarray, index: pd.Index, columns: tuple[str, ...],
) -> pd.DataFrame:
    """Wrap a solved ``(n_elements, n_columns)`` float matrix as a result table.

    Built from one 2-D array rather than a column dict: ``pd.DataFrame({...})`` sanitizes and
    boxes every column separately, which on case30 costs ~100 us for ``res_line`` against ~58 us
    here. Six tables are rebuilt on every solve, so that is ~130 us per power flow -- worth
    having on a backend whose whole point is a ~2.5 ms solve. The values are unchanged: the
    matrix is float64 throughout, exactly as the per-column frames were.

    :param values: one column per entry of ``columns``, in that order.
    :type values: np.ndarray
    :param index: the element index the results belong to (``net.line.index`` and friends).
    :type index: pd.Index
    :param columns: the result column names, matching ``values``' column order.
    :type columns: tuple[str, ...]
    :return: the result table, a fresh object per solve as pandapower's own results are.
    :rtype: pd.DataFrame
    """
    return pd.DataFrame(values, index=index, columns=list(columns), copy=False)


# Terminals rewired onto busbar 0 when the mirror is built: every element of
# ``_ELEMENT_TERMINALS``, plus the slack. The slack is deliberately *not* in that table, because
# lightsim2grid folds an ``ext_grid`` into a generator instead of exposing a ``change_bus_*``
# setter for it -- so it can be placed once, but not moved per solve the way the others are. It
# still has to be rewired here: leaving it on an auxiliary bus that ``build_mirror_net`` then
# deletes makes ``init_from_pandapower`` raise ``KeyError`` on every net whose slack sits inside a
# double-busbar substation. case14 and case30 keep theirs on a plain bus, which is why the N-0
# parity runs never reached it; case118's sits on bus 384, an auxiliary bus, and could not build
# a backend at all. :meth:`LightsimBackend._slack_moved` covers the states this cannot express.
_MIRROR_TERMINALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    *((table, tuple(column for column, _setter in terminals)) for table, terminals in _ELEMENT_TERMINALS),
    ("ext_grid", ("bus",)),
)


def build_mirror_net(net: pandapowerNet) -> tuple[pandapowerNet, dict]:
    """Build the switch-free mirror of a double-busbar net, plus the assignment plan.

    Every element that hangs off a substation's auxiliary bus is rewired onto that
    substation's busbar 0, and the auxiliary buses and substation switches are dropped. The
    resulting net is electrically the *fully coupled* configuration, which is the state the
    ``GridModel`` is built from; per-step assignments then move terminals onto other busbars.

    :param net: the expanded double-busbar network.
    :type net: pandapowerNet
    :return: ``(mirror_net, plan)`` where ``plan`` describes how to read assignments back.
    :rtype: tuple[pandapowerNet, dict]
    :raises ValueError: if ``net`` carries no ``multi_bb_substation`` table.
    """
    if "multi_bb_substation" not in net:
        msg = "build_mirror_net needs a net with double-busbar substations (multi_bb_substation)."
        raise ValueError(msg)

    mirror = _copy_tables(net)

    substations = net.multi_bb_substation
    # auxiliary bus -> (busbar buses of its substation, per-busbar switch ids, coupler switches)
    terminal_plan: dict[int, dict] = {}
    busbar_columns = [column for column in substations.columns if column.startswith("bus_")]

    for i_sub, row in substations.iterrows():
        busbars = [int(row[column]) for column in busbar_columns if not _is_missing(row[column])]
        switch_columns = [f"b{k}_switches" for k in range(len(busbars))]
        for position, aux_bus in enumerate(row["connected_buses"]):
            terminal_plan[int(aux_bus)] = {
                "substation": int(i_sub),
                "busbars": busbars,
                "switches": [int(row[column][position]) for column in switch_columns],
            }

    # Rewire every terminal off its auxiliary bus onto busbar 0, then delete the aux buses.
    for table, bus_columns in _MIRROR_TERMINALS:
        if table not in mirror or not len(mirror[table]):
            continue
        for bus_column in bus_columns:
            buses = mirror[table][bus_column].to_numpy()
            mirror[table][bus_column] = [
                terminal_plan[int(bus)]["busbars"][0] if int(bus) in terminal_plan else int(bus)
                for bus in buses
            ]

    aux_buses = sorted(terminal_plan)
    mirror.bus = mirror.bus.drop(index=[b for b in aux_buses if b in mirror.bus.index])
    mirror.switch = mirror.switch.iloc[0:0]  # the whole switch layer is what ls2g cannot read

    plan = {
        "terminal_plan": terminal_plan,
        "couplers": {
            int(i_sub): [
                int(row[column]) for column in substations.columns
                if column.endswith("_switch") and not _is_missing(row[column])
            ]
            for i_sub, row in substations.iterrows()
        },
    }
    return mirror, plan


def _copy_tables(net: pandapowerNet) -> pandapowerNet:
    """Copy the element tables that matter for a power flow, leaving profiles behind."""
    mirror = pp.create_empty_network(sn_mva=net.sn_mva, f_hz=net.f_hz)
    for table in ("bus", "line", "trafo", "trafo3w", "load", "sgen", "gen", "ext_grid",
                  "shunt", "switch", "impedance", "ward", "xward"):
        if table in net and len(net[table]):
            mirror[table] = net[table].copy()
    return mirror


def _is_missing(value: object) -> bool:
    """Report whether a substation-table cell is NaN or None."""
    return value is None or (isinstance(value, float) and np.isnan(value))


def current_bus_assignment(net: pandapowerNet, plan: dict) -> dict[int, int]:
    """Read the live switch state as ``expanded bus -> mirror busbar bus``.

    Element *i* of a substation sits on busbar *k* when ``b{k}_switches[i]`` is closed. When the
    substation's bus coupler is closed the busbars are a single electrical node, so every
    element is reported on busbar 0 -- which is exactly the state ``build_mirror_net`` wired.

    Busbar buses map to themselves, except under a closed coupler where they collapse onto
    busbar 0 as well. Those entries carry no weight for the element terminals (no element sits
    directly on a busbar bus) but they are what lets the result write-back give a fused busbar
    the voltage pandapower reports for it.

    :param net: the live expanded network, read for ``switch.closed``.
    :type net: pandapowerNet
    :param plan: the plan returned by :func:`build_mirror_net`.
    :type plan: dict
    :return: mapping of expanded bus to the mirror bus it is electrically part of.
    :rtype: dict[int, int]
    """
    closed = net.switch["closed"].to_numpy()
    coupled = {
        i_sub: all(closed[switch] for switch in switches) and bool(switches)
        for i_sub, switches in plan["couplers"].items()
    }

    assignment: dict[int, int] = {}
    for aux_bus, entry in plan["terminal_plan"].items():
        busbars = entry["busbars"]
        if coupled.get(entry["substation"], False):
            assignment[aux_bus] = busbars[0]
            for busbar in busbars:
                assignment[busbar] = busbars[0]
            continue
        # First closed busbar switch wins; a fully isolated element keeps busbar 0 and is
        # handled by the caller's connectivity check.
        target = busbars[0]
        for busbar, switch in zip(busbars, entry["switches"]):
            if closed[switch]:
                target = busbar
                break
        assignment[aux_bus] = target
        for busbar in busbars:
            assignment.setdefault(busbar, busbar)
    return assignment


class LightsimBackend:
    """Solve a double-busbar network through lightsim2grid's native ``GridModel``.

    Built once per environment from the *mirror* net (see :func:`build_mirror_net`), it then
    solves each timestep by pushing the live bus assignment and injections straight into the
    C++ model -- skipping pandapower's per-call ppc rebuild, which is ~98% of a normal step.

    :param net: the expanded double-busbar network this backend will solve.
    :type net: pandapowerNet
    :raises ImportError: if lightsim2grid is unavailable.
    """

    def __init__(self, net: pandapowerNet) -> None:
        from lightsim2grid.gridmodel import init_from_pandapower

        self.mirror, self.plan = build_mirror_net(net)
        self.model = init_from_pandapower(self.mirror)
        self.n_mirror_buses = len(self.mirror.bus)
        # Mirror bus label -> row position, because GridModel indexes buses positionally.
        self._bus_position = {int(label): pos for pos, label in enumerate(self.mirror.bus.index)}
        self._v_init = np.ones(self.n_mirror_buses, dtype=complex)
        # Where each slack terminal sat when the model was built; the model cannot move it, so
        # this is what :meth:`_slack_moved` compares the live switch state against.
        self._mirror_slack_buses = [int(bus) for bus in self.mirror.ext_grid["bus"].to_numpy()]
        self._applied_assignment: dict[tuple, int] = {}
        # Seeded from the mirror rather than from all-True: init_from_pandapower honours
        # in_service, so an element that was already out when the model was built must not be
        # remembered as live -- the first solve would then never push the real state.
        self._branch_in_service = {
            table: self.mirror[table]["in_service"].to_numpy(dtype=bool).copy()
            for table in _MONITORED_ELEMENTS
        }
        # Busbars with nothing attached are injection-free isolated nodes, which make the
        # Newton solve singular -- lightsim2grid returns an empty array rather than raising.
        # Every mirror bus starts active; _sync_active_buses deactivates the unused ones.
        self._bus_active = np.ones(self.n_mirror_buses, dtype=bool)

    def solve(self, net: pandapowerNet, *, max_iter: int = 20, tol: float = 1e-8) -> bool:
        """Solve ``net``'s current state and write the results back onto it.

        :param net: the live expanded network; read for topology and injections, written for results.
        :type net: pandapowerNet
        :param max_iter: Newton-Raphson iteration cap.
        :type max_iter: int
        :param tol: convergence tolerance.
        :type tol: float
        :return: True if the power flow converged.
        :rtype: bool
        """
        assignment = self._push_topology(net)
        if self._slack_moved(net, assignment):
            return _solve_with_pandapower(net)
        self._sync_active_buses(net, assignment)
        self._push_injections(net)
        voltages = self.model.ac_pf(self._v_init, max_iter, tol)
        if voltages.shape[0] == 0:
            net.converged = False
            return False
        self._write_results(net, assignment)
        net.converged = True
        return True

    def _slack_moved(self, net: pandapowerNet, assignment: dict[int, int]) -> bool:
        """Report whether the live topology has switched the slack off the busbar it was built on.

        The model's slack is a *generator* lightsim2grid synthesised from the ``ext_grid`` when the
        mirror was built (see :data:`_MIRROR_TERMINALS`), and it is pinned to the busbar it sat on
        then. A substation action that splits the slack's substation and moves its terminal to the
        other busbar is therefore a grid this model cannot express: pandapower would feed the
        network from the busbar the switch state actually names, and answering from the original
        one would be a *wrong* result rather than a failed one.

        Rare enough to pay for with the slow solver -- it needs the slack's own substation to be
        split *and* the slack switched across -- and it is the only state where the two backends
        would otherwise disagree by more than solver tolerance.

        :param net: the live expanded network.
        :type net: pandapowerNet
        :param assignment: the bus assignment :func:`current_bus_assignment` derived from it.
        :type assignment: dict[int, int]
        :return: True if the caller must fall back to pandapower.
        :rtype: bool
        """
        for element, live_bus in enumerate(net.ext_grid["bus"].to_numpy()):
            if assignment.get(int(live_bus), int(live_bus)) != self._mirror_slack_buses[element]:
                return True
        return False

    def solve_nminus1(
        self,
        net: pandapowerNet,
        *,
        topk_percent: float = 100.0,
        max_iter: int = 20,
        tol: float = 1e-8,
    ) -> bool:
        """Run the N-1 sweep through lightsim2grid and write the same columns pandapower does.

        One contingency at a time, the element is switched out of service on ``net`` and the grid
        is re-solved with :meth:`solve` -- so a contingency sees exactly the topology translation
        the N-0 path uses, split substations included. The per-contingency loadings are reduced
        into ``res_line`` / ``res_trafo`` (``max_loading_percent``, ``min_loading_percent``,
        ``cause_element``, ``cause_index``, ``causes_overloading``) and ``res_bus``
        (``max_vm_pu``, ``min_vm_pu``), matching ``pandapower.contingency.run_contingency``:
        the aggregates cover the N-1 cases only, the N-0 values stay in ``loading_percent`` /
        ``vm_pu``, and a contingency that no solver can answer is skipped rather than fatal.

        **A contingency that splits the grid falls back to pandapower** (:func:`_solve_with_pandapower`).
        lightsim2grid has one slack, so an outage that leaves an island returns an empty voltage
        vector, while pandapower promotes a generator in the island and solves it. Skipping those
        cases would silently *under*-report the N-1 risk -- on case30 one line outage islands a bus,
        and dropping it moved ``max_loading_percent`` by 0.3 pp and ``min_loading_percent`` by 69 pp.
        They are rare (1 of 41 contingencies there), so the slow solver is paid for rarely.

        The net is left carrying the N-0 result, because the sweep ends on one more N-0 solve --
        again mirroring ``run_contingency``, whose caller reads ``loading_percent`` afterwards.

        One deliberate difference from pandapower: when the running maximum of an element is still
        NaN (every contingency so far left it unsolved), pandapower's ``val > NaN`` test fails and
        the *cause* columns keep their uninitialised value while the maximum is updated. Here the
        first finite loading wins the cause, so ``cause_element`` / ``cause_index`` are always the
        contingency behind ``max_loading_percent``.

        :param net: the live expanded network; mutated in place with the results.
        :type net: pandapowerNet
        :param topk_percent: percentage of lines evaluated as contingencies, ranked by N-0 flow
            (see :func:`pandapower_env.toolbox.utils.select_topk_line_contingencies`); trafo
            contingencies are unaffected, and every element stays *monitored*.
        :type topk_percent: float
        :param max_iter: Newton-Raphson iteration cap per contingency.
        :type max_iter: int
        :param tol: convergence tolerance.
        :type tol: float
        :return: True if the N-0 power flow converged (individual contingencies may not have).
        :rtype: bool
        :raises NotImplementedError: if the net carries three-winding transformers, whose
            contingencies this backend cannot evaluate -- skipping them silently would report a
            more optimistic N-1 than the pandapower path.
        """
        if len(net.trafo3w):
            msg = "The lightsim2grid backend cannot run N-1 on a net with trafo3w elements."
            raise NotImplementedError(msg)

        if not self.solve(net, max_iter=max_iter, tol=tol):
            return False

        totals = _new_contingency_totals(net)
        for element, index in _contingency_cases(net, topk_percent):
            # .at, not .loc: this is a scalar write per contingency, the same one run_contingency
            # makes, and .loc would build an indexer for it.
            net[element].at[index, "in_service"] = False  # noqa: PD008
            try:
                if self.solve(net, max_iter=max_iter, tol=tol) or _solve_with_pandapower(net):
                    _update_contingency_totals(totals, net, element, index)
            finally:
                net[element].at[index, "in_service"] = True  # noqa: PD008

        # Back to N-0: solve() rebuilds the result tables from scratch, so the aggregates have to
        # be written onto the tables the caller will actually read.
        if not self.solve(net, max_iter=max_iter, tol=tol):
            return False
        _write_contingency_totals(net, totals)
        return True

    def _push_topology(self, net: pandapowerNet) -> dict[int, int]:
        """Apply the live switch state as per-element bus assignments, and branch outages.

        :return: the assignment that was applied, for :meth:`_sync_active_buses`.
        :rtype: dict[int, int]
        """
        assignment = current_bus_assignment(net, self.plan)
        for table, terminals in _ELEMENT_TERMINALS:
            if table not in self.mirror or not len(self.mirror[table]):
                continue
            for bus_column, setter_name in terminals:
                setter = getattr(self.model, setter_name)
                live_buses = net[table][bus_column].to_numpy()
                for element, live_bus in enumerate(live_buses):
                    target = assignment.get(int(live_bus), int(live_bus))
                    key = (table, bus_column, element)
                    if self._applied_assignment.get(key) != target:
                        setter(element, self._bus_position[target])
                        self._applied_assignment[key] = target

        self._push_branch_outages(net)
        return assignment

    def _push_branch_outages(self, net: pandapowerNet) -> None:
        """Mirror the live ``in_service`` flags of lines and transformers into the model.

        Both a line-disconnection action and an N-1 contingency are expressed the same way -- as
        an ``in_service`` flag on the net -- so pushing both here is what lets
        :meth:`solve_nminus1` reuse :meth:`solve` unchanged for every contingency.
        """
        for table, (activate, deactivate) in (
            ("line", (self.model.reactivate_powerline, self.model.deactivate_powerline)),
            ("trafo", (self.model.reactivate_trafo, self.model.deactivate_trafo)),
        ):
            cached = self._branch_in_service[table]
            for element, live in enumerate(net[table]["in_service"].to_numpy()):
                if bool(live) == bool(cached[element]):
                    continue
                (activate if live else deactivate)(element)
                cached[element] = live

    def _sync_active_buses(self, net: pandapowerNet, assignment: dict[int, int]) -> None:
        """Deactivate mirror buses that currently carry no element, and reactivate the rest.

        A busbar only exists electrically while something is switched onto it. Left active and
        empty it is an isolated, injection-free node: the Newton solve goes singular and
        lightsim2grid signals that by returning an *empty* voltage vector rather than raising.
        """
        used = np.zeros(self.n_mirror_buses, dtype=bool)
        for table, terminals in _ELEMENT_TERMINALS:
            if table not in self.mirror or not len(self.mirror[table]):
                continue
            in_service = net[table]["in_service"].to_numpy() if "in_service" in net[table] else None
            for bus_column, _setter in terminals:
                for element, live_bus in enumerate(net[table][bus_column].to_numpy()):
                    if in_service is not None and not in_service[element]:
                        continue
                    target = assignment.get(int(live_bus), int(live_bus))
                    used[self._bus_position[target]] = True
        for bus in net.ext_grid["bus"].to_numpy():
            target = assignment.get(int(bus), int(bus))
            used[self._bus_position[target]] = True

        for position, should_be_active in enumerate(used):
            if bool(should_be_active) == bool(self._bus_active[position]):
                continue
            (self.model.reactivate_bus if should_be_active else self.model.deactivate_bus)(position)
            self._bus_active[position] = should_be_active

    def _push_injections(self, net: pandapowerNet) -> None:
        """Copy the timestep's load / generation setpoints into the model."""
        for element, value in enumerate(net.load["p_mw"].to_numpy()):
            self.model.change_p_load(element, float(value))
        for element, value in enumerate(net.load["q_mvar"].to_numpy()):
            self.model.change_q_load(element, float(value))
        for element, value in enumerate(net.sgen["p_mw"].to_numpy()):
            self.model.change_p_sgen(element, float(value))
        for element, value in enumerate(net.sgen["q_mvar"].to_numpy()):
            self.model.change_q_sgen(element, float(value))
        for element, value in enumerate(net.gen["p_mw"].to_numpy()):
            self.model.change_p_gen(element, float(value))
        for element, value in enumerate(net.gen["vm_pu"].to_numpy()):
            self.model.change_v_gen(element, float(value))

    def _write_results(self, net: pandapowerNet, assignment: dict[int, int]) -> None:
        """Rebuild every ``net.res_*`` table the environment reads, from the model's answer.

        The environment never calls lightsim2grid: it reads ``net.res_line`` /
        ``net.res_bus`` / ``net.res_trafo`` / ``net.res_load`` / ``net.res_gen`` /
        ``net.res_sgen`` exactly as pandapower leaves them. So this writes the same column
        layout, and nothing downstream needs to know which solver ran.

        :param net: the live expanded network, whose result tables are replaced.
        :type net: pandapowerNet
        :param assignment: the bus assignment applied for this solve.
        :type assignment: dict[int, int]
        """
        positions, live = self._resolve_bus_positions(net, assignment)
        self._write_bus_lookup(net, positions, live)
        self._write_bus_results(net, positions, live)
        self._write_line_results(net)
        self._write_trafo_results(net)
        self._write_injection_results(net)

    def _resolve_bus_positions(
        self, net: pandapowerNet, assignment: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Locate every expanded bus in the mirror, and say which of those carry a result.

        Both :meth:`_write_bus_lookup` and :meth:`_write_bus_results` need exactly this mapping
        -- one to publish the grouping, the other to read the voltages -- so it is derived once
        per solve here rather than twice.

        A bus with no mirror counterpart gets position ``-1``. Indexing ``_bus_active`` with that
        wraps to the last busbar, which is why the ``positions >= 0`` term comes first: it makes
        the ``&`` False regardless of what the wrapped read returned.

        :param net: the live expanded network, read for its bus labels.
        :type net: pandapowerNet
        :param assignment: the bus assignment applied for this solve.
        :type assignment: dict[int, int]
        :return: the mirror row position of each bus, and a mask of the ones that are live.
        :rtype: tuple[np.ndarray, np.ndarray]
        """
        positions = np.array(
            [
                self._bus_position.get(assignment.get(int(b), int(b)), -1)
                for b in net.bus.index.to_numpy()
            ],
        )
        return positions, (positions >= 0) & self._bus_active[positions]

    def _write_bus_lookup(
        self, net: pandapowerNet, positions: np.ndarray, live: np.ndarray,
    ) -> None:
        """Publish the bus grouping as ``net._pd2ppc_lookups["bus"]``.

        The graph observations fold the auxiliary busbar buses back onto the node that actually
        carries the flow, and they read that grouping out of pandapower's internal bus lookup --
        which only a pandapower solve writes. The bus assignment this backend already computes
        *is* that grouping, so publishing it keeps every graph observation working unchanged.

        Node numbering does not depend on the values written here, only on which buses share
        one: ``_compute_pp_to_canonical`` keys each group on its lowest bus label. So the node
        order matches the pandapower backend's as long as the grouping does.

        :param net: the live expanded network; ``_pd2ppc_lookups`` is replaced on it.
        :type net: pandapowerNet
        :param positions: mirror row position per bus, from :meth:`_resolve_bus_positions`.
        :type positions: np.ndarray
        :param live: which buses carry a result, from :meth:`_resolve_bus_positions`.
        :type live: np.ndarray
        """
        labels = net.bus.index.to_numpy()
        lookup = np.empty(int(labels.max()) + 1, dtype=np.int32)
        lookup[labels[live]] = positions[live]
        # Buses on a deactivated busbar are electrically isolated; pandapower gives each its own
        # ppc row, so they must not collapse into one shared group here either. The ids stay
        # consecutive in bus order, which is the order the equivalent per-bus loop assigned them.
        isolated = labels[~live]
        lookup[isolated] = self.n_mirror_buses + np.arange(len(isolated), dtype=np.int32)
        net["_pd2ppc_lookups"] = {"bus": lookup}

    def _write_bus_results(
        self, net: pandapowerNet, positions: np.ndarray, live: np.ndarray,
    ) -> None:
        """Map the mirror's bus voltages back onto every expanded bus.

        An expanded bus is electrically the mirror busbar it is switched onto, so it reports
        that busbar's voltage; a busbar that currently carries nothing was deactivated before
        the solve and reports NaN, which is what pandapower reports for an isolated bus and
        what ``BaseEnvPP._grid_is_disconnected`` looks for.

        :param net: the live expanded network; ``res_bus`` is replaced on it.
        :type net: pandapowerNet
        :param positions: mirror row position per bus, from :meth:`_resolve_bus_positions`.
        :type positions: np.ndarray
        :param live: which buses carry a result, from :meth:`_resolve_bus_positions`.
        :type live: np.ndarray
        """
        vm_mirror = self.model.get_Vm()
        va_mirror = np.rad2deg(self.model.get_Va())

        values = np.full((len(positions), len(_RES_BUS_COLUMNS)), np.nan)
        values[live, 0] = vm_mirror[positions[live]]
        values[live, 1] = va_mirror[positions[live]]
        net["res_bus"] = _result_frame(values, net.bus.index, _RES_BUS_COLUMNS)

    def _write_line_results(self, net: pandapowerNet) -> None:
        """Write branch flows, currents and thermal loading into ``net.res_line``."""
        p_or, q_or, _v_or, a_or = self.model.get_lineor_res_full()[:4]
        p_ex, q_ex, _v_ex, a_ex = self.model.get_lineex_res_full()[:4]
        # get_line*_res_full reports current in kA already, matching res_line.i_*_ka.
        i_ka = np.maximum(a_or, a_ex)
        limits = net.line["max_i_ka"].to_numpy() * net.line["df"].to_numpy() * net.line["parallel"].to_numpy()
        # A line that is switched off carries no current, and pandapower reports its loading as
        # 0, not NaN. The distinction is not cosmetic: the greedy worker maps a NaN loading to its
        # worst-case placeholder, so NaN here would score every line-disconnection action as a
        # catastrophe.
        loading = np.where(net.line["in_service"].to_numpy(), i_ka / limits * 100.0, 0.0)
        values = np.column_stack(
            (p_or, q_or, p_ex, q_ex, p_or + p_ex, q_or + q_ex, a_or, a_ex, i_ka, loading),
        )
        net["res_line"] = _result_frame(values, net.line.index, _RES_LINE_COLUMNS)

    def _write_trafo_results(self, net: pandapowerNet) -> None:
        """Write transformer flows and loading into ``net.res_trafo``.

        The loading follows pandapower's default ``trafo_loading="current"``: each side's
        current is expressed against the rating implied by ``sn_mva`` at that side's rated
        voltage, and the worse side wins.
        """
        if "trafo" not in net or not len(net.trafo):
            net["res_trafo"] = _empty_result_frame()
            return
        p_hv, q_hv, _v_hv, a_hv = self.model.get_trafohv_res_full()[:4]
        p_lv, q_lv, _v_lv, a_lv = self.model.get_trafolv_res_full()[:4]
        trafo = net.trafo
        sn_mva = trafo["sn_mva"].to_numpy()
        scale = trafo["parallel"].to_numpy() * trafo["df"].to_numpy()
        loading_hv = a_hv * trafo["vn_hv_kv"].to_numpy() * np.sqrt(3.0) / sn_mva * 100.0
        loading_lv = a_lv * trafo["vn_lv_kv"].to_numpy() * np.sqrt(3.0) / sn_mva * 100.0
        loading = np.maximum(loading_hv, loading_lv) / scale
        values = np.column_stack((
            p_hv, q_hv, p_lv, q_lv, p_hv + p_lv, q_hv + q_lv, a_hv, a_lv,
            np.where(trafo["in_service"].to_numpy(), loading, 0.0),
        ))
        net["res_trafo"] = _result_frame(values, trafo.index, _RES_TRAFO_COLUMNS)

    def _write_injection_results(self, net: pandapowerNet) -> None:
        """Write the solved load / generator / static-generator injections."""
        for table, getter in (("load", "get_loads_res"), ("sgen", "get_sgens_res")):
            if table not in net or not len(net[table]):
                net[f"res_{table}"] = _empty_result_frame()
                continue
            p_mw, q_mvar = getattr(self.model, getter)()[:2]
            net[f"res_{table}"] = _result_frame(
                np.column_stack((p_mw, q_mvar)), net[table].index, _RES_INJECTION_COLUMNS,
            )
        if "gen" not in net or not len(net.gen):
            net["res_gen"] = _empty_result_frame()
            return
        # lightsim2grid has no separate ext_grid: init_from_pandapower appends each one to the
        # generator list as a slack, so the model reports more generators than net.gen has.
        # The pandapower generators keep their order and come first.
        n_gen = len(net.gen)
        p_mw, q_mvar, vm_pu = self.model.get_gen_res()[:3]
        values = np.column_stack((p_mw[:n_gen], q_mvar[:n_gen], vm_pu[:n_gen]))
        net["res_gen"] = _result_frame(values, net.gen.index, _RES_GEN_COLUMNS)


def _contingency_cases(net: pandapowerNet, topk_percent: float) -> list[tuple[str, int]]:
    """List the outages to evaluate, as ``(element table, element index)`` pairs.

    Lines come first and transformers second -- the order
    ``pandapower.contingency.run_contingency`` walks -- because ties for the worst loading are
    resolved first-wins, so the order decides which contingency is named as the cause. Elements
    that are already out of service are skipped, as pandapower skips them.

    :param net: the live network, read for ``in_service`` and (via the top-k filter) ``res_line``.
    :type net: pandapowerNet
    :param topk_percent: percentage of lines to keep as contingencies; 100 keeps all.
    :type topk_percent: float
    :return: the contingencies in evaluation order.
    :rtype: list[tuple[str, int]]
    """
    selected = {
        "line": select_topk_line_contingencies(net, topk_percent),
        "trafo": net.trafo.index.to_numpy(),
    }
    return [
        (element, int(index))
        for element in _MONITORED_ELEMENTS
        for index in selected[element]
        if bool(net[element].at[index, "in_service"])  # noqa: PD008 -- scalar read, .loc is slower
    ]


def _solve_with_pandapower(net: pandapowerNet) -> bool:
    """Solve ``net`` with pandapower, for the contingencies lightsim2grid cannot answer.

    lightsim2grid models a single slack bus, so a contingency that splits the grid into islands
    comes back as an empty voltage vector. pandapower instead picks a generator in the island as
    its slack and returns a full result, which is the answer the pandapower N-1 backend records --
    so this is what keeps the two paths agreeing on exactly those cases.

    :param net: the network carrying the contingency, solved in place.
    :type net: pandapowerNet
    :return: True if pandapower converged; False leaves the contingency out of the aggregates.
    :rtype: bool
    """
    try:
        run_powerflow(net)
    except pp.LoadflowNotConverged:
        return False
    return True


def _new_contingency_totals(net: pandapowerNet) -> dict[str, dict[str, np.ndarray]]:
    """Allocate the per-element accumulators the N-1 sweep reduces into.

    ``max_loading_percent`` / ``min_loading_percent`` start as NaN so an element no contingency
    ever solved stays NaN, which is what the pandapower path reports for it.
    """
    totals: dict[str, dict[str, np.ndarray]] = {
        element: {
            "max_loading_percent": np.full(len(net[element]), np.nan),
            "min_loading_percent": np.full(len(net[element]), np.nan),
            "cause_element": np.empty(len(net[element]), dtype=object),
            "cause_index": np.zeros(len(net[element]), dtype=np.int64),
            "causes_overloading": np.zeros(len(net[element]), dtype=bool),
            "loading_limit_percent": _loading_limit_percent(net, element),
        }
        for element in _MONITORED_ELEMENTS
    }
    totals["bus"] = {
        "max_vm_pu": np.full(len(net.bus), np.nan),
        "min_vm_pu": np.full(len(net.bus), np.nan),
    }
    return totals


def _loading_limit_percent(net: pandapowerNet, element: str) -> np.ndarray:
    """Return the loading above which an element counts as overloaded during a contingency.

    Follows pandapower: the N-1 specific limit wins where the net states one, otherwise the
    ordinary ``max_loading_percent``, otherwise the 100% placeholder
    :func:`pandapower_env.toolbox.utils.run_nminus1_powerflow` fills in for a net that names none.
    """
    table = net[element]
    for column in ("max_loading_percent_nminus1", "max_loading_percent"):
        if column in table.columns:
            return table[column].to_numpy(dtype=float)
    return np.full(len(table), _DEFAULT_LOADING_LIMIT_PERCENT)


def _update_contingency_totals(
    totals: dict[str, dict[str, np.ndarray]],
    net: pandapowerNet,
    cause_element: str,
    cause_index: int,
) -> None:
    """Fold one solved contingency into the accumulators.

    Only elements that are in service *during this contingency* and that produced a finite result
    take part, so the outaged element itself and anything islanded by it are ignored -- the same
    ``in_service & ~isnan`` mask ``run_contingency`` applies.

    :param totals: the accumulators from :func:`_new_contingency_totals`, updated in place.
    :param net: the network carrying this contingency's solved results.
    :param cause_element: the element table whose outage produced these results.
    :param cause_index: the index of the outaged element.
    """
    for element in _MONITORED_ELEMENTS:
        table = net[element]
        if not len(table):
            continue
        entry = totals[element]
        loading = net[f"res_{element}"]["loading_percent"].to_numpy(dtype=float)
        solved = table["in_service"].to_numpy(dtype=bool) & ~np.isnan(loading)

        if np.any(loading > entry["loading_limit_percent"]):
            cause_position = net[cause_element].index.get_loc(cause_index)
            totals[cause_element]["causes_overloading"][cause_position] = True

        improves = solved & (loading > np.nan_to_num(entry["max_loading_percent"], nan=-np.inf))
        entry["cause_element"][improves] = cause_element
        entry["cause_index"][improves] = cause_index

        np.fmax(loading, entry["max_loading_percent"], out=entry["max_loading_percent"], where=solved)
        np.fmin(loading, entry["min_loading_percent"], out=entry["min_loading_percent"], where=solved)

    vm_pu = net.res_bus["vm_pu"].to_numpy(dtype=float)
    solved_bus = net.bus["in_service"].to_numpy(dtype=bool) & ~np.isnan(vm_pu)
    np.fmax(vm_pu, totals["bus"]["max_vm_pu"], out=totals["bus"]["max_vm_pu"], where=solved_bus)
    np.fmin(vm_pu, totals["bus"]["min_vm_pu"], out=totals["bus"]["min_vm_pu"], where=solved_bus)


def _write_contingency_totals(net: pandapowerNet, totals: dict[str, dict[str, np.ndarray]]) -> None:
    """Publish the accumulated N-1 aggregates as the ``res_*`` columns the environment reads."""
    for element in _MONITORED_ELEMENTS:
        if not len(net[element]):
            continue
        res_table = net[f"res_{element}"]
        for column in ("max_loading_percent", "min_loading_percent",
                       "causes_overloading", "cause_element", "cause_index"):
            res_table[column] = totals[element][column]
    net.res_bus["max_vm_pu"] = totals["bus"]["max_vm_pu"]
    net.res_bus["min_vm_pu"] = totals["bus"]["min_vm_pu"]
