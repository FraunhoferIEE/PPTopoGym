"""Tests for the opt-in lightsim2grid backend.

The contract this backend is held to is *not* byte parity -- it assembles the model differently --
but **tolerance parity** against the pandapower path, and **decision parity**: the same actions
must converge and the same lines must come out overloaded. Those are the two ways a wrong
topology translation would actually hurt an agent.

The interesting case throughout is a **split** substation: the fully coupled state is easy and
would pass even with the bus assignment ignored entirely.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest

from pandapower_env.toolbox.ls2g_backend import (
    # The oracle must walk exactly the terminals the backend does, so it shares the table.
    _ELEMENT_TERMINALS as ELEMENT_TERMINALS,
)
from pandapower_env.toolbox.ls2g_backend import (
    LightsimBackend,
    build_mirror_net,
    current_bus_assignment,
)

# The mirror solves the same physics through a different Newton implementation; agreement is
# ~1e-11 in practice, so this leaves three orders of magnitude of headroom.
LOADING_TOLERANCE = 1e-6


def test_mirror_net_drops_the_switch_layer(test_grid_dbb_plus_simbench) -> None:
    """The mirror keeps the busbar buses and loses every auxiliary bus and switch."""
    net = test_grid_dbb_plus_simbench
    mirror, _plan = build_mirror_net(net)

    assert len(mirror.switch) == 0, "lightsim2grid ignores switches; the mirror must have none"
    assert len(mirror.bus) < len(net.bus), "auxiliary busbar buses should be gone"
    # Every busbar bus of every substation survives -- they are what elements attach to.
    busbar_columns = [c for c in net.multi_bb_substation.columns if c.startswith("bus_")]
    for _, row in net.multi_bb_substation.iterrows():
        for column in busbar_columns:
            if not np.isscalar(row[column]) or row[column] == row[column]:  # not NaN
                assert int(row[column]) in mirror.bus.index


def test_mirror_reproduces_the_coupled_grid(test_grid_dbb_plus_simbench) -> None:
    """With every coupler closed, the mirror is electrically the same grid."""
    net = test_grid_dbb_plus_simbench
    pp.runpp(net)
    mirror, _plan = build_mirror_net(net)
    pp.runpp(mirror)

    assert np.allclose(
        net.res_line["p_from_mw"].to_numpy(),
        mirror.res_line["p_from_mw"].to_numpy(),
        atol=1e-9,
    )


def test_assignment_follows_the_closed_switch(test_grid_dbb_plus_simbench) -> None:
    """An element moves to busbar 1 exactly when its b1 switch is the closed one."""
    net = test_grid_dbb_plus_simbench
    _mirror, plan = build_mirror_net(net)
    row = net.multi_bb_substation.iloc[0]

    # Split the substation: open the coupler, move element 0 to busbar 1.
    net.switch.loc[int(row["b01_switch"]), "closed"] = False
    net.switch.loc[int(row["b0_switches"][0]), "closed"] = False
    net.switch.loc[int(row["b1_switches"][0]), "closed"] = True

    assignment = current_bus_assignment(net, plan)
    assert assignment[int(row["connected_buses"][0])] == int(row["bus_1"])
    # An untouched element stays on busbar 0.
    assert assignment[int(row["connected_buses"][1])] == int(row["bus_0"])


def test_closed_coupler_collapses_both_busbars(test_grid_dbb_plus_simbench) -> None:
    """A closed coupler makes the busbars one node, so everything reports busbar 0."""
    net = test_grid_dbb_plus_simbench
    _mirror, plan = build_mirror_net(net)
    row = net.multi_bb_substation.iloc[0]

    net.switch.loc[int(row["b01_switch"]), "closed"] = True
    net.switch.loc[int(row["b1_switches"][0]), "closed"] = True

    assignment = current_bus_assignment(net, plan)
    assert assignment[int(row["connected_buses"][0])] == int(row["bus_0"])


@pytest.mark.parametrize("action", [0, 1, 2, 3])
def test_backend_matches_pandapower_over_actions(simenv30, action: int) -> None:
    """Tolerance + decision parity against the pandapower solver, split substations included."""
    env = simenv30
    env.reset(options={"index": 12})
    backend = LightsimBackend(env.net)

    env.reset(options={"index": 12})
    env.step(action)
    if not env.net.converged:
        pytest.skip(f"action {action} does not converge on the pandapower path")

    reference = env.net.res_line["loading_percent"].to_numpy().copy()
    assert backend.solve(env.net), "backend diverged where pandapower converged"
    produced = env.net.res_line["loading_percent"].to_numpy()

    assert np.nanmax(np.abs(reference - produced)) < LOADING_TOLERANCE
    # Decision parity: the same lines are reported overloaded.
    assert np.array_equal(reference > 100.0, produced > 100.0)  # noqa: PLR2004


def test_backend_rebuilds_every_result_table(simenv30) -> None:
    """The write-back covers the tables the observations read, not just ``res_line``.

    A missing table is not a small error: ``res_bus.vm_pu`` is what the environment's
    disconnection check reads, so an empty one would report a healthy grid as islanded.
    """
    env = simenv30
    env.reset(options={"index": 12})
    backend = LightsimBackend(env.net)

    env.reset(options={"index": 12})
    env.step(0)
    columns = (("res_bus", "vm_pu"), ("res_gen", "p_mw"), ("res_load", "p_mw"))
    expected = {table: env.net[table][column].to_numpy().copy() for table, column in columns}
    assert backend.solve(env.net)

    for table, column in columns:
        produced = env.net[table][column].to_numpy()
        assert len(produced) == len(expected[table]), f"{table} lost rows"
        assert np.allclose(produced, expected[table], atol=1e-6, equal_nan=True), table
    # The graph observations need the bus grouping, which only a pandapower solve normally writes.
    assert len(env.net["_pd2ppc_lookups"]["bus"]) == len(env.net.bus)


def test_env_on_lightsim_backend_scores_the_same_rewards(simenv30) -> None:
    """A whole environment configured onto the backend agrees with the pandapower one.

    This is the end-to-end version of the parity check: it exercises ``run_pf``, the reward and
    the observation build, not just the solver call.
    """
    reference = simenv30
    config = dict(reference.orig_config)
    config["backend"] = "lightsim"
    from pandapower_env.environments.simulation_env import PPTopoGym
    lightsim_env = PPTopoGym(config)

    for action in (0, 5, 7):
        reference.reset(options={"index": 12})
        _obs, expected_reward, *_ = reference.step(action)
        lightsim_env.reset(options={"index": 12})
        _obs, reward, *_ = lightsim_env.step(action)
        assert reward == pytest.approx(expected_reward, abs=1e-6), f"action {action}"


def bus_lookup_by_loop(backend: LightsimBackend, net, assignment: dict[int, int]) -> np.ndarray:
    """Build the bus lookup one bus at a time -- the oracle the vectorized writer must match.

    This is the implementation ``_write_bus_lookup`` replaced, kept here rather than in the
    backend because the isolated-bus numbering it defines is easy to get subtly wrong: the ids
    must stay consecutive *in bus order*, which a vectorized rewrite only preserves as long as
    it does not reorder the masked labels.
    """
    labels = net.bus.index.to_numpy()
    lookup = np.empty(int(labels.max()) + 1, dtype=np.int32)
    next_isolated = backend.n_mirror_buses
    for label in labels:
        target = assignment.get(int(label), int(label))
        position = backend._bus_position.get(target, -1)
        if position < 0 or not backend._bus_active[position]:
            lookup[label] = next_isolated
            next_isolated += 1
        else:
            lookup[label] = position
    return lookup


def test_absent_element_tables_are_empty_and_never_shared(simenv30) -> None:
    """A table the net has no elements for is written as a fresh, genuinely empty frame.

    ``_empty_result_frame`` copies a module-level prototype instead of calling
    ``pd.DataFrame()``, which is 15x cheaper. Two things have to survive that: the frame must
    still compare equal to what ``pd.DataFrame()`` builds (shape, index, columns, dtypes), and
    each solve must hand out its **own** object -- sharing the prototype would let a caller that
    mutates last step's table corrupt every later solve.
    """
    import pandas as pd

    env = simenv30
    env.reset(options={"index": 12})
    backend = LightsimBackend(env.net)
    assert not len(env.net.trafo), "case30 is the no-transformer case this test relies on"

    assert backend.solve(env.net)
    first = env.net["res_trafo"]
    assert first.equals(pd.DataFrame())
    assert first.empty

    assert backend.solve(env.net)
    second = env.net["res_trafo"]
    assert second.equals(pd.DataFrame())
    assert second is not first, "each solve must publish its own frame"

    # Mutating one solve's table must not reach the next one.
    first["injected"] = []
    assert backend.solve(env.net)
    assert "injected" not in env.net["res_trafo"].columns


class PerElementPushOracle:
    """Push topology into a ``GridModel`` one element at a time -- the loop the fast path replaced.

    Kept here rather than in the backend because what has to be proven is a *state* equality:
    the vectorized push may call the C++ setters in a different order and skip elements whose
    target is unchanged, so the only meaningful oracle is "the model ends up in the same state".
    It carries its own applied-assignment / active-bus memory so it can drive a second
    ``GridModel`` independently of the backend under test.

    :param backend: the backend whose mirror, plan and model this oracle drives.
    :type backend: LightsimBackend
    """

    def __init__(self, backend: LightsimBackend) -> None:
        self.backend = backend
        self.applied: dict[tuple, int] = {}
        self.bus_active = np.ones(backend.n_mirror_buses, dtype=bool)
        self.branch_in_service = {
            table: backend.mirror[table]["in_service"].to_numpy(dtype=bool).copy()
            for table in ("line", "trafo")
        }

    def push(self, net) -> None:
        """Apply the live topology, branch outages and bus activation, per element."""
        assignment = current_bus_assignment(net, self.backend.plan)
        self._push_terminals(net, assignment)
        self._push_branch_outages(net)
        self._sync_active_buses(net, assignment)

    def _push_terminals(self, net, assignment: dict[int, int]) -> None:
        for table, terminals in ELEMENT_TERMINALS:
            if table not in self.backend.mirror or not len(self.backend.mirror[table]):
                continue
            for bus_column, setter_name in terminals:
                setter = getattr(self.backend.model, setter_name)
                for element, live_bus in enumerate(net[table][bus_column].to_numpy()):
                    target = assignment.get(int(live_bus), int(live_bus))
                    key = (table, bus_column, element)
                    if self.applied.get(key) != target:
                        setter(element, self.backend._bus_position[target])
                        self.applied[key] = target

    def _push_branch_outages(self, net) -> None:
        model = self.backend.model
        for table, (activate, deactivate) in (
            ("line", (model.reactivate_powerline, model.deactivate_powerline)),
            ("trafo", (model.reactivate_trafo, model.deactivate_trafo)),
        ):
            cached = self.branch_in_service[table]
            for element, live in enumerate(net[table]["in_service"].to_numpy()):
                if bool(live) == bool(cached[element]):
                    continue
                (activate if live else deactivate)(element)
                cached[element] = live

    def _sync_active_buses(self, net, assignment: dict[int, int]) -> None:
        backend = self.backend
        used = np.zeros(backend.n_mirror_buses, dtype=bool)
        for table, terminals in ELEMENT_TERMINALS:
            if table not in backend.mirror or not len(backend.mirror[table]):
                continue
            in_service = (
                net[table]["in_service"].to_numpy() if "in_service" in net[table] else None
            )
            for bus_column, _setter in terminals:
                for element, live_bus in enumerate(net[table][bus_column].to_numpy()):
                    if in_service is not None and not in_service[element]:
                        continue
                    target = assignment.get(int(live_bus), int(live_bus))
                    used[backend._bus_position[target]] = True
        for bus in net.ext_grid["bus"].to_numpy():
            target = assignment.get(int(bus), int(bus))
            used[backend._bus_position[target]] = True

        for position, should_be_active in enumerate(used):
            if bool(should_be_active) == bool(self.bus_active[position]):
                continue
            model = backend.model
            (model.reactivate_bus if should_be_active else model.deactivate_bus)(position)
            self.bus_active[position] = should_be_active


TERMINAL_GETTERS = (
    ("line_or", "line", "get_bus_powerline_or"),
    ("line_ex", "line", "get_bus_powerline_ex"),
    ("trafo_hv", "trafo", "get_bus_trafo_hv"),
    ("trafo_lv", "trafo", "get_bus_trafo_lv"),
    ("load", "load", "get_bus_load"),
    ("sgen", "sgen", "get_bus_sgen"),
    ("gen", "gen", "get_bus_gen"),
    ("shunt", "shunt", "get_bus_shunt"),
)


def model_topology_state(backend: LightsimBackend) -> dict[str, np.ndarray]:
    """Read back everything the topology push writes into a ``GridModel``.

    The per-terminal getters are scalar (``get_bus_powerline_or(i)``), so they are walked
    element by element; the status getters are already vectorized.

    :param backend: the backend whose model and mirror tables are read.
    :type backend: LightsimBackend
    :return: per-element bus assignments plus line / trafo / bus activation status.
    :rtype: dict[str, np.ndarray]
    """
    model = backend.model
    state: dict[str, np.ndarray] = {}
    for key, table, getter_name in TERMINAL_GETTERS:
        count = len(backend.mirror[table]) if table in backend.mirror else 0
        getter = getattr(model, getter_name)
        state[key] = np.array([getter(element) for element in range(count)], dtype=np.int64)
    state["line_status"] = np.asarray(model.get_lines_status())
    state["trafo_status"] = np.asarray(model.get_trafo_status())
    state["bus_status"] = np.asarray(model.get_bus_status())
    return state


@pytest.mark.parametrize("action", [0, 1, 2, 3, 5, 7])
def test_vectorized_push_matches_the_per_element_oracle(simenv30, action: int) -> None:
    """The vectorized topology push leaves the model in exactly the per-element loop's state.

    Two independent ``GridModel``s are driven over the *same* action from the same starting
    point -- one by the backend's fast path, one by :class:`PerElementPushOracle`. Every bus
    assignment, every in-service flag and every bus activation must match exactly, and so must
    the voltages the two models then solve. A split substation is the case that matters: it is
    what moves a terminal onto busbar 1 and what leaves a busbar empty.
    """
    env = simenv30
    env.reset(options={"index": 12})
    fast = LightsimBackend(env.net)
    oracle_backend = LightsimBackend(env.net)
    oracle = PerElementPushOracle(oracle_backend)

    env.reset(options={"index": 12})
    env.step(action)

    fast.solve(env.net)
    oracle.push(env.net)
    oracle_backend._push_injections(env.net)

    produced = model_topology_state(fast)
    expected = model_topology_state(oracle_backend)
    for key, expected_values in expected.items():
        np.testing.assert_array_equal(produced[key], expected_values, err_msg=key)
    np.testing.assert_array_equal(fast._bus_active, oracle.bus_active)

    # Identical model state must give bit-identical voltages, not merely close ones.
    oracle_backend.model.ac_pf(np.ones(oracle_backend.n_mirror_buses, dtype=complex), 20, 1e-8)
    np.testing.assert_array_equal(fast.model.get_Vm(), oracle_backend.model.get_Vm())
    np.testing.assert_array_equal(fast.model.get_Va(), oracle_backend.model.get_Va())


def test_vectorized_push_tracks_repeated_actions(simenv30) -> None:
    """The cached applied-assignment memory stays correct across a sequence of actions.

    The fast path only calls a setter when an element's target *changed*, so a stale cache
    would show up not on the first push but on a later one. This walks several actions through
    one backend and re-checks against a fresh oracle each time.
    """
    env = simenv30
    env.reset(options={"index": 12})
    fast = LightsimBackend(env.net)
    oracle_backend = LightsimBackend(env.net)
    oracle = PerElementPushOracle(oracle_backend)

    for action in (0, 5, 7, 0, 5, 1):
        env.reset(options={"index": 12})
        env.step(action)
        fast.solve(env.net)
        oracle.push(env.net)

        produced = model_topology_state(fast)
        expected = model_topology_state(oracle_backend)
        for key, expected_values in expected.items():
            np.testing.assert_array_equal(produced[key], expected_values,
                                          err_msg=f"action {action}: {key}")
        np.testing.assert_array_equal(fast._bus_active, oracle.bus_active)


@pytest.mark.parametrize("action", [0, 1, 2, 3, 5, 7])
def test_bus_lookup_matches_the_per_bus_oracle(simenv30, action: int) -> None:
    """The vectorized bus lookup is element-for-element what the per-bus loop produced.

    Split substations are the interesting case: they are what puts buses on a second busbar and
    what leaves busbars empty, so they exercise both branches of the isolated-bus numbering.
    """
    env = simenv30
    env.reset(options={"index": 12})
    backend = LightsimBackend(env.net)

    env.reset(options={"index": 12})
    env.step(action)
    if not backend.solve(env.net):
        pytest.skip(f"action {action} does not converge on the backend")

    assignment = current_bus_assignment(env.net, backend.plan)
    produced = env.net["_pd2ppc_lookups"]["bus"]
    expected = bus_lookup_by_loop(backend, env.net, assignment)

    # Only the entries the buses actually name are written; the gaps stay uninitialised.
    labels = env.net.bus.index.to_numpy()
    np.testing.assert_array_equal(produced[labels], expected[labels])
    assert produced.dtype == expected.dtype


def test_a_slack_on_an_auxiliary_bus_is_rewired_into_the_mirror(test_grid_dbb_plus_simbench) -> None:
    """An ``ext_grid`` on an auxiliary bus follows its substation onto busbar 0.

    The rewiring loop originally covered only ``_ELEMENT_TERMINALS``, which has no ``ext_grid``
    entry, so a slack inside a double-busbar substation kept a bus the mirror then deleted and
    ``init_from_pandapower`` died on the slack lookup. case118 is the real instance (``ext_grid``
    on bus 384); here the fixture's slack is moved onto an auxiliary bus to reproduce it.
    """
    net = test_grid_dbb_plus_simbench
    auxiliary = [
        int(bus)
        for _, row in net.multi_bb_substation.iterrows()
        for bus in row["connected_buses"]
    ]
    net.ext_grid.loc[net.ext_grid.index[0], "bus"] = auxiliary[0]

    mirror, _plan = build_mirror_net(net)

    slack_bus = int(mirror.ext_grid["bus"].iloc[0])
    assert slack_bus not in auxiliary, "the slack must not keep an auxiliary bus"
    assert slack_bus in mirror.bus.index, "the slack's bus must survive into the mirror"
    # The whole point: the model can now be built at all.
    assert LightsimBackend(net) is not None
