from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandapower as pp
import pandapower.contingency
import pandas as pd
from pandapower.powerflow import _powerflow

if TYPE_CHECKING:
    from pandapower import pandapowerNet
logger = logging.getLogger(__name__)

# Net key marking which (pf_type, use_ls2g, init_vm_pu) the runpp options on the net were
# parsed for. Stored on the net so it lives and dies with ``net._options`` -- see
# :func:`_run_powerflow_warm`.
_WARM_OPTIONS_KEY = "_ppenv_warm_options"

def warn_unavailable_ls2g(net: pandapowerNet, pf_type: str, use_ls2g: str | bool) -> None:
    """Warn when lightsim2grid was asked for but pandapower did not use it.

    ``use_ls2g="auto"`` falls back silently in two cases -- DC power flow, and a net
    lightsim2grid cannot consume -- so the results then come from the slower native solver with
    no indication. Shared by the serial and parallel N-1 backends and by
    :func:`run_powerflow`, which each carried their own copy of these two checks.

    :param net: a solved pandapower network (``net._options`` must be populated).
    :type net: pandapowerNet
    :param pf_type: the power-flow type that ran, ``"ac"`` or ``"dc"``.
    :type pf_type: str
    :param use_ls2g: what the caller requested.
    :type use_ls2g: str | bool
    """
    if pf_type == "ac" and use_ls2g is not False and net._options["lightsim2grid"] is False:  # noqa: SLF001
        logger.warning("Warning: use_ls2g is %s, but lightsim2grid can't be used as backend.", use_ls2g)
    if pf_type == "dc" and use_ls2g:
        logger.warning("Warning: use_ls2g is %s, but lightsim2grid can't be used for DC powerflow.", use_ls2g)


def run_nminus1_powerflow(
    net: pandapowerNet,
    pf_type: str = "ac",
    use_ls2g: str | bool = "auto",
    topk_percent: float = 100.0,
) -> None:
    """
    Run n-1 powerflows.

    Runs one power flow per single-element (line / trafo / trafo3w) outage and stores the
    worst-case results on ``net`` (``res_line.max_loading_percent`` etc.) via
    ``pandapower.contingency.run_contingency``.

    For speed, the runpp options are initialised once (a single warm power flow) and every
    contingency is then evaluated through the low-level ``pandapower.powerflow._powerflow``,
    which skips pandapower's per-call option re-parsing -- the dominant cost. Results are
    identical to running ``pp.runpp`` per contingency.

    When ``topk_percent < 100`` only the most heavily loaded lines are switched off as
    contingencies (see :func:`select_topk_line_contingencies`); trafo / trafo3w contingencies
    are unaffected. All lines are still *monitored*, so ``res_line.max_loading_percent`` is
    produced for every line -- just over a smaller contingency set.

    :param net: a pandapower network
    :type net: pandapowerNet
    :param pf_type: the powerflow type, either 'ac' oder 'dc'
    :type pf_type: str
    :param use_ls2g: Whether lightsim2grid should be used as backend or not.
    :type use_ls2g: str | bool
    :param topk_percent: percentage of lines to evaluate as N-1 contingencies (default 100 = all)
    :type topk_percent: float
    """
    if pf_type not in {"ac", "dc"}:
        msg = "pf_type must be 'ac' or 'dc'."
        raise ValueError(msg)

    if use_ls2g != "auto" and not isinstance(use_ls2g, bool):
        msg = "use_ls2g must be bool or 'auto'."
        raise ValueError(msg)

    nminus1_cases = {
        "line": {"index": net.line.index.to_numpy()},
        "trafo": {"index": net.trafo.index.to_numpy()},
        "trafo3w": {"index": net.trafo3w.index.to_numpy()},
    }

    # Initialise the runpp options once; every contingency then reuses them via _powerflow.
    base_powerflow = pp.runpp if pf_type == "ac" else pp.rundcpp
    base_powerflow(net, lightsim2grid=use_ls2g)

    # Restrict the line contingencies to the top-k% most loaded lines (needs the N-0 result
    # from the warm power flow above). A no-op at the default 100%.
    nminus1_cases["line"]["index"] = select_topk_line_contingencies(net, topk_percent)

    def evaluate_contingency(net: pandapowerNet, **_kwargs: object) -> None:
        # Options are already set by the warm power flow above; skip re-parsing them.
        _powerflow(net)

    # run_contingency monitors loading of every element type present, which needs a
    # max_loading_percent column. Add it temporarily where missing so trafo/trafo3w
    # contingencies don't raise (and log-spam) a swallowed KeyError; restore net afterwards.
    elements_missing_limit = [
        element
        for element in ("trafo", "trafo3w")
        if len(net[element]) and "max_loading_percent" not in net[element].columns
    ]
    for element in elements_missing_limit:
        net[element]["max_loading_percent"] = 100.0

    try:
        pp.contingency.run_contingency(
            net=net,
            nminus1_cases=nminus1_cases,
            contingency_evaluation_function=evaluate_contingency,
        )
    finally:
        for element in elements_missing_limit:
            del net[element]["max_loading_percent"]

    warn_unavailable_ls2g(net, pf_type, use_ls2g)


def select_topk_line_contingencies(net: pandapowerNet, topk_percent: float = 100.0) -> np.ndarray:
    """Select the lines to switch off as N-1 contingencies: the top ``topk_percent`` % by load.

    Lines are ranked by their N-0 apparent power flow ``S = sqrt(P^2 + Q^2)`` in MVA (the max of
    the from/to ends); the ``ceil(topk_percent/100 * n_lines)`` highest are kept (at least one
    whenever any line exists). This trims N-1 cost: only the most heavily loaded lines -- whose
    outage redistributes the most flow -- are evaluated. Both N-1 backends call this after their
    warm base power flow, so the same lines are selected serially and in parallel.

    :param net: a pandapower network with a solved N-0 power flow (reads ``net.res_line``)
    :type net: pandapowerNet
    :param topk_percent: percentage of lines to keep; ``>= 100`` (or no lines) keeps all
    :type topk_percent: float
    :return: the kept line indices, sorted ascending (preserving contingency evaluation order)
    :rtype: np.ndarray
    :raises ValueError: if ``topk_percent <= 0``
    """
    line_index = net.line.index.to_numpy()
    if topk_percent >= 100.0 or len(line_index) == 0:  # noqa: PLR2004
        return line_index
    if topk_percent <= 0.0:
        msg = "topk_percent must be in (0, 100]."
        raise ValueError(msg)

    n_keep = int(np.ceil(len(line_index) * topk_percent / 100.0))
    apparent_power = _line_apparent_power(net.res_line.loc[line_index])
    top_positions = np.argsort(apparent_power, kind="stable")[::-1][:n_keep]
    return np.sort(line_index[top_positions])


def _line_apparent_power(res_line: pd.DataFrame) -> np.ndarray:
    """Per-line apparent power ``S = sqrt(P^2 + Q^2)`` in MVA, the larger of the from/to ends.

    NaN reactive power (DC power flow) counts as 0; a fully-NaN row (e.g. an out-of-service
    line) becomes ``-inf`` so it ranks last and is never selected while lines remain.
    """
    s_from = np.hypot(res_line["p_from_mw"].to_numpy(), np.nan_to_num(res_line["q_from_mvar"].to_numpy(), nan=0.0))
    s_to = np.hypot(res_line["p_to_mw"].to_numpy(), np.nan_to_num(res_line["q_to_mvar"].to_numpy(), nan=0.0))
    return np.nan_to_num(np.fmax(s_from, s_to), nan=-np.inf)


def run_powerflow(
    net: pandapowerNet,
    pf_type: str = "ac",
    use_ls2g: str | bool = "auto",
) -> None:
    """Run the powerflow.

    Roughly two thirds of a ``pp.runpp`` call is pandapower re-parsing its options
    (``_init_runpp_options`` / ``_check_lightsim2grid_compatibility``, which run several
    ``DataFrame.query`` calls each). Once those options are on the net, repeating the power
    flow through the low-level :func:`pandapower.powerflow._powerflow` skips that work and is
    ~2.8x faster. ``_powerflow`` re-derives the internal ppc from the live net tables on every
    call, so switch positions, ``in_service`` flags and tap changes are always picked up --
    only the parsed *options* are reused. See :func:`_run_powerflow_warm` for when they are
    re-parsed. Results are identical to calling ``pp.runpp`` every time.

    :param net: a pandapower network
    :type net: pandapowerNet
    :param pf_type: the powerflow type, either 'ac' oder 'dc'
    :type pf_type: str
    :param use_ls2g: Whether lightsim2grid should be used as backend or not.
    :type use_ls2g: str | bool
    """
    if pf_type not in {"ac", "dc"}:
        msg = "pf_type must be 'ac' or 'dc'."
        raise ValueError(msg)

    if use_ls2g != "auto" and not isinstance(use_ls2g, bool):
        msg = "use_ls2g must be bool or 'auto'."
        raise ValueError(msg)

    _run_powerflow_warm(net, pf_type, use_ls2g)

    warn_unavailable_ls2g(net, pf_type, use_ls2g)


def _init_vm_pu_signature(net: pandapowerNet) -> float:
    """Return the flat-start voltage pandapower would derive from the current net.

    Under the default ``init="auto"``, ``_init_runpp_options`` sets ``init_vm_pu`` to the mean
    ``vm_pu`` of the in-service ``gen`` and ``ext_grid`` elements. Reusing warm options freezes
    that value, so a net whose generator voltage setpoints change (a ``gen_vm`` profile) would
    keep starting the Newton-Raphson iteration from a stale point. That still converges to the
    same solution, but not to the same last bits -- so this signature is part of the warm-options
    key and any change to it forces a re-parse. Costs ~120 us against the ~15 ms it protects.

    :param net: a pandapower network
    :type net: pandapowerNet
    :return: The mean in-service gen/ext_grid voltage setpoint, or 1.0 if there are none.
    :rtype: float
    """
    gen_setpoints = net.gen["vm_pu"].to_numpy()[net.gen["in_service"].to_numpy()]
    ext_grid_setpoints = net.ext_grid["vm_pu"].to_numpy()[net.ext_grid["in_service"].to_numpy()]
    n_setpoints = len(gen_setpoints) + len(ext_grid_setpoints)
    if not n_setpoints:
        return 1.0
    return float((gen_setpoints.sum() + ext_grid_setpoints.sum()) / n_setpoints)


def _run_powerflow_warm(net: pandapowerNet, pf_type: str, use_ls2g: str | bool) -> None:
    """Run one power flow, reusing already-parsed runpp options when it is safe to do so.

    The options are re-parsed (a full ``pp.runpp`` / ``pp.rundcpp``) whenever anything they
    depend on may have changed: a different ``pf_type`` or ``use_ls2g`` than they were parsed
    for, a different flat-start voltage (see :func:`_init_vm_pu_signature`), missing
    ``net._options``, or an ``_options`` whose AC/DC mode disagrees with ``pf_type`` -- which
    catches a foreign ``pp.rundcpp`` call on the same net. Otherwise the cheap
    :func:`pandapower.powerflow._powerflow` path is taken.

    The marker is stored *on the net* so it shares the exact lifetime of ``net._options``: both
    survive ``copy.deepcopy`` and ``pickle`` together, and both are dropped by the
    ``to_json``/``from_json`` roundtrip that ``greedy_worker`` and ``nminus1_parallel`` use to
    ship nets into child processes. A net therefore can never carry a marker without the options
    it describes.

    ``use_ls2g`` is compared as the value the caller *requested*, not the value pandapower
    resolved it to, so switching ``"auto"`` -> ``True`` on an ls2g-incompatible net still misses
    the cache, takes the full path, and raises exactly as it does today.

    :param net: a pandapower network, modified in place
    :type net: pandapowerNet
    :param pf_type: the powerflow type, either 'ac' or 'dc'
    :type pf_type: str
    :param use_ls2g: Whether lightsim2grid should be used as backend or not.
    :type use_ls2g: str | bool
    """
    key = (pf_type, use_ls2g, _init_vm_pu_signature(net))
    options = net.get("_options")

    if (
        net.get(_WARM_OPTIONS_KEY) == key
        and options is not None
        and options.get("ac") == (pf_type == "ac")
    ):
        _powerflow(net)
        return

    base_powerflow = pp.runpp if pf_type == "ac" else pp.rundcpp
    base_powerflow(net, lightsim2grid=use_ls2g)
    net[_WARM_OPTIONS_KEY] = key


def _bus_vn_kv(net: pandapowerNet, bus_labels: np.ndarray) -> np.ndarray:
    """Nominal voltage of each named bus, avoiding a label lookup when labels are positions.

    ``net.bus["vn_kv"].loc[bus_labels]`` costs ~221 us on case30 -- pandas builds a full
    reindexer for what is, in every pandapower net this package produces, a plain positional
    take. Checking that the bus index really is ``0..n-1`` costs ~11 us and makes the take
    (~5 us) safe, so the common path is ~11x cheaper; a net indexed any other way still goes
    through ``.loc`` and gets the same answer. This runs twice per environment step via
    :func:`total_active_overload_mva`.

    :param net: a pandapower network.
    :type net: pandapowerNet
    :param bus_labels: bus *labels* (as stored in ``line.from_bus`` and friends).
    :type bus_labels: np.ndarray
    :return: the nominal voltage in kV of each named bus, in the order given.
    :rtype: np.ndarray
    """
    vn_kv = net.bus["vn_kv"]
    if net.bus.index.equals(pd.RangeIndex(len(net.bus))):
        return vn_kv.to_numpy()[bus_labels]
    return vn_kv.loc[bus_labels].to_numpy()


def total_active_overload_mva(
    net: pandapowerNet,
) -> float:
    """
    Total ACTIVE POWER overload (MW) in the network for a single snapshot.

    Parameters
    ----------
    net : pandapowerNet

    Returns
    -------
    float
        Total active power overload in MW (sum over all elements).
        0.0 if no element is overloaded.
    """
    # Make sure results are available. Routed through run_powerflow, not a plain pp.runpp: a
    # bare runpp re-parses the options onto the net without touching the warm-options marker,
    # so the *next* run_powerflow would take the warm path and silently reuse foreign options.
    if net.res_line is None or net.res_line.empty or \
       (len(net.trafo) and (net.res_trafo is None or net.res_trafo.empty)):
        run_powerflow(net)

    total_overload_mva = 0.0

    # Lines: rating in MVA from Imax & nominal voltage. The derating factor ``df`` and the
    # number of parallel systems must be included: pandapower's own ``loading_percent`` divides
    # by ``max_i_ka * df * parallel``, so leaving them out overstates the overload of a derated
    # line and understates it for a double circuit. Both default to 1 in most grids, which is
    # why this went unnoticed on case30.
    if len(net.line):
        line = net.line
        max_i_ka = line.max_i_ka.to_numpy() * line.df.to_numpy() * line.parallel.to_numpy()
        rating_mva = np.sqrt(3.0) * _bus_vn_kv(net, line.from_bus.to_numpy()) * max_i_ka
        total_overload_mva += _overload_sum(
            _worst_apparent_power(net.res_line, (("p_from_mw", "q_from_mvar"), ("p_to_mw", "q_to_mvar"))),
            rating_mva,
        )

    # 2W transformers: rated apparent power scaled by parallel systems and the derating factor
    # for the same reason -- this is the denominator of res_trafo.loading_percent.
    if len(net.trafo):
        trafo = net.trafo
        rating_mva = trafo.sn_mva.to_numpy() * trafo.parallel.to_numpy() * trafo.df.to_numpy()
        total_overload_mva += _overload_sum(
            _worst_apparent_power(net.res_trafo, (("p_hv_mw", "q_hv_mvar"), ("p_lv_mw", "q_lv_mvar"))),
            rating_mva,
        )

    # 3W transformers: unlike the two above, every winding is rated separately and their
    # overloads add up, rather than the worst end being taken.
    if len(net.trafo3w):
        trafo = net.trafo3w
        res_tr = net.res_trafo3w
        windings = (("p_hv_mw", "q_hv_mvar", "sn_hv_mva"),
                    ("p_mv_mw", "q_mv_mvar", "sn_mv_mva"),
                    ("p_lv_mw", "q_lv_mvar", "sn_lv_mva"))
        # Summed elementwise first and reduced once, matching the original operation order --
        # summing each winding separately would round differently.
        overload_tr_mva = sum(
            np.maximum(0.0, _apparent_power(res_tr, p_col, q_col) - trafo[sn_col].to_numpy())
            for p_col, q_col, sn_col in windings
        )
        total_overload_mva += float(np.sum(overload_tr_mva))

    return total_overload_mva


def _apparent_power(res: pd.DataFrame, p_column: str, q_column: str) -> np.ndarray:
    """Apparent power ``sqrt(P^2 + Q^2)`` in MVA at one element end.

    Deliberately not ``np.hypot``: it is the numerically safer spelling but not the
    bit-identical one, and these values feed rewards that must not drift.
    """
    return np.sqrt(res[p_column].to_numpy() ** 2 + res[q_column].to_numpy() ** 2)


def _worst_apparent_power(res: pd.DataFrame, ends: tuple[tuple[str, str], ...]) -> np.ndarray:
    """Largest apparent power across an element's ends, per element."""
    return np.maximum.reduce([_apparent_power(res, p_col, q_col) for p_col, q_col in ends])


def _overload_sum(actual_mva: np.ndarray, rating_mva: np.ndarray) -> float:
    """Total MVA by which elements exceed their rating; 0.0 when none do."""
    return float(np.sum(np.maximum(0.0, actual_mva - rating_mva)))
