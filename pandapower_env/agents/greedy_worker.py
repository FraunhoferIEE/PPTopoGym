from __future__ import annotations

import contextlib
import io
import logging
from typing import TYPE_CHECKING, Generator

import numpy as np
import pandapower as pp

# Use your existing utils
from pandapower_env.toolbox.utils import run_nminus1_powerflow, run_powerflow

if TYPE_CHECKING:
    import pandas as pd

    from pandapower_env.toolbox.ls2g_backend import LightsimBackend

# Process-local cache of the deserialized net, keyed on the blob it was built from.
_NET: pp.pandapowerNet | None = None
_NET_KEY: int | None = None
# The lightsim2grid backend built from that net, cached alongside it: constructing the mirror
# net and the GridModel is the expensive part, and it is valid for as long as the net is.
_LS2G: object | None = None




@contextlib.contextmanager
def _suppress_pp_output(level: int=logging.ERROR) -> Generator[None, None, None]:
    """Temporarily silence pandapower logs and stdout/stderr."""
    pp_logger = logging.getLogger("pandapower")
    prev_level = pp_logger.level
    pp_logger.setLevel(level)
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            yield
        finally:
            pp_logger.setLevel(prev_level)

def _ensure_net_from_blob(static_net_blob: bytes) -> pp.pandapowerNet:
    """
    Build (or reuse) a process-local pandapowerNet from a serialized JSON blob.

    The deserialized net is cached per process and reused while the *same* blob is
    passed -- the common case, where every action in one ``act()`` call shares one grid.
    A *different* blob (e.g. another grid evaluated later in the same process) rebuilds
    the net: keying only on ``_NET is None`` would wrongly keep the first grid ever seen
    and then apply a mismatched topology snapshot to it (a length-mismatch crash).

    Parameters
    ----------
    static_net_blob : bytes

    Returns
    -------
    net : pp.pandapowerNet
    """
    global _NET, _NET_KEY, _LS2G #noqa: PLW0603
    key = hash(static_net_blob)
    if _NET is None or key != _NET_KEY:
        json_str = static_net_blob.decode("utf-8")
        with io.StringIO(json_str) as fp:
            _NET = pp.from_json(fp)
        _NET_KEY = key
        _LS2G = None
    return _NET


def _ensure_lightsim_backend(net: pp.pandapowerNet) -> LightsimBackend:
    """Build (or reuse) the process-local lightsim2grid backend for the cached net.

    Building it means constructing the switch-free mirror net and the ``GridModel``, which costs
    far more than a solve; the backend is therefore kept for as long as the net it mirrors is,
    and dropped by :func:`_ensure_net_from_blob` when a different grid arrives.

    :param net: the worker's process-local network.
    :type net: pp.pandapowerNet
    :return: the backend that solves that network.
    :rtype: LightsimBackend
    """
    global _LS2G #noqa: PLW0603
    if _LS2G is None:
        from pandapower_env.toolbox.ls2g_backend import LightsimBackend
        _LS2G = LightsimBackend(net)
    return _LS2G  # type: ignore[return-value]


def _apply_topology(net: pp.pandapowerNet, topo: dict[str, np.ndarray]) -> None:
    if "switch_closed" in topo and len(net.switch):
        net.switch["closed"] = topo["switch_closed"].astype(bool)
    if "line_in_service" in topo and len(net.line):
        net.line["in_service"] = topo["line_in_service"].astype(bool)
    if "trafo_tap_pos" in topo and len(net.trafo):
        net.trafo["tap_pos"] = topo["trafo_tap_pos"].astype(int)

def _inject_profile(net: pp.pandapowerNet, prof: dict[str, np.ndarray]) -> None:
    if "load_p_mw" in prof and len(net.load):
        net.load["p_mw"] = prof["load_p_mw"]
    if "load_q_mvar" in prof and len(net.load):
        net.load["q_mvar"] = prof["load_q_mvar"]
    if "sgen_p_mw" in prof and len(net.sgen):
        net.sgen["p_mw"] = prof["sgen_p_mw"]
    if "sgen_q_mvar" in prof and len(net.sgen):
        net.sgen["q_mvar"] = prof["sgen_q_mvar"]
    if "gen_vm_pu" in prof and len(net.gen):
        net.gen["vm_pu"] = prof["gen_vm_pu"]
    if "gen_p_mw" in prof and len(net.gen):
        net.gen["p_mw"] = prof["gen_p_mw"]

def _apply_grid_snapshot(net: pp.pandapowerNet, grid_snapshot: dict[str, pd.DataFrame]) -> None:
    """Write a ``{element: DataFrame}`` grid snapshot back into ``net``.

    The counterpart of a snapshot that captured a slice of the mutable element tables (the
    columns an action may change -- see ``PPTopoGym.supported_action_types``). Columns are
    assigned outright rather than merged, so a snapshot restores the state it captured exactly,
    NaN tap positions included; that matches ``BaseEnvPP.restore_topology``.

    Elements the deserialized net does not carry are skipped, so a snapshot taken from a richer
    grid still applies what it can.

    :param net: the worker's process-local network, rebuilt from the static blob.
    :type net: pp.pandapowerNet
    :param grid_snapshot: element tables holding the columns to restore.
    :type grid_snapshot: dict[str, pd.DataFrame]
    """
    for element, snapshot_df in grid_snapshot.items():
        if element not in net:
            continue
        table = net[element]
        for column in snapshot_df.columns:
            table[column] = snapshot_df[column].to_numpy()


def _apply_action_delta(net: pp.pandapowerNet, action_row: dict) -> None:
    if "open_switches" in action_row:
        net.switch.loc[action_row["open_switches"], "closed"] = False
    if "closed_switches" in action_row:
        net.switch.loc[action_row["closed_switches"], "closed"] = True
    if "lines" in action_row and "disconnect_lines" in action_row:
        lines = action_row["lines"]
        disconnects = np.array(action_row["disconnect_lines"], dtype=bool)
        net.line.loc[lines, "in_service"] = ~disconnects
    if "trafos" in action_row and "tap_pos" in action_row:
        net.trafo.loc[action_row["trafos"], "tap_pos"] = np.asarray(action_row["tap_pos"], dtype=int)

def evaluate_action(static_net_blob: bytes, # noqa: PLR0913
                    action_row: dict,
                    base_topology: dict[str, np.ndarray] | None = None,
                    profile_slice: dict | None = None,
                    grid_snapshot: dict[str, pd.DataFrame] | None = None,
                    pf_mode: str = "ac",
                    need_n1: bool = False, #noqa: FBT001, FBT002
                    ) -> dict:
    """
    Score one action against a grid state, in a worker process.

    The pre-action state can be handed in two equivalent ways, and both may be combined:

    - ``base_topology`` + ``profile_slice`` -- the packed numpy arrays produced by
      ``PPTopoGym.snapshot_topology`` / ``get_profile_slice``. This is what the greedy agents
      use: the injections travel separately, so one static blob serves every timestep.
    - ``grid_snapshot`` -- the ``{element: DataFrame}`` slice produced by a caller that
      snapshots ``PPTopoGym.supported_action_types`` off the live net. Note this restores
      *topology only*; the injections must already be in ``static_net_blob``, so a caller using
      it has to re-dump the blob per episode.

    :param static_net_blob: JSON bytes of the network, cached per process.
    :type static_net_blob: bytes
    :param action_row: one row of ``df_actions`` as a dict, the action to apply.
    :type action_row: dict
    :param base_topology: packed switch/line/trafo arrays to restore before acting.
    :type base_topology: dict[str, np.ndarray] | None
    :param profile_slice: packed per-element injections for the timestep being scored.
    :type profile_slice: dict | None
    :param grid_snapshot: element tables to restore before acting.
    :type grid_snapshot: dict[str, pd.DataFrame] | None
    :param pf_mode: ``"ac"`` or ``"dc"``.
    :type pf_mode: str
    :param need_n1: also run the N-1 sweep and report ``nminus1``.
    :type need_n1: bool
    :return: ``crashed`` / ``reward`` / ``max_loading`` / ``line_loadings`` (+ ``nminus1``).
    :rtype: dict
    """
    return _evaluate_on_net(
        _ensure_net_from_blob(static_net_blob),
        action_row,
        base_topology=base_topology,
        profile_slice=profile_slice,
        grid_snapshot=grid_snapshot,
        pf_mode=pf_mode,
        need_n1=need_n1,
    )


def evaluate_actions(static_net_blob: bytes, # noqa: PLR0913
                     action_rows: list[dict],
                     base_topology: dict[str, np.ndarray] | None = None,
                     profile_slice: dict | None = None,
                     pf_mode: str = "ac",
                     need_n1: bool = False, #noqa: FBT001, FBT002
                     backend: str = "pandapower",
                     ) -> list[dict]:
    """
    Score a whole chunk of actions against one grid state, in a worker process.

    This is what the greedy agents dispatch, one chunk per worker, rather than one task per
    candidate action. The payload -- ``static_net_blob`` above all, which is the entire network
    serialized as JSON -- is identical for every candidate, so sending it per task made it cross
    the process boundary once per *action* instead of once per *worker*. Rebuilding the net from
    the blob (and hashing the blob to check the process-local cache) likewise happens once here
    rather than per action.

    Results are returned in the order of ``action_rows``, which the caller relies on to line
    scores up with the actions it asked about.

    :param static_net_blob: JSON bytes of the network, cached per process.
    :type static_net_blob: bytes
    :param action_rows: the ``df_actions`` rows to score, as dicts.
    :type action_rows: list[dict]
    :param base_topology: packed switch/line/trafo arrays to restore before each action.
    :type base_topology: dict[str, np.ndarray] | None
    :param profile_slice: packed per-element injections for the timestep being scored.
    :type profile_slice: dict | None
    :param pf_mode: ``"ac"`` or ``"dc"``.
    :type pf_mode: str
    :param need_n1: also run the N-1 sweep and report ``nminus1``.
    :type need_n1: bool
    :param backend: ``"pandapower"`` or ``"lightsim"``; the latter solves the same grid through
        ``toolbox.ls2g_backend``, for the N-1 sweep as well as for the single power flow.
    :type backend: str
    :return: one result dict per entry of ``action_rows``, in the same order.
    :rtype: list[dict]
    """
    net = _ensure_net_from_blob(static_net_blob)
    ls2g = _ensure_lightsim_backend(net) if backend == "lightsim" else None
    return [
        _evaluate_on_net(
            net, action_row,
            base_topology=base_topology, profile_slice=profile_slice,
            grid_snapshot=None, pf_mode=pf_mode, need_n1=need_n1, ls2g=ls2g,
        )
        for action_row in action_rows
    ]


def _solve(net: pp.pandapowerNet, *, pf_mode: str, need_n1: bool,
           ls2g: LightsimBackend | None) -> bool:
    """Run the power flow the caller asked for and report whether it failed.

    The two backends signal failure differently -- pandapower raises, lightsim2grid returns
    False -- so this funnels both into one boolean and keeps the scoring path free of it.

    :param net: the worker's net, already carrying the action to score.
    :type net: pp.pandapowerNet
    :param pf_mode: ``"ac"`` or ``"dc"``.
    :type pf_mode: str
    :param need_n1: run the N-1 sweep instead of a single power flow.
    :type need_n1: bool
    :param ls2g: the lightsim2grid backend to solve with, or None for pandapower.
    :type ls2g: LightsimBackend | None
    :return: True if the power flow did *not* converge.
    :rtype: bool
    """
    try:
        with _suppress_pp_output(logging.CRITICAL):
            if ls2g is not None:
                return not (ls2g.solve_nminus1(net) if need_n1 else ls2g.solve(net))
            if need_n1:
                run_nminus1_powerflow(net, pf_type=pf_mode)
            else:
                run_powerflow(net, pf_type=pf_mode)
    except pp.LoadflowNotConverged:
        return True
    return False


def _evaluate_on_net(net: pp.pandapowerNet, # noqa: PLR0913
                     action_row: dict,
                     *,
                     base_topology: dict[str, np.ndarray] | None,
                     profile_slice: dict | None,
                     grid_snapshot: dict[str, pd.DataFrame] | None,
                     pf_mode: str,
                     need_n1: bool,
                     ls2g: LightsimBackend | None = None,
                     ) -> dict:
    """Restore the pre-action state onto ``net``, apply one action, solve it and score it.

    The state is re-applied per action rather than once per chunk: every action is scored
    against the same starting grid, and the previous action's writes have to be undone.

    When ``ls2g`` is given the power flow goes through lightsim2grid instead of pandapower. It
    signals non-convergence by returning False rather than by raising, so both are funnelled
    into the same crashed result.
    """
    worst_loading = 1000.0 # bad worst loading and reward
    clipping = 200.0 # clipping for reward

    if grid_snapshot is not None:
        _apply_grid_snapshot(net, grid_snapshot)
    if base_topology is not None:
        _apply_topology(net, base_topology)
    if profile_slice is not None:
        _inject_profile(net, profile_slice)
    _apply_action_delta(net, action_row)

    if _solve(net, pf_mode=pf_mode, need_n1=need_n1, ls2g=ls2g):
        n_lines = len(net.line)
        return {
            "crashed": True,
            "reward": -worst_loading,
            "max_loading": worst_loading,
            "line_loadings": np.full(n_lines, worst_loading, dtype=np.float32),
            "nminus1": worst_loading,
        }

    vals = net.res_line["loading_percent"].to_numpy(dtype=float, na_value=worst_loading)
    max_load = np.nanmax(vals)

    n1_max = None
    if need_n1:
        if "max_loading_percent" in net.res_line:
            n1_vals = net.res_line["max_loading_percent"].to_numpy(dtype=float, na_value=worst_loading)
            n1_max = np.nanmax(n1_vals)
        else:
            n1_max = worst_loading


    line_loadings = np.nan_to_num(vals, nan=worst_loading).astype(np.float32, copy=False)
    capped = max(0.0, min(max_load, clipping)) if np.isfinite(max_load) else clipping
    reward = clipping - capped

    result = {
        "crashed": False,
        "reward": reward,
        "max_loading": max_load,
        "line_loadings": line_loadings,
    }
    if n1_max is not None:
        result["nminus1"] = n1_max
    return result
