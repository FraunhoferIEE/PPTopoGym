from __future__ import annotations

import contextlib
import io
import logging
from typing import Generator

import numpy as np
import pandapower as pp

# Use your existing utils
from pandapower_env.toolbox.utils import run_nminus1_powerflow, run_powerflow

# Process-local cache
_NET: pp.pandapowerNet | None = None




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

    Parameters
    ----------
    static_net_blob : bytes

    Returns
    -------
    net : pp.pandapowerNet
    """
    global _NET #noqa: PLW0603
    if _NET is None:
        json_str = static_net_blob.decode("utf-8")
        with io.StringIO(json_str) as fp:
            _NET = pp.from_json(fp)
    return _NET


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
                    base_topology: dict[str, np.ndarray],
                    profile_slice: dict,
                    action_row: dict,
                    pf_mode: str = "ac",
                    need_n1: bool = False, #noqa: FBT001, FBT002
                    ) -> dict:
    worst_loading = 1000.0 # bad worst loading and reward
    clipping = 200.0 # clipping for reward
    net = _ensure_net_from_blob(static_net_blob)

    _apply_topology(net, base_topology)
    _inject_profile(net, profile_slice)
    _apply_action_delta(net, action_row)

    try:
        with _suppress_pp_output(logging.CRITICAL):
            run_nminus1_powerflow(net, pf_type=pf_mode) if need_n1 \
                else run_powerflow(net, pf_type=pf_mode)
    except pp.LoadflowNotConverged:
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
