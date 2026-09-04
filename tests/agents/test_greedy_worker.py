from __future__ import annotations

import io

import numpy as np
import pandapower as pp

import pandapower_env.agents.greedy_worker as gw


def _net_to_blob(net: pp.pandapowerNet) -> bytes:
    """Serialize a pandapower net to bytes blob that greedy_worker expects."""
    buf = io.StringIO()
    pp.to_json(net, buf)
    return buf.getvalue().encode("utf-8")


def test_apply_topology_with_real_net(test_grid) -> None:
    """_apply_topology should correctly set fields in a real pandapowerNet."""
    net = test_grid

    switch_closed = np.ones(len(net.switch), dtype=np.int8)
    line_in_service = np.zeros(len(net.line), dtype=np.int8)
    trafo_tap_pos = np.full(len(net.trafo), 2, dtype=np.int16)

    topo: dict[str, np.ndarray] = {
        "switch_closed": switch_closed,
        "line_in_service": line_in_service,
        "trafo_tap_pos": trafo_tap_pos,
    }

    gw._apply_topology(net, topo)

    if len(net.switch):
        assert np.all(net.switch["closed"].to_numpy(dtype=bool))
    if len(net.line):
        assert not np.any(net.line["in_service"].to_numpy(dtype=bool))
    if len(net.trafo):
        assert np.all(net.trafo["tap_pos"].to_numpy() == 2) # noqa: PLR2004


def test_inject_profile_with_real_net(test_grid) -> None:
    """_inject_profile should correctly update load and sgen values."""
    net = test_grid

    load_p = np.arange(len(net.load), dtype=np.float32) + 1.0
    load_q = np.arange(len(net.load), dtype=np.float32) + 0.5
    sgen_p = np.arange(len(net.sgen), dtype=np.float32) + 2.0
    sgen_q = np.arange(len(net.sgen), dtype=np.float32) + 0.1

    prof: dict[str, np.ndarray] = {
        "load_p_mw": load_p,
        "load_q_mvar": load_q,
        "sgen_p_mw": sgen_p,
        "sgen_q_mvar": sgen_q,
    }

    gw._inject_profile(net, prof)

    if len(net.load):
        assert np.allclose(net.load["p_mw"].to_numpy(), load_p)
        assert np.allclose(net.load["q_mvar"].to_numpy(), load_q)
    if len(net.sgen):
        assert np.allclose(net.sgen["p_mw"].to_numpy(), sgen_p)
        assert np.allclose(net.sgen["q_mvar"].to_numpy(), sgen_q)


def test_evaluate_action_real_pf(test_grid_dbb_plus_simbench) -> None:
    """Test evaluate_action.

    Asserts result contract and basic sanity of values.
    """
    net = test_grid_dbb_plus_simbench

    # Build base topology arrays from the current net
    switch_closed = net.switch["closed"].to_numpy(dtype=np.int8) if len(net.switch) else np.zeros(0, dtype=np.int8)
    line_in_service = net.line["in_service"].to_numpy(dtype=np.int8) if len(net.line) else np.zeros(0, dtype=np.int8)
    trafo_tap_pos = net.trafo["tap_pos"].to_numpy(dtype=np.int16) if len(net.trafo) else np.zeros(0, dtype=np.int16)

    base_topology: dict[str, np.ndarray] = {
        "switch_closed": switch_closed,
        "line_in_service": line_in_service,
        "trafo_tap_pos": trafo_tap_pos,
    }

    # Build a profile slice from current values
    prof: dict[str, np.ndarray] = {}
    if len(net.load):
        prof["load_p_mw"] = net.load["p_mw"].to_numpy(dtype=np.float32)
        prof["load_q_mvar"] = net.load["q_mvar"].to_numpy(dtype=np.float32)
    if len(net.sgen):
        prof["sgen_p_mw"] = net.sgen["p_mw"].to_numpy(dtype=np.float32)
        prof["sgen_q_mvar"] = net.sgen["q_mvar"].to_numpy(dtype=np.float32)
    if len(net.gen):
        prof["gen_vm_pu"] = net.gen["vm_pu"].to_numpy(dtype=np.float32)

    # No-op action row
    action_row: dict = {}

    blob = _net_to_blob(net)
    result = gw.evaluate_action(
        static_net_blob=blob,
        base_topology=base_topology,
        profile_slice=prof,
        action_row=action_row,
        pf_mode="ac",
        need_n1=False,
    )

    # Contract checks
    assert isinstance(result, dict)
    assert "crashed" in result
    assert isinstance(result["crashed"], bool)
    assert "reward" in result
    assert isinstance(result["reward"], float)
    assert "max_loading" in result
    assert isinstance(result["max_loading"], float)
    assert "line_loadings" in result
    assert isinstance(result["line_loadings"], np.ndarray)

    # Value sanity
    line_loadings = result["line_loadings"]
    assert line_loadings.ndim == 1
    assert len(line_loadings) == len(net.line)
    assert np.all(np.isfinite(line_loadings))
    assert np.isfinite(result["reward"])
    assert np.isfinite(result["max_loading"])
