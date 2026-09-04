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


def test_ensure_net_from_blob_rebuilds_for_different_blob() -> None:
    """The process-local net cache must key on the blob, not just ``_NET is None``.

    Regression for an order-dependent crash: the first grid deserialized in a process
    was cached and reused for *every* later blob, so a topology snapshot from a
    different grid (more switches) got applied to the stale net (fewer switches),
    raising ``Length of values (N) does not match length of index (M)``.
    """
    from pandapower.networks import case14, case30

    gw._NET = None
    gw._NET_KEY = None
    try:
        net_a, net_b = case14(), case30()
        blob_a, blob_b = _net_to_blob(net_a), _net_to_blob(net_b)
        assert len(net_a.bus) != len(net_b.bus)

        got_a = gw._ensure_net_from_blob(blob_a)
        assert len(got_a.bus) == len(net_a.bus)

        # A different blob must rebuild rather than serve the cached net_a.
        got_b = gw._ensure_net_from_blob(blob_b)
        assert len(got_b.bus) == len(net_b.bus)

        # The same blob reuses the cached object (identity), avoiding re-deserialization.
        assert gw._ensure_net_from_blob(blob_b) is got_b
    finally:
        gw._NET = None
        gw._NET_KEY = None


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


def test_evaluate_action_grid_snapshot_matches_packed_topology(test_grid_dbb_plus_simbench) -> None:
    """Restoring state from a ``{element: DataFrame}`` snapshot scores an action identically.

    Two ways to hand ``evaluate_action`` the pre-action grid: the packed numpy arrays the greedy
    agents build, or an element-table snapshot of the columns an action may change. Both describe
    the same topology, so both must produce the same power flow.
    """
    net = test_grid_dbb_plus_simbench
    blob = _net_to_blob(net)

    base_topology = {
        "switch_closed": net.switch["closed"].to_numpy(dtype=np.int8),
        "line_in_service": net.line["in_service"].to_numpy(dtype=np.int8),
    }
    grid_snapshot = {
        "switch": net.switch[["closed"]].copy(),
        "line": net.line[["in_service"]].copy(),
        # An element the deserialized net does not carry must be skipped, not raise.
        "not_a_table": net.line[["in_service"]].copy(),
    }
    action_row: dict = {}

    packed = gw.evaluate_action(
        static_net_blob=blob, base_topology=base_topology, action_row=action_row, pf_mode="ac",
    )
    snapshotted = gw.evaluate_action(
        static_net_blob=blob, grid_snapshot=grid_snapshot, action_row=action_row, pf_mode="ac",
    )

    assert snapshotted["crashed"] == packed["crashed"]
    assert snapshotted["max_loading"] == packed["max_loading"]
    np.testing.assert_array_equal(snapshotted["line_loadings"], packed["line_loadings"])


def test_evaluate_action_grid_snapshot_restores_a_changed_switch(test_grid_dbb_plus_simbench) -> None:
    """A snapshot overwrites what an earlier action left behind, dtypes intact."""
    net = test_grid_dbb_plus_simbench
    grid_snapshot = {"switch": net.switch[["closed"]].copy()}
    first_switch = net.switch.index[0]

    dirty = pp.pandapowerNet(net)  # a shallow view is enough: we only read the blob below
    dirty_blob = _net_to_blob(dirty)
    worker_net = gw._ensure_net_from_blob(dirty_blob)
    worker_net.switch.loc[first_switch, "closed"] = not net.switch.loc[first_switch, "closed"]

    gw._apply_grid_snapshot(worker_net, grid_snapshot)

    assert worker_net.switch["closed"].dtype == net.switch["closed"].dtype
    np.testing.assert_array_equal(
        worker_net.switch["closed"].to_numpy(), net.switch["closed"].to_numpy(),
    )
