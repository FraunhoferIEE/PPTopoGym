from __future__ import annotations

import numpy as np

from pandapower_env.toolbox.utils_graph_obs import make_obs_cache, n_nodes


def test_obs_edge_case30(simenv30) -> None:
    n_trafos = 0
    obs, _, _, _, _ = simenv30.step(0)
    n_trafos = 0
    assert len(obs["transformer_status"]) == n_trafos


def test_cache_valid(simenv) -> None:
    simenv.create_observation()
    id_lookup_table = simenv._obs_cache["lookup_obj_id"]
    simenv.step(1)
    new_id_lookup_table = simenv._obs_cache["lookup_obj_id"]
    assert new_id_lookup_table != id_lookup_table

def test_obs_space_dimension_node(simenv) -> None:
    """Default mode (``fix_obs_space=True``): node-mapped observations have length n_nodes.

    TABLE-type observations (bus values, gen runpf/status) are scattered onto the
    electrical nodes, so their length is ``n_nodes``. PROFILE-type observations (gen
    profile setpoints) are *not* node-aggregated -- they keep the raw element-table
    length (see ``_resolve_shape`` and ``_get_profile_value`` in simulation_env.py).
    Line/trafo observations always stay per-element.
    """
    obs = simenv.create_observation()
    n_nodes_count = n_nodes(simenv.net, simenv._obs_cache)
    n_gen = len(simenv.net.gen)
    n_lines = len(simenv.net.line)
    n_trafos = len(simenv.net.trafo)

    assert n_nodes_count > n_gen, "fixture must have more nodes than generators for the fill check"

    # bus + node-aggregated TABLE observations -> n_nodes
    assert len(obs["bus_voltage_angle"]) == n_nodes_count
    assert len(obs["bus_voltage_magnitude"]) == n_nodes_count
    assert len(obs["gen_power_p_mw_runpf"]) == n_nodes_count
    assert len(obs["gen_status"]) == n_nodes_count

    # PROFILE observations are NOT node-aggregated -> raw element-table length
    assert len(obs["gen_power_p_mw_profile"]) == n_gen
    assert len(obs["gen_vm_pu_profile"]) == n_gen

    # line / trafo observations stay per-element
    assert len(obs["line_loadings"]) == n_lines
    assert len(obs["line_thermal_limit"]) == n_lines
    assert len(obs["transformer_loading_percent"]) == n_trafos
    assert len(obs["transformer_power_flow_p_mw"]) == n_trafos

    # Fill check: a node-aggregated TABLE observation has one slot per node; nodes
    # without a generator are filled with 0 (np.bincount default), never NaN. Since
    # there are more nodes than generators, zero-filled slots must exist.
    gen_runpf = np.asarray(obs["gen_power_p_mw_runpf"], dtype=float)
    assert not np.isnan(gen_runpf).any()
    assert np.count_nonzero(gen_runpf) <= n_gen
    assert np.count_nonzero(gen_runpf == 0.0) >= n_nodes_count - n_gen

    # Adjacency matrix changes after a topology-changing step.
    adj1 = obs["adjacency_matrix"]
    simenv.step(1)
    adj2 = simenv.create_observation()["adjacency_matrix"]
    assert not np.array_equal(adj1, adj2), "Arrays should have been different!"


def test_obs_space_dimension_oldobs(simenv_oldobs) -> None:
    """``fix_obs_space=False``: every observation keeps its raw pandapower table length.

    Uses the ``simenv_oldobs`` fixture (same grid/actions as ``simenv`` but with
    ``fix_obs_space=False``). Bus observations span all physical busbars of the
    dbb-expanded table, and element observations match their table element counts.
    """
    env = simenv_oldobs
    obs = env.create_observation()
    n_bus = len(env.net.bus)
    n_gen = len(env.net.gen)
    n_lines = len(env.net.line)
    n_trafos = len(env.net.trafo)

    assert len(obs["bus_voltage_angle"]) == n_bus
    assert len(obs["bus_voltage_magnitude"]) == n_bus
    assert len(obs["gen_power_p_mw_runpf"]) == n_gen
    assert len(obs["gen_power_p_mw_profile"]) == n_gen
    assert len(obs["gen_vm_pu_profile"]) == n_gen
    assert len(obs["gen_status"]) == n_gen
    assert len(obs["line_loadings"]) == n_lines
    assert len(obs["line_thermal_limit"]) == n_lines
    assert len(obs["transformer_loading_percent"]) == n_trafos
    assert len(obs["transformer_power_flow_p_mw"]) == n_trafos

def test_obs_edge_features(simenv) -> None:
    obs = simenv.create_observation()
    n_lines = 15
    n_trafos = 5
    assert len(obs["line_status"]) == n_lines
    assert len(obs["transformer_status"]) == n_trafos
    assert len(obs["transformer_loading_percent"]) == n_trafos


def test_topology_change(simenv) -> None:
    simenv.step(1) # topology change in 1 substation
    obs = simenv.create_observation()
    assert "bus_voltage_angle" in obs
    # assert len(obs["bus_voltage_angle"]) == n_buses+1 does not work, outputs all 57 buses again


def test_topology_caches_key_on_lookup_content(simenv30) -> None:
    """The obs caches key on the *content* of the bus lookup, not on the lookups object.

    Regression for an id()-reuse bug: a freed ``_pd2ppc_lookups`` object whose id() was
    recycled by a different object produced a false cache hit, serving a stale
    node-mapping for a new topology (the
    ``Length of values (116) does not match length of index (79)`` crash). Comparing the
    bus-lookup *content* against a copy held in the cache removes that failure mode
    entirely -- there is no object whose identity could be recycled -- and additionally
    keeps the cache valid across the fresh lookups object that every power flow allocates.
    """
    env = simenv30
    env.reset(options={"index": 0})
    env.create_observation()
    cache = env._obs_cache
    lookups = env.net._pd2ppc_lookups

    # Each per-topology cache stores its own copy of the bus-lookup fingerprint.
    for ref_key in ("_mapping_lookups_ref", "table_cache_lookups_ref", "canonical_lookups_ref"):
        np.testing.assert_array_equal(cache[ref_key], lookups["bus"])

    n_before = cache["n_nodes"]

    # A fresh power flow allocates a new _pd2ppc_lookups object but leaves the topology
    # alone, so the content is unchanged and the cache must stay valid. ``converged`` has to be
    # cleared first, because ``run_pf`` now serves an unchanged net from the results already on
    # it -- and a skipped solve would leave the *same* lookups object, which is not the case
    # under test here.
    env.net.converged = None
    env.run_pf()
    assert env.net._pd2ppc_lookups is not lookups
    obs = env.create_observation()
    assert cache["n_nodes"] == n_before
    assert len(obs["bus_voltage_magnitude"]) == cache["n_nodes"]


def _find_splitting_action(env) -> int:
    """Return an action index that changes the node count (a real busbar split).

    Derived from ``df_actions`` rather than hard-coded so the test survives a change in
    action ordering.

    :param env: a PPTopoGym whose action table is searched
    :return: the first action index that changes ``n_nodes``.
    :raises AssertionError: if the grid has no action that splits a busbar.
    """
    env.reset(options={"index": 0})
    env.step(0)
    base = n_nodes(env.net, env._obs_cache)
    for action in env.df_actions.index[1:]:
        env.reset(options={"index": 0})
        env.step(int(action))
        if n_nodes(env.net, env._obs_cache) != base:
            return int(action)
    msg = "fixture grid has no busbar-splitting action"
    raise AssertionError(msg)


def test_obs_cache_survives_powerflow_without_topology_change(simenv30) -> None:
    """A power flow that leaves the topology alone must not invalidate the topology cache.

    pandapower allocates a fresh ``_pd2ppc_lookups`` on every power flow. Keying the cache on
    that object's identity rebuilt the whole node mapping once per step (~1.5 ms) for nothing;
    keying on content keeps the cached arrays alive.
    """
    env = simenv30
    env.reset(options={"index": 0})
    env.create_observation()
    cache = env._obs_cache

    mapping_before = cache["pp_to_canonical"]
    edges_before = cache["edges"]
    nodes_before = cache["n_nodes"]

    env.run_pf()
    env.create_observation()

    assert cache["pp_to_canonical"] is mapping_before, "mapping was rebuilt despite same topology"
    assert cache["edges"] is edges_before, "adjacency was rebuilt despite same topology"
    assert cache["n_nodes"] == nodes_before


def test_obs_cache_invalidates_on_busbar_split(simenv30) -> None:
    """A busbar split must invalidate the cache, and restoring the topology must invalidate back."""
    env = simenv30
    action = _find_splitting_action(env)

    env.reset(options={"index": 0})
    env.step(0)
    env.create_observation()
    base_nodes = env._obs_cache["n_nodes"]

    env.reset(options={"index": 0})
    env.step(action)
    env.create_observation()
    split_nodes = env._obs_cache["n_nodes"]
    assert split_nodes != base_nodes, "busbar split must change the node count"

    # ... and back: invalidation has to work in both directions, not just monotonically.
    env.reset(options={"index": 0})
    env.step(0)
    env.create_observation()
    assert env._obs_cache["n_nodes"] == base_nodes


def test_obs_cache_matches_fresh_cache_after_line_outage(simenv30) -> None:
    """The cached path must equal a from-scratch computation for the same net state."""
    env = simenv30
    env.reset(options={"index": 0})
    env.step(0)
    env.create_observation()

    env.net.line.loc[env.net.line.index[0], "in_service"] = False
    env.run_pf()
    cached = env.create_observation()

    env._obs_cache = make_obs_cache()
    fresh = env.create_observation()

    for key, value in cached.items():
        np.testing.assert_array_equal(
            np.asarray(value), np.asarray(fresh[key]), err_msg=f"{key} differs from fresh cache",
        )
