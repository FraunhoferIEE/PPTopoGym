"""
Tests for the deepcopy-free in-place ``reset()`` of ``BaseEnvPP`` / ``PPTopoGym``.

``reset()`` restores the network topology in place instead of deep-copying a
pristine baseline. These tests prove that

1. a greedy agent produces the same output after an (in-place) reset as on a
   fresh init to the same scenario index, and
2. the in-place restore is byte-for-byte equivalent to the legacy deepcopy
   restore (still available via ``reset(options={"deep": True})``).
"""

import numpy as np
import pytest

from pandapower_env.agents.benchmark_agents import GreedyAgent
from pandapower_env.data.example_configs import config_case30
from pandapower_env.environments.simulation_env import PPTopoGym


def _assert_obs_equal(obs1: dict, obs2: dict) -> None:
    """Assert two observation dicts are numerically identical."""
    assert obs1.keys() == obs2.keys()
    for key in obs1:
        np.testing.assert_allclose(
            np.asarray(obs1[key], dtype=float),
            np.asarray(obs2[key], dtype=float),
            err_msg=f"observation key {key!r} differs after reset",
        )


def test_greedy_agent_same_output_after_reset(simenv30) -> None:
    """A greedy agent picks the same action after reset as on a fresh init.

    This exercises the in-place reset path: the env is dirtied with topology
    changing steps, reset to the same scenario index, and the greedy agent must
    reproduce both its observation and its chosen action.
    """
    env = simenv30
    # GreedyAgent builds its own internal env from the config; using a separate
    # config_case30() instance keeps state_from_info() off the same instance.
    agent = GreedyAgent(env.action_space, config_case30())

    index = 50

    # --- fresh init run ---------------------------------------------------
    env.reset(options={"index": index})
    obs1 = env.create_observation()
    info1 = env.state_to_info()
    action1 = agent.act(obs1, info1)  # full action set -> deterministic

    # --- dirty the env, then in-place reset to the same scenario ----------
    env.step(1)
    env.step(2)
    env.reset(options={"index": index})
    obs2 = env.create_observation()
    info2 = env.state_to_info()
    action2 = agent.act(obs2, info2)

    assert info1["index_profile"] == info2["index_profile"]
    assert info1["current_step"] == info2["current_step"] == 0
    _assert_obs_equal(obs1, obs2)
    assert action1 == action2, "greedy action changed after in-place reset"


def test_envs_from_one_config_own_independent_nets(env_config) -> None:
    """Envs built from the same config must not share a mutable net.

    Regression for the shared-net aliasing that, combined with the deepcopy-free
    in-place reset, let ``setup_profiles`` re-scale an already-scaled load on every
    newly built env until the power flow diverged (load.p_mw compounded
    259 -> 777 -> 2331 ...). Each env must own a private deep copy of the grid.
    """
    env_a = PPTopoGym(env_config)
    env_a.reset(options={"index": 0})
    load_a = float(env_a.net.load.p_mw.sum())

    env_b = PPTopoGym(env_config)
    env_b.reset(options={"index": 0})
    load_b = float(env_b.net.load.p_mw.sum())

    assert env_a.net is not env_b.net, "envs must not share the same net object"
    assert env_a.net is not env_config["net"], "env must own a copy, not the config's net"
    assert load_b == pytest.approx(load_a), "profile scaling compounded across envs"
    assert env_a.net.converged is True
    assert env_b.net.converged is True


def _capture_state(net) -> dict:
    """Copy the element + result tables that an episode can touch."""
    return {
        "switch": net.switch[["closed"]].copy(),
        "line": net.line[["in_service"]].copy(),
        "trafo_tap": net.trafo[["tap_pos"]].copy(),
        "load": net.load[["p_mw", "q_mvar"]].copy(),
        "sgen": net.sgen[["p_mw", "q_mvar"]].copy(),
        "gen": net.gen[["p_mw", "vm_pu"]].copy() if len(net.gen) else None,
        "res_line": net.res_line["loading_percent"].to_numpy(copy=True),
        "res_bus": net.res_bus["vm_pu"].to_numpy(copy=True),
    }


def test_inplace_vs_deepcopy_reset_parity(simenv30) -> None:
    """In-place reset matches the legacy deepcopy reset across the board."""
    env = simenv30
    rng = np.random.default_rng(0)

    for index in (0, 96, 288):
        actions = [int(a) for a in rng.integers(1, env.action_space.n, size=4)]

        # in-place restore (default path)
        env.reset(options={"index": index})
        for action in actions:
            env.step(action)
        env.reset(options={"index": index})
        state_inplace = _capture_state(env.net)

        # legacy deepcopy restore (escape hatch / oracle)
        for action in actions:
            env.step(action)
        env.reset(options={"index": index, "deep": True})
        state_deep = _capture_state(env.net)

        for table in ("switch", "line", "trafo_tap", "load", "sgen", "gen"):
            new, old = state_inplace[table], state_deep[table]
            if new is None and old is None:
                continue
            assert new.equals(old), f"element table {table!r} differs at index {index}"

        np.testing.assert_allclose(
            state_inplace["res_line"],
            state_deep["res_line"],
            rtol=1e-8,
            atol=1e-8,
            err_msg=f"res_line loadings differ at index {index}",
        )
        np.testing.assert_allclose(
            state_inplace["res_bus"],
            state_deep["res_bus"],
            rtol=1e-8,
            atol=1e-8,
            err_msg=f"res_bus voltages differ at index {index}",
        )
