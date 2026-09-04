"""
Tests for the deepcopy-free ``save_state`` / ``restore_state`` used by AlphaZero MCTS.

The ``PPTopoSerializer`` mutates a single scratch env in place instead of
deep-copying the pandapower network per MCTS node. These tests prove the
save/restore round-trip matches the deepcopy path that ``alphazero_search`` uses
by default (``_DeepcopySerializer``):

1. ``restore_state(parent) -> step(a)`` produces the same reward, done flags,
   ``log_actions`` and net tables as ``deepcopy(parent) -> step(a)``, and
2. ``restore_state(root, run_pf=True)`` (the serializer's ``finalize``) returns
   the env to its root state so the actor's next real step is correct.
"""

import copy

import numpy as np

from pandapower_env.environments.simulation_env import PPTopoGym


def _capture_state(net) -> dict:
    """Copy the element + result tables a step can touch."""
    return {
        "switch": net.switch[["closed"]].copy(),
        "line": net.line[["in_service"]].copy(),
        "trafo_tap": net.trafo[["tap_pos"]].copy(),
        "load": net.load[["p_mw", "q_mvar"]].copy(),
        "sgen": net.sgen[["p_mw", "q_mvar"]].copy(),
        "gen": net.gen[["p_mw", "vm_pu"]].copy() if len(net.gen) else None,
        "res_line": net.res_line["loading_percent"].to_numpy(copy=True),
    }


def _assert_net_equal(new: dict, old: dict, context: str) -> None:
    for table in ("switch", "line", "trafo_tap", "load", "sgen", "gen"):
        a, b = new[table], old[table]
        if a is None and b is None:
            continue
        assert a.equals(b), f"element table {table!r} differs ({context})"
    np.testing.assert_allclose(
        new["res_line"], old["res_line"], rtol=1e-8, atol=1e-8,
        err_msg=f"res_line loadings differ ({context})",
    )


def test_restore_state_step_matches_deepcopy(simenv30: PPTopoGym) -> None:
    """``restore_state(parent) -> step`` matches ``deepcopy(parent) -> step``."""
    env = simenv30
    rng = np.random.default_rng(0)

    for index in (0, 96, 288):
        env.reset(options={"index": index})
        # Build a non-trivial parent: topology changes + a populated action log.
        env.step(1)
        env.step(2)
        parent_oracle = copy.deepcopy(env)   # what _DeepcopySerializer stores
        parent_state = env.save_state()      # what PPTopoSerializer stores

        for action in (int(a) for a in rng.integers(1, env.action_space.n, size=5)):
            # Deepcopy path: deepcopy the parent and step it.
            child = copy.deepcopy(parent_oracle)
            _obs_d, rew_d, term_d, trunc_d, _ = child.step(action)

            # Lightweight path: restore the scratch env to the parent and step it.
            env.restore_state(parent_state, run_pf=False)
            _obs_l, rew_l, term_l, trunc_l, _ = env.step(action)

            ctx = f"index={index}, action={action}"
            assert (term_l, trunc_l) == (term_d, trunc_d), f"done flags differ ({ctx})"
            np.testing.assert_allclose(rew_l, rew_d, rtol=1e-9, atol=1e-9, err_msg=f"reward differs ({ctx})")
            assert list(env.log_actions) == list(child.log_actions), f"log_actions differ ({ctx})"
            _assert_net_equal(_capture_state(env.net), _capture_state(child.net), ctx)


def test_finalize_restores_root_state(simenv30: PPTopoGym) -> None:
    """``restore_state(root, run_pf=True)`` returns the env to its root state."""
    env = simenv30
    env.reset(options={"index": 96})
    env.step(1)

    root_state = env.save_state()
    env.run_pf(pf_type=env.pf_type)
    root_net = _capture_state(env.net)
    root_log = list(env.log_actions)
    root_pos = (env.index, env.current_step)

    # A search mutates the scratch env across several branches.
    for action in (2, 3, 4):
        env.restore_state(root_state, run_pf=False)
        env.step(action)

    # finalize back to the root.
    env.restore_state(root_state, run_pf=True)

    assert list(env.log_actions) == root_log
    assert (env.index, env.current_step) == root_pos
    _assert_net_equal(_capture_state(env.net), root_net, "finalize")
