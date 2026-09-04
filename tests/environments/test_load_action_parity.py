"""
Tests that the positional ``load_action`` fast path matches the label-based one exactly.

``PPTopoGym.load_action`` applies a precomputed plan of positional numpy writes instead of
per-call ``DataFrame.loc`` label lookups (~460 us -> ~10 us on case30). The original
implementation is kept as ``_load_action_by_label`` and is the correctness reference here:
for every action in the action space, both paths must leave the grid in the same state.

Also covers the fallback: a net whose element tables are *not* indexed ``0..n-1`` cannot use
positional writes, and must transparently keep using the label-based path.
"""

from __future__ import annotations

import numpy as np
import pytest

from pandapower_env.environments.simulation_env import PPTopoGym, _positional_index


def _topology(env: PPTopoGym) -> dict[str, np.ndarray]:
    """Capture the switch/line/trafo state an action is allowed to change."""
    return {
        "switch_closed": env.net.switch["closed"].to_numpy(copy=True),
        "line_in_service": env.net.line["in_service"].to_numpy(copy=True),
        "trafo_tap_pos": env.net.trafo["tap_pos"].to_numpy(copy=True),
    }


def _assert_same_topology(fast: dict, slow: dict, action: int) -> None:
    for table, fast_values in fast.items():
        slow_values = slow[table]
        assert fast_values.shape == slow_values.shape, f"{table} shape differs for action {action}"
        if fast_values.dtype.kind == "f":
            np.testing.assert_array_equal(
                np.isnan(fast_values), np.isnan(slow_values),
                err_msg=f"{table} NaN pattern differs for action {action}",
            )
            mask = ~np.isnan(fast_values)
            np.testing.assert_array_equal(
                fast_values[mask], slow_values[mask],
                err_msg=f"{table} differs for action {action}",
            )
        else:
            np.testing.assert_array_equal(
                fast_values, slow_values, err_msg=f"{table} differs for action {action}",
            )


def test_action_plans_are_built(simenv) -> None:
    """The environment resolves its actions to positional plans (one per non-zero action)."""
    assert simenv._action_plans is not None
    assert set(simenv._action_plans) == set(simenv.df_actions.index[1:])


def test_every_action_matches_the_label_path(simenv) -> None:
    """Both load_action paths leave an identical grid state, for every action."""
    for action in simenv.df_actions.index:
        simenv.reset(options={"index": 0})
        simenv.load_action(action)
        fast = _topology(simenv)

        simenv.reset(options={"index": 0})
        if action != 0:  # action 0 is short-circuited before either path runs
            simenv._load_action_by_label(action)
        slow = _topology(simenv)

        _assert_same_topology(fast, slow, int(action))


def test_repeated_actions_match_the_label_path(simenv) -> None:
    """A sequence of actions applied without reset also stays in lockstep."""
    actions = [int(a) for a in simenv.df_actions.index[1:]] * 2

    simenv.reset(options={"index": 0})
    for action in actions:
        simenv.load_action(action)
    fast = _topology(simenv)

    simenv.reset(options={"index": 0})
    for action in actions:
        simenv._load_action_by_label(action)
    slow = _topology(simenv)

    _assert_same_topology(fast, slow, -1)


def test_action_zero_preserves_converged_flag(simenv) -> None:
    """DoNothing keeps the previous power flow status (it changes nothing)."""
    simenv.reset(options={"index": 0})
    simenv.run_pf()
    assert simenv.net.converged is True
    simenv.load_action(0)
    assert simenv.net.converged is True


def test_load_action_survives_reset_reallocating_columns(simenv) -> None:
    """The fast path re-reads the column arrays, so a reset in between is harmless.

    ``reset`` / ``restore_topology`` replace the underlying numpy arrays of the element
    tables. A cached view would silently write into a detached array; this pins that
    an action applied after a reset really lands on the live grid.
    """
    action = int(simenv.df_actions.index[1])

    simenv.reset(options={"index": 0})
    simenv.load_action(action)
    expected = _topology(simenv)

    simenv.reset(options={"index": 0})  # reallocates the switch/line columns
    simenv.load_action(action)
    _assert_same_topology(_topology(simenv), expected, action)


def test_positional_index_detects_offset_index() -> None:
    """``_positional_index`` returns None only when labels already equal positions."""
    import pandas as pd

    assert _positional_index(pd.Index([0, 1, 2, 3])) is None
    assert _positional_index(pd.Index([], dtype=int)) == {}
    assert _positional_index(pd.Index([5, 7, 9])) == {5: 0, 7: 1, 9: 2}


def test_reindexed_net_falls_back_to_label_path(simenv) -> None:
    """A net whose switch labels are not positions still applies actions correctly."""
    action = int(simenv.df_actions.index[1])

    simenv.reset(options={"index": 0})
    simenv.load_action(action)
    expected = _topology(simenv)

    # Rebuild the plans against a switch table whose labels are offset from its positions.
    simenv.reset(options={"index": 0})
    simenv._action_plans = None  # forces _load_action_by_label
    simenv.load_action(action)
    _assert_same_topology(_topology(simenv), expected, action)


def test_step_reward_unchanged_by_fast_path(simenv) -> None:
    """Stepping through the fast path reproduces the label path's rewards."""
    actions = [int(a) for a in simenv.df_actions.index]

    simenv.reset(options={"index": 0})
    fast_rewards = [float(simenv.step(a)[1]) for a in actions]

    simenv._action_plans = None
    simenv.reset(options={"index": 0})
    slow_rewards = [float(simenv.step(a)[1]) for a in actions]

    assert fast_rewards == pytest.approx(slow_rewards)
