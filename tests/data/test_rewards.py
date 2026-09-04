"""Tests for the named reward functions in ``pandapower_env.data.rewards``."""

import numpy as np
import pytest

from pandapower_env.data.rewards import (
    OVERLOAD_THRESHOLD_PERCENT,
    reward_better_than_donothing,
    reward_normalized,
)
from pandapower_env.environments.simulation_env import PPTopoGym

START_INDEX = 0


@pytest.fixture()
def donothing_env(env_config) -> PPTopoGym:
    """Build an environment scored by ``reward_better_than_donothing``."""
    env_config["reward"] = "reward_better_than_donothing"
    env = PPTopoGym(env_config)
    env.reset(options={"index": START_INDEX})
    return env


def test_reward_better_than_donothing_runs(donothing_env) -> None:
    """The reward must produce a number.

    It used to recurse without bound: the DoNothing rollout drove ``env.step()``, which called
    the reward again, which started another rollout. It also unpacked four values from the
    five-tuple ``step`` returns. Selecting it by name therefore crashed the environment on the
    very first step, and nothing tested more than that it had been bound to the env.
    """
    _, reward, _, _, _ = donothing_env.step(0)
    assert np.isfinite(reward)
    assert 0.0 < reward <= 1.0


def test_reward_better_than_donothing_preserves_env_state(donothing_env) -> None:
    """The DoNothing rollout must leave no trace on the environment it measures.

    Topology (with the scored action applied), profile index, episode counters and the power
    flow results all have to survive, because ``step`` builds the returned observation from
    them *after* the reward has been taken.
    """
    env = donothing_env
    action = 1

    # What the state looks like when only the action is applied and no reward runs.
    env.load_action(action)
    env.run_pf()
    expected_switches = env.net.switch["closed"].to_numpy().copy()
    expected_loadings = env.net.res_line["loading_percent"].to_numpy().copy()

    env.reset(options={"index": START_INDEX})
    env.step(action)

    assert np.array_equal(env.net.switch["closed"].to_numpy(), expected_switches)
    assert np.allclose(env.net.res_line["loading_percent"].to_numpy(), expected_loadings)
    assert env.index == START_INDEX + 1  # step advanced the timeseries exactly once
    assert env.current_step == 1
    assert env.episode_step_counter == 1


def test_reward_better_than_donothing_caches_baseline(donothing_env) -> None:
    """The DoNothing baseline is measured once per episode, not once per step."""
    env = donothing_env
    env.step(0)
    baseline = dict(env.cache["DoNothing_worst_loading"])
    assert baseline  # a baseline was recorded for this episode

    env.step(0)
    assert env.cache["DoNothing_worst_loading"] == baseline


def test_reward_better_than_donothing_is_one_below_overload(donothing_env) -> None:
    """A grid that never overloads scores full reward whatever DoNothing achieved."""
    env = donothing_env
    _, reward, _, _, _ = env.step(0)
    agent_loading = max(env.cache["max_line_loading"].values())
    if agent_loading < OVERLOAD_THRESHOLD_PERCENT:
        assert reward == 1.0


def test_reward_normalized_handles_nan(simenv) -> None:
    """A NaN maximum loading (no converged result) yields ``worst_reward``."""
    simenv.reset(options={"index": 0})
    simenv.net.res_line["loading_percent"] = np.nan
    assert reward_normalized(simenv) == simenv.worst_reward


def test_reward_better_than_donothing_returns_zero_inside_rollout(donothing_env) -> None:
    """The recursion guard short-circuits the reward while a rollout is in flight."""
    donothing_env.cache["_donothing_rollout_active"] = True
    assert reward_better_than_donothing(donothing_env) == 0.0
