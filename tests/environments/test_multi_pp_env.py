"""Tests for the vectorized-environment helpers.

``multi_pp_env`` wraps a ``PPTopoGym`` factory into a Gymnasium vector env and exposes a
thin batched step/reset API on top of it. The integration test runs a real ``SyncVectorEnv``
with ``static_obs_space=True``: node-aggregated observations grow when a substation splits,
so without the static upper bound the vector env raises a broadcast error rather than a
readable failure (see the observation-space contract tests).

``AsyncVectorEnv`` is deliberately not exercised here -- it spawns interpreters that each
rebuild the grid, which costs minutes for no additional coverage of this module's logic.
"""
from __future__ import annotations

import copy
from typing import Any, cast

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv

from pandapower_env.environments.multi_pp_env import (
    SimpleVecEnvWrapper,
    StepResult,
    create_vectorized_env,
    make_env_fn,
)
from pandapower_env.environments.simulation_env import PPTopoGym

NUM_ENVS = 2
FAKE_ACTION_SIZE = 7
FAKE_RESET_SEED = 11


@pytest.fixture()
def static_obs_config(env_config: dict) -> dict:
    """Return the shared env config with the static observation-space upper bound enabled."""
    config = copy.deepcopy(env_config)
    config["static_obs_space"] = True
    return config


class _FakeVecEnv:
    """Duck-types the parts of a Gymnasium vector env that ``SimpleVecEnvWrapper`` touches."""

    def __init__(self, action_space: spaces.Space, num_envs: int = NUM_ENVS) -> None:
        self.num_envs = num_envs
        self.single_action_space = action_space
        self.single_observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,))
        self.closed = False
        self.reset_seed: int | None = None

    def reset(self, seed: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.reset_seed = seed
        return {"obs": np.zeros((self.num_envs, 3))}, {}

    def step(self, actions: np.ndarray) -> tuple[Any, ...]:
        return (
            {"obs": np.ones((self.num_envs, 3))},
            np.array([1, 2], dtype=np.int64),  # ints, to prove the float64 cast
            np.array([False, True]),
            np.array([False, False]),
            {"action": actions},
        )

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def test_make_env_fn_is_lazy_and_seeds_per_index(static_obs_config: dict) -> None:
    """The factory must not build anything until called, then seed with ``seed + env_idx``."""
    built: list[int] = []
    seeds: list[int | None] = []

    class _Recorder(PPTopoGym):
        def reset(
            self, seed: int | None = None, options: dict[str, Any] | None = None,
        ) -> tuple[dict, dict[str, Any]]:
            seeds.append(seed)
            return super().reset(seed=seed, options=options)

    def factory() -> PPTopoGym:
        built.append(1)
        return _Recorder(copy.deepcopy(static_obs_config))

    env_fn = make_env_fn(factory, env_idx=3, seed=100)
    assert built == [], "make_env_fn must not build the environment eagerly"

    env = env_fn()
    assert built == [1]
    assert seeds == [103]
    env.close()


def test_make_env_fn_without_seed_does_not_reset(static_obs_config: dict) -> None:
    """With ``seed=None`` the env is handed back untouched, so the caller controls the first reset."""
    resets: list[int] = []

    class _Recorder(PPTopoGym):
        def reset(
            self, seed: int | None = None, options: dict[str, Any] | None = None,
        ) -> tuple[dict, dict[str, Any]]:
            resets.append(1)
            return super().reset(seed=seed, options=options)

    env = make_env_fn(lambda: _Recorder(copy.deepcopy(static_obs_config)), env_idx=0)()

    assert resets == []
    env.close()


def test_create_vectorized_env_sync_steps_all_envs(static_obs_config: dict) -> None:
    """A synchronous vector env of PPTopoGyms resets and steps as one batch."""
    vec_env = create_vectorized_env(
        env_factory=lambda: PPTopoGym(copy.deepcopy(static_obs_config)),
        num_envs=NUM_ENVS,
        use_async=False,
        seed=42,
    )
    try:
        assert isinstance(vec_env, SyncVectorEnv)
        assert vec_env.num_envs == NUM_ENVS

        wrapper = SimpleVecEnvWrapper(vec_env)
        obs, _ = wrapper.reset(seed=0)
        assert set(obs), "reset must return a non-empty observation dict"

        result = wrapper.step(np.zeros(NUM_ENVS, dtype=np.int64))
        assert isinstance(result, StepResult)
        assert result.rewards.shape == (NUM_ENVS,)
        assert result.terminateds.shape == (NUM_ENVS,)
    finally:
        vec_env.close()


# ---------------------------------------------------------------------------
# SimpleVecEnvWrapper
# ---------------------------------------------------------------------------


def test_wrapper_exposes_single_env_spaces_and_action_size() -> None:
    """The wrapper reports the *single* env spaces, not the batched ones."""
    fake = _FakeVecEnv(spaces.Discrete(FAKE_ACTION_SIZE))
    wrapper = SimpleVecEnvWrapper(cast("SyncVectorEnv", fake))

    assert wrapper.num_envs == NUM_ENVS
    assert wrapper.action_size == FAKE_ACTION_SIZE
    assert wrapper.action_space is fake.single_action_space
    assert wrapper.observation_space is fake.single_observation_space


def test_wrapper_rejects_non_discrete_action_space() -> None:
    """A continuous action space cannot be sampled as an action index, so it is refused up front."""
    fake = _FakeVecEnv(spaces.Box(low=-1.0, high=1.0, shape=(2,)))

    with pytest.raises(TypeError, match="requires a Discrete action space"):
        SimpleVecEnvWrapper(cast("SyncVectorEnv", fake))


def test_wrapper_step_packs_a_step_result_with_float_rewards() -> None:
    """``step`` bundles the 5-tuple into a StepResult and normalizes rewards to float64."""
    fake = _FakeVecEnv(spaces.Discrete(3))
    wrapper = SimpleVecEnvWrapper(cast("SyncVectorEnv", fake))
    actions = np.array([1, 2], dtype=np.int64)

    result = wrapper.step(actions)

    assert result.rewards.dtype == np.float64
    np.testing.assert_array_equal(result.rewards, [1.0, 2.0])
    np.testing.assert_array_equal(result.terminateds, [False, True])
    np.testing.assert_array_equal(result.truncateds, [False, False])
    np.testing.assert_array_equal(result.infos["action"], actions)


def test_wrapper_reset_and_close_delegate() -> None:
    """The wrapper forwards the seed to the vector env and closes it exactly once."""
    fake = _FakeVecEnv(spaces.Discrete(3))
    wrapper = SimpleVecEnvWrapper(cast("SyncVectorEnv", fake))

    wrapper.reset(seed=FAKE_RESET_SEED)
    assert fake.reset_seed == FAKE_RESET_SEED

    wrapper.close()
    assert fake.closed
