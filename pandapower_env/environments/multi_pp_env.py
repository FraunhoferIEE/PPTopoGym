# Create multiple PPTopoGym Environments
"""Simplified vectorized environment utilities for PPTopoGym."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np
from gymnasium import spaces
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

if TYPE_CHECKING:
    import gymnasium as gym
    from numpy.typing import NDArray

    from pandapower_env.environments.simulation_env import PPTopoGym


def make_env_fn(
    env_factory: Callable[[], PPTopoGym],
    env_idx: int,
    seed: int | None = None,
) -> Callable[[], gym.Env]:
    """
    Create a callable that returns an environment instance.

    Args:
        env_factory: Factory function to create the base environment.
        env_idx: Index of this environment (for seeding).
        seed: Optional base seed.

    Returns
    -------
        Callable that creates and configures an environment.
    """

    def _make() -> gym.Env:
        env = env_factory()
        if seed is not None:
            env.reset(seed=seed + env_idx)
        return env

    return _make


def create_vectorized_env(
    env_factory: Callable[[], PPTopoGym],
    num_envs: int,
    *,
    use_async: bool = True,
    seed: int | None = None,
) -> AsyncVectorEnv | SyncVectorEnv:
    """
    Create a vectorized environment.

    Args:
        env_factory: Factory function to create individual environments.
        num_envs: Number of parallel environments.
        use_async: If True, use AsyncVectorEnv (separate processes).
        seed: Optional random seed.

    Returns
    -------
        Vectorized environment instance.
    """
    env_fns = [make_env_fn(env_factory, idx, seed) for idx in range(num_envs)]

    if use_async:
        return AsyncVectorEnv(env_fns, context="spawn")
    return SyncVectorEnv(env_fns)


@dataclass
class StepResult:
    """Result of a vectorized environment step."""

    observations: dict[str, NDArray[np.floating | np.integer]]
    rewards: NDArray[np.float64]
    terminateds: NDArray[np.bool_]
    truncateds: NDArray[np.bool_]
    infos: dict[str, NDArray]


class SimpleVecEnvWrapper:
    """
    Simple wrapper around Gymnasium's vectorized environments.

    Handles batched step/reset operations with minimal overhead.
    """

    __slots__ = ("vec_env", "num_envs", "action_size")

    def __init__(self, vec_env: AsyncVectorEnv | SyncVectorEnv) -> None:
        """
        Initialize the wrapper.

        Args:
            vec_env: Vectorized environment instance.
        """
        self.vec_env = vec_env
        self.num_envs: int = vec_env.num_envs
        action_space = vec_env.single_action_space
        if not isinstance(action_space, spaces.Discrete):
            msg = f"SimpleVecEnvWrapper requires a Discrete action space, got {type(action_space).__name__}."
            raise TypeError(msg)
        self.action_size: int = int(action_space.n)

    def reset(
        self, seed: int | None = None,
    ) -> tuple[dict[str, NDArray], dict[str, NDArray]]:
        """
        Reset all environments.

        Args:
            seed: Optional seed for reproducibility.

        Returns
        -------
            Tuple of (observations, infos).
        """
        return self.vec_env.reset(seed=seed)

    def step(self, actions: NDArray[np.int64]) -> StepResult:
        """
        Step all environments.

        Args:
            actions: Array of shape [num_envs] with action indices.

        Returns
        -------
            StepResult containing observations, rewards, terminateds, truncateds, infos.
        """
        obs: dict[str, NDArray]
        rewards: NDArray[np.floating]
        terminateds: NDArray[np.bool_]
        truncateds: NDArray[np.bool_]
        obs, rewards, terminateds, truncateds, infos = self.vec_env.step(actions)
        return StepResult(
            observations=obs,
            rewards=rewards.astype(np.float64),
            terminateds=terminateds,
            truncateds=truncateds,
            infos=infos,
        )

    @property
    def observation_space(self) -> spaces.Space:
        """Get the single environment's observation space."""
        return self.vec_env.single_observation_space

    @property
    def action_space(self) -> spaces.Space:
        """Get the single environment's action space."""
        return self.vec_env.single_action_space

    def close(self) -> None:
        """Close vectorized environment."""
        self.vec_env.close()

