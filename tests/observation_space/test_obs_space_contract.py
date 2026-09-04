"""Contract tests for the observation space.

Every observation the env emits must satisfy the contract its ``spaces.Dict`` declares:
correct dtype, correct shape, values inside the Box bounds -- at reset and across
topology-changing steps (busbar splits). These tests guard against the recurring class
of "obs boxes wrong" bugs (wrong bounds, dtype upcasts, shape drift after splits).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pandapower_env.environments.simulation_env import PPTopoGym


def _assert_obs_matches_space(env: PPTopoGym, obs: dict) -> None:
    """Assert every space key is present with matching shape, dtype, and bounds.

    :param env: The environment whose ``observation_space`` declares the contract.
    :param obs: The observation dict to validate against that contract.
    :raises AssertionError: If a shape, dtype, or Box bound is violated.
    """
    for key, space in env.observation_space.spaces.items():
        value = np.asarray(obs[key])
        assert value.shape == space.shape, f"{key}: shape {value.shape} != declared {space.shape}"
        assert value.dtype == space.dtype, f"{key}: dtype {value.dtype} != declared {space.dtype}"
        assert space.contains(value), (
            f"{key}: value outside declared Box "
            f"[min={value.min() if value.size else None}, "
            f"max={value.max() if value.size else None}]"
        )


def test_observation_matches_space_at_reset(simenv: PPTopoGym) -> None:
    """The reset observation must satisfy the declared observation space exactly.

    Regression: ``transformer_tap_position`` declared int32 but float clip bounds
    upcast the values to float64, so ``observation_space.contains()`` was False at t=0.
    """
    obs, _ = simenv.reset(options={"index": 0})
    _assert_obs_matches_space(simenv, obs)


def test_observation_matches_space_across_busbar_split(simenv: PPTopoGym) -> None:
    """Shapes/dtypes/bounds must hold after a busbar split adds an electrical node."""
    obs_before, _ = simenv.reset(options={"index": 0})
    obs_after, _, _, _, _ = simenv.step(1)  # action 1 splits substation 0

    _assert_obs_matches_space(simenv, obs_after)
    for key in simenv.observation_space.spaces:
        assert np.asarray(obs_after[key]).shape == np.asarray(obs_before[key]).shape, (
            f"{key}: shape changed across a topology-changing step"
        )


def test_switch_positions_are_non_negative(simenv: PPTopoGym) -> None:
    """``switch_positions`` must stay within its declared ``[0, 1]`` Box.

    Regression: the declared Box allowed ``low=-1``, which admits values the
    0/1 switch states can never take.
    """
    obs, _ = simenv.reset(options={"index": 0})
    assert obs["switch_positions"].min() >= 0
    assert obs["switch_positions"].max() <= 1
