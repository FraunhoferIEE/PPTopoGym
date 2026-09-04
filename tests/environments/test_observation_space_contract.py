"""
Pins the observation-space contract that vectorized environments depend on.

Node-aggregated observations have length ``n_nodes``, but ``n_nodes`` *grows as substations
split*, while ``define_observation_space`` freezes the shape at the reset topology. The
declared space is therefore violated as soon as an action splits a substation, which is what
makes ``SyncVectorEnv`` / ``AsyncVectorEnv`` fail with::

    ValueError: could not broadcast input array from shape (31,) into shape (30,)

That is a pre-existing bug and is **not fixed here**, because padding the observations to a
static bound changes their shape and so would break consumers' network input dimensions.
:func:`test_observation_shapes_are_stable_across_topologies` is marked ``xfail`` to record
it: it flips to a pass the moment the shapes are made static, which is the signal that the
vectorized path is safe again.

The static bound to pad to is ``n_nodes(reset topology) + n_double_busbar_substations`` --
splitting one substation can add at most one electrical node.
"""

from __future__ import annotations

import numpy as np
import pytest

from pandapower_env.environments.simulation_env import PPTopoGym

N_STEPS = 40


def _observation_shapes(env: PPTopoGym, steps: int = N_STEPS) -> dict[str, set[tuple[int, ...]]]:
    """Collect the shapes each observation key takes while stepping through the action space."""
    shapes: dict[str, set[tuple[int, ...]]] = {}
    rng = np.random.default_rng(0)
    env.reset(options={"index": 0})

    for _ in range(steps):
        action = int(rng.integers(0, len(env.df_actions)))
        observation, _reward, terminated, truncated, _info = env.step(action)
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                shapes.setdefault(key, set()).add(value.shape)
        if terminated or truncated:
            env.reset(options={"index": 0})
    return shapes


@pytest.mark.xfail(
    reason="Known bug: node-aggregated observations resize when a substation splits.",
    strict=False,
)
def test_observation_shapes_are_stable_across_topologies(simenv) -> None:
    """Every observation keeps one shape, whatever the topology -- required for vector envs."""
    varying = {key: sorted(s) for key, s in _observation_shapes(simenv).items() if len(s) > 1}
    assert not varying, f"observation keys changed shape across topologies: {varying}"


@pytest.mark.xfail(
    reason="Known bug: observations outgrow the declared space when a substation splits.",
    strict=False,
)
def test_observations_stay_inside_the_declared_space(simenv) -> None:
    """Every emitted observation satisfies ``observation_space.contains``."""
    rng = np.random.default_rng(0)
    simenv.reset(options={"index": 0})

    for step in range(N_STEPS):
        action = int(rng.integers(0, len(simenv.df_actions)))
        observation, _reward, terminated, truncated, _info = simenv.step(action)
        assert simenv.observation_space.contains(observation), (
            f"observation left the declared space at step {step} (action {action})"
        )
        if terminated or truncated:
            simenv.reset(options={"index": 0})


def test_reset_observation_matches_the_declared_space(simenv) -> None:
    """The reset observation -- the one the space is built from -- always fits it."""
    observation, _info = simenv.reset(options={"index": 0})
    assert simenv.observation_space.contains(observation)


def test_static_obs_space_keeps_shapes_constant(env_config) -> None:
    """``static_obs_space`` pads node observations, so their shape never changes."""
    static_env = PPTopoGym({**env_config, "static_obs_space": True})
    varying = {key: sorted(s) for key, s in _observation_shapes(static_env).items() if len(s) > 1}
    assert not varying, f"static_obs_space still let keys change shape: {varying}"


def test_static_obs_space_observations_stay_in_the_space(env_config) -> None:
    """With ``static_obs_space`` every emitted observation satisfies the declared space."""
    env = PPTopoGym({**env_config, "static_obs_space": True})
    rng = np.random.default_rng(0)
    env.reset(options={"index": 0})

    for step in range(N_STEPS):
        action = int(rng.integers(0, len(env.df_actions)))
        observation, _reward, terminated, truncated, _info = env.step(action)
        assert env.observation_space.contains(observation), (
            f"observation left the declared space at step {step} (action {action})"
        )
        if terminated or truncated:
            env.reset(options={"index": 0})


def test_static_obs_space_does_not_change_rewards(env_config) -> None:
    """Padding is cosmetic: the rewards of both modes are identical."""
    default_env = PPTopoGym(env_config)
    static_env = PPTopoGym({**env_config, "static_obs_space": True})
    for env in (default_env, static_env):
        env.reset(options={"index": 0})

    actions = [int(a) for a in default_env.df_actions.index] * 2
    default_rewards = [float(default_env.step(a)[1]) for a in actions]
    static_rewards = [float(static_env.step(a)[1]) for a in actions]
    assert default_rewards == pytest.approx(static_rewards)


def test_static_obs_space_is_off_by_default(simenv) -> None:
    """The flag is opt-in, so existing consumers keep their observation shapes."""
    assert simenv.static_obs_space is False
    node_length = simenv.observation_space["bus_voltage_magnitude"].shape[0]
    assert node_length <= simenv._max_n_nodes
