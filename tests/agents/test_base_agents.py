"""
Test all classes from base_agents.py.

Attention! This test file only accepts nets with 3 actions and fails as soon a 4th action is done.
Please reset the environment after all 3 steps.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

# Hope you don't be imprisoned by legacy Python code :)
# Get the current directory (tests directory)
from pandapower_env.action_space.action_space import (
    _create_unitary_line_actions,  # noqa: F401
    add_actions_substation_line_switching,  # noqa: F401
    create_actions_df,  # noqa: F401
    create_unitary_substation_action,  # noqa: F401
    enforce_rules,  # noqa: F401
    return_donothing_action,  # noqa: F401
)
from pandapower_env.agents.base_agents import (
    BaseAgent,
    BaseGreedyAgent,
)
from pandapower_env.environments.gym_env_pp import BaseEnvPP  # noqa: F401
from pandapower_env.environments.simulation_env import PPTopoGym


def test_baseagent(env_config: dict) -> None:
    """
    Test the BaseAgent class.

    :param env_config: Environment configuration.
    :type env_config: dict
    """

    class DummyAgent(BaseAgent):
        def act(self) -> int:
            return 0

    # test initialization
    env = PPTopoGym(env_config)
    agent = DummyAgent(env.action_space, env_config)  # Nutze DummyAgent statt BaseAgent
    assert agent.action_space == env.action_space


def test_basegreedyagent(env_config: dict) -> None:
    """
    Test the BaseGreedyAgent class.

    :param env_config: Environment configuration.
    :type env_config: dict
    """
    # test initialization
    env_config = copy.deepcopy(env_config)
    env = PPTopoGym(env_config)
    agent = BaseGreedyAgent(
        env.action_space,
        env_config,
        feedback_type="line_loadings",
    )
    assert agent.action_space == env.action_space

    agent = BaseGreedyAgent(
        env.action_space,
        env_config,
        feedback_type="line_loadings",
    )
    assert agent.action_space == env.action_space

    # New contract: feedback_func expects a dict[str, float]
    true_reward = 1.0
    line_loadings = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    max_loading = float(np.max(line_loadings))
    nminus1_value = 0.37

    mock_result: dict[str, float] = {
        "reward": true_reward,
        # store the scalar criterion used by greedy selection
        "line_loadings": max_loading,
        "max_loading": max_loading,  # some implementations use this key
        "nminus1": nminus1_value,
    }

    # feedback_func lookups
    assert agent.feedback_func(mock_result, "reward") == pytest.approx(true_reward)
    assert agent.feedback_func(mock_result, "line_loadings") == pytest.approx(max_loading)
    assert agent.feedback_func(mock_result, "nminus1") == pytest.approx(nminus1_value)

    # act must return a valid action
    observation = agent.env.create_observation()
    info = agent.env.state_to_info()
    result = agent.act(observation, info)
    assert isinstance(result, (int, np.integer))
    assert result in agent.action_space

    # act with constrained candidate list
    result = agent.act(observation, info, max_actions=1, action_list=[0])
    assert isinstance(result, (int, np.integer))
    assert result == 0

    # even if max_actions > len(list), it should still work and pick from provided list
    result = agent.act(observation, info, max_actions=5, action_list=[0])
    assert isinstance(result, (int, np.integer))
    assert result == 0





