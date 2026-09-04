import numpy as np
import pytest

from pandapower_env.agents.base_agents import BaseAgent
from pandapower_env.agents.benchmark_agents import (
    DoNothingAgent,
    GreedyAgent,
    RandomAgent,
)
from pandapower_env.environments.simulation_env import PPTopoGym


def test_donothingagent(env_config) -> None:
    env = PPTopoGym(env_config)
    # Create an instance of DoNothingAgent
    agent = DoNothingAgent(env.action_space)
    action = agent.act()
    assert action == 0
    assert env.net.switch["closed"].all(), "Not all switches are closed."
    assert env.net.line.loc[:, "in_service"].all(), "Not all lines are in service."
    assert isinstance(agent, BaseAgent), "agent is not an instance of BaseAgent"


def test_randomagent(env_config) -> None:
    env = PPTopoGym(env_config)
    agent = RandomAgent(env.action_space)
    # Call the act() method
    action = agent.act()
    # Assert that the action is an integer
    assert isinstance(
        action,
        (int, np.integer),
    ), f"action {action} of type {type(action)} is not an integer or integer-like value."
    # Assert that the action is within the action space
    assert (
        action in env.action_space
    ), f"action {action} of type {type(action)} is not in the action space."
    assert isinstance(agent, BaseAgent), "agent is not an instance of BaseAgent"


def test_greedyagent(env_config) -> None:
    # Create a mock environment
    env = PPTopoGym(env_config)
    # Create an instance of GreedyAgent
    agent = GreedyAgent(env.action_space, env_config)
    observation = agent.env.create_observation()
    info = agent.env.state_to_info()
    action = agent.act(observation, info)
    assert isinstance(action, (int, np.integer)), "action is not an integer."
    assert action in agent.action_space, "action is not in the action space."
    assert isinstance(agent, BaseAgent), "agent is not an instance of BaseAgent"

def test_greedynminus1agent(env_config) -> None:
    env = PPTopoGym(env_config)
    agent = GreedyAgent(env.action_space, env_config, feedback_type="nminus1")

    # Prepare mock result dict[str, float] as expected by feedback_func
    mock_reward = 2.0
    mock_lineloading = np.array([0.25, 0.75, 0.6], dtype=float)
    mock_nminus1_value = 0.42

    mock_result = {
        "reward": float(mock_reward),
        "line_loadings": float(np.max(mock_lineloading)),  # greedy uses scalar criterion
        "max_loading": float(np.max(mock_lineloading)),
        "nminus1": float(mock_nminus1_value),
    }

    # Check feedback_func for each supported key
    assert agent.feedback_func(mock_result, "nminus1") == pytest.approx(mock_nminus1_value)
    assert agent.feedback_func(mock_result, "reward") == pytest.approx(mock_reward)
    assert agent.feedback_func(mock_result, "line_loadings") == pytest.approx(float(np.max(mock_lineloading)))

    # act should be robust and return a valid action
    observation = agent.env.create_observation()
    info = agent.env.state_to_info()
    try:
        best_action = agent.act(observation=observation, info=info)
        assert isinstance(best_action, (int, np.integer))
        assert best_action in agent.action_space
        assert agent.feedback_type == "nminus1"
    except ValueError:
        pytest.fail("ValueError not correctly handled for GreedyAgent.act")
    assert isinstance(agent, BaseAgent)
