"""
Test all classes from base_agents.py.

Attention! This test file only accepts nets with 3 actions and fails as soon a 4th action is done.
Please reset the environment after all 3 steps.
"""

from __future__ import annotations

import copy
from typing import ClassVar

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

# Feedback values planted in a fake worker result, one per supported feedback type.
REWARD_VALUE = 4.2
MAX_LOADING_VALUE = 99.0
NMINUS1_VALUE = 120.0


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
    agent2 = copy.deepcopy(agent)
    info = agent2.env.state_to_info()
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




def test_baseagent_act_is_abstract(env_config: dict) -> None:
    """A subclass that forwards to ``BaseAgent.act`` must be told to implement it itself.

    :param env_config: Environment configuration.
    :type env_config: dict
    """

    class ForwardingAgent(BaseAgent):
        def act(self, *args: object, **kwargs: object) -> int | np.integer:
            return super().act(*args, **kwargs)

    env = PPTopoGym(env_config)
    agent = ForwardingAgent(env.action_space)

    with pytest.raises(NotImplementedError, match="Subclass must implement act method"):
        agent.act()


def test_basegreedyagent_requires_an_env_config(env_config: dict) -> None:
    """A greedy agent simulates on its own env, so an empty config leaves it unusable.

    :param env_config: Environment configuration.
    :type env_config: dict
    """
    env = PPTopoGym(env_config)

    with pytest.raises(ValueError, match="Environment must be provided"):
        BaseGreedyAgent(env.action_space, env_config={})


def test_feedback_func_defaults_to_the_agents_feedback_type(env_config: dict) -> None:
    """Omitting ``feedback_type`` falls back to the one the agent was built with.

    :param env_config: Environment configuration.
    :type env_config: dict
    """
    env_config = copy.deepcopy(env_config)
    env = PPTopoGym(env_config)
    agent = BaseGreedyAgent(env.action_space, copy.deepcopy(env_config), feedback_type="reward")

    result = {"reward": REWARD_VALUE, "max_loading": 99.0, "nminus1": 120.0}

    assert agent.feedback_func(result) == agent.feedback_func(result, "reward") == REWARD_VALUE


@pytest.mark.parametrize(
    ("feedback_type", "expected"),
    [("reward", REWARD_VALUE), ("line_loadings", MAX_LOADING_VALUE), ("nminus1", NMINUS1_VALUE)],
)
def test_feedback_func_reads_the_matching_result_key(
    feedback_type: str, expected: float, env_config: dict,
) -> None:
    """Each feedback type maps to its own key in the worker result.

    :param feedback_type: The feedback type under test.
    :type feedback_type: str
    :param expected: The value the metric should read out of the result dict.
    :type expected: float
    :param env_config: Environment configuration.
    :type env_config: dict
    """
    env_config = copy.deepcopy(env_config)
    env = PPTopoGym(env_config)
    agent = BaseGreedyAgent(env.action_space, copy.deepcopy(env_config))

    result = {"reward": REWARD_VALUE, "max_loading": MAX_LOADING_VALUE, "nminus1": NMINUS1_VALUE}

    assert agent.feedback_func(result, feedback_type) == expected


def test_feedback_func_rejects_an_unknown_feedback_type(env_config: dict) -> None:
    """An unrecognised feedback type is a configuration error, not a silent default.

    :param env_config: Environment configuration.
    :type env_config: dict
    """
    env_config = copy.deepcopy(env_config)
    env = PPTopoGym(env_config)
    agent = BaseGreedyAgent(env.action_space, copy.deepcopy(env_config))

    with pytest.raises(ValueError, match="Unsupported selection criterion: bogus"):
        agent.feedback_func({}, "bogus")


def test_act_returns_donothing_below_the_overload_threshold(env_config: dict) -> None:
    """Within limits the search cannot beat DoNothing, so it is skipped entirely.

    :param env_config: Environment configuration.
    :type env_config: dict
    """
    env_config = copy.deepcopy(env_config)
    env = PPTopoGym(env_config)
    agent = BaseGreedyAgent(
        env.action_space, copy.deepcopy(env_config), overload_threshold=100,
    )

    # An empty info dict would otherwise be replayed onto the env; reaching the early
    # return means neither the state restore nor the simulation ran.
    action = agent.act({"line_loadings": np.array([10.0, 20.0])}, info={})

    assert action == 0


def test_dc_agent_verifies_its_pick_with_an_ac_power_flow(env_config: dict) -> None:
    """DC scoring is only a ranking; the returned action must still converge under AC.

    :param env_config: Environment configuration.
    :type env_config: dict
    """
    env_config = copy.deepcopy(env_config)
    env = PPTopoGym(env_config)
    agent = BaseGreedyAgent(env.action_space, copy.deepcopy(env_config), pf_type="dc")
    assert agent.dc_approximation

    obs, info = agent.env.reset(options={"index": 0})
    action = agent.act(obs, info)

    assert 0 <= int(action) < agent.action_space.n


def test_dc_agent_falls_back_to_donothing_when_nothing_converges(
    env_config: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If every candidate fails the AC check, the agent keeps the grid as it is.

    :param env_config: Environment configuration.
    :type env_config: dict
    :param monkeypatch: pytest monkeypatch fixture.
    :type monkeypatch: pytest.MonkeyPatch
    """
    env_config = copy.deepcopy(env_config)
    env = PPTopoGym(env_config)
    agent = BaseGreedyAgent(env.action_space, copy.deepcopy(env_config), pf_type="dc")

    obs, info = agent.env.reset(options={"index": 0})

    class _NonConverging:
        info: ClassVar[dict] = {"powerflow_converged": False}

    monkeypatch.setattr(agent.env, "simulation", lambda _actions: [_NonConverging()])

    assert agent.act(obs, info) == 0
