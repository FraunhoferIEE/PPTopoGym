from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium import spaces

from pandapower_env.agents.base_agents import (
    BaseAgent,
    BaseGreedyAgent,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    import numpy as np
    from gymnasium import spaces



class DoNothingAgent(BaseAgent):
    """
    Simple agent doing nothing.

    The DoNothing action is 0.
    The agent does not change the state of the environment.
    It can be applied to GymEnvPP directly.
    """

    def __init__(self, action_space: spaces.Discrete) -> None:
        """
        Nothing needs to be initialized.

        It still takes the action_space as an argument to be compatible with the other agents.

        :param action_space: The action space of the environment.
        :type action_space: spaces.Discrete
        """
        super().__init__(action_space)

    def act(self, observation: dict | None = None) -> int:  # noqa: ARG002
        """
        Return the action to do nothing (0 by default).

        :return: The action corresponding to "doing nothing" (assumed to be `0`).
        :rtype: int
        """
        return 0


class RandomAgent(BaseAgent):
    """
    Random agent which selects actions randomly.

    The agent can be applied to GymEnvPP directly.
    """

    def __init__(self, action_space: spaces.Discrete) -> None:
        """
        Initialize the action space.

        :param action_space: The action space of the environment.
        :type action_space: spaces.Discrete
        """
        super().__init__(action_space)

    def act(self, observation: dict | None = None) -> np.integer:  # noqa: ARG002
        """
        Randomly sample an action.

        :return: The randomly sampled action.
        :rtype: np.integer
        """
        # Randomly sample an action
        return self.action_space.sample()

class GreedyAgent(BaseGreedyAgent):
    """
    Greedy agent which selects the action with the highest reward.

    Goal: simulate all actions and take the best action.
    """

    def __init__( #noqa: PLR0913
        self,
        action_space: spaces.Discrete,
        env_config: dict,
        feedback_type: str = "line_loadings",
        n_workers: int | None = None,
        dc_approximation: str = "ac",
        overload_threshold: int = 0,
    ) -> None:
        """
        Greedy agent which directly selects the action with the highest reward.

        It directly inherits from the BaseGreedyAgent class.


        :param action_space: The action space of the environment.
        :type action_space: spaces.Discrete
        :param env_config: The simulation environment. (as config)
        :type env_config: dict
        :param selection_criterion: The selection criterion for the best action.
        :type selection_criterion: str
        :param feedback_type: The type of feedback to be used.
        :type feedback_type: str
        """
        super().__init__(action_space, env_config, feedback_type, n_workers, dc_approximation, overload_threshold)

    def act(
        self,
        observation: dict,
        info: dict,
        max_actions: int | None = None,
        action_list: (
            list[int | np.integer]
            | Generator[int, None, None]
            | Iterable[int]
            | range
            | None
        ) = None,
    ) -> int | np.integer:
        """
        Agent acts by selecting the best action.

        For explanations to the parameters, see the BaseGreedyAgent class.

        :param observation: The environment observation at the current state.
        :type observation: dict
        :param max_actions: The maximum number of actions to consider.
        :type max_actions: int | None
        :param action_list: A subset of actions to evaluate.
        :type action_list: list[int] | Generator[int, None, None] | Iterable[int] | range | None
        :return: The best action to take.
        :rtype: int | np.integer
        """
        return super().act(
            observation,
            info,
            max_actions,
            action_list,
        )
