from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium import spaces

from pandapower_env.agents.base_agents import (
    BaseAgent,
    BaseGreedyAgent,
)

if TYPE_CHECKING:

    import numpy as np
    from gymnasium import spaces



class DoNothingAgent(BaseAgent):
    """
    Simple agent doing nothing.

    The DoNothing action is 0.
    The agent does not change the state of the environment.
    It can be applied to GymEnvPP directly.
    """

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
        seed: int | None = None,
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
        :param seed: Seed for the ``max_actions`` subsampling RNG.
        :type seed: int | None
        """
        super().__init__(
            action_space, env_config, feedback_type, n_workers, dc_approximation,
            overload_threshold, seed,
        )
