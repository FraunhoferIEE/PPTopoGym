from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np

from pandapower_env.environments.simulation_env import PPTopoGym

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from gymnasium import spaces

from joblib import Parallel, delayed

from pandapower_env.agents.greedy_worker import evaluate_actions


class BaseAgent(ABC):
    """
    Base class for all agents.

    It creates a environment, if a env_config is provided.

    The agent class initializes the action space and the environment.
    :param action_space: The action space of the environment.
    :type action_space: spaces.Discrete
    :param env_config: The environment the agent interacts with. (as config)
    :type env_config: dict

    Each agent must have a basic function "act" which returns the output of an action.
    """

    def __init__(
        self,
        action_space: spaces.Discrete,
        env_config: dict | None = None,
    ) -> None:
        self.action_space = action_space
        if env_config:
            self.env = PPTopoGym(env_config)
            self.env.reset()

    @abstractmethod
    def act(self, *args: Any, **kwargs: Any) -> int | np.integer:  # noqa: ANN401
        """
        Use an action to act on the environment, given an observation.

        :return: The action the agent takes.
        :rtype: int | np.integer
        """
        msg = "Subclass must implement act method"
        raise NotImplementedError(msg)


class BaseGreedyAgent(BaseAgent):
    """
    Greedy agent which selects the action with the best feedback.

    This agent evaluates all possible actions in the action space by simulating their
    effects on the environment. It selects the action that optimizes the feedback
    based on the specified `selection_criterion` (e.g., "min" or "max"). The feedback
    can be derived from rewards or other metrics like line loading.

    The Workflow of all inherited classes should be:
        1. Call super().act
        2. call simulate_feedback to get all outputs
        3. Gather the correct feedback with the feedback_func

        The inherited class can overwrite the feedback_func and the simulate_feedback func
        to implement custom behavior.

    Attributes
    ----------
        action_space (list): The list of possible actions for the agent.
        env (PPTopoGym): The environment in which the agent operates.
        selection_criterion (str): The criterion for selecting the best action ("min" or "max").
        feedback_type (str): The type of feedback used for decision-making (e.g., "reward" or "line_loadings").

    Methods
    -------
        feedback_func(output, feedback_type):
            Computes feedback based on the given output and feedback type.
        simulate_feedback(action, feedback_type):
            Simulates the effect of an action and computes the corresponding feedback.
        act(calculate_simulation):
            Simulates all actions, evaluates their feedback, and selects the best action.
    """

    def __init__( #noqa: PLR0913
        self,
        action_space: spaces.Discrete,
        env_config: dict,
        feedback_type: str = "line_loadings",
        n_workers: int | None = None,
        pf_type: str  = "ac",
        overload_threshold: int = 0,
        seed: int | None = None,
    ) -> None:
        """
        Initialize the BaseGreedyAgent.

        :param action_space: The list of possible actions.
        :type action_space: spaces.Discrete
        :param env_config: The environment in which the agent operates.
        :type env_config: dict
        :param selection_criterion: Criterion for selecting the best action ("min" or "max").
        :type selection_criterion: str
        :param feedback_type: Type of feedback used for decision-making ("reward" or "line_loadings").
        :type feedback_type: str
        :param n_workers: How many workers to use for simulation parallelization
        :type n_workers: int
        :param seed: Seed for this agent's RNG, which drives the ``max_actions``
            subsampling in :meth:`act`. Pass a seed to make a capped greedy run
            reproducible; ``None`` keeps the previous non-reproducible behaviour.
        :type seed: int | None

        :raises ValueError: If the selection_criterion is not supported.
        """
        super().__init__(action_space, env_config)
        if not hasattr(self, "env"):
            msg = "Environment must be provided."
            raise ValueError(msg)
        self.n_workers = n_workers if n_workers is not None else 1
        self.feedback_type = feedback_type
        self.dc_approximation = pf_type == "dc"
        self.overload_threshold = overload_threshold
        # Per-agent RNG. Previously act() drew from an unseeded np.random.default_rng(),
        # so a capped greedy run could not be reproduced even with a fully seeded env.
        self.np_random = np.random.default_rng(seed)

    def _subsample_actions(
        self,
        actions: np.ndarray,
        max_actions: int,
    ) -> np.ndarray:
        """
        Draw ``max_actions`` distinct candidate actions from ``actions``.

        Uses this agent's seeded RNG, so the same seed yields the same subsample.
        Returns ``actions`` unchanged when it already fits within the budget.

        :param actions: Candidate action indices to choose from.
        :type actions: np.ndarray
        :param max_actions: Maximum number of actions to keep.
        :type max_actions: int
        :return: The retained action indices, without repeats.
        :rtype: np.ndarray
        """
        if len(actions) <= max_actions:
            return actions
        return self.np_random.choice(actions, size=max_actions, replace=False)

    def feedback_func(self,
            result: dict[str, float],
            feedback_type: str | None = None,
    ) -> float:
        """
        Evaluate the feedback for one action.

        selection_criterion is min or max.
        feedback_type is reward or line_loadings.
        These are lists or numpy arrays in the observation-dict.

        :param result: The result of an environment step.
        :type result: dict[str, Any]
        :param feedback_type: The type of feedback to compute ("reward" or "line_loadings").
        :type feedback_type: str
        :return: The computed feedback value.
        :rtype: float
        :raises ValueError: If an unsupported feedback_type or selection_criterion is provided.

        This function can be more modularized as soon as metrics are implemented.
        """
        if feedback_type is None:
            feedback_type = self.feedback_type
        worst_loading = 1000.0  # bad worst loading and reward

        if feedback_type == "reward":
            return result.get("reward", -worst_loading)
        if feedback_type == "line_loadings":
            return result.get("max_loading", worst_loading)
        if feedback_type == "nminus1":
            return result.get("nminus1", worst_loading)
        msg = f"Unsupported selection criterion: {feedback_type}"
        raise ValueError(msg)

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
        Greedy agent which selects the action with the best feedback.

        Simulate all actions and select the best one based on the feedback.+

        :param observation: The observation of the environment.
        :type observation: dict
        :param info: Additional information about the environment.
        :type info: dict
        :param max_actions: The maximal number of actions to simulate.
        :type max_actions: int | None
        :param action_list: A list of actions to simulate.
        :type action_list: list[int | np.integer] | Generator[int, None, None] | Iterable[int] | range | None
        :raises ValueError: If no valid actions are available for aggregation.
        :return: The best action based on the selection criterion.
        :rtype: int
        """
        # workflow:
        # simulate action
        # use output to calculate feedback
        # select best action

        # Decided before any work is done. The grid is within limits, so DoNothing is the answer
        # whatever the search would find -- and reaching this check further down meant first
        # replaying the whole action log through state_from_info (a power flow) and packing the
        # worker payloads, all for an action that never depended on them.
        if max(observation["line_loadings"]) <= self.overload_threshold:
            return 0

        if info:  # if observation is provided, convert it to state
            self.env.state_from_info(info)

        temp_actions = list(range(self.action_space.n)) if action_list is None \
            else [int(a) for a in action_list]
        actions_to_simulate = np.array(temp_actions, dtype=int)
        if max_actions is not None:
            actions_to_simulate = self._subsample_actions(actions_to_simulate, max_actions)

        # minimal payloads
        static_blob = getattr(self.env, "static_net_blob", None) or self.env.dump_static_net_bytes()
        base_topology = self.env.snapshot_topology()
        prof = self.env.get_profile_slice(self.env.index)

        pf_mode = "dc" if self.dc_approximation else "ac"
        need_n1 =  self.feedback_type == "nminus1"

        # One task per worker, not one per candidate: static_blob is the whole net serialized
        # as JSON and is identical for every candidate, so a task-per-action dispatch pickled
        # it across the process boundary O(n_actions) times. df_actions is likewise read once
        # here instead of a .loc row build per candidate.
        action_rows = self.env.df_actions.to_dict("index")
        chunks = [
            chunk for chunk in np.array_split(
                actions_to_simulate, min(max(self.n_workers, 1), len(actions_to_simulate)),
            ) if len(chunk)
        ]
        chunked_results = Parallel(
            n_jobs=self.n_workers,
            backend="loky",
            prefer="processes",
        )(
            delayed(evaluate_actions)(
                static_net_blob=static_blob,
                base_topology=base_topology,
                profile_slice=prof,
                action_rows=[action_rows[int(a)] for a in chunk],
                pf_mode=pf_mode,
                need_n1=need_n1,
                backend=self.env.backend_name,
            )
            for chunk in chunks
        )
        # joblib preserves task order, so flattening keeps results aligned with
        # actions_to_simulate -- which the scoring below indexes into positionally.
        results = [result for chunk_results in chunked_results for result in chunk_results]

        scores = [self.feedback_func(r, self.feedback_type) for r in results]
        order = np.argsort(scores)
        if self.feedback_type == "reward":
            order = order[::-1]  # max reward

        if self.dc_approximation:
            # DC was used for scoring verify with  AC on the env
            for idx in order:
                a = int(actions_to_simulate[idx])
                out = self.env.simulation([a])[0]  # AC check on live env
                if out.info.get("powerflow_converged", False):
                    return a
            return 0
        return int(actions_to_simulate[order[0]])
