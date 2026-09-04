from __future__ import annotations

import copy
import importlib
import io
import logging
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Callable

import gymnasium as gym
import numpy as np
import pandapower as pp
from gymnasium import spaces
from typing_extensions import override

from pandapower_env.action_space.action_space import (
    create_actions_df,
    verify_action,
)
from pandapower_env.environments.gym_env_pp import BaseEnvPP, deepcopy_net_sharing_profiles
from pandapower_env.observation_space.obs_space_utils import (
    ObservationConfig,
    ObsType,
    build_info_observation_registry,
    build_observation_registry,
)
from pandapower_env.observation_space.pp_to_observation import (
    has_gen_results,
    has_load_results,
    line_loading_max,
    nminus1_line_loading_max,
    system_losses_sum,
    total_gen_p,
    total_load_p,
)
from pandapower_env.toolbox.env_specs import LoggedArray, Output
from pandapower_env.toolbox.utils import (
    total_active_overload_mva,
)
from pandapower_env.toolbox.utils_graph_obs import (
    batch_observations,
    create_adjacency_matrix,
    get_observation,
    get_raw_observation,
    make_obs_cache,
    n_nodes,
    n_static_slots,
    node_slot_map,
)

if TYPE_CHECKING:
    from collections.abc import Generator

    import pandas as pd

logger = logging.getLogger(__name__)


def _validated_custom_observations(observations: list[dict] | None) -> dict[str, Callable]:
    """Check the ``observation`` config entries and return them as ``{name: function}``.

    Each entry must carry a ``name``, a ``function`` (called with the environment) and a
    ``spaces`` box. An empty or missing config means the environment has no custom
    observations.

    :param observations: the ``env_config["observation"]`` list, or None if not configured
    :type observations: list[dict] | None
    :return: mapping of observation name to its function (empty if none are configured)
    :rtype: dict[str, Callable]
    :raises ValueError: if an entry is missing ``name``, ``function`` or ``spaces``
    """
    if not observations:
        return {}
    for observation in observations:
        for required_key in ("name", "function", "spaces"):
            if required_key not in observation:
                msg = f"No {required_key} in observation. Please change!"
                raise ValueError(msg)
    return {observation["name"]: observation["function"] for observation in observations}


def _copy_config_sharing_profiles(env_config: dict) -> dict:
    """Deep-copy an env config, sharing the timeseries tables instead of copying them.

    Equivalent to ``copy.deepcopy(env_config)`` for every purpose the package has -- the
    resulting config builds an identical environment -- except that the profile DataFrames,
    whether they sit in ``config["net"].profiles`` or in ``config["profiles"]``, are the *same
    objects* as in the source config rather than duplicates. They are read-only once
    :meth:`BaseEnvPP.setup_profiles` / :meth:`BaseEnvPP.setup_profiles_from_config` has consumed
    them, and they dominate an environment's memory (see :func:`deepcopy_net_sharing_profiles`).

    The ``config["profiles"]`` dicts themselves *are* copied, one shallow level deep, so a caller
    that later swaps a frame in its own config cannot reach into the stored one.

    :param env_config: the environment configuration to copy
    :type env_config: dict
    :return: a copy safe to hand to another ``PPTopoGym``
    :rtype: dict
    """
    shared_keys = ("net", "profiles")
    config = copy.deepcopy({key: value for key, value in env_config.items() if key not in shared_keys})

    if "net" in env_config:
        net = env_config["net"]
        config["net"] = deepcopy_net_sharing_profiles(net) if isinstance(net, pp.pandapowerNet) else copy.deepcopy(net)
    if "profiles" in env_config:
        config["profiles"] = {
            element: dict(variables) for element, variables in env_config["profiles"].items()
        }
    return config


@dataclass(slots=True)
class _ActionPlan:
    """Precomputed positional writes for one action of ``df_actions``.

    Each field is a *positional* index array into the corresponding pandapower table column
    (or ``None`` when the action does not touch that table), paired with the values to write.
    Built once per environment by :meth:`PPTopoGym._build_action_plans` so that applying an
    action is a handful of numpy fancy-index assignments instead of pandas label lookups.
    """

    open_switches: np.ndarray | None = None
    closed_switches: np.ndarray | None = None
    lines: np.ndarray | None = None
    line_in_service: np.ndarray | None = None
    trafos: np.ndarray | None = None
    tap_pos: np.ndarray | None = None


def _positional_index(table_index: pd.Index) -> dict[int, int] | None:
    """Map table labels to row positions, or ``None`` when labels already are positions.

    ``None`` means the table is indexed ``0..n-1`` so a label *is* its position and no
    translation is needed -- the common pandapower case.

    :param table_index: the index of a pandapower element table
    :return: label -> position mapping, or None if the index is already ``0..n-1``
    """
    labels = table_index.to_numpy()
    if len(labels) and np.array_equal(labels, np.arange(len(labels))):
        return None
    return {int(label): position for position, label in enumerate(labels)}


def _fill_nan(arr: np.ndarray, fill: float) -> np.ndarray:
    """Replace NaN with ``fill`` -- a faster, NaN-only ``np.nan_to_num``.

    ``np.nan_to_num`` also scans for and replaces +/-inf and always allocates a copy;
    observation arrays only ever need NaN filled. The common case (converged power flow,
    valid profiles) has no NaN at all, so we return the array untouched after a single
    ``isnan`` pass. Non-float arrays cannot hold NaN and are returned as-is.
    """
    if arr.dtype.kind != "f":
        return arr
    mask = np.isnan(arr)
    if not mask.any():
        return arr
    out = arr.copy()
    out[mask] = fill
    return out




class PPTopoGym(BaseEnvPP):
    """
    Initialize an Environment for agents to perform.

    It includes the possibility for several simulations of actions.
    This ensures the agents can experience the consequences of their actions.
    """

    @override
    def __init__(self, env_config: dict) -> None:
        """
          Initialize the PPTopoGym environment.

          Note that env_config must be a serializable dict in order to be e.g. used in a RLLib-style
          environment.

          Required configuration keys in env_config:
          - 'net' (from gym_env_pp): pp.pandapowerNet or path to pandapower
          network file
          - 'n_episodes' (from gym_env_pp): Number of episodes to run
          - 'action_space': A dictionary corresponding to the action space

        Custom functions in the config:
        - reward: Function returning float OR function-name from /data/rewards.py;
            Example usage:
            def my_custom_reward(env_instance):
                return -env_instance.net.res_line.loading_percent.mean()
            env_config = {"reward": my_custom_reward}
            OR
            env_config = "reward_normalized"
        - observation: list[{"name": str, "function": Callable(self), "spaces": spaces.Box)]
            list[dict]: The custom observation function gets self as input.
        - observation_keys: list[str]
            list[str]: The names of the observation-keys the environment calculates
        - fix_obs_space: bool (default: False)
            If True, table observations are aggregated to the electrical nodes so
            each node-mapped observation has length n_nodes (line/trafo stay
            per-element). If False, observations keep the full pandapower table length.
        - static_obs_space: bool (default: False)
            If True, node-mapped observations are declared at (and zero-padded to) their
            static upper bound instead of the node count of the reset topology. Splitting a
            substation adds an electrical node, so without this the emitted observations
            outgrow the declared ``observation_space`` and the vectorized environments fail
            with a broadcast error. Off by default because enabling it changes observation
            shapes; rewards are unaffected either way.

        Attributes
        ----------
          - `self.net` (`pp.pandapowerNet`): The power network model used for simulation.
          - `self.df_actions` (`pd.DataFrame`): A DataFrame representing available actions.
          - `self.action_space` (`gym.spaces.Discrete`): The discrete action space.
          - `self.observation_space` (`gym.spaces.Space`): The observation space, defined
          via `self.define_observation_space()`.
          - `self.current_step` (`int`): The current step within an episode.
          - `self.worst_reward` (`float`): The defined worst-case value for reward. (Hard-coded to -2)
          - `self.log_actions` (`list[int | np.integer]`): A log of taken actions.
          - 'self.resolution' (float): The resolution of timesteps (in hours).
        This can be used to restore the network to a previous state.
        When an action is executed in the step function, it is logged here.
          - `self.current_simulation_log` (`list[dict]`): A log of simulation states
        for tracking the start state of the simulation.

        Parameters
        ----------
          :param env_config: environment configuration
          :type env_config: dict
        """
        # Snapshot of the config as handed in, so `orig_config` can rebuild a sibling env.
        # The net's profile tables are shared, not copied (read-only after setup_profiles);
        # everything else is deep-copied as before so later config edits cannot leak in.
        self._orig_config = _copy_config_sharing_profiles(env_config)
        super().__init__(env_config)
        # actions
        action_space = env_config["action_space"]
        # NB: do not re-assign self.net = env_config["net"] here -- BaseEnvPP.__init__ has
        # already taken an owned deep copy. Re-aliasing to the config's net would make
        # several envs built from one config share a mutable grid (see gym_env_pp.py).
        self.df_actions = create_actions_df(self.net, action_space)
        # Positional switch/line/trafo writes per action, resolved once (see load_action).
        # None -> the net is not positionally indexed and load_action falls back to .loc.
        self._action_plans: dict[int, _ActionPlan] | None = self._build_action_plans()
        # Define action space:
        self.action_space: spaces.Discrete = spaces.Discrete(len(self.df_actions))
        self.resolution: float = env_config.get("resolution", 1.0)
        # If True, table observations are aggregated to the electrical nodes
        # (length == n_nodes); if False, they keep the full table length.
        self.fix_obs_space: bool = env_config.get("fix_obs_space", True)

        # Initial power flow + graph observation lookup table. Done here (before
        # define_observation_space) so n_nodes is available for node-length shapes.
        self.run_pf()
        self._obs_cache: dict = make_obs_cache()
        # Opt-in: declare node observations at their static upper bound and pad up to it, so
        # the observation space stays valid when a substation splits (needed by the vector
        # envs). Off by default -- turning it on changes observation shapes.
        self.static_obs_space: bool = env_config.get("static_obs_space", False)
        self._max_n_nodes: int = self._compute_max_n_nodes()

        # Set observation keys (if not provided, default to all keys)
        # Observation space
        self.default_obs_items: dict = build_observation_registry()
        # Define observation space:
        all_configs: dict[str, ObservationConfig] = self.default_obs_items
        requested_keys = env_config.get("observation_keys", list(all_configs.keys()))
        self.active_obs_configs = {
            key: all_configs[key]
            for key in requested_keys
            if key in all_configs
        }
        # Aggregates that ``info_observations`` / the evaluation metrics may ask for by name but
        # that are intentionally *not* in the observation space (see
        # build_info_observation_registry). Only the lookup used to compute an explicitly
        # requested key consults them; define_observation_space stays on active_obs_configs, so
        # the observation shapes an agent was trained against do not move.
        self._computable_obs_configs: dict[str, ObservationConfig] = {
            **build_info_observation_registry(),
            **self.active_obs_configs,
        }
        observation_space: spaces.Dict = self.define_observation_space()



        self.custom_obs = _validated_custom_observations(env_config.get("observation"))
        if self.custom_obs:
            # spaces for all custom observations
            custom_observations = PPTopoGym._insert_custom_observations(env_config["observation"])
            self.observation_space: spaces.Dict = spaces.Dict({**observation_space, **custom_observations})
        else:
            self.observation_space = observation_space

        # ``active_obs_configs`` and ``custom_obs`` are assigned once here and only ever read
        # afterwards, so the full-observation key sets and the sorted output order are fixed
        # for the lifetime of the env and are precomputed instead of rebuilt on every call.
        self._all_default_keys: tuple[str, ...] = tuple(self.active_obs_configs)
        self._all_custom_keys: tuple[str, ...] = tuple(self.custom_obs)
        self._sorted_obs_keys: tuple[str, ...] = tuple(
            sorted((*self._all_default_keys, *self._all_custom_keys)),
        )

        self.log_actions: LoggedArray = LoggedArray(self.episode_length)
        # this is used for simulation to store the previous state
        self.current_simulation_log: list[dict] = []
        # custom reward
        custom_reward: str | Callable | None = env_config.get("reward")
        if isinstance(custom_reward, str):
            module_name = "pandapower_env.data.rewards"
            reward_module = importlib.import_module(module_name)
            try:
                custom_reward = getattr(reward_module, custom_reward)
            except AttributeError:
                msg = f"Reward function '{custom_reward}' not found in {module_name}"
                raise ValueError(msg) from AttributeError
        if callable(custom_reward):
            # Override the calculate_reward method behavior
            self.reward_function: Callable[[], float] = partial(custom_reward, self)
        else:
            self.reward_function = self._default_reward_function

        self.static_net_blob: None | bytes = None

    def start_simulation(self) -> None:
        """
        Start a new simulation session.

        Snapshots the state :meth:`end_simulation` restores: the switch / line / trafo
        topology arrays, the profile index, the step counter and the action log.

        It is not private, to enable users to start simulations in simulations.
        This is similar to Grid2OP.
        """
        self.current_simulation_log.append(self.save_state())

    def end_simulation(self) -> None:
        """
        End the current simulation session and save the log.

        Restores the network to the state snapshotted by the matching
        :meth:`start_simulation` -- topology, profile index and action log -- and re-solves it.
        """
        last_state = self.current_simulation_log.pop()
        # Restore the captured topology directly instead of rewinding to the baseline and
        # replaying the action log: ``start_simulation`` already snapshotted the exact switch /
        # line / trafo arrays, so the replay only re-derived a state we were holding. It also
        # could not reproduce the live grid after a crashed step, which appends no action to
        # the log (see CLAUDE.md) -- the replay restored a *different* topology there.
        self.restore_state(last_state)
        # end_simulation's own contract for the two counters, unchanged: current_step follows
        # the action log, and episode_step_counter is zeroed the way _reset_state zeroed it.
        self.current_step = len(self.log_actions)
        self.episode_step_counter = 0
        self.run_pf(pf_type = self.pf_type)
        if self.net.converged is False:
            msg = f"Power flow did not converge at step {self.current_step}"
            logger.warning(msg)
            self._handle_loadflow_failure()

    def simulation(
        self,
        actions: int | np.integer | list[int | np.integer] | Generator[int, None, None],
    ) -> list[Output]:
        """
        Simulate the environment for a sequence of actions.

        This method allows the environment to process one or more actions and evaluate their effects
        on the network. Each action represents a set of topology changes (e.g., switch states or
        line connections) to be applied to the grid. The simulation runs through all actions,
        executing each sequentially, and returns the outcomes.

        :param actions: A single action (integer) or a sequence of actions (list or generator).
        :type actions: int | np.integer | list[int | np.integer] | Generator[int, None, None]
        :return: A list of Output dataclasses for each action, containing:
            - observation (dict): The state of the environment after the action.
            - reward (float): The reward value associated with the action's outcome.
            - terminated (bool): Indicates whether the simulation has reached a terminal state.
            - truncated (bool): Indicates whether the simulation ended prematurely due to an error.
            - info (dict): Additional information about the action and environment state.
        :rtype: list[Output]

        Notes
        -----
            - If an action is invalid (e.g., out of bounds), it is skipped, and a warning is logged.
            - If a power flow calculation fails (e.g., `pp.LoadflowNotConverged`), the reward is set
              to `worst_reward`, and the simulation continues with subsequent actions.
        """
        return self._simulate(actions, with_nminus1=False)

    def simulation_nminus1(
        self,
        actions: int | np.integer | list[int | np.integer] | Generator[int, None, None],
    ) -> list[Output]:
        """
        Simulate the environment for a sequence of actions.

        This method allows the environment to process one or more actions and evaluate their effects
        on the network. Each action represents a set of topology changes (e.g., switch states or
        line connections) to be applied to the grid. The simulation runs through all actions,
        executing each sequentially, and returns the outcomes.

        :param actions: A single action (integer) or a sequence of actions (list or generator).
        :type actions: int | np.integer | list[int | np.integer] | Generator[int, None, None]
        :return: A list of Output dataclasses for each action, containing:
            - observation (dict): The state of the environment after the action.
            - reward (float): The reward value associated with the action's outcome.
            - terminated (bool): Indicates whether the simulation has reached a terminal state.
            - truncated (bool): Indicates whether the simulation ended prematurely due to an error.
            - info (dict): Additional information about the action and environment state.
        :rtype: list[Output]

        Notes
        -----
            - If an action is invalid (e.g., out of bounds), it is skipped, and a warning is logged.
            - If a power flow calculation fails (e.g., `pp.LoadflowNotConverged`), the reward is set
              to `worst_reward`, and the simulation continues with subsequent actions.
        """
        return self._simulate(actions, with_nminus1=True)

    def _simulate(
        self,
        actions: int | np.integer | list[int | np.integer] | Generator[int, None, None],
        *,
        with_nminus1: bool,
    ) -> list[Output]:
        """Drive a sequence of actions between ``start_simulation`` and ``end_simulation``.

        The shared body of :meth:`simulation` and :meth:`simulation_nminus1`, which differed
        only in whether each output carries an ``nminus1`` observation.

        :param actions: a single action or a sequence of them.
        :param with_nminus1: also report the worst N-1 line loading per action.
        :return: one :class:`Output` per action, in order.
        :rtype: list[Output]
        """
        outputs: list = []
        if isinstance(actions, (int, np.integer)):
            actions = [actions]
        self.start_simulation()
        for action in actions:
            self._validate_action(action)
            observation, reward, terminated, truncated, info = self.step(action)
            if not info.get("crashed", False):
                info["powerflow_converged"] = True
                output = Output(observation, reward, terminated, truncated, info)
                if with_nminus1:
                    output.observation["nminus1"] = nminus1_line_loading_max(self.net)
            else:
                label = "simulation N-1" if with_nminus1 else "simulation"
                logger.warning(
                    "Power flow did not converge in %s at step %s", label, self.current_step,
                )
                output = self._handle_loadflow_failure()
                if with_nminus1:
                    # _handle_loadflow_failure already sets these; the N-1 path restated them.
                    output.truncated = False
                    output.terminated = True
                    output.observation["nminus1"] = float("inf")
            outputs.append(output)
        self.end_simulation()  # also running run_pp
        return outputs

    def _validate_action(self, action: int | np.integer) -> None:
        """Reject an action index that is not a row of ``df_actions``.

        The check accepts ``np.integer`` as well as ``int`` and rejects negative indices. Both
        matter in practice: an agent hands back whatever ``action_space.sample()`` or an
        ``argmax`` produced, which is a numpy integer, and the previous ``isinstance(action, int)``
        guard let those straight through into ``load_action`` -- where an out-of-range index
        surfaced as a bare ``KeyError`` and a negative one silently applied the *last* action
        instead of raising.

        :param action: the action index to check.
        :type action: int | np.integer
        :raises ValueError: if ``action`` is not an integer in ``[0, len(df_actions))``.
        """
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < len(self.df_actions):
            msg = (
                f"Invalid action {action!r}. Must be an integer within the action space range "
                f"[0, {len(self.df_actions)})."
            )
            raise ValueError(msg)

    def verify_action(self, action: int | np.integer) -> bool:
        """
        Apply actions rules.

        This runs a simulation of an action for verification.
        Actions rules include:
        - passes_two_bus_symmetry_rule
        - passes_islanded_elements_rule
        - passes_n_elements_rule

        These work only for substations with 2 busbars, not more.
        """
        self.start_simulation()
        # as net is altered in verify function
        verify = verify_action(self.net, self.df_actions.loc[action])
        self.end_simulation()
        return verify

    @override
    def step(self, action: int | np.integer) -> tuple:
        """
        Execute a single action and return the environment's response.

        It logs the action taken and the current step.

        :param action: The action index to execute
        :type action: int | np.integer
        :return: A tuple of (observation, reward, terminated, truncated, info)
        :rtype: tuple
        """
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated:
            return self._handle_loadflow_failure().unpack()
        self.log_actions.append(action)
        if truncated:
            return obs, reward, terminated, truncated, info
        info.update(self.state_to_info())
        return obs, reward, terminated, truncated, info

    def _empty_obs(self) -> dict[str, np.ndarray | float]:
        empty_obs: dict[str, np.ndarray | float] = {
            k: np.zeros(space.shape or (), dtype=space.dtype or np.float32)
            for k, space in self.observation_space.spaces.items()
        }
        for key, config in self.active_obs_configs.items():
            if config.obs_type == ObsType.TOPOLOGY:
                empty_obs[key] = self._get_topology_value(config)
        if "adjacency_matrix" in self.active_obs_configs:
            empty_obs["adjacency_matrix"] = create_adjacency_matrix(self.net, self._obs_cache).astype(np.int32)
        if "node_slot_map" in self.active_obs_configs:
            # Like the adjacency: a real topology map, not zeros. Zeros would send every node to
            # slot 0, so a consumer scattering into slots would silently stack them on one row.
            empty_obs["node_slot_map"] = node_slot_map(self.net, self._obs_cache).astype(np.int32)
        return empty_obs
    def _handle_loadflow_failure(self) -> Output:
        output = Output()
        output.observation = self._empty_obs()
        output.reward = self.worst_reward
        output.terminated = True
        output.truncated = False
        output.info.update({
            "powerflow_converged": False,
            "crashed": True,
            "loading_percent": None,
        })
        output.info.update(self.state_to_info())
        return output


    @staticmethod
    def _insert_custom_observations(list_observations: list[dict]) -> spaces.Dict:
        """Insert custom observations into the observation space."""
        obs_spaces_dict = {obs["name"]: obs["spaces"] for obs in list_observations}
        return spaces.Dict(obs_spaces_dict)

    def _get_table_length(self, table_name: str) -> int:
        """Get the length of a pandapower network table."""
        if table_name == "adjacency":
            return len(self.net.line) + len(self.net.trafo)
        if table_name == "node_slots":
            return n_static_slots(self.net, self._obs_cache)
        return len(self.net[table_name])

    def _node_observation_length(self) -> int:
        """Length of a node-aggregated observation: either the live or the static node count.

        ``n_nodes`` is not constant: splitting a double-busbar substation adds an electrical
        node, so an observation built at the reset topology is shorter than one built after a
        split. Declaring the reset length -- the historical behaviour -- makes the environment
        emit observations that violate its own ``observation_space`` and breaks the vectorized
        envs (see the ``static_obs_space`` config key).

        With ``static_obs_space`` enabled the length is the static upper bound instead, and
        :meth:`_pad_to_node_length` pads every node observation up to it.

        :return: the number of entries a node-mapped observation has.
        :rtype: int
        """
        if self.static_obs_space:
            return self._max_n_nodes
        return n_nodes(self.net, self._obs_cache)

    def _compute_max_n_nodes(self) -> int:
        """Compute the static upper bound on the node count, over every reachable topology.

        Every busbar of a multi-busbar substation can carry at most one electrical node, so
        the grid can never have more nodes than the reset topology has, plus one extra node
        per additional busbar that the substations can split off. That bound is topology
        independent, which is exactly what a fixed observation space needs.

        :return: the largest node count any action sequence can produce.
        :rtype: int
        """
        base_nodes = n_nodes(self.net, self._obs_cache)
        substations = self.net.get("multi_bb_substation")
        if substations is None or not len(substations):
            return base_nodes
        extra_busbars = (substations["n_busbars_in_substation"].to_numpy(dtype=int) - 1).clip(min=0)
        return base_nodes + int(extra_busbars.sum())

    def _pad_to_node_length(self, values: np.ndarray) -> np.ndarray:
        """Pad a node-aggregated observation with zeros up to the static node length.

        A no-op unless ``static_obs_space`` is on. Padding with zeros keeps the entries of the
        nodes that do exist at their own indices, and reads as "no element here" for the
        trailing slots that a less-split topology does not use.

        :param values: a node-aggregated observation array
        :type values: np.ndarray
        :return: the array, padded to ``self._max_n_nodes`` if needed
        :rtype: np.ndarray
        """
        if not self.static_obs_space:
            return values
        missing = self._max_n_nodes - len(values)
        if missing <= 0:
            return values
        return np.concatenate([values, np.zeros(missing, dtype=values.dtype)])

    def _resolve_shape(self, obs_config: ObservationConfig) -> tuple[int, ...]:
        """Resolve the observation shape from the config specification."""
        shape_spec = obs_config.spaces_shape
        if shape_spec is None or shape_spec == "scalar":
            return (1,)

        if isinstance(shape_spec, tuple) and len(shape_spec) == 2:  # noqa: PLR2004
            dim0 = (
                self._get_table_length(shape_spec[0])
                if isinstance(shape_spec[0], str)
                else shape_spec[0]
            )
            return (dim0, shape_spec[1])

        if isinstance(shape_spec, str):
            if (
                self.fix_obs_space
                and obs_config.obs_type == ObsType.TABLE
                and shape_spec not in {"line", "trafo"}
            ):
                return (self._node_observation_length(),)
            return (self._get_table_length(shape_spec),)
        msg = f"resolve_shape for type {shape_spec} not solveable." # type: ignore[unreachable]
        raise ValueError(msg)


    def define_observation_space(self) -> spaces.Dict:
        """
        Define the observation space for the environment.

        This method specifies the possible range of observations that the environment might return.
        It is essential for reinforcement learning frameworks to:
        - Initialize the agent with the correct state space.
        - Allocate memory for the observations.
        - Understand the environment's state space structure and bounds.

        :return: A Dict space containing all observation spaces.
        :rtype: spaces.Dict
        """
        def create_box(obs_config: ObservationConfig) -> spaces.Box:
            """Create a gymnasium Box space from an ObservationConfig."""
            shape = self._resolve_shape(obs_config)

            # Ensure low/high are floats and replace None with infinity
            # This satisfies the SupportsFloat requirement
            low_val = float(obs_config.low) if obs_config.low is not None else -np.inf
            high_val = float(obs_config.high) if obs_config.high is not None else np.inf

            return spaces.Box(
                low=low_val,
                high=high_val,
                shape=shape,
                dtype=obs_config.dtype,
            )

        obs_space_dict: dict[str, gym.Space] = {
            key: create_box(config)
            for key, config in self.active_obs_configs.items()
        }

        return spaces.Dict(obs_space_dict)


    def _get_default_observation(self, key: str) -> np.ndarray:
        """
        Get observation value using the ObservationConfig specification.

        Parameters
        ----------
        key : str
            The observation key from active_obs_configs.

        Returns
        -------
        np.ndarray
            The observation data as numpy array.

        Raises
        ------
        ValueError
            If obs_type is unknown or custom column is not recognized.
        """
        config = self._computable_obs_configs[key]

        if config.handler is not None:
            return config.handler(self.net)

        match config.obs_type:
            case ObsType.PROFILE:
                return self._get_profile_value(config)
            case ObsType.AGGREGATE:
                return self._get_aggregate_value(config.column, config.dtype, config.nan_value)
            case ObsType.CUSTOM:
                return self._get_custom_value(config.column, config.dtype)
            case ObsType.TABLE:
                return self._get_table_value(config)
            case ObsType.TOPOLOGY:
                return self._get_topology_value(config)
            case _:
                msg = f"Unknown observation type: {config.obs_type}" # type:ignore [unreachable]
                raise ValueError(msg)

    def _get_custom_value(self, column: str, dtype: type) -> np.ndarray:
        """
        Get custom observation values requiring special computation.

        Parameters
        ----------
        column : str
            The custom observation identifier.
        dtype : type
            Target numpy dtype.

        Returns
        -------
        np.ndarray
            The computed custom observation.

        Raises
        ------
        ValueError
            If the custom observation column is not recognized.
        """
        match column:
            case "adjacency_matrix":
                return create_adjacency_matrix(self.net, self._obs_cache).astype(dtype)
            case "node_slot_map":
                return node_slot_map(self.net, self._obs_cache).astype(dtype)
            case _:
                msg = f"Unknown custom observation: {column}"
                raise ValueError(msg)


    def _get_table_value(self, config: ObservationConfig) -> np.ndarray:
        """
        Get observation value from a standard pandapower table and enforce datatype.

        Parameters
        ----------
        config : ObservationConfig
            The observation configuration containing table, column, and dtype info.

        Returns
        -------
        np.ndarray
            Values from the specified table column, cast to config.dtype.
        """
        # 1. Extract data (aggregated to nodes if fix_obs_space, else raw table length)
        if self.fix_obs_space:
            column_data = get_observation(
                self.net, self._obs_cache, config.table, config.column,
            )
            # Node observations grow when a substation splits; pad them back to the declared
            # length so the observation space keeps holding. Guarded on the flag first so the
            # default path costs one attribute lookup and never enters the padding helper --
            # this runs for every table observation on every step.
            # ``spaces_shape`` is the same discriminator define_observation_space uses, so the
            # padded keys are exactly the ones declared at the node length.
            if self.static_obs_space and config.spaces_shape not in {"line", "trafo"}:
                column_data = self._pad_to_node_length(column_data)
        else:
            column_data = get_raw_observation(
                self.net, self._obs_cache, config.table, config.column,
            )

        # 2. Handle NaNs and convert to the target dtype immediately
        # We cast to config.dtype here to ensure nan_val is compatible
        values: np.ndarray = _fill_nan(column_data, config.nan_value).astype(config.dtype)

        # 3. Clip values if bounds are provided. In place: the .astype above always returns a
        # freshly allocated array (copy=True by default), never a view into a net table.
        if config.low is not None and config.high is not None:
            np.clip(values, config.low, config.high, out=values)

        return values

    def _current_profile_rows(self) -> dict[tuple[str, str], np.ndarray]:
        """Profile rows for the current timestep as numpy arrays, cached per index.

        All profile observations and profile aggregates at one timestep read the same
        rows; caching them avoids a pandas ``.loc`` label lookup per observation key
        (and per ``create_observation`` call while the index is unchanged -- the profile
        tables are immutable after ``setup_profiles``).
        """
        if self._profile_rows_index == self.index:
            return self._profile_rows_cache
        idx = self.index
        rows: dict[tuple[str, str], np.ndarray] = {}
        specs = (
            ("profile_load", "p_mw", self.df_profiles_load_p),
            ("profile_load", "q_mvar", self.df_profiles_load_q),
            ("profile_gen", "p_mw", self.df_profiles_gen_p),
            ("profile_gen", "vm_pu", self.df_profiles_gen_vm),
            ("profile_sgen", "p_mw", self.df_profiles_sgen_p),
            ("profile_sgen", "q_mvar", self.df_profiles_sgen_q),
        )
        for table, column, df in specs:
            if not df.empty and idx in df.index:
                rows[table, column] = df.loc[idx].to_numpy()
        self._profile_rows_cache = rows
        self._profile_rows_index = idx
        return rows

    def _get_profile_value(
        self,
        config: ObservationConfig,
    ) -> np.ndarray:
        """Get observation value from profile dataframes."""
        base_table = config.table.removeprefix("profile_")
        if getattr(self.net, base_table).empty:
            return np.array([], dtype=config.dtype)

        values = self._current_profile_rows().get((config.table, config.column))
        if values is None:
            msg = f"Unknown profile: {config.table}.{config.column}"
            raise ValueError(msg)

        values = _fill_nan(values, config.nan_value).astype(config.dtype)
        # In place: .astype always allocates, so this never writes into the profile table.
        if config.low is not None and config.high is not None:
            np.clip(values, config.low, config.high, out=values)
        return values

    def _get_topology_value(self, config: ObservationConfig) -> np.ndarray:
        """Get raw topology information (bus IDs, lookup table)."""
        if config.column == "bus_lookup_table":
            return self.net._pd2ppc_lookups["bus"].astype(config.dtype)  # noqa: SLF001

        table = getattr(self.net, config.table)
        if len(table) == 0:
            return np.array([], dtype=config.dtype)
        return table[config.column].to_numpy().astype(config.dtype)

    def _get_aggregate_value(  # noqa: C901, PLR0912
        self,
        column: str,
        dtype: type,
        nan_val: float,
    ) -> np.ndarray:
        """Get scalar aggregate observation values as 1D array with requested dtype."""
        value: float = nan_val

        match column:
            case "total_load_p_profile":
                row = self._current_profile_rows().get(("profile_load", "p_mw"))
                if row is not None and not self.net.load.empty:
                    value = float(row.sum())
            case "total_gen_p_profile":
                rows = self._current_profile_rows()
                total = 0.0
                found = False
                gen_row = rows.get(("profile_gen", "p_mw"))
                if gen_row is not None and not self.net.gen.empty:
                    total += float(gen_row.sum())
                    found = True
                sgen_row = rows.get(("profile_sgen", "p_mw"))
                if sgen_row is not None and not self.net.sgen.empty:
                    total += float(sgen_row.sum())
                    found = True
                value = total if found else nan_val
            case "total_load_p_runpf":
                if has_load_results(self.net):
                    value = total_load_p(self.net)
            case "total_gen_p_runpf":
                if has_gen_results(self.net):
                    value = total_gen_p(self.net)
            case "system_losses":
                value = system_losses_sum(self.net)
            case "total_energy_overload":
                value = total_active_overload_mva(self.net)
            case "max_loading_percent":
                if not self.net.res_line.empty:
                    value = line_loading_max(self.net)
            case _:
                msg = f"Unknown aggregate column: {column}"
                raise ValueError(msg)

        if np.isnan(value):
            value = nan_val
        return np.array([value], dtype=dtype)

    @override
    def create_observation(
    self,
    keys: list[str] | None = None,
) -> dict[str, np.ndarray | float]:
        """
        Generate an observation of the current network state.

        Observations capture the key features of the network state that agents use to
        make decisions.

        Parameters
        ----------
        keys : list[str] | None
            Specific observation keys to include. If None, all active observations
            (both default and custom) are included.

        Returns
        -------
        dict[str, np.ndarray | float]
            A dictionary containing all requested observation arrays. A non-converged power
            flow yields the zero-filled fallback from :meth:`_empty_obs`, not an exception.
        """
        if self.net.converged is None:
            self.run_pf()

        if not self.net.converged:
            msg = f"Power flow did not converge at creating an observation at step {self.current_step}"
            logger.warning(msg)
            return self._empty_obs()

        observation: dict[str, np.ndarray | float] = {}

        if keys is None: # Include all active default observations and all custom observations
            default_keys: tuple[str, ...] | set[str] = self._all_default_keys
            custom_keys: tuple[str, ...] | set[str] = self._all_custom_keys
            output_keys: tuple[str, ...] = self._sorted_obs_keys
        else: # Filter to only requested keys (info-only aggregates included, see __init__)
            default_keys = set(keys) & set(self._computable_obs_configs.keys())
            custom_keys = set(keys) & set(self.custom_obs.keys())
            output_keys = tuple(sorted((*default_keys, *custom_keys)))

        with batch_observations(self.net, self._obs_cache):
            for key in default_keys:
                observation[key] = self._get_default_observation(key)
            for key in custom_keys:
                observation[key] = self.custom_obs[key](self)
        return {k: observation[k] for k in output_keys}



    def _observation_before_to_info(self) -> dict:
        return self.create_observation(keys=self.info_intermediate_obs)

    def _observation_after_to_info(self) -> dict:
        obs_keys = []
        for key in self.info_intermediate_obs:
            if key in self.active_obs_configs:
                continue
            obs_keys.append(key)
        return self.create_observation(keys=obs_keys)

    @override
    def observation_to_info(self, metric_key: str) -> dict:
        """Report intermediate loadflow results to info."""
        info: dict = super().observation_to_info(metric_key = metric_key)
        new_info = {}
        if metric_key == "before":
            new_info = self._observation_before_to_info()
        elif metric_key == "after":
            new_info = self._observation_after_to_info()

        new_info = {f"{k}_{metric_key}": v for k, v in new_info.items()}

        info.update(new_info)
        return info

    def state_to_info(self) -> dict:
        """
        Convert the current state of the environment to a dictionary.

        This method is useful for logging or debugging purposes, as it provides a
        structured representation of the environment's current state.

        :return: A dictionary containing the current state information.
        :rtype: dict
        """
        return {
            "prev_actions": self.log_actions,
            "current_step": self.current_step,
            "index_profile": self.index,
            "_source_instance_id": id(self),
        }

    def state_from_info(self, info: dict) -> None:
        """
        Restore the environment's state from a given information dictionary.

        Function to restore from: state_to_info.

        !! This cannot be called from the same instance, only from another instance.

        Args:
            info (dict): A dictionary containing the state information to restore.
        """
        # Check if this is being called on the same instance that created the info
        if "_source_instance_id" in info and info["_source_instance_id"] == id(self):
            msg = (
                "state_from_info() cannot be called on the same instance that created the "
                "info dictionary. Use a different environment instance to restore state."
            )
            raise ValueError(msg)
        index = info.get("index_profile")
        if index is not None:
            self.reset(options={"index": index})
        else:
            self.reset()
        if len(info["prev_actions"]) == 0:
            self.run_pf(pf_type = self.pf_type) # run powerflow
            if self.net.converged is True:
                return
            msg = "Power flow did not converge for the step 0 in create_observation."
            raise pp.LoadflowNotConverged(msg)
        self.log_actions = copy.deepcopy(info["prev_actions"])
        self.log_actions.reset()
        for action in info["prev_actions"][:-1]:
            self.load_action(action)
            self.log_actions.append(action)
        self.index = info["index_profile"] -1 #gets incremented in step
        self.current_step = info["current_step"] - 1  # gets incremented in step
        # run powerflow for the last action
        self.step(info["prev_actions"][-1])



    def _default_reward_function(self) -> float:
        """
        Calculate the reward for the current network state.

        Default function, which may be overriden by the __init__.
        :return: The reward value for the current state.
        :rtype: float

        The default reward is defined as the negative maximum line loading percentage.
        """
        clipped_loading =  np.clip(self.net.res_line["loading_percent"].max(),0,self.clip_max_loading)
        return self.clip_max_loading - clipped_loading if not np.isnan(clipped_loading) else self.worst_reward


    @override
    def calculate_reward(self) -> float:
        """
        Calculate the reward for the current network state.

        :return: The reward value for the current state.
        :rtype: float

        The default reward is defined as the negative maximum line loading percentage.
        """
        return self.reward_function()

    def load_action(self, action: int | np.integer) -> None:
        """
        Load a specific action to the pandapower network without running the power flow.

        Each action represents a set of topology changes, such as opening/closing switches,
        disconnecting/reconnecting lines or changing the tap position of a phase shift transformer.
        This method updates the network's state based on the specified action.

        Applies the plan precomputed by :meth:`_build_action_plans` -- positional numpy writes
        into the ``switch``/``line``/``trafo`` columns instead of per-call label-based
        ``DataFrame.loc`` lookups, which dominated this method (~460 us -> ~2 us on case30).
        Nets whose tables are not positionally indexed fall back to the original ``.loc``
        path (see :meth:`_build_action_plans`), so behaviour is unchanged either way.

        :param action: The index of the action to apply.
        :type action: int | np.integer
        :raises KeyError: If the `df_actions` DataFrame does not contain the required columns
        """
        prev_converged = self.net.converged
        super().load_action(action)
        if action == 0:
            self.net.converged = prev_converged
            return # 0 does nothing

        plan = self._action_plans.get(int(action)) if self._action_plans is not None else None
        if plan is None:
            self._load_action_by_label(action)
            return

        # The column arrays are re-fetched per call on purpose: reset() / restore_topology()
        # replace them, so a cached view would silently write into a detached array.
        if plan.open_switches is not None or plan.closed_switches is not None:
            closed = self.net.switch["closed"].to_numpy()
            if plan.open_switches is not None:
                closed[plan.open_switches] = False
            if plan.closed_switches is not None:
                closed[plan.closed_switches] = True

        if plan.lines is not None:
            self.net.line["in_service"].to_numpy()[plan.lines] = plan.line_in_service

        if plan.trafos is not None:
            self.net.trafo["tap_pos"].to_numpy()[plan.trafos] = plan.tap_pos

    def _build_action_plans(self) -> dict[int, _ActionPlan] | None:
        """Translate every row of ``df_actions`` into positional numpy writes.

        Each action's switch / line / trafo *labels* are resolved once to row positions in the
        corresponding pandapower table, so :meth:`load_action` can apply the action with plain
        fancy-index assignment instead of repeating a label lookup on every call.

        Returns ``None`` -- disabling the fast path in favour of :meth:`_load_action_by_label`
        -- when any label cannot be resolved to a position (an action referring to an element
        the net does not contain). That keeps a malformed action space behaving exactly as
        before rather than failing in a new place.

        :return: action index -> plan, or None if the actions cannot be resolved positionally.
        :rtype: dict[int, _ActionPlan] | None
        """
        columns = self.df_actions.columns
        has_switch_action = {"open_switches", "closed_switches"}.issubset(columns)
        has_line_action = {"lines", "disconnect_lines"}.issubset(columns)
        has_trafo_action = {"trafos", "tap_pos"}.issubset(columns)

        switch_positions = _positional_index(self.net.switch.index)
        line_positions = _positional_index(self.net.line.index)
        trafo_positions = _positional_index(self.net.trafo.index)

        def to_positions(labels: object, mapping: dict[int, int] | None) -> np.ndarray:
            """Resolve element labels to row positions (raises KeyError on an unknown label)."""
            label_array = np.asarray(labels, dtype=np.int64).ravel()
            if mapping is None:
                return label_array.astype(np.intp)
            return np.array([mapping[int(label)] for label in label_array], dtype=np.intp)

        # Column-at-a-time, not row-at-a-time: ``df_actions.loc[action]`` builds a mixed-dtype
        # object Series per action (~0.57 ms on case89's 155k actions, which was most of this
        # environment's construction cost). The columns are read once and indexed by position.
        needed = (
            (["open_switches", "closed_switches"] if has_switch_action else [])
            + (["lines", "disconnect_lines"] if has_line_action else [])
            + (["trafos", "tap_pos"] if has_trafo_action else [])
        )
        cells = {name: self.df_actions[name].to_numpy() for name in needed}

        plans: dict[int, _ActionPlan] = {}
        try:
            for position, action in enumerate(self.df_actions.index):
                if action == 0:  # DoNothing is short-circuited in load_action
                    continue
                plan = _ActionPlan()
                if has_switch_action:
                    plan.open_switches = to_positions(cells["open_switches"][position], switch_positions)
                    plan.closed_switches = to_positions(cells["closed_switches"][position], switch_positions)
                if has_line_action:
                    plan.lines = to_positions(cells["lines"][position], line_positions)
                    plan.line_in_service = ~np.asarray(
                        cells["disconnect_lines"][position], dtype=bool,
                    ).ravel()
                if has_trafo_action:
                    plan.trafos = to_positions(cells["trafos"][position], trafo_positions)
                    plan.tap_pos = np.asarray(cells["tap_pos"][position]).ravel()
                plans[int(action)] = plan
        except (KeyError, TypeError, ValueError):
            logger.debug("Could not build positional action plans; using the label-based path.")
            return None
        return plans

    def _load_action_by_label(self, action: int | np.integer) -> None:
        """Apply an action through label-based ``.loc`` writes (fallback path).

        Used when the net's ``switch``/``line``/``trafo`` tables are not positionally indexed,
        so the precomputed positional plans of :meth:`_build_action_plans` do not apply. This
        is the original implementation and is kept as the correctness reference.

        :param action: The index of the action to apply.
        :type action: int | np.integer
        """
        if {"open_switches", "closed_switches"}.issubset(self.df_actions.columns):
            # open switches of action
            open_switches = self.df_actions.loc[action, "open_switches"]
            self.net.switch.loc[open_switches, "closed"] = False
            # close switches of action
            closed_switches = self.df_actions.loc[action, "closed_switches"]
            self.net.switch.loc[closed_switches, "closed"] = True

        # lines are stored in "lines"
        # disconnect lines are stored in "disconnect_lines"
        if {"lines", "disconnect_lines"}.issubset(self.df_actions.columns):
            lines = self.df_actions.loc[action, "lines"]
            disconnects = np.array(self.df_actions.loc[action, "disconnect_lines"], dtype = bool)
            self.net.line.loc[lines, "in_service"] = ~disconnects

        # PST's are stored in "trafos"
        # PST tap positions are stored in "tap_pos"
        if {"trafos", "tap_pos"}.issubset(self.df_actions.columns):
            trafos = self.df_actions.loc[action, "trafos"]
            positions = self.df_actions.loc[action, "tap_pos"]
            self.net.trafo.loc[trafos, "tap_pos"] = positions

    def _reset_state(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Restore a clean episode start *without* solving the grid or building an observation.

        The state half of :meth:`reset`: drop the stale results, clear the action log and step
        counter, restore the baseline topology and load the profile row for the chosen timestep.
        It is split out because :meth:`end_simulation` needs exactly this and nothing more -- it
        replays the action log and solves afterwards, so a power flow and an observation taken
        here would describe the pristine topology and be thrown away immediately.

        :param seed: seed for this environment's RNG, forwarded to :meth:`BaseEnvPP.reset`.
        :type seed: int | None
        :param options: ``{"index": N}`` to start at a given timestep; see :meth:`BaseEnvPP.reset`.
        :type options: dict[str, Any] | None
        """
        self.net.converged = None # will be set again in self.run_pf

        # Reset environment variables
        self.log_actions.reset()
        self.current_step = 0

        # Call the parent class reset
        super().reset(seed=seed, options=options)

    @override
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        """
        Reset the environment and restore the topology to its initial state.

        :param seed: timeseries index to reset to
        :type seed: int | None
        :param options: Additional options for the reset (not used, included for gym compatibility)
        :type options: dict[str, Any] | None
        :return: Tuple containing the initial observation and an empty info dictionary.
        :rtype: tuple[dict, dict]
        """
        self._reset_state(seed=seed, options=options)

        # Return initial observation and empty info dictionary
        self.run_pf(pf_type = self.pf_type) # run powerflow
        if self.net.converged is False:
            logger.warning(
                "Warning: net did not converge. skipping profile index %s.",
                self.index,
            )
        return self.create_observation(), {}

    def __str__(self) -> str:
        # Determine the types of actions available
        busstation_actions = {"open_switches", "closed_switches"}.issubset(
            self.df_actions.columns,
        )
        line_actions = {"lines", "disconnect_lines"}.issubset(self.df_actions.columns)

        # Construct action description
        action_desc = []
        if busstation_actions:
            action_desc.append("substation actions")
        if line_actions:
            action_desc.append("line actions")
        action_info = " and ".join(action_desc) if action_desc else "no special actions"

        if hasattr(self, "observation_space") and isinstance(
            self.observation_space,
            spaces.Dict,
        ):
            obs_space_keys = list(
                self.observation_space.spaces.keys(),
            )  # Get just the keys
            obs_space_str = f"[{', '.join(obs_space_keys)}]"
        else:
            obs_space_str = "N/A"

        return (
            "PPTopoGym(\n"
            f"  net={self.net}),\n"
            f"  current profile index={getattr(self, 'index', 'Unnamed index')},\n"
            f"  num_actions={len(self.df_actions)} with {action_info},\n"
            f"  current_step={getattr(self, 'current_step', 'N/A')},\n"
            f"  logged_actions={self.log_actions},\n"
            f"  worst_reward={self.worst_reward},\n"
            f"  action_space={self.action_space},\n"
            f"  observation_space={obs_space_str}\n"
            ")"
        )

    def dump_static_net_bytes(self) -> bytes:
        """
        Make a JSON byte representation of the static network.

        Serialize a *static* copy of the network to JSON bytes:
        - strips net.profiles (we pass profile slices explicitly)
        - strips result tables
        This avoids deepcopy by temporarily removing & restoring attributes.
        """
        # Save and remove heavy attrs in-place
        saved_profiles = None
        had_profiles = hasattr(self.net, "profiles")
        if had_profiles:
            saved_profiles = self.net.profiles
            delattr(self.net, "profiles")

        saved_res = {}
        for res_key in ("res_bus", "res_line", "res_trafo", "res_sgen", "res_load", "res_gen", "res_switch"):
            if hasattr(self.net, res_key):
                saved_res[res_key] = getattr(self.net, res_key)
                delattr(self.net, res_key)

        try:
            # Write JSON to memory (no file)
            s = io.StringIO()
            pp.to_json(self.net, s)
            json_str = s.getvalue()
        finally:
            # Restore everything we temporarily removed
            if had_profiles:
                self.net.profiles = saved_profiles
            for k, v in saved_res.items():
                setattr(self.net, k, v)

        blob: bytes = json_str.encode("utf-8")
        self.static_net_blob = blob
        return blob

    def snapshot_topology(self) -> dict[str, np.ndarray]:
        topo: dict[str, np.ndarray] = {}
        if len(self.net.switch):
            topo["switch_closed"] = self.net.switch["closed"].to_numpy(dtype=np.int8, na_value=0)
        if len(self.net.line):
            topo["line_in_service"] = self.net.line["in_service"].to_numpy(dtype=np.int8, na_value=1)
        if len(self.net.trafo):
            topo["trafo_tap_pos"] = self.net.trafo["tap_pos"].to_numpy(dtype=np.int32, na_value=0)
        return topo

    def save_state(self) -> dict[str, Any]:
        """
        Capture the full episode state for cheap random-access restore.

        Returns topology arrays + profile index + step counter + a copy of the
        action log. This avoids deep-copying the pandapower network (see
        ``restore_state``): power setpoints are re-derived from the profile index
        and results are recomputed by the power flow on the next step.
        """
        return {
            "index": self.index,
            "current_step": self.current_step,
            "log_actions": copy.deepcopy(self.log_actions),
            "topology": self._capture_topology(),
        }

    def restore_state(self, state: dict[str, Any], *, run_pf: bool = False) -> None:
        """
        Restore a state captured by ``save_state`` in place (no deepcopy, no replay).

        Restores topology, reloads the profile for the saved index, and restores
        the step counter and action log. ``res_*`` are left stale on purpose: the
        caller's next ``step`` runs the power flow before reading any result. Pass
        ``run_pf=True`` to refresh results immediately (used when finalizing a
        search back to the actor's real state).
        """
        self.restore_topology(state["topology"])
        self.load_profile_timestep_into_net(state["index"])
        self.index = state["index"]
        self.current_step = state["current_step"]
        self.log_actions = copy.deepcopy(state["log_actions"])
        if run_pf:
            self.run_pf(pf_type=self.pf_type)

    def get_profile_slice(self, index: int) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if len(self.net.load):
            out["load_p_mw"] = self.df_profiles_load_p.loc[index].T.to_numpy()
            out["load_q_mvar"] = self.df_profiles_load_q.loc[index].T.to_numpy()
        if len(self.net.sgen):
            out["sgen_p_mw"] = self.df_profiles_sgen_p.loc[index].T.to_numpy()
            out["sgen_q_mvar"] = self.df_profiles_sgen_q.loc[index].T.to_numpy()
        if len(self.net.gen):
            out["gen_p_mw"] = self.df_profiles_gen_p.loc[index].T.to_numpy()
            out["gen_vm_pu"] = self.df_profiles_gen_vm.loc[index].T.to_numpy()
        return out
    @property
    def orig_config(self) -> dict:
        """
        Return the original config for recreational purposes.

        Clone the configuration for agent's use.
        Agents may need an own environment; as the .net object changes from the config,
        here the original config is deep-copied.

        The net's Simbench profile tables are shared with the stored config rather than
        copied (see :func:`deepcopy_net_sharing_profiles`) -- they are read-only once an
        environment has been built from them, and copying them per agent-environment is what
        made an extra environment cost tens of MB.
        """
        return _copy_config_sharing_profiles(self._orig_config)
