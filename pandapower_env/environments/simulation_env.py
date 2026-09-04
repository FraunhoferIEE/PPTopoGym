from __future__ import annotations

import copy
import cProfile
import io
import logging
import pstats
from dataclasses import dataclass, field
from functools import partial, wraps
from typing import TYPE_CHECKING, Any, Callable, ParamSpec, TypeVar

import numpy as np
import pandapower as pp
from gymnasium import spaces

from pandapower_env.action_space.action_space import (
    create_actions_df,
    verify_action,
)
from pandapower_env.environments.gym_env_pp import BaseEnvPP
from pandapower_env.observation_space.obs_space_utils import aggregate_generators_to_buses, aggregate_loads_to_buses
from pandapower_env.observation_space.pp_to_observation import (
    nminus1_line_loading_max,
)
from pandapower_env.toolbox.utils import create_adjacency_matrix

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

logger = logging.getLogger(__name__)


P = ParamSpec("P")  # Parameter specification for the wrapped function
R = TypeVar("R")  # Return type of the wrapped function


def profile_execution_time(func: Callable[P, R]) -> Callable[P, R]:
    """
    Time the execution time of a function.

    :param func: The function to be profiled.
    :type func: Callable[P, R]
    :return: A wrapped function that profiles and prints the top 10 cumulative stats.
    :rtype: Callable[P, R]
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        profiler = cProfile.Profile()
        profiler.enable()  # Start profiling
        result = func(*args, **kwargs)  # Run the function
        profiler.disable()  # Stop profiling

        # Print top k results
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumulative")
        stats.print_stats(10)  # Display top 10 lines of stats

        return result

    return wrapper


@dataclass(slots=True)
class Output:
    """
    NamedTuple for storing the output of the environment.

    These are compatible with "normal" tuples, but provide more context.
    The order of the defined variables is important.
    Hence, they can be used in rllib, etc.
    """

    observation: dict = field(default_factory=dict)
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize prev_actions and index_profile always for observation."""
        # Ensure observation is initialized correctly
        self.info["prev_actions"] = []
        self.info["index_profile"] = 0

    def unpack(self) -> tuple:
        """
        Unpacks the tuple into its individual components.

        :return: a tuple of (observation, reward, terminated, truncated, info).
        :rtype: tuple
        """
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def __iter__(self) -> Iterator:
        """Allow the object to be unpacked like a tuple."""
        return iter(self.unpack())

    @classmethod
    def from_step(cls, step_result: tuple) -> Output:
        """
        Create an Output instance from the result of env.step(action).

        :param step_result: Tuple (observation, reward, terminated, truncated, info)
        :return: An Output instance
        """
        return cls(*step_result)


class LoggedArray:
    """
    A fixed-capacity logging buffer backed by a NumPy array, initialized empty.

    Attributes
    ----------
    data : np object array; for ints + lists
        The underlying NumPy array holding logged values (with unlogged slots as np.nan).
    capacity : int
        Maximum number of values that can be logged.
    _log_idx : int
        Current number of logged values; next insertion index.
    """

    __slots__ = ("data", "_log_idx")

    def __init__(self, capacity: np.integer | int) -> None:
        """
        Initialize the logging buffer with NaNs.

        Parameters
        ----------
        capacity : np.integer | int
            Maximum number of values to store.
        """
        self.data = np.empty(capacity, dtype=object)
        self._log_idx = 0

    @property
    def capacity(self) -> int:
        return len(self.data)

    def append(self, value: int | np.integer | Iterator) -> None:
        """
        Log a new value into the buffer at the next available position.

        Parameters
        ----------
        value : int | np.integer
            The value to store.

        Raises
        ------
        IndexError
            If the buffer is already full.
        TypeError
            If something else than a list or an integer is appended.
        """
        if self._log_idx >= self.capacity:
            msg = "Exceeded capacity for logging"
            raise IndexError(msg)
        if isinstance(value, (int, np.integer)):
            self.data[self._log_idx] = int(value)
        elif isinstance(value, (list, np.ndarray)):
            self.data[self._log_idx] = np.array(value, dtype=int)
        else:
            msg = f"Unsupported input type: {type(value)}"
            raise TypeError(msg)
        self._log_idx += 1

    def __len__(self) -> int:
        """
        Return the number of logged values.

        Returns
        -------
        int
            Count of values logged so far.
        """
        return self._log_idx

    def reset(self) -> None:
        """Clear the buffer, resetting all entries to np.nan and the counter to zero."""
        self.data[:self._log_idx] = [None] * self._log_idx
        self._log_idx = 0

    def __iter__(self) -> Iterator[int]:
        """Allow iteration only over logged (non-NaN) values."""
        return iter(self.data[:self._log_idx])

    def __getitem__(self, key: int | slice) -> float |np.ndarray:
        """
        Access logged data using indexing or slicing.

        Parameters
        ----------
        key : int or slice
            The index or range of indices to access.

        Raises
        ------
        IndexError
            If a value is called that is not present.

        Returns
        -------
        float or np.ndarray
            The logged value(s) at the specified position(s).
        """
        if isinstance(key, int) and (key >= self._log_idx or key < -self._log_idx):
            msg = "Index out of bounds for logged values"
            raise IndexError(msg)
        return self.data[: self._log_idx][key]

    def __deepcopy__(self, memo: dict) -> LoggedArray:
        new_obj = LoggedArray(self.capacity)
        new_obj._log_idx = self._log_idx # noqa: SLF001
        new_obj.data[:self._log_idx] = self.data[:self._log_idx]
        new_obj.data[self._log_idx:] = np.nan
        return new_obj

    def __repr__(self) -> str:
        return f"LoggedArray({self.data})"

    def __str__(self) -> str:
        """User-friendly string representation without None values."""
        return str(self.data[:self._log_idx])



class PPTopoGym(BaseEnvPP):
    """
    Initialize an Environment for agents to perform.

    It includes the possibility for several simulations of actions.
    This ensures the agents can experience the consequences of their actions.
    """

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
        - reward: Function returning float;
            Example usage:
            def my_custom_reward(env_instance):
                return -env_instance.net.res_line.loading_percent.mean()
            env_config = {"reward": my_custom_reward}
        - observation: list[{"name": str, "function": Callable(self), "spaces": spaces.Box)]
            The custom observation function gets self as input.

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
        This can be used to restore the network to a previous state.
        When an action is executed in the step function, it is logged here.
          - `self.current_simulation_log` (`list[dict]`): A log of simulation states
        for tracking the start state of the simulation.

        Parameters
        ----------
          :param env_config: environment configuration
          :type env_config: dict
        """
        super().__init__(env_config)
        action_space = env_config["action_space"]
        self.net = env_config["net"]
        self.df_actions = create_actions_df(self.net, action_space)
        # Define action space:
        self.action_space: spaces.Discrete = spaces.Discrete(len(self.df_actions))

        # Set observation keys (if not provided, default to all keys)
        # Set observation keys (if not provided, default to all keys)
        self.default_obs_keys: list[str] = env_config.get("default_obs_keys", [
            "bus_voltage_magnitude",
            "bus_voltage_angle",
            "bus_loads",
            "bus_generators",
            "line_loadings",
            "line_power_flow_p_mw",
            "line_power_flow_q_mvar",
            "line_status",
            "line_thermal_limit",
            "transformer_loading_percent",
            "transformer_power_flow_p_mw",
            "transformer_power_flow_q_mvar",
            "transformer_tap_position",
            "transformer_status",
            "generator_power_p_mw",
            "generator_power_q_mvar",
            "generator_status",
            "load_power_p_mw",
            "load_power_q_mvar",
            "load_status",
            "switch_positions",
            "total_power_demand",
            "total_power_generation",
            "system_losses",
            "adjacency_matrix",
        ])

        # Define observation space:
        observation_space: spaces.Dict = self.define_observation_space()
        if "observation" in env_config and len(env_config["observation"]) > 0:
            # check validity, observation has form [{name, function, spaces}]
            for obs in env_config["observation"]:
                if "name" not in obs:
                    msg = "No name in observation. Please change!"
                    raise ValueError(msg)
                if "function" not in obs:
                    msg = "No function in observation. Please change!"
                    raise ValueError(msg)
                if "spaces" not in obs:
                    msg = "No spaces in observation. Please change!"
                    raise ValueError(msg)
            self.custom_obs = {obs["name"]: obs["function"] for obs in env_config["observation"]}
            # spaces for all custom observations
            custom_observations = PPTopoGym._insert_custom_observations(env_config["observation"])
            self.observation_space: spaces.Dict = spaces.Dict({**observation_space, **custom_observations})
        else:
            self.custom_obs = {}
            self.observation_space = observation_space

        self.current_step = 0  # current steps in the episode

        self.log_actions: LoggedArray = LoggedArray(self.episode_length)
        # this is used for simulation to store the previous state
        self.current_simulation_log: list[dict] = []

        self.worst_reward = env_config.get("worst_reward", -1000)
        # custom reward
        custom_reward: partial | None = env_config.get("reward")
        if custom_reward is not None:
            # Override the calculate_reward method behavior
            self.reward_function: Callable[[], float] = partial(custom_reward, self)
        else:
            self.reward_function = self._default_reward_function

        self.static_net_blob: None | bytes = None

    # @profile_execution_time
    # @profile
    def start_simulation(self) -> None:
        """
        Start a new simulation session.

        This fct stores the current state of the network:
        - Topology: By re-doing actions
        - Current episode length
        - Index of the profile
        - Profile of the network

        It is not private, to enable users to start simulations in simulations.
        This is similar to Grid2OP.
        """
        current_net: dict[str, list | int | LoggedArray] = {}
        current_net["prev_actions"] = copy.deepcopy(self.log_actions)
        current_net["index_profile"] = self.index
        self.current_simulation_log.append(current_net)

    # @profile_execution_time
    # @profile
    def end_simulation(self) -> None:
        """
        End the current simulation session and save the log.

        This fct restores the network to the previous state,
        saved in latest entry of the current_simulation_log.
        It reruns all actions with load_action, and then runs the powerflow.
        """
        last_state = self.current_simulation_log.pop()
        # return to the previous state
        index = last_state["index_profile"]
        self.reset(options={"index": index})
        self.log_actions = copy.deepcopy(last_state["prev_actions"])
        for action in self.log_actions:
            self.load_action(action)
        self.index = last_state["index_profile"]
        self.current_step = len(self.log_actions)
        # loaded into the net in super().step()
        self.run_pf()
        if self.net.converged is False:
            msg = f"Power flow did not converge at step {self.current_step}"
            logger.exception(msg)
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
        outputs: list = []
        if isinstance(actions, (int, np.integer)):
            actions = [actions]
        self.start_simulation()
        for action in actions:
            if isinstance(action, int) and action >= len(self.df_actions):
                msg = f"Invalid action {action}. Must be within action space range."
                raise ValueError(msg)

            observation, reward, terminated, truncated, info = self.step(action)
            if  not info.get("crashed", False):
                info["powerflow_converged"] = True
                output = Output(observation, reward, terminated, truncated, info)
                outputs.append(output)
            else:
                msg = f"Power flow did not converge in simulation at step {self.current_step}"
                logger.exception(msg)
                outputs.append(self._handle_loadflow_failure())
        self.end_simulation()  # also running run_pp
        return outputs

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
        outputs: list = []
        if isinstance(actions, (int, np.integer)):
            actions = [actions]
        self.start_simulation()
        for action in actions:
            if isinstance(action, int) and action >= len(self.df_actions):
                msg = f"Invalid action {action}. Must be within action space range."
                raise ValueError(msg)
            observation, reward, terminated, truncated, info = self.step(action)
            if  not info.get("crashed", False):
                info["powerflow_converged"] = True
                output = Output(observation, reward, terminated, truncated, info)
                output.observation["nminus1"] = nminus1_line_loading_max(self.net)
                outputs.append(output)
            else:
                msg = f"Power flow did not converge in simulation N-1 at step {self.current_step}"
                logger.exception(msg)
                output = self._handle_loadflow_failure()
                output.truncated = False
                output.terminated = True
                output.observation["nminus1"] = float("inf")
                outputs.append(output)
        self.end_simulation()  # also running run_pp
        return outputs

    def verify_action(self, action: int | np.integer) -> bool:
        """Load action and do tests."""
        self.start_simulation()
        # as net is altered in verify function
        verfify = verify_action(self.net, self.df_actions.loc[action])
        self.end_simulation()
        return verfify

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
        observation = obs if isinstance(obs, dict) else {"line_loadings": obs}
        if truncated:
            return obs, reward, terminated, truncated, info
        info.update(self.state_to_info())
        return observation, reward, terminated, truncated, info

    def _empty_obs(self) -> dict[str, np.ndarray]:
        empty_obs =  {k : np.zeros(space.shape or (), dtype=space.dtype or np.float32)
                      for k, space in self.observation_space.spaces.items()}
        empty_obs["adjacency_matrix"] = create_adjacency_matrix(self.net).astype(np.int32)
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

    def define_observation_space(self) -> spaces.Dict:
        """
        Define the observation space for the environment.

        This method specifies the possible range of observations that the environment might return.
        It is essential for reinforcement learning frameworks to:
        - Initialize the agent with the correct state space.
        - Allocate memory for the observations.
        - Understand the environment's state space structure and bounds.

        Current observations:
        - Line loadings in percentage.

        :return: A Box space defining the bounds and shape of the observation space.
        :rtype: spaces.Box

        The bounds are defined as:
        - Low: 0 (minimum loading percentage).
        - High: 200 (maximum loading percentage, accounting for possible overloading).
        - Shape: (len(self.net.line),) where `len(self.net.line)` is the number of lines.
        - Data type: `np.float32`.
        """
        full_obs_space = spaces.Dict({
            "bus_voltage_magnitude": spaces.Box(
                low=0.0, high=1.2,
                shape=(len(self.net.bus),),
                dtype=np.float32,
            ),
            "bus_voltage_angle": spaces.Box(
                low=-360, high=+360,
                shape=(len(self.net.bus),),
                dtype=np.float32,
            ),
            "bus_loads": spaces.Box(
                low=-1000, high=1000,
                shape=(len(self.net.bus),),
                dtype=np.float32,
            ),
            "bus_generators": spaces.Box(
                low=0, high=1000,
                shape=(len(self.net.bus),),
                dtype=np.float32,
            ),
            "line_loadings": spaces.Box(
                low=0, high=200,
                shape=(len(self.net.line),),
                dtype=np.float32,
            ),
            "line_power_flow_p_mw": spaces.Box(
                low=-1000, high=1000,
                shape=(len(self.net.line),),
                dtype=np.float32,
            ),
            "line_power_flow_q_mvar": spaces.Box(
                low=-1000, high=1000,
                shape=(len(self.net.line),),
                dtype=np.float32,
            ),
            "line_status": spaces.Box(
                low=0, high=1,
                shape=(len(self.net.line),),
                dtype=np.int32,
            ),
            "line_thermal_limit": spaces.Box(
                low=0, high=100000,
                shape=(len(self.net.line),),
                dtype=np.float32,
            ),
            "transformer_loading_percent": spaces.Box(
                low=0, high=150,
                shape=(len(self.net.trafo),),
                dtype=np.float32,
            ),
            "transformer_power_flow_p_mw": spaces.Box(
                low=-1000, high=1000,
                shape=(len(self.net.trafo),),
                dtype=np.float32,
            ),
            "transformer_power_flow_q_mvar": spaces.Box(
                low=-1000, high=1000,
                shape=(len(self.net.trafo),),
                dtype=np.float32,
            ),
            "transformer_tap_position": spaces.Box(
                low=np.iinfo(np.int32).min,
                high=np.iinfo(np.int32).max,
                shape=(len(self.net.trafo),),
                dtype=np.int32,
            ),
            "transformer_status": spaces.Box(
                low=0, high=1,
                shape=(len(self.net.trafo),),
                dtype=np.int32,
            ),
            "generator_power_p_mw": spaces.Box(
                low=0, high=1000,
                shape=(len(self.net.gen),),
                dtype=np.float32,
            ),
            "generator_power_q_mvar": spaces.Box(
                low=-1000, high=1000,
                shape=(len(self.net.gen),),
                dtype=np.float32,
            ),
            "generator_status": spaces.Box(
                low=0, high=1,
                shape=(len(self.net.gen),),
                dtype=np.int32,
            ),
            "load_power_p_mw": spaces.Box(
                low=-1000, high=1000,
                shape=(len(self.net.load),),
                dtype=np.float32,
            ),
            "load_power_q_mvar": spaces.Box(
                low=-1000, high=1000,
                shape=(len(self.net.load),),
                dtype=np.float32,
            ),
            "load_status": spaces.Box(
                low=0, high=1,
                shape=(len(self.net.load),),
                dtype=np.int32,
            ),
            "switch_positions": spaces.Box(
                low=-1, high=1,
                shape=(len(self.net.switch),),
                dtype=np.int32,
            ),

            "total_power_demand": spaces.Box(
                low=0, high=1e6,
                shape=(1,),
                dtype=np.float32,
            ),
            "total_power_generation": spaces.Box(
                low=0, high=1e6,
                shape=(1,),
                dtype=np.float32,
            ),
            "system_losses": spaces.Box(
                low=0, high=1e6,
                shape=(1,),
                dtype=np.float32,
            ),
            "adjacency_matrix": spaces.Box(
                low=0,
                high=np.iinfo(np.int32).max,
                shape=(len(self.net.line) + len(self.net.trafo), 2),
                dtype=np.int32,
            ),
        })
        # Filter out keys not in self.default_obs_keys
        filtered_space = {key: space for key, space in full_obs_space.items() if key in self.default_obs_keys}
        return spaces.Dict(filtered_space)


    def _get_default_observation(self, key: str) -> np.ndarray: #noqa: PLR0911, PLR0912, C901
        """
        Get a default observation value by key.

        This method extracts the actual observation data from the pandapower network
        with proper error handling and data type conversion.

        :param key: The observation key name
        :return: The observation data as numpy array
        """
        match key:
            case "bus_voltage_magnitude":
                return np.nan_to_num(self.net.res_bus["vm_pu"].to_numpy(dtype=np.float32), nan=0.0)
            case "bus_voltage_angle":
                return np.nan_to_num(self.net.res_bus["va_degree"].to_numpy(dtype=np.float32), nan=360.0)
            case "bus_loads":
                return aggregate_loads_to_buses(self.net)
            case "bus_generators":
                return aggregate_generators_to_buses(self.net)
            case "line_loadings":
                return np.clip(
                    np.nan_to_num(self.net.res_line["loading_percent"].to_numpy(dtype=np.float32), nan=200.0),
                    0.0, 200.0,
                )
            case "line_power_flow_p_mw":
                return np.nan_to_num(self.net.res_line["p_from_mw"].to_numpy(dtype=np.float32), nan=0.0)
            case "line_power_flow_q_mvar":
                return np.nan_to_num(self.net.res_line["q_from_mvar"].to_numpy(dtype=np.float32), nan=0.0)
            case "line_status":
                return self.net.line["in_service"].to_numpy(dtype=np.int32, na_value=0)
            case "line_thermal_limit":
                return np.nan_to_num(self.net.line["max_i_ka"].to_numpy(dtype=np.float32), nan=1e-6)
            # Transformer information
            case "transformer_loading_percent":
                return np.nan_to_num(self.net.res_trafo["loading_percent"].to_numpy(dtype=np.float32), nan=0.0)
            case "transformer_power_flow_p_mw":
                return np.nan_to_num(self.net.res_trafo["p_hv_mw"].to_numpy(dtype=np.float32), nan=0.0)
            case "transformer_power_flow_q_mvar":
                return np.nan_to_num(self.net.res_trafo["q_hv_mvar"].to_numpy(dtype=np.float32), nan=0.0)
            case "transformer_tap_position":
                return self.net.trafo["tap_pos"].to_numpy(dtype=np.int32, na_value=0)
            case "transformer_status":
                return self.net.trafo["in_service"].to_numpy(dtype=np.int32, na_value=0)
            # Generator information
            case "generator_power_p_mw":
                return np.nan_to_num(self.net.res_gen["p_mw"].to_numpy(dtype=np.float32), nan=0.0)
            case "generator_power_q_mvar":
                return np.nan_to_num(self.net.res_gen["q_mvar"].to_numpy(dtype=np.float32), nan=0.0)
            case "generator_status":
                return self.net.gen["in_service"].to_numpy(dtype=np.int32, na_value=0)
            case "load_power_p_mw":
                return np.nan_to_num(self.net.res_load["p_mw"].to_numpy(dtype=np.float32), nan=0.0)
            case "load_power_q_mvar":
                return np.nan_to_num(self.net.res_load["q_mvar"].to_numpy(dtype=np.float32), nan=0.0)
            case "load_status":
                return self.net.load["in_service"].to_numpy(dtype=np.int32, na_value=0)
            case "switch_positions":
                return self.net.switch["closed"].to_numpy(dtype=np.int32, na_value=0)
            case "total_power_demand":
                return np.array([np.nan_to_num(self.net.res_load["p_mw"].sum(), nan=0.0)], dtype=np.float32)
            case "total_power_generation":
                return np.array([np.nan_to_num(self.net.res_gen["p_mw"].sum(), nan=0.0)], dtype=np.float32)
            case "system_losses":
                loss = self.net.res_line["pl_mw"].sum() + self.net.res_trafo["pl_mw"].sum()
                return np.array([np.nan_to_num(loss, nan=0.0)], dtype=np.float32)
            case "adjacency_matrix":
                return create_adjacency_matrix(self.net).astype(np.int32)
            case _:
                msg = f"Unknown observation key: {key}"
                raise ValueError(msg)


    def create_observation(self) -> dict:
        """
        Generate an observation of the current network state.

        Observations capture the key features of the network state that agents use to
        make decisions. Currently, this includes the loading percentage of all lines.

        :return: A dictionary containing the current line loadings.
        :rtype: dict

        """
        if not hasattr(self.net, "converged"):
            msg = "Power flow has not been run yet. Please run power flow before creating an observation."
            raise ValueError(msg)
        observation = {}
        converged = self.net.converged

        # Adjacency information
        create_adjacency_matrix(self.net).astype(np.int32)
        if converged:
            for key in self.default_obs_keys:
                observation[key] = self._get_default_observation(key)
            for key, fct in self.custom_obs.items():
                observation[key] = fct(self)
        else:
            msg = f"Power flow did not converge at creating an observation at step {self.current_step}"
            logger.exception(msg)
            observation = self._empty_obs()
        return dict(sorted(observation.items()))


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
            self.run_pf() # run powerflow
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
        clipped_loading =  np.clip(self.net.res_line["loading_percent"].max(),0 ,200)
        return 200 - clipped_loading if not np.isnan(clipped_loading) else self.worst_reward



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

        :param action: The index of the action to apply.
        :type action: int | np.integer
        :raises KeyError: If the `df_actions` DataFrame does not contain the required columns
        """
        prev_converged = self.net.converged if hasattr(self.net, "converged") else None
        super().load_action(action)
        if action == 0:
            self.net.converged = prev_converged
            return # 0 does nothing
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
        # Clear previous results
        if hasattr(self.net, "res_line"):
            del self.net.res_line
        if hasattr(self.net, "res_bus"):
            del self.net.res_bus

        # Reset environment variables
        self.log_actions.reset()
        self.current_step = 0

        # Call the parent class reset
        super().reset(seed=seed, options=options)

        # Return initial observation and empty info dictionary
        self.run_pf() # run powerflow
        if self.net.converged is False:
            logger.warning(
                "Warning: net did not converge. skipping profile index %s.",
                self.index,
            )
            return self.create_observation(), {}
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

        # Construct the string representation
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
