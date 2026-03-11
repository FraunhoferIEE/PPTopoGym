from __future__ import annotations

import copy
import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, SupportsFloat, TypeVar

import gymnasium as gym
import pandapower as pp
import pandapower.contingency
import pandas as pd

from pandapower_env.toolbox.utils import run_nminus1_powerflow, run_powerflow

if TYPE_CHECKING:
    import numpy as np

ObsType = TypeVar("ObsType")

logger = logging.getLogger(__name__)


class BaseEnvPP(gym.Env, ABC):
    """
    Prototype for a simple electrical grid RL environment.

    This is a base class which will not work correctly on its own, but
    must be used in a derived class that implements load_action,
    create_observation, and calculate_reward.

    The specific behavior of the environment is described in the step()
    function (which can also be rewritten in the derived class).
    """

    def __init__(self, env_config: dict[Any, Any]) -> None:
        """
        Initialize the gym_env_pp environment.

        Required configuration keys in env_config:
        - net: pp.pandapowerNet or path to pandapower network file
        - n_episodes: Number of episodes to run
        - episode_length: the length (number of timesteps) of a single episode.

        :param env_config: environment configuration
        :type env_config: dict
        """
        super().__init__()

        # The number of timesteps prepared
        self.n_episodes = env_config["n_episodes"]

        # Episode length (number of timesteps per episode)
        self.episode_length = env_config["episode_length"]
        self.current_step = 0

        # If n-1 powerflows should be calculated
        self.nminus1 = env_config.get("nminus1", False)

        # The current timeseries index of the episode
        self.index = 0

        # Set up the network grid (you can also use set_grid to do this.)
        if isinstance(env_config["net"], pp.pandapowerNet):
            self.net = env_config["net"]
        else:
            self.net = pp.from_json(env_config["net_file"])
        self.net_copy_from = copy.deepcopy(self.net)

        if "profiles" not in self.net:
            msg = "Error - no net.profiles in the network. Stopping."
            raise RuntimeError(msg)

        self.df_profiles_load_p = pd.DataFrame()
        self.df_profiles_load_q = pd.DataFrame()
        self.df_profiles_sgen_p = pd.DataFrame()
        self.df_profiles_sgen_q = pd.DataFrame()
        self.df_profiles_gen_p = pd.DataFrame()
        self.df_profiles_gen_vm = pd.DataFrame()

        # Set up the Simbench-style profiles
        self.setup_profiles()

        # test if the episode length is smaller than the number of timesteps
        if self.episode_length > len(self.df_profiles_load_p):
            msg = (
                f"Episode length {self.episode_length} is larger than "
                f"number of timesteps {len(self.df_profiles_load_p)} in the profiles."
            )
            raise RuntimeError(msg)

        self.worst_reward = env_config.get("worst_reward", -1000.)

    def setup_profiles(self) -> None:
        """
        Configure profiles for load, gen, etc. timeseries.

        The code expects that the pandapowerNet contains a key-value pair "profiles"
        containing a dictionary:
        net.profiles = {'load': [Dataframe], 'renewables': [Dataframe], 'powerplants': [Dataframe]}

        The dataframes hold the timeseries, each with a unique [column] name.
        For loads, timeseries must have column names "NAME_pload" and "NAME_qload" for active
        and reactive power, respectively.

        The e.g. net.load Dataframe should contain a column called "profile" which
        contains the column in the Dataframe net.profiles['load'] to use as a profile for each
        load. (Same for net.sgen and net.gen.)

        (This profile format matches the conventions used in simbench.)
        """
        if self.net.load["name"].unique().shape != self.net.load["name"].shape:
            msg = "Load names must be unique!"
            raise RuntimeError(msg)

        if self.net.gen["name"].unique().shape != self.net.gen["name"].shape:
            msg = "Generator names must be unique!"
            raise RuntimeError(msg)

        if self.net.sgen["name"].unique().shape != self.net.sgen["name"].shape:
            msg = "Static generator names must be unique!"
            raise RuntimeError(msg)

        self.df_profiles_load_p = pd.DataFrame(
            {
                ld["name"]: ld.p_mw * self.net.profiles["load"]["{}_pload".format(ld["profile"])]
                for i, ld in self.net.load.iterrows()
            },
        )

        self.df_profiles_load_q = pd.DataFrame(
            {
                ld["name"]: ld.q_mvar * self.net.profiles["load"]["{}_qload".format(ld["profile"])]
                for i, ld in self.net.load.iterrows()
            },
        )

        self.df_profiles_sgen_p = pd.DataFrame(
            {
                sgen["name"]: sgen.p_mw * self.net.profiles["renewables"][sgen["profile"]]
                for i, sgen in self.net.sgen.iterrows()
            },
        )

        # simbench does not habe sgen_q profiles, so we set them to 0
        self.df_profiles_sgen_q = pd.DataFrame(
            {
                sgen["name"]: pd.Series(
                    0.0,
                    index=self.net.profiles["renewables"][sgen["profile"]].index,
                )
                for i, sgen in self.net.sgen.iterrows()
            },
        )

        self.df_profiles_gen_p = pd.DataFrame(
            {
                gen["name"]: gen.p_mw * self.net.profiles["powerplants"][gen["profile"]]
                for i, gen in self.net.gen.iterrows()
            },
        )

        self.df_profiles_gen_vm = pd.DataFrame(
            {
                gen["name"]: pd.Series(
                    gen.vm_pu,
                    index=self.net.profiles["powerplants"][gen["profile"]].index,
                )
                for _, gen in self.net.gen.iterrows()
            },
        )


        if self.n_episodes <= 0:
            self.n_episodes = len(self.df_profiles_load_p)

    def load_profile_timestep_into_net(self, index: int) -> None:
        """
        Load profiles for a given timestep into the net.load, etc. Dataframes.

        :param index: The index of the desired timestep, in the timeseries Dataframe.
        :type index: int
        """
        # Replace the load p and q with the values stored in the profiles_load dataframe
        if len(self.net.load):
            self.net.load["p_mw"] = self.df_profiles_load_p.loc[index].T.to_numpy()
            self.net.load["q_mvar"] = self.df_profiles_load_q.loc[index].T.to_numpy()

        if len(self.net.sgen):
            self.net.sgen["p_mw"] = self.df_profiles_sgen_p.loc[index].T.to_numpy()

        if len(self.net.gen):
            self.net.gen["p_mw"] = self.df_profiles_gen_p.loc[index].T.to_numpy()

    def run_pf(
        self,
        pf_type: str = "ac",
        use_ls2g: bool | str = "auto",
        nminus1: bool | None = None,
    ) -> bool:
        """
        Run the powerflow. Return True if successful, False if not.

        :param pf_type: the powerflow type, either 'ac' oder 'dc'
        :type pf_type: str
        :param use_ls2g: Whether lightsim2grid should be used as backend or not.
        :type use_ls2g: bool | str
        :param nminus1: Whether to run the n-1 powerflow or not.
        :type nminus1: bool | None
        :return: True if the powerflow converged, False if not.
        :rtype: bool
        """
        if nminus1 is None:
            nminus1 = self.nminus1
        try:
            if nminus1:
                run_nminus1_powerflow(self.net, pf_type)
                if "max_loading_percent" not in self.net.res_line:
                    msg = "N-1 analysis ran but didn't store max_loading_percent"
                    raise RuntimeError(msg)
            else:
                run_powerflow(self.net, pf_type, use_ls2g)
        except pp.LoadflowNotConverged:
            return False
        return True

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[dict | ObsType | list[float], SupportsFloat, bool, bool, dict]:
        """
        Perform the "step" action in the environment.

        In this implementation, the action is performed and the reward is calculated based
        on the observation that the agent sees. In other words, the action is performed
        "instantaneously" without changing anything else. After the reward is calculated,
        the timeseries moves forward one step and the next observation is calculated, *with*
        the selected action still in place.

        :param action: to be defined by the user in the derived class.
        :type action: TBD
        :return: Gymnasium-style output (observation, reward, terminated, truncated, info)
        :rtype: tuple[observation, reward, episode_over, truncated, info]
        """
        self.current_step += 1
        self.load_action(action)

        # Run the powerflow
        if not self.run_pf(): # should actually verify action and do DoNothing instead
            #if action is 0, then skip the next line
            #else set action to 0 and call this function again

            terminated = True
            truncated = False
            logger.warning("Net did not converge.")
            return (
                self.create_observation(run_pf=False),
                self.worst_reward,
                terminated,
                truncated,
                {"message": "network did not converge (next step).",
                 "crashed": True,
                 "loading_percent": None,
                 },
            )
        terminated = False
        truncated = self.current_step >= self.episode_length
        # For now, because one timestep = one episode, the episode is always
        # immediately over
        reward = self.calculate_reward()
        info: dict = {
            "message": "step completed successfully.",
            "crashed": False,
            "loading_percent": self.net.res_line.loading_percent.max(),
        }
        if truncated:
            observation = self.create_observation(run_pf=True)
            return observation, reward, terminated, truncated, info
        self.index += 1
        self.load_profile_timestep_into_net(self.index)
        observation = self.create_observation()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment."""
        txt = "Current timestep index: {}"
        logger.info(txt.format(self.index))

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        """
        Reset the environment.

        Loads a given index of the timeseries given by the "seed" parameter.
        (index = 0 if None is provided.)

        :param seed: timeseries index to reset to
        :type seed: int
        :param options: not used (kept for compatibility with gym)
        :type options: None
        :return: observation and [empty] info dict
        :rtype: tuple[observation type, dict]
        """
        self.net = copy.deepcopy(self.net_copy_from)
        self.current_step = 0
        if seed is not None:
            random.seed(seed)  # Set seed for random module
        if options is not None and "index" in options:
            self.index = options["index"]
        else:
            # set the scenario-start
            random_max_number = max((len(self.df_profiles_load_p) // self.episode_length) - 1, 0)
            scenario_index = random.randint(  # noqa: S311 # random needed for setting seed for rllib
                0,
                random_max_number,
            )
            self.index = scenario_index*self.episode_length

        logger.debug("Options: %s", options)

        self.load_profile_timestep_into_net(self.index)

        return {}, {}



    def close(self) -> None:
        """Close function -- kept here for compatibility with gym."""
        return

    @abstractmethod
    def load_action(self, action: int | np.integer) -> None:
        """
        Apply the action to the pandapower network.

        This function must be implemented in the derived class.

        :param action: The action
        :type action: TBD
        """
        msg = "The load_action method must be implemented in a derived class."
        raise NotImplementedError(msg)

    @abstractmethod
    def create_observation(self, run_pf: bool | None = None) -> list[float] | dict:
        """
        Create the observation from the result of the powerflow calculation in net.res_line and net.res_bus.

        This function must be implemented in the derived class.

        :param run_pf: Flag indicating whether the power-flow calculation should be done in the function.
        :type run_pf: bool
        :return: observation
        :rtype: TBD
        """
        msg = "The create_observation method must be implemented in a derived class."
        raise NotImplementedError(msg)

    @abstractmethod
    def calculate_reward(self) -> SupportsFloat:
        """
        Calculate the step reward.

        This function must be implemented in the derived class.

        :return: reward
        :rtype: TBD
        """
        msg = "The calculate_reward method must be implemented in a derived class."
        raise NotImplementedError(msg)
