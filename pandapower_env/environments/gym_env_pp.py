from __future__ import annotations

import copy
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np
import pandapower as pp
import pandapower.contingency
import pandas as pd

from pandapower_env.toolbox.utils import run_nminus1_powerflow, run_powerflow

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
        self.net.converged = None  # initially, no powerflow has been run
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

    def setup_profiles(self) -> None: #noqa: PLR0915, PLR0912, C901
        """
        Configure profiles for load, gen, etc. timeseries.

        The code expects that the pandapowerNet contains a key-value pair "profiles"
        containing a dictionary:
        net.profiles = {
            'load': [Dataframe], #load.p_mw AND load.q_mvar
            'renewables': [Dataframe], # sgen.p_mw
            'powerplants': [Dataframe], # gen.p_mw
            'gen_vm': [Dataframe], # gen.vm_pu (voltage PER UNIT, scaling the bus-voltages)
            'sgen_q': [Dataframe], # sgen.q_mvar
            }
        (This profile format matches the conventions used in simbench.)

        The dataframes hold the timeseries, each with a unique [column] name.
        For loads, timeseries must have column names "NAME_pload" and "NAME_qload" for active
        and reactive power, respectively.

        Then, the workflow is:
        1. extract the dfs for load_p, load_q, gen_p, gen_vm, sgen_p, sgen_q
        2. Make the DFs match the length of the number of corresponding net-elements
        3. Multiply the profile-values with the current load.p_mw, load.p_mvar, ... values
        4. Store the results in (then unchangeable!) profile dataframes.

        There must be one profile column per net-element.
        """
        # get number of elements:
        n_load = len(self.net.load)
        n_gen = len(self.net.gen)
        n_sgen = len(self.net.sgen)
        # get length of profiles:
        len_profiles = 0
        if "load" in self.net.profiles:
            len_profiles = len(self.net.profiles["load"])
        if "powerplants" in self.net.profiles:
            len_pp = len(self.net.profiles["powerplants"])
            if (len_profiles > 0) and (len_pp > 0) and (len_pp != len_profiles):
                msg = "The profiles have different number of timesteps."
                raise RuntimeError(msg)
            len_profiles = max(len_profiles, len_pp)
        if "renewables" in self.net.profiles:
            len_renewables = len(self.net.profiles["renewables"])
            if (len_profiles > 0) and (len_renewables > 0) and (len_renewables != len_profiles):
                msg = "The profiles have different number of timesteps."
                raise RuntimeError(msg)
            len_profiles = max(len_renewables, len_profiles)
        if "gen_vm" in self.net.profiles:
            len_genvm = len(self.net.profiles["gen_vm"])
            if (len_profiles > 0) and (len_genvm > 0) and (len_genvm != len_profiles):
                msg = "The profiles have different number of timesteps."
                raise RuntimeError(msg)
            len_profiles = max(len_profiles, len_genvm)
        if "sgen_q" in self.net.profiles:
            len_sgenq = len(self.net.profiles["sgen_q"])
            if (len_profiles > 0) and (len_sgenq > 0) and (len_sgenq != len_profiles):
                msg = "The profiles have different number of timesteps."
                raise RuntimeError(msg)
            len_profiles = max(len_profiles, len_sgenq)
        if len_profiles == 0:
            return # no profiles to add




        #extract load_p, load_q dataframes:
        if n_load > 0:
            if "load" not in self.net.profiles:
                msg = "No profiles for pp net.load defined (DF 'load' storing NAME_pload, NAME_qload columns.)."
                raise RuntimeError(msg)
            n_needed_columns = 2*n_load +1 if "time" in self.net.profiles["load"].columns else 2*n_load
            if self.net.profiles["load"].shape[1] < n_needed_columns: # has one time column
                msg = "load does not have enough profiles for all load elements in pp net."
                raise RuntimeError(msg)
            df_load_p = self.net.profiles["load"].filter(regex="_pload$").iloc[:, :n_load]
            #extract load_q dataframe:
            df_load_q = self.net.profiles["load"].filter(regex="_qload$").iloc[:, :n_load]
        # extract sgen_p dataframe:
        if n_sgen > 0:
            if "renewables" not in self.net.profiles:
                msg = "No profiles for pp net.sgen defined ('renewables' for sgen_p, 'sgen_q' for sgen_q.)."
                raise RuntimeError(msg)
            df_sgen_p = self.net.profiles["renewables"].drop(columns="time", errors="ignore").iloc[:, :n_sgen]
            # Optional: extract sgen_q from net.profiles["sgen_q]
            if "sgen_q" in self.net.profiles:
                if self.net.profiles["sgen_q"].shape[1] < n_sgen:
                    msg = "sgen_q does not have enough profiles for all sgen elements in pp net."
                    raise RuntimeError(msg)
                df_sgen_q = self.net.profiles["sgen_q"].drop(columns="time", errors="ignore").iloc[:, :n_sgen]
            # else-case handled below

        if n_gen > 0:
            if "powerplants" not in self.net.profiles:
                msg = "No profiles for pp net.gen defined ('powerplants' for gen_p)."
                raise RuntimeError(msg)
            if self.net.profiles["powerplants"].shape[1] < n_gen:
                msg = "The DF renewables does not have enough profiles for all gen elements in pp net."
                raise RuntimeError(msg)
            # extract gen_p dataframe:
            df_gen_p = self.net.profiles["powerplants"].drop(columns="time", errors="ignore").iloc[:, :n_gen]

            if "gen_vm" in self.net.profiles:
                if self.net.profiles["gen_vm"].shape[1] < n_gen:
                    msg = "gen_vm does not have enough profiles for all gen elements in pp net."
                    raise RuntimeError(msg)
                df_gen_vm = self.net.profiles["gen_vm"].drop(columns="time", errors="ignore").iloc[:, :n_gen]
            # else-case handled below


        # build dataframes:
        if n_load > 0:
            self.df_profiles_load_p = df_load_p @ np.diag(self.net.load.p_mw.to_numpy())
            self.df_profiles_load_q = df_load_q @ np.diag(self.net.load.q_mvar.to_numpy())
        if n_gen > 0:
            self.df_profiles_gen_p = df_gen_p @ np.diag(self.net.gen.p_mw.to_numpy())
        if n_sgen > 0:
            self.df_profiles_sgen_p = df_sgen_p @ np.diag(self.net.sgen.p_mw.to_numpy())

        #optional dataframes:
        if "sgen_q" in self.net.profiles:
            self.df_profiles_sgen_q = df_sgen_q @ np.diag(self.net.sgen.q_mvar.to_numpy())
        else: # set to 0
            self.df_profiles_sgen_q = pd.DataFrame(
                np.zeros((len_profiles, n_sgen)),
            )
        if "gen_vm" in self.net.profiles:
            self.df_profiles_gen_vm = df_gen_vm @ np.diag(self.net.gen.vm_pu.to_numpy())
        else: # set constantly to gen.vm_pu value (as this seldomly changes in applications)
            vm_values = self.net.gen.vm_pu.to_numpy()
            self.df_profiles_gen_vm = pd.DataFrame(
                np.tile(vm_values, (len_profiles, 1)),  # shape (n_rows, n_cols)
                columns=[f"gen {i + 1}_vm" for i in range(len(vm_values))],  # valid column names
            )


        if self.n_episodes <= 0:
            self.n_episodes = len(self.df_profiles_load_p)

    def load_profile_timestep_into_net(self, index: int) -> None:
        """
        Load profiles for a given timestep into the net.load, etc. Dataframes.

        Replace the load p and q with the values stored in the profiles_load dataframe

        :param index: The index of the desired timestep, in the timeseries Dataframe.
        :type index: int
        """
        if len(self.net.load):
            self.net.load["p_mw"] = self.df_profiles_load_p.loc[index].T.to_numpy()
            self.net.load["q_mvar"] = self.df_profiles_load_q.loc[index].T.to_numpy()

        if len(self.net.sgen):
            self.net.sgen["p_mw"] = self.df_profiles_sgen_p.loc[index].T.to_numpy()
            self.net.sgen["q_mvar"] = self.df_profiles_sgen_q.loc[index].T.to_numpy()

        if len(self.net.gen):
            self.net.gen["p_mw"] = self.df_profiles_gen_p.loc[index].T.to_numpy()
            self.net.gen["vm_pu"] = self.df_profiles_gen_vm.loc[index].T.to_numpy()
        self.net.converged = None  # reset converged flag

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
                if self.net.converged is not None:
                    return self.net.converged
                run_powerflow(self.net, pf_type, use_ls2g)
        except pp.LoadflowNotConverged:
            self.net.converged = False
            return False
        self.net.converged = True
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
        self.run_pf()
        if self.net.converged is False: # should actually verify action and do DoNothing instead
            #if action is 0, then skip the next line
            #else set action to 0 and call this function again

            terminated = True
            truncated = False
            logger.warning("Net did not converge.")
            return (
                self.create_observation(),
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
            observation = self.create_observation()
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

    def load_action(self, *_: Any) -> None: # noqa: ANN401
        """
        Apply the action to the pandapower network.

        This base implementation resets the power flow status. Derived classes
        should override this method with their specific signature and call
        super().load_action() to ensure the status is reset.

        :param args: Positional arguments (defined by derived classes)
        """
        self.net.converged = None  # reset converged flag for new topology

    @abstractmethod
    def create_observation(self) -> list[float] | dict:
        """
        Create the observation from the result of the powerflow calculation in net.res_line and net.res_bus.

        This function must be implemented in the derived class.
        The powerflow must have been run before, saving "self.net.converged" flag to True.

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
