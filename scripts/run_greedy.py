import copy
import numpy as np
import simbench
import pandas as pd
import pandapower as pp
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower.networks import case14, case30, case57, case118, case89pegase
from pandapower_env.substation.create_double_busbar_substation import (
    create_all_double_busbar_substations,
)
from pandapower_env.action_space.action_space import (
    add_actions_substation_line_switching,
)
from pandapower_env.toolbox.utils_profiles import deterministic_profiles
from pandapower_env.toolbox.utils_profiles import get_orig_profiles
from pandapower_env.toolbox.utils_profiles import get_first_sb_profiles
from pandapower_env.toolbox.utils_scaling import ensure_slack_gen
from pandapower_env.toolbox.utils_scaling import ensure_no_zero_values
from pandapower_env.toolbox.utils_scaling import find_scaling_recursive
from pandapower_env.toolbox.utils_profiles import create_simbench_data_from_profiles
from pandapower_env.data.example_configs import config_case30
from pandapower_env.agents.benchmark_agents import GreedyAgent, DoNothingAgent


env_config = config_case30()
agent_env_config = copy.deepcopy(env_config)
env = PPTopoGym(env_config)

greedy = GreedyAgent(env.action_space, agent_env_config, n_workers = 16)

all_actions = list()
done = False
obs, info = env.reset(options={"index": 0})
while not done:
    action = greedy.act(obs, info)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(obs["line_loadings"].max())
    print(f"action: {action}")
    all_actions.append(action)
print("Training finished")
print(all_actions)