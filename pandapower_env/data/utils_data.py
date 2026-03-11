import copy
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count

import numpy as np
from gymnasium import spaces

from pandapower_env.agents.benchmark_agents import DoNothingAgent
from pandapower_env.environments.simulation_env import PPTopoGym


def poly_fit_error(y: np.ndarray, degree: int=3) -> float:
    x = np.arange(len(y))
    coeffs = np.polyfit(x, y, degree)
    y_pred = np.polyval(coeffs, x)
    return np.sum((y - y_pred) ** 2)


def run_single_scenario(args: list) -> list[float]:
    """Run a single scenario simulation."""
    ind, config_data = args
    episode_length = config_data["episode_length"]

    # Recreate environment for this process
    actions = spaces.Discrete(len(config_data["action_space"]))
    config_env = copy.deepcopy(config_data)
    env = PPTopoGym(config_env)
    agent_dn = DoNothingAgent(actions)

    index = ind * episode_length
    obs, _ = env.reset(options={"index": index})
    overloads_net = []

    for _ in range(episode_length):
        action = agent_dn.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        overloads_net.append(round(float(env.net.res_line.loading_percent.max()), 2))

    return overloads_net


def find_smooth_scenario(my_config: dict, threshold: int = 100) -> list[np.integer]:
    """
    Find smooth scenario - parallelized version.

    Make each of the 366 simbench profiles fit to a polynomial of degree <=5.
    Whichever fits best wins, and is selected.

    Parameters
    ----------
    my_config Config of a PPTopoGym Environment

    Returns
    -------
    int: the time-day.
    """
    # Prepare arguments for parallel execution
    max_scenarions = len(my_config["net"].profiles["load"]) // my_config["episode_length"]
    args_list = [(ind, copy.deepcopy(my_config)) for ind in range(max_scenarions)]

    # Method 1: Using ProcessPoolExecutor (recommended)
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        all_overload_lists = list(executor.map(run_single_scenario, args_list))
    filtered_lists = [lst for lst in all_overload_lists if any(val > threshold for val in lst)]
    # fit all lists (this is fast, no need to parallelize)
    errors = [poly_fit_error(np.array(lst), 5) for lst in filtered_lists]
    return list(np.argsort(errors))
