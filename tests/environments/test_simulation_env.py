import contextlib
import copy

import numpy as np
import pandapower as pp
from gymnasium import spaces

from pandapower_env.environments.simulation_env import LoggedArray, Output, PPTopoGym


def test_loggedarray() -> None:
    buf = LoggedArray(5)
    magcap = 5
    assert buf.capacity == magcap  # should print from `capacity`
    assert len(buf) == 0      # should print from `__len__`
    mag1 = 42
    mag2 = 3
    mag_len = 2
    buf.append(mag1)
    buf.append(mag2)
    assert len(buf) == mag_len
    assert buf[0] == mag1
    assert list(buf) == [mag1, mag2]
    buf.append(1)
    buf.append(2)
    buf.append(5)
    assert len(buf) == buf.capacity

    with contextlib.suppress(IndexError):
        buf.append(99)
    buf.reset()
    assert len(buf) == 0
    assert all(x is None for x in buf.data)


def test_observation_space(simenv) -> None:
    """Test if observation space is correctly defined."""
    obs_space = simenv.observation_space
    assert isinstance(
        obs_space,
        spaces.Dict,
    ), "Observation space should be a gym.spaces.Dict"


def test_state_to_info(simenv) -> None:
    """Test if state is correctly converted to info."""
    info = simenv.state_to_info()
    assert isinstance(info, dict), f"Expected dict, got {type(info)}"
    assert "prev_actions" in info, "'log_actions' key missing in info"
    assert "current_step" in info, "'current_step' key missing in info"
    assert "index_profile" in info, "'index_profile' key missing in info"




def test_create_observation(simenv) -> None:
    """Test if observations are correctly generated and structured."""
    observation = simenv.create_observation()

    assert isinstance(observation, dict), f"Expected dict, got {type(observation)}"
    assert "line_loadings" in observation, "'line_loadings' key missing in observation"

    observation_line = observation["line_loadings"]
    assert isinstance(
        observation_line,
        np.ndarray,
    ), f"Expected np.ndarray, got {type(observation_line)}"


def test_step_function(simenv, env_config) -> None:
    """Test if step function correctly updates state and logs actions."""
    simenv_2 = PPTopoGym(env_config)

    current_step = copy.deepcopy(simenv_2.current_step)
    current_log_actions = copy.deepcopy(simenv.log_actions)

    outputs = simenv.step(0) # 1 step for simenv
    simenv_2.step(1)
    simenv_2.step(0) # 2 steps for simenv_dict
    num_outputs = 5  # RLlib-compatible output format
    assert simenv.current_step == 1
    assert (simenv.current_step + 1) == simenv_2.current_step, (
        "Step count did not increase correctly",
        current_step,
        simenv_2.current_step,
    )
    current_log_actions.append(0)
    assert np.allclose(current_log_actions, simenv.log_actions, equal_nan=True), \
        f"Log actions {current_log_actions} != {simenv.log_actions} did not update correctly"
    assert (
        len(outputs) == num_outputs
    ), "Step function did not return correct number of outputs"

def test_load_action(simenv) -> None:
    simenv.net.converged = "test"
    prev_converged = simenv.net.converged
    simenv.load_action(0)
    assert simenv.net.converged == prev_converged, "Load action with 0 should not change convergence status"
    prev_net_switch = copy.deepcopy(simenv.net.switch["closed"])
    simenv.load_action(1)
    assert not (simenv.net.switch["closed"] == prev_net_switch).all(), "Load action with 1 should change switch states"



def test_reset_function(simenv) -> None:
    """Test if reset function correctly restores the initial environment state."""
    simenv.reset(options={"index": 1})
    assert simenv.current_step == 0, "Current step was not reset to 0"
    assert np.all(np.isnan(simenv.log_actions)), "Log actions should be empty after reset"
    assert (
        simenv.net.switch["closed"]
    ).all(), "Not all switches are closed after reset"
    assert (
        simenv.net.line["in_service"]
    ).all(), "Not all lines are in service after reset"


def test_load_action_does_not_log(simenv) -> None:
    """Ensure that load_action does not modify action logs."""
    simenv.load_action(0)
    assert len(simenv.log_actions) == 0, "load_action should not modify log_actions"


def test_simulation_process(simenv) -> None:
    """Test simulation start, end, and step functionality."""
    simenv.step(0)
    simenv.step(1)
    simenv.reset()

    simenv.simulation(2)
    assert (
        len(simenv.current_simulation_log) == 0
    ), "Simulation log should be empty after simulation"

    simenv.start_simulation()
    assert (
        len(simenv.current_simulation_log) == 1
    ), "Start simulation should add entry to simulation log"

    simenv.end_simulation()
    assert (
        len(simenv.current_simulation_log) == 0
    ), "End simulation should clear simulation log"

    simenv.simulation([3, 1])
    assert np.all(np.isnan(simenv.log_actions)), "Actions log should be cleared after simulation"
    assert (
        simenv.net.switch["closed"]
    ).all(), "Not all switches are closed after simulation"
    assert (
        simenv.net.line["in_service"]
    ).all(), "Not all lines are in service after simulation"


def test_simulation_log_deletion(simenv) -> None:
    """Ensure that simulation logs are correctly deleted after simulations."""
    simenv.step(1)
    simenv.simulation(2)
    simenv.simulation(3)
    simenv.step(2)

    assert (
        len(simenv.current_simulation_log) == 0
    ), "Simulation log should be empty after multiple simulations"


def test_observation_keys_match_space(simenv) -> None:
    """Ensure all observation keys are defined in the observation space."""
    keys = simenv.create_observation().keys()
    for key in keys:
        assert key in simenv.observation_space.spaces, (
            f"Key '{key}' missing in observation space. "
            f"All observation keys must be defined in observation space! "
            f"Available keys: {list(simenv.observation_space.spaces.keys())}"
        )


def test_state_from_info(simenv, simenv2) -> None:
    """Ensure that state is correctly extracted from observation."""
    simenv.reset()
    simenv.step(2)
    info = simenv.state_to_info()
    simenv2.reset()
    pp.runpp(simenv.net)
    pp.runpp(simenv2.net)
    assert len(info["prev_actions"]) == 1, \
        f"actual length of prev_actions {info['prev_actions']} is {len(info['prev_actions'])}"
    simenv2.state_from_info(info)
    # check that self.index has the correct value
    assert simenv2.index == simenv.index, f"Index does not match: {simenv2.index} != {simenv.index}"
    simenv.reset(options={"index": 0})
    simenv.step(0)
    current_step_simenv = simenv.current_step
    info = simenv.state_to_info()
    assert len(info["prev_actions"]) != 0 # passes
    info_index = copy.deepcopy(info["index_profile"])
    simenv2.state_from_info(info)
    assert current_step_simenv == simenv.current_step # does not pass
    assert info_index == simenv.index, f"Index does not match: {info_index} != {simenv.index}"
    assert simenv2.current_step == len(info["prev_actions"]), \
        f"Step does not match: {simenv.current_step} != {len(info['prev_actions'])}"

    pp.runpp(simenv2.net)
    assert (
        np.allclose(simenv2.log_actions, simenv.log_actions, equal_nan=True)
    ), f"actions do not match: {simenv2.log_actions} != {simenv.log_actions} for array {info['prev_actions'][:-1]}"
    assert (
        simenv2.current_step == simenv.current_step
    ), f"step does not match: {simenv2.current_step} != {simenv.current_step}"

    arr1 = simenv.net.res_line["loading_percent"].to_numpy()
    arr2 = simenv2.net.res_line["loading_percent"].to_numpy()
    for i, (v1, v2) in enumerate(zip(arr1, arr2)):
        # Use np.isclose to check floating-point equality (including NaNs if desired)
        is_close = np.isclose(v1, v2, rtol=1e-5, atol=1e-8, equal_nan=True)
        assert (
            is_close
        ), f"Line loading mismatch at index={i} -> simenv2={v1} vs simenv={v2}"


def test_verify_action(simenv) -> None:
    """Only calls the function to ensure it does something."""
    assert simenv.verify_action(0), "Action verification failed"


def test_handle_loadflow_failure(simenv) -> None:
    """Only calls the function to ensure it does something."""
    output = simenv._handle_loadflow_failure()
    assert isinstance(output, Output)
    assert "line_loadings" in output.observation

def test_custom_reward(env_config) -> None:
    mag_compare_value = 4.2
    def custom_reward(obs: dict) -> float: #noqa: ARG001
        return mag_compare_value
    env_config["reward"] = custom_reward
    env = PPTopoGym(env_config)
    env.reset()
    reward = env.calculate_reward()
    assert reward == mag_compare_value



def test_custom_observation(env_config) -> None:
    mag_compare_value = 4.2
    def custom_obs(obs: dict) -> float:  #noqa: ARG001
        return mag_compare_value
    space_test = spaces.Box(
                low=0.0, high=1.2,
                shape=(1,),
                dtype=np.float32,
            )
    name = "test_obs"
    env_config["n_episodes"] = 2
    env_config["observation"] = [{"name": name, "function": custom_obs, "spaces": space_test}]
    env = PPTopoGym(env_config)
    env.reset()
    obs = env.create_observation()
    assert name in env.custom_obs
    assert obs["test_obs"] == mag_compare_value

def test_current_step(simenv) -> None:
    episode_length = simenv.episode_length
    simenv.reset(options = {"index": 0})
    for _ in range(episode_length):
        simenv.step(0)
    assert (simenv.log_actions == np.full(episode_length, 0)).all()

