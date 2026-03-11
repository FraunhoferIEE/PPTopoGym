import copy

import pandapower as pp

from pandapower_env.action_space.action_space import create_unitary_line_actions_and_donothing
from pandapower_env.data.example_configs import config_case30
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.toolbox.utils_profiles import (
    create_simbench_data_from_profiles,
    get_orig_profiles,
    scale_profiles,
)
from pandapower_env.toolbox.utils_scaling import find_scaling_recursive


def test_config_case30() -> None:
    """Tests the config, and also the used scaling function."""
    config = config_case30()
    net = config["net"]
    net_copy = copy.deepcopy(net)
    assert isinstance(config, dict)
    assert config["nminus1"] is False
    assert isinstance(config["net"], pp.pandapowerNet)
    assert isinstance(config["action_space"], list)
    assert isinstance(config["episode_length"], int)
    assert isinstance(config["n_episodes"], int)
    # check that scaling works
    # test if scaled net has other values in res_line then simb profiles with 1-values in net
    # test if the scaling was reached for one example.
    orig_profiles = get_orig_profiles(net)
    max_perc = 150
    find_scaling_recursive(net, init_scaling=5, orig_profiles=orig_profiles, max_percent=max_perc, overloaded_lines=2)
    max_loading = max(net.res_line["loading_percent"])
    assert max_loading > max_perc
    scale_profiles(net, orig_profiles)
    # test if scaling was correctly applied
    orig_profiles = get_orig_profiles(net_copy)
    create_simbench_data_from_profiles(net_copy, orig_profiles)
    actions = create_unitary_line_actions_and_donothing(net_copy)
    config = {
        "net": net_copy,
        "n_episodes": 1,
        "episode_length": 96,
        "action_space": actions,
        "nminus1": False,
    }
    env = PPTopoGym(config)
    max_env = max(env.net.res_line["loading_percent"])
    assert max_env != max_loading
