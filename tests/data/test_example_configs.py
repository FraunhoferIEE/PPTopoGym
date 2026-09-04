import copy

import pandapower as pp

from pandapower_env.action_space.action_space import create_unitary_line_actions_and_donothing
from pandapower_env.data.example_configs import (
    config_30pst,
    config_case30,
    config_case89,
)
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.toolbox.utils_profiles import (
    create_simbench_data_from_profiles,
    get_orig_profiles,
    scale_profiles,
)
from pandapower_env.toolbox.utils_scaling import find_scaling_recursive


def test_config_case30() -> None:  # noqa: C901, PLR0915
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

    # Test that multiple calls to config_case30() create independent configurations.

    # Create three separate configs
    config1 = config_case30()
    config2 = config_case30()
    config3 = config_case30()

    # Test 1: Configs should not be the same object
    assert config1 is not config2
    assert config2 is not config3
    assert config1 is not config3

    # Test 2: Networks should not be the same object
    net1 = config1["net"]
    net2 = config2["net"]
    net3 = config3["net"]

    assert net1 is not net2
    assert net2 is not net3
    assert net1 is not net3

    # Test 3: DataFrames within networks should be independent
    # Check common pandapower tables
    for table in ["bus", "line", "load", "gen", "sgen", "trafo"]:
        if table in net1 and len(net1[table]) > 0:
            assert net1[table] is not net2[table], f"{table} DataFrames are the same object"
            assert net2[table] is not net3[table], f"{table} DataFrames are the same object"

    # Test 4: Profiles should be independent
    if hasattr(net1, "profiles") and net1.profiles:
        for key in net1.profiles:
            if key in net2.profiles:
                assert net1.profiles[key] is not net2.profiles[key], \
                    f"Profile '{key}' DataFrames are the same object"

    # Test 5: Modify one network and verify others are unaffected
    # Modify a bus voltage in net1
    if len(net1.bus) > 0:
        original_vn_net2 = net2.bus["vn_kv"].iloc[0]
        original_vn_net3 = net3.bus["vn_kv"].iloc[0]

        net1.bus.loc[0, "vn_kv"] = 999.999  # Unlikely value

        assert net2.bus["vn_kv"].iloc[0] == original_vn_net2, \
            "Modifying net1 affected net2"
        assert net3.bus["vn_kv"].iloc[0] == original_vn_net3, \
            "Modifying net1 affected net3"

    # Test 6: Modify a load value
    if len(net1.load) > 0:
        original_p_net2 = net2.load["p_mw"].iloc[0]

        net1.load.loc[0, "p_mw"] = 888.888

        assert net2.load["p_mw"].iloc[0] == original_p_net2, \
            "Modifying net1.load affected net2.load"

    # Test 7: Modify profiles if they exist
    if hasattr(net1, "profiles") and net1.profiles:
        first_profile_key = next(iter(net1.profiles.keys()))
        if len(net1.profiles[first_profile_key]) > 0:
            original_value = net2.profiles[first_profile_key].iloc[0, 0]

            net1.profiles[first_profile_key].iloc[0, 0] = 777.777

            assert net2.profiles[first_profile_key].iloc[0, 0] == original_value, \
                "Modifying net1 profiles affected net2 profiles"

    # Test 8: Action spaces should be independent
    actions1 = config1["action_space"]
    actions2 = config2["action_space"]

    assert actions1 is not actions2, "Action spaces are the same object"

    # Test 9: Modify action space and verify independence
    if isinstance(actions1, dict) and len(actions1) > 0:
        first_key = next(iter(actions1.keys()))
        original_actions2 = len(actions2)

        actions1[first_key] = "MODIFIED"

        assert len(actions2) == original_actions2, \
            "Modifying actions1 affected actions2"


def test_config_case89() -> None:  # noqa: C901, PLR0915
    """Tests the config, and also the used scaling function."""
    config = config_case89()
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
    max_perc = 100
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

    # Test that multiple calls to config_case30() create independent configurations.

    # Create three separate configs
    config1 = config_case89()
    config2 = config_case89()
    config3 = config_case89()

    # Test 1: Configs should not be the same object
    assert config1 is not config2
    assert config2 is not config3
    assert config1 is not config3

    # Test 2: Networks should not be the same object
    net1 = config1["net"]
    net2 = config2["net"]
    net3 = config3["net"]

    assert net1 is not net2
    assert net2 is not net3
    assert net1 is not net3

    # Test 3: DataFrames within networks should be independent
    # Check common pandapower tables
    for table in ["bus", "line", "load", "gen", "sgen", "trafo"]:
        if table in net1 and len(net1[table]) > 0:
            assert net1[table] is not net2[table], f"{table} DataFrames are the same object"
            assert net2[table] is not net3[table], f"{table} DataFrames are the same object"

    # Test 4: Profiles should be independent
    if hasattr(net1, "profiles") and net1.profiles:
        for key in net1.profiles:
            if key in net2.profiles:
                assert net1.profiles[key] is not net2.profiles[key], \
                    f"Profile '{key}' DataFrames are the same object"

    # Test 5: Modify one network and verify others are unaffected
    # Modify a bus voltage in net1
    if len(net1.bus) > 0:
        original_vn_net2 = net2.bus["vn_kv"].iloc[0]
        original_vn_net3 = net3.bus["vn_kv"].iloc[0]

        net1.bus.loc[0, "vn_kv"] = 999.999  # Unlikely value

        assert net2.bus["vn_kv"].iloc[0] == original_vn_net2, \
            "Modifying net1 affected net2"
        assert net3.bus["vn_kv"].iloc[0] == original_vn_net3, \
            "Modifying net1 affected net3"

    # Test 6: Modify a load value
    if len(net1.load) > 0:
        original_p_net2 = net2.load["p_mw"].iloc[0]

        net1.load.loc[0, "p_mw"] = 888.888

        assert net2.load["p_mw"].iloc[0] == original_p_net2, \
            "Modifying net1.load affected net2.load"

    # Test 7: Modify profiles if they exist
    if hasattr(net1, "profiles") and net1.profiles:
        first_profile_key = next(iter(net1.profiles.keys()))
        if len(net1.profiles[first_profile_key]) > 0:
            original_value = net2.profiles[first_profile_key].iloc[0, 0]

            net1.profiles[first_profile_key].iloc[0, 0] = 777.777

            assert net2.profiles[first_profile_key].iloc[0, 0] == original_value, \
                "Modifying net1 profiles affected net2 profiles"

    # Test 8: Action spaces should be independent
    actions1 = config1["action_space"]
    actions2 = config2["action_space"]

    assert actions1 is not actions2, "Action spaces are the same object"

    # Test 9: Modify action space and verify independence
    if isinstance(actions1, dict) and len(actions1) > 0:
        first_key = next(iter(actions1.keys()))
        original_actions2 = len(actions2)

        actions1[first_key] = "MODIFIED"

        assert len(actions2) == original_actions2, \
            "Modifying actions1 affected actions2"


def test_config_case30_pst() -> None:  # noqa: C901, PLR0915
    """Tests the config, and also the used scaling function."""
    config = config_30pst()
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
    max_perc = 100
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

    # Test that multiple calls to config_case30() create independent configurations.

    # Create three separate configs
    config1 = config_30pst()
    config2 = config_30pst()
    config3 = config_30pst()

    # Test 1: Configs should not be the same object
    assert config1 is not config2
    assert config2 is not config3
    assert config1 is not config3

    # Test 2: Networks should not be the same object
    net1 = config1["net"]
    net2 = config2["net"]
    net3 = config3["net"]

    assert net1 is not net2
    assert net2 is not net3
    assert net1 is not net3

    # Test 3: DataFrames within networks should be independent
    # Check common pandapower tables
    for table in ["bus", "line", "load", "gen", "sgen", "trafo"]:
        if table in net1 and len(net1[table]) > 0:
            assert net1[table] is not net2[table], f"{table} DataFrames are the same object"
            assert net2[table] is not net3[table], f"{table} DataFrames are the same object"

    # Test 4: Profiles should be independent
    if hasattr(net1, "profiles") and net1.profiles:
        for key in net1.profiles:
            if key in net2.profiles:
                assert net1.profiles[key] is not net2.profiles[key], \
                    f"Profile '{key}' DataFrames are the same object"

    # Test 5: Modify one network and verify others are unaffected
    # Modify a bus voltage in net1
    if len(net1.bus) > 0:
        original_vn_net2 = net2.bus["vn_kv"].iloc[0]
        original_vn_net3 = net3.bus["vn_kv"].iloc[0]

        net1.bus.loc[0, "vn_kv"] = 999.999  # Unlikely value

        assert net2.bus["vn_kv"].iloc[0] == original_vn_net2, \
            "Modifying net1 affected net2"
        assert net3.bus["vn_kv"].iloc[0] == original_vn_net3, \
            "Modifying net1 affected net3"

    # Test 6: Modify a load value
    if len(net1.load) > 0:
        original_p_net2 = net2.load["p_mw"].iloc[0]

        net1.load.loc[0, "p_mw"] = 888.888

        assert net2.load["p_mw"].iloc[0] == original_p_net2, \
            "Modifying net1.load affected net2.load"

    # Test 7: Modify profiles if they exist
    if hasattr(net1, "profiles") and net1.profiles:
        first_profile_key = next(iter(net1.profiles.keys()))
        if len(net1.profiles[first_profile_key]) > 0:
            original_value = net2.profiles[first_profile_key].iloc[0, 0]

            net1.profiles[first_profile_key].iloc[0, 0] = 777.777

            assert net2.profiles[first_profile_key].iloc[0, 0] == original_value, \
                "Modifying net1 profiles affected net2 profiles"

    # Test 8: Action spaces should be independent
    actions1 = config1["action_space"]
    actions2 = config2["action_space"]

    assert actions1 is not actions2, "Action spaces are the same object"

    # Test 9: Modify action space and verify independence
    if isinstance(actions1, dict) and len(actions1) > 0:
        first_key = next(iter(actions1.keys()))
        original_actions2 = len(actions2)

        actions1[first_key] = "MODIFIED"

        assert len(actions2) == original_actions2, \
            "Modifying actions1 affected actions2"

