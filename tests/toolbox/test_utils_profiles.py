import pandas as pd

from pandapower_env.toolbox.utils_profiles import (
    _add_column_names,
    add_random_profiles,
    create_simbench_data_from_profiles,
    deterministic_profiles,
    get_first_sb_profiles,
    get_orig_profiles,
    get_scenario_profiles,
    setup_profiles,
)


def test_deterministic_profiles(test_grid) -> None:
    net= test_grid
    profiles = deterministic_profiles(net, sb_index=0)

    # Check keys
    assert set(profiles.keys()) == {"load", "renewables", "powerplants"}

    # Check types
    for df in profiles.values():
        assert isinstance(df, pd.DataFrame)

    # Ensure dimensions match expectations
    assert profiles["load"].shape[1] <= len(net.load) * 2 + 1
    assert profiles["renewables"].shape[1] <= len(net.sgen) + 1
    assert profiles["powerplants"].shape[1] <= len(net.gen) + 1

def test_add_column_names(test_grid) -> None:
    net = test_grid
    net.profiles = deterministic_profiles(net)
    _add_column_names(net)
    for eltype in ("gen", "sgen", "load"):
        assert hasattr(net[eltype], "profile")

def test_get_first_sb_profiles(test_grid) -> None:
    net = test_grid
    ret_dict = get_first_sb_profiles(net)
    assert isinstance(ret_dict, dict)

def test_get_orig_profiles(test_grid) -> None:
    net = test_grid
    net.profiles = get_first_sb_profiles(net)
    orig_profiles = get_orig_profiles(net)
    assert isinstance(orig_profiles, dict)
    assert set(orig_profiles.keys()) == {"gen_p", "gen_vm", "sgen_p", "sgen_q", "load_p", "load_q"}

# test scale_profiles should be done when testing scaling, as this requires the column "scenario_scaling".

def test_create_simbench_data_from_profiles(test_grid) -> None:
    net = test_grid
    net.profiles = get_first_sb_profiles(net)
    orig_profiles = get_orig_profiles(net)
    sb_data = create_simbench_data_from_profiles(net, orig_profiles)
    assert isinstance(sb_data, dict)
    assert set(sb_data.keys()) == {"load", "renewables", "powerplants"}
    assert sb_data["load"].shape[1] == len(net.load)*2 +1
    assert (net.gen.profile.to_numpy() == orig_profiles["gen_p"].columns.to_numpy()).all()


def test_get_scenario_profiles(test_grid_with_sgens_plus_simbench) -> None:
    """
    Test the get_scenario_profiles function.

    :param test_grid_with_sgens_plus_simbench: test_grid_with_sgens_plus_simbench fixture (see conftest.py)
    :type test_grid_with_sgens_plus_simbench: pandapowerNet
    :param test_grid_multi_bb_substations: test_grid_multi_bb_substations fixture (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_with_sgens_plus_simbench

    indices = get_scenario_profiles(net, window="D", wp=None, pv=None, bm=None, hy=None)
    assert len(indices) == 366  # noqa: PLR2004

    indices = get_scenario_profiles(
        net,
        window="D",
        wp="high",
        pv="high",
        bm=None,
        hy=None,
    )
    assert len(indices) > 0


def test_add_random_profiles(test_grid_multi_bb_substations) -> None:
    """
    Test the add_random_sgen function.

    :param test_grid_multi_bb_substations: test_grid_multi_bb_substations fixture (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations

    # Add random sgens to the grid
    add_random_profiles(net, 0.3, 0.3, 0.2, 0.2, 42)
    assert hasattr(net, "profiles"), "Error: net.profiles does not exist!"
    assert hasattr(net.sgen, "profile")


def test_setup_profiles(test_grid_multi_bb_substations) -> None:
    """
    Test the setup_profiles function.

    :param test_grid_multi_bb_substations: test_grid_multi_bb_substations fixture (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations
    add_random_profiles(net, 0.3, 0.3, 0.2, 0.2, 42)

    # Add random sgens to the grid
    outputs = setup_profiles(net)

    # Check if the sgens were added correctly
    for out in outputs:
        assert isinstance(out, pd.DataFrame), "Error: output is not a DataFrame!"
