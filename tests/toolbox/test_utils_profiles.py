import pandas as pd
import pytest
import simbench

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


def test_add_column_names_ceiling_is_documented_and_enforced() -> None:
    """Pin the grid-size ceiling imposed by positional Simbench profile assignment.

    ``_add_column_names`` assigns one distinct Simbench profile per element positionally,
    so a grid cannot carry more loads than the library has distinct load profiles. On
    ``sb_index`` 0-2 that is 96; on 3-5 it is 27. This is why ``case118`` (99 loads) and
    ``case300`` (193 loads) cannot be built at all -- see the cycle-4 section of
    ``profiling/PERF_LEDGER.md``.

    The test documents the limit rather than asserting an exact catalogue size, so a
    Simbench upgrade that ships more profiles relaxes it instead of breaking the suite.
    """
    import pandapower.networks as pn

    net = pn.case118()
    n_available = _count_unique_load_profiles(sb_index=2)

    assert len(net.load) > n_available, (
        "case118 is expected to exceed the positional profile ceiling; if Simbench now "
        f"ships >= {len(net.load)} load profiles this limit has been relaxed."
    )

    # The failure today is a raw pandas length error from the positional assignment.
    with pytest.raises(ValueError, match="Length of values"):
        get_first_sb_profiles(net, 2)


def _count_unique_load_profiles(sb_index: int) -> int:
    """Count distinct load profiles the Simbench library offers for one scenario index.

    Mirrors the de-duplication ``_add_column_names`` performs (``_pload``/``_qload``
    suffixes collapse to one profile), so the count is the number of loads that can
    actually be assigned.

    :param sb_index: Simbench scenario index.
    :return: Number of distinct load profile names.
    """
    profiles = simbench.get_all_simbench_profiles(sb_index)
    names = profiles["load"].columns[1:].to_series().apply(
        lambda x: x.replace("_pload", "").replace("_qload", ""),
    )
    return int(len(names.unique()))
