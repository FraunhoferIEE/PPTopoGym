import pandapower as pp

from pandapower_env.toolbox.utils_profiles import (
    get_first_sb_profiles,
    get_orig_profiles,
)
from pandapower_env.toolbox.utils_scaling import (
    _bracket_search_bounds,
    _run_pf_with_scaling,
    adjust_values_w_scaling,
    ensure_no_zero_values,
    ensure_slack_gen,
    find_max_timestep,
    find_scaling_binarysearch,
    find_scaling_recursive,
    load_profile_timestep_into_net,
    readjust_gen_values_for_convergence,
    run_pf,
    scale_scenario_scaling,
    target_loading,
)


def test_find_max_timestep(simenv) -> None:
    """
    Test the find_max_timestep function.

    :param test_grid_dbb_plus_simbench: test_grid_dbb_plus_simbench fixture (see conftest.py)
    :type test_grid_dbb_plus_simbench: pandapowerNet
    """
    p = simenv.df_profiles_load_p
    q = simenv.df_profiles_load_q
    sp = simenv.df_profiles_sgen_p
    gp = simenv.df_profiles_gen_p
    max_timestep = find_max_timestep(p, q, sp, gp)
    assert isinstance(max_timestep, int)
    # assure, that the function can be called several times
    max_steps = [find_max_timestep(p, q, sp, gp) for _ in range(10)]
    assert all(max_steps[i] == max_steps[i + 1] for i in range(len(max_steps) - 1))
    assert max_steps[0] == max_timestep

def test_load_profile_timestep_into_net(test_grid_dbb_plus_simbench) -> None:
    net = test_grid_dbb_plus_simbench
    orig_net_load_0 = net.load.loc[0, "p_mw"]
    orig_net_gen_0 = net.gen.loc[0, "p_mw"]
    net.profiles = get_first_sb_profiles(net)
    profiles = get_orig_profiles(net)
    load_profile_timestep_into_net(net, profiles)
    new_net_load_0 = net.load.loc[0, "p_mw"]
    new_net_gen_0 = net.gen.loc[0, "p_mw"]
    assert new_net_load_0 != orig_net_load_0
    assert new_net_gen_0 != orig_net_gen_0

def test_run_pf(test_grid) -> None:
    net = test_grid
    assert run_pf(net)
    net.load.p_mw = -10
    net.load.q_mvar = 0
    net.gen.p_mw = 10000
    assert not run_pf(net)

def test_target_loading(test_grid) -> None:
    net = test_grid
    assert run_pf(net) is True
    lines = target_loading(net, 100)
    assert lines == 0


def test_ensure_slack_gen(test_grid) -> None:
    net = test_grid
    ensure_slack_gen(net)
    assert net.gen.loc[0, "slack"]

def test_readjust_gen_values_for_convergence(test_grid) -> None:
    net = test_grid
    for elem in ("load", "gen", "sgen"):
        net[elem]["scenario_scaling"] = 1
    readjust_gen_values_for_convergence(net)
    sum_loads = net.load["p_mw"].sum()
    sum_gens = net.gen["p_mw"].sum() + (net.sgen["p_mw"].sum() if len(net.sgen) > 0 else 0)
    assert sum_loads*0.99 <= sum_gens <= sum_loads*1.05

def test_ensure_no_zero_values(test_grid) -> None:
    net = test_grid
    ensure_no_zero_values(net)
    for elem in ("load", "gen", "sgen"):
        assert (net[elem]["p_mw"] == 0).sum() == 0, f"Zero p_mw still present in {elem}"
    # Check that no zeros remain in q_mvar for load
    assert (net.load["q_mvar"] == 0).sum() == 0, "Zero q_mvar still present in load"


def test_adjust_values_w_scaling(test_grid) -> None:
    net = test_grid
    net.profiles = get_first_sb_profiles(net)
    profiles = get_orig_profiles(net)
    elems = ("gen", "sgen", "load")
    for elem in elems:
        net[elem]["scenario_scaling"] = 0
    adjust_values_w_scaling(net, profiles)
    for elem in elems:
        assert net[elem].p_mw.sum() == 0


# scale_scenario_scaling only used internally in scaling_recursive
def test_scale_scenario_scaling(test_grid) -> None:
    net = test_grid
    elems = ("gen", "sgen", "load")
    for elem in elems:
        net[elem]["scenario_scaling"] = 2
    value = 0
    scale_scenario_scaling(net, value)
    for elem in elems:
        assert net[elem]["scenario_scaling"].sum() == 0


def test_find_scaling_recursive(test_grid) -> None:
    net = test_grid
    max_percent = 2
    n_lines = 1
    assert find_scaling_recursive(net, max_percent=max_percent, overloaded_lines=n_lines) is True



def test_run_pf_with_scaling(test_grid_dbb_plus_simbench) -> None:
    """Test the run_pf_with_scaling function."""
    net = test_grid_dbb_plus_simbench
    return_value = _run_pf_with_scaling(net=test_grid_dbb_plus_simbench, scaling=1.0, min_percent_overload=100)
    assert return_value is None or return_value == 0
    pp.runpp(net)
    return_low_scaling = _run_pf_with_scaling(
        net=test_grid_dbb_plus_simbench, scaling=0.0000001, min_percent_overload=0.1,
    )
    assert return_low_scaling is None or return_low_scaling >= 0


def test_bracket_search_bounds(test_grid_dbb_plus_simbench) -> None:
    net = test_grid_dbb_plus_simbench
    low, high = _bracket_search_bounds(net=net, min_percent_overload=0.1, min_overloaded_lines=1)
    assert low < high


def test_find_scaling_binarysearch(test_grid_dbb_plus_simbench) -> None:
    net = test_grid_dbb_plus_simbench
    goal_overloaded = 1
    scaling_found, final_overloaded, line_scaling_found = find_scaling_binarysearch(net=net,
                                                                       min_percent_overload=0.1,
                                                                       min_overloaded_lines=goal_overloaded,
                                                                       )
    assert final_overloaded >= goal_overloaded
    assert scaling_found > 0
    assert line_scaling_found > 0
