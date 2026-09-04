from typing import TYPE_CHECKING

from pandapower.networks import case30, case89pegase

from pandapower_env.action_space.action_space import (
    add_actions_substation_line_pst,
    add_actions_substation_line_switching,
    verify_all_actions,
)
from pandapower_env.substation.create_double_busbar_substation import (
    create_all_dbb_or_3bbwpst_substations,
    create_all_double_busbar_substations,
)
from pandapower_env.toolbox.utils_profiles import (
    create_simbench_data_from_profiles,
    get_first_sb_profiles,
    get_orig_profiles,
)
from pandapower_env.toolbox.utils_scaling import ensure_no_zero_values, find_scaling_recursive

if TYPE_CHECKING:
    import pandapower as pp


def config_case30(max_percent: int = 40, overloaded_lines: int = 3, init_scaling: int = 1) -> dict:
    net: pp.Pandapowernet = case30()
    get_first_sb_profiles(net, 2)
    ensure_no_zero_values(net)
    for key, df in net.profiles.items():
        net.profiles[key] = df.replace(0.0, 1.0)
    orig_profiles = get_orig_profiles(net)
    find_scaling_recursive(net, init_scaling=init_scaling,
                           orig_profiles=orig_profiles,
                           max_percent=max_percent,
                           overloaded_lines=overloaded_lines,
                           )
    create_simbench_data_from_profiles(net, orig_profiles)
    create_all_double_busbar_substations(net)
    actions = add_actions_substation_line_switching(net)
    actions = verify_all_actions(net, actions)
    # delete net columns
    for eltype in ("gen", "sgen", "load"):
        if hasattr(net[eltype], "scenario_scaling"):
            del net[eltype]["scenario_scaling"]
    return  {
    "net": net,
    "n_episodes": 366,
    "episode_length": 96,
    "action_space": actions,
    "nminus1": False,
    }


def config_case89() -> dict:
    net: pp.Pandapowernet = case89pegase()
    get_first_sb_profiles(net, 2)
    ensure_no_zero_values(net)
    for key, df in net.profiles.items():
        net.profiles[key] = df.replace(0.0, 1.0)
    orig_profiles = get_orig_profiles(net)
    find_scaling_recursive(net, init_scaling=100, orig_profiles=orig_profiles, max_percent=80, overloaded_lines=4)
    create_simbench_data_from_profiles(net, orig_profiles)
    create_all_double_busbar_substations(net)
    actions = add_actions_substation_line_switching(net)
    actions = verify_all_actions(net, actions)
    # delete net columns
    for eltype in ("gen", "sgen", "load"):
        if hasattr(net[eltype], "scenario_scaling"):
            del net[eltype]["scenario_scaling"]
    return  {
    "net": net,
    "n_episodes": 366,
    "episode_length": 96,
    "action_space": actions,
    "nminus1": False,
    }

def config_30pst() -> dict:
    net: pp.Pandapowernet = case30()
    get_first_sb_profiles(net, 2)
    ensure_no_zero_values(net)
    for key, df in net.profiles.items():
        net.profiles[key] = df.replace(0.0, 1.0)
    orig_profiles = get_orig_profiles(net)
    find_scaling_recursive(net, init_scaling=1, orig_profiles=orig_profiles, max_percent=40, overloaded_lines=3)
    create_simbench_data_from_profiles(net, orig_profiles)
    create_all_dbb_or_3bbwpst_substations(net, [5])
    actions = add_actions_substation_line_pst(net)
    actions = verify_all_actions(net, actions)
    # delete net columns
    for eltype in ("gen", "sgen", "load"):
        if hasattr(net[eltype], "scenario_scaling"):
            del net[eltype]["scenario_scaling"]
    return {
        "net": net,
        "n_episodes": 366,
        "episode_length": 96,
        "action_space": actions,
        "nminus1": False,
    }
