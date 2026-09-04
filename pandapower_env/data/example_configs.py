from typing import TYPE_CHECKING

from pandapower.networks import case30

from pandapower_env.action_space.action_space import (
    create_unitary_substation_action,
)
from pandapower_env.substation.create_double_busbar_substation import create_all_double_busbar_substations
from pandapower_env.toolbox.utils_profiles import (
    create_simbench_data_from_profiles,
    get_first_sb_profiles,
    get_orig_profiles,
    scale_profiles,
)
from pandapower_env.toolbox.utils_scaling import ensure_no_zero_values, find_scaling_recursive

if TYPE_CHECKING:
    import pandapower as pp


def config_case30() -> dict:
    net: pp.Pandapowernet = case30()
    get_first_sb_profiles(net, 2)
    ensure_no_zero_values(net)
    for key, df in net.profiles.items():
        net.profiles[key] = df.replace(0.0, 1.0)
    orig_profiles = get_orig_profiles(net)
    find_scaling_recursive(net, init_scaling=1, orig_profiles=orig_profiles, max_percent=35, overloaded_lines=3)
    scale_profiles(net, orig_profiles)
    create_simbench_data_from_profiles(net, orig_profiles)
    create_all_double_busbar_substations(net)
    actions = create_unitary_substation_action(net)
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
