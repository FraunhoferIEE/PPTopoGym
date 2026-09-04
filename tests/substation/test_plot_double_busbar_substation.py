
from pandapower_env.substation.plot_double_busbar_substation import (
    create_double_busbar_plotting_net,
    separate_substation_buses_visually,
)
from pandapower_env.toolbox.plotting_helpers import (
    calculate_externals_locations,
)


def test_create_double_busbar_plotting_net(test_grid_multi_bb_substations) -> None:
    """
    Test the function create_double_busbar_plotting_net.

    :param test_grid_multi_bb_substations:
    :type test_grid_multi_bb_substations:
    """
    net = test_grid_multi_bb_substations
    create_double_busbar_plotting_net(net, 0)


def test_substation_plotting_functions(test_grid_multi_bb_substations) -> None:
    """
    Test substation plotting functions.

    :param test_grid_multi_bb_substations: net fixture with multi_bb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations
    calculate_externals_locations(net, offset=55)

    action_dict = {"substations": [0], "states": ["0x110101"]}
    separate_substation_buses_visually(net, action_dict, r_split_bus=20)
