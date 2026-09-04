
import matplotlib.pyplot as plt

from pandapower_env.toolbox.plotting_helpers import (
    calculate_externals_locations,
    create_bus2bus_geodata,
    label_buses,
    label_lines,
    label_substations,
    plot_all_peripherals,
)


def test_create_bus2bus_geodata(test_grid) -> None:
    """
    Test the create_bus2bus_geodata function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid

    create_bus2bus_geodata(net, net.line["from_bus"], net.line["to_bus"])
    create_bus2bus_geodata(net, net.trafo["lv_bus"], net.trafo["hv_bus"])


def test_plotting_functions(test_grid_multi_bb_substations) -> None:
    """
    Test a litany of plotting functions.

    :param test_grid_multi_bb_substations: net fixture with multi_bb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations
    calculate_externals_locations(net, offset=55)

    _, ax = plt.subplots(1, 1, figsize=(10, 10))

    plot_all_peripherals(net, ax)
    label_buses(net, ax)
    label_lines(net, ax)
    label_substations(net, ax)
