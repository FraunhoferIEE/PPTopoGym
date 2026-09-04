
import pytest

from pandapower_env.substation.plot_double_busbar_substation import (
    create_double_busbar_plotting_net,
    create_double_busbar_plotting_not_abbc,
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


def test_create_double_busbar_plotting_not_abbc(test_grid_multi_bb_substations) -> None:
    """The non-ABBC layout builds one plotting bus per connected element plus the two busbars.

    :param test_grid_multi_bb_substations: net fixture with a multi_bb_substation Dataframe.
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations
    sub = net.multi_bb_substation.loc[0]

    tmpnet = create_double_busbar_plotting_not_abbc(net, 0)

    n_busbars = 2
    assert len(tmpnet.bus) == n_busbars + len(sub.connected_buses)
    assert len(tmpnet.line) == len(sub.connected_buses)
    assert len(tmpnet.bus_geodata) == len(tmpnet.bus)
    # Elements alternate between the two busbars.
    assert set(tmpnet.line["from_bus"]) == {0, 1}


@pytest.mark.parametrize(
    "plotting_fn",
    [create_double_busbar_plotting_net, create_double_busbar_plotting_not_abbc],
)
def test_plotting_net_requires_substation_table(plotting_fn, test_grid) -> None:
    """Both plotting builders refuse a net that was never converted to double busbars.

    :param plotting_fn: the plotting-net builder under test.
    :param test_grid: a plain case14 net without a multi_bb_substation Dataframe.
    :type test_grid: pandapowerNet
    """
    assert not hasattr(test_grid, "multi_bb_substation")

    with pytest.raises(RuntimeError, match="does not have a multi_bb_substation"):
        plotting_fn(test_grid, 0)


def test_separate_substation_buses_skips_fully_connected_substations(
    test_grid_multi_bb_substations,
) -> None:
    """A state with every element on one busbar has nothing to pull apart, so geodata is unchanged.

    :param test_grid_multi_bb_substations: net fixture with a multi_bb_substation Dataframe.
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations
    calculate_externals_locations(net, offset=55)
    n_elements = len(net.multi_bb_substation.loc[0, "connected_buses"])

    untouched, _ = separate_substation_buses_visually(
        net, {"substations": [], "states": []}, r_split_bus=20,
    )
    fully_connected, _ = separate_substation_buses_visually(
        net, {"substations": [0], "states": ["0x" + "0" * n_elements]}, r_split_bus=20,
    )

    assert fully_connected.equals(untouched)


def test_separate_substation_buses_skips_more_than_two_busbars(
    test_grid_multi_bb_substations,
) -> None:
    """Only 2-busbar substations are drawn; a 3-busbar one is passed over rather than mis-drawn.

    :param test_grid_multi_bb_substations: net fixture with a multi_bb_substation Dataframe.
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations
    calculate_externals_locations(net, offset=55)
    n_elements = len(net.multi_bb_substation.loc[0, "connected_buses"])
    split_state = "0x" + ("01" * n_elements)[:n_elements]

    untouched, _ = separate_substation_buses_visually(
        net, {"substations": [], "states": []}, r_split_bus=20,
    )

    net.multi_bb_substation.loc[0, "n_busbars_in_substation"] = 3
    skipped, _ = separate_substation_buses_visually(
        net, {"substations": [0], "states": [split_state]}, r_split_bus=20,
    )

    assert skipped.equals(untouched)

