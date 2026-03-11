from pandapower_env.substation.double_busbar_substation import (
    close_all_substation_switches,
    get_all_substation_switches,
    get_list_of_closed_and_open_substation_switches,
    reset_all_substations,
    select_element_indices,
    set_substation_switches,
)


def test_switch_functions(test_grid_multi_bb_substations) -> None:
    """
    Test various functions in double_busbar_substation.py.

    :param test_grid_multi_bb_substations: net fixture with multi_bb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations

    result = [1, 3, 5, 7, 9, 11, 2, 4, 6, 8, 10, 12, 0]
    assert (
        get_all_substation_switches(net, 0) == result
    ), "get_all_substation_switches failed."

    result_closed = {2, 4, 5, 8, 9, 12}
    result_open = {0, 1, 3, 7, 11, 6, 10}
    closed_list, open_list = get_list_of_closed_and_open_substation_switches(net, 0, "0x110101")

    assert (set(closed_list) == result_closed), "get_list_of_closed_and_open_substation_switches failed."
    assert (set(open_list) == result_open), "get_list_of_closed_and_open_substation_switches failed."


def test_multi_bb_substation_helper_functions(test_grid_multi_bb_substations) -> None:
    """
    Test the double-busbar substation helper functions.

    :param test_grid_multi_bb_substations: net fixture with multi_bb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations

    result = [0, 2, 3, 4]
    assert (
        select_element_indices(net.multi_bb_substation.loc[0], "line") == result
    ), "select_element_indices failed."


def test_set_substation_switches(test_grid_multi_bb_substations) -> None:
    """
    Check that the correct switches are set.

    :param test_grid_multi_bb_substations: net fixture with multi_bb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet
    """
    net = test_grid_multi_bb_substations

    # Index 0, 1, 3, and 5 are set to bus_1
    # Index 2, 4 are set to bus_0
    set_substation_switches(net, 0, "0x110101")
    open_switches = net.switch[~net.switch["closed"]].index
    b01_open = net.multi_bb_substation.loc[0, "b01_switch"]
    check_b0_open = [
        net.multi_bb_substation.loc[0, "b0_switches"][i] for i in [0, 1, 3, 5]
    ]
    check_b1_open = [net.multi_bb_substation.loc[0, "b1_switches"][i] for i in [2, 4]]
    assert set(open_switches) == set(
        check_b0_open + check_b1_open + [b01_open],
    ), "set_substation_switches failed."

    # Check that substation switches are closed properly.
    set_substation_switches(net, 5, "0x00101")
    set_substation_switches(net, 6, "0x1101")
    close_all_substation_switches(net, 6)
    sub_6_switches = [net.multi_bb_substation.loc[6, "b01_switch"]]
    sub_6_switches.extend(net.multi_bb_substation.loc[6, "b0_switches"])
    sub_6_switches.extend(net.multi_bb_substation.loc[6, "b1_switches"])
    assert net.switch.loc[
        sub_6_switches,
        "closed",
    ].all(), "close_all_substation_switches failed."

    # Check that resetting all substations worked
    set_substation_switches(net, 3, "0x10101")
    set_substation_switches(net, 4, "0x010100")
    reset_all_substations(net)
    assert net.switch.closed.all(), "reset_all_substations failed."
