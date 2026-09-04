import copy

from pandapower_env.substation.create_double_busbar_substation import (
    can_convert_to_n_busbar_substation,
    create_all_dbb_or_3bbwpst_substations,
    create_all_double_busbar_substations,
    create_multi_bb_substations_from_list,
    create_n_busbar_substation,
)


def test_create_n_busbar_substation(test_grid) -> None:
    """
    Test the creation of double-busbar substations and the can_convert_to_doublebusbar_substation function.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = test_grid

    is_dbbs_list = [1, 2, 3, 4, 5, 8, 12]
    not_dbbs_list = [0, 6, 7, 9, 10, 11, 13]

    for ibus in net.bus.index:
        is_dbbs = can_convert_to_n_busbar_substation(net, ibus)

        if is_dbbs:
            assert ibus in is_dbbs_list, f"{ibus} should not be double-busbarrable."
            create_n_busbar_substation(
                net,
                ibus,
            )  # only run function for busbars that are double-busbarrable
        else:
            assert ibus in not_dbbs_list, f"{ibus} should be double-busbarrable."


def test_create_all_double_busbar_substations(test_grid) -> None:
    """
    Test the function create_all_double_busbar_substations.

    :param test_grid: test_grid fixture (see conftest.py)
    :type test_grid: pandapowerNet
    """
    net = copy.deepcopy(test_grid)
    create_all_double_busbar_substations(net)
    mbb1 = copy.deepcopy(net.multi_bb_substation)
    net2 = copy.deepcopy(test_grid)
    create_multi_bb_substations_from_list(net2, [(5, 3)])
    create_all_double_busbar_substations(net2)
    mbb2 = copy.deepcopy(net2.multi_bb_substation)
    assert not mbb1.equals(
        mbb2,
    ), "The multi_bb_substation DataFrame should be different."
    # test, if it only created dbbs if the busbars are not in another sustation

def test_create_all_dbb_or_3bbwpst_substations(test_grid) -> None:
    net = copy.deepcopy(test_grid)
    n_trafos = len(net.trafo)
    create_all_dbb_or_3bbwpst_substations(net, buses_for_pst = [5])
    assert (n_trafos+1) == len(net.trafo)
    assert hasattr(net, "multi_bb_substation")
    assert "bus_2" in net.multi_bb_substation.columns
    conn_element_type = net.multi_bb_substation["element_type"].iloc[4]
    assert "trafo<PST>" in conn_element_type[-1], f"{conn_element_type} is missing new pst"
    assert "trafo<PST>" in conn_element_type[-2], f"{conn_element_type} is missing new pst"
