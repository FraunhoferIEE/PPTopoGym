from __future__ import annotations

import copy
import math

# Hope you don't be imprisoned by legacy Python code :)
# Get the current directory (tests directory)
from pandapower_env.action_space.action_space import (
    _create_unitary_line_actions,
    add_actions_substation_line_pst,
    add_actions_substation_line_switching,
    create_actions_df,
    create_all_unitary_pst_actions,
    create_unitary_line_actions_and_donothing,
    create_unitary_pst_actions_and_donothing,
    create_unitary_substation_action,
    enforce_rules,
    return_donothing_action,
)


def test_enforce_rules(test_grid_multi_bb_substations) -> None:
    """
    Test if all rules are enforced.

    :param test_grid_double_bb_substations: net fixture with dbb_substation Dataframe (see conftest.py)
    :type test_grid_double_bb_substations: pandapowerNet

    Verifies:
    - All rules are enforced.
    - The correct number of actions is returned if all rules are passed.
    """
    net = test_grid_multi_bb_substations
    actions = create_unitary_substation_action(net)
    n_all_actions = 145
    assert len(actions) == n_all_actions, f"Unexpected number of actions: {len(actions)}"
    actions = add_actions_substation_line_switching(net)
    assert len(actions) > n_all_actions, f"Unexpected number of actions: {len(actions)}"
    actions = enforce_rules(net, actions)
    true_actions = 136  # hard-coded for net = case14()
    assert len(actions) == true_actions, f"Unexpected number of actions: {len(actions)}"


def test_create_unitary_substation_action(test_grid_multi_bb_substations) -> None:
    """
    Test on a simple case-14 net, if all actions are found.

    :param test_grid_multi_bb_substations: net fixture with dbb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet

    Verifies:
    - Number of actions matches the expected count.
    - All actions only involve one busbar.
    - All actions have at least two 1s in their states.
    """
    net = test_grid_multi_bb_substations
    net_actions = 145  # hard-coded for net = case14()
    result = create_unitary_substation_action(net)

    assert len(result) == net_actions, f"Unexpected length: {len(result)}"

    for item in result:
        assert isinstance(item, dict), f"Item is not a dict: {item}"
        assert "action" in item, f"'action' key missing in item: {item}"
        assert isinstance(
            item["action"],
            int,
        ), f"'action' is not an int: {item['action']}"
        if item["action"] == 0:
            continue
        assert "substations" in item, f"'substations' key missing in item: {item}"
        assert isinstance(
            item["substations"],
            list,
        ), f"'substations' is not a list: {item['substations']}"
        assert (
            len(item["substations"]) == 1
        ), f"Only one substation should be changed, instead of: {len(item['substations'])}"
        assert "states" in item, f"'states' key missing in item: {item}"


def test_return_donothing_action() -> None:
    """
    Test the do-nothing action.

    Verifies:
    - A single do-nothing action is added to the actions list.
    """
    dict_actions: list[dict[str, list | int]] = []
    dict_actions.insert(0, return_donothing_action())
    assert len(dict_actions) == 1, len(dict_actions)
    assert dict_actions[0]["action"] == 0, dict_actions[0]["action"]


def test_create_unitary_line_actions(test_grid_multi_bb_substations) -> None:
    """
    Test all possible line switches.

    :param test_grid_multi_bb_substations: net fixture with dbb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet

    Verifies:
    - Number of actions matches twice the number of lines.
    """
    net = test_grid_multi_bb_substations
    number_lines = len(net.line) * 2
    actions = _create_unitary_line_actions(net)
    assert len(actions) == number_lines, len(actions)


def test_create_unitary_line_actions_and_donothing(
    test_grid_multi_bb_substations,
) -> None:
    """
    Test all possible line switches.

    :param test_grid_multi_bb_substations: net fixture with dbb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet

    Verifies:
    - Number of actions matches twice the number of lines.
    """
    net = test_grid_multi_bb_substations
    number_lines = len(net.line) * 2 + 1
    actions = create_unitary_line_actions_and_donothing(net)
    assert len(actions) == number_lines, len(actions)


def test_create_actions_df(test_grid_multi_bb_substations) -> None:
    """
    Test the creation of the actions DataFrame.

    :param test_grid_multi_bb_substations: net fixture with dbb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet

    Verifies:
    - Required columns are present in the DataFrame.
    - All entries in the DataFrame are lists.
    """
    net = test_grid_multi_bb_substations
    actions = add_actions_substation_line_switching(net)
    df_actions = create_actions_df(net, actions)
    assert "disconnect_lines" in df_actions.columns
    assert "lines" in df_actions.columns
    # assert "states_binary_str" in df_actions.columns # not used anymore, we directly store the hex strings
    assert "states" in df_actions.columns

    df_wo_action = copy.deepcopy(df_actions)
    df_wo_action = df_wo_action.drop(columns=["action"], errors="ignore")
    all_lists = df_wo_action.map(lambda x: isinstance(x, list)).all().all()
    assert all_lists, ("Not all entries in the DataFrame are lists.", df_wo_action)


def test_add_actions_substation_line_switching(test_grid_multi_bb_substations) -> None:
    """
    Test the addition of substation and line switching actions.

    :param test_grid_multi_bb_substations: net fixture with dbb_substation Dataframe (see conftest.py)
    :type test_grid_multi_bb_substations: pandapowerNet

    Verifies:
    - Total actions match the sum of substation and line actions.
    - Action 0 (do-nothing action) behaves as expected.
    """
    net = test_grid_multi_bb_substations
    actions_total = add_actions_substation_line_switching(net)
    actions_buses = create_unitary_substation_action(net)
    actions_lines = _create_unitary_line_actions(net)

    assert len(actions_total) == len(actions_buses) + len(
        actions_lines,
    ), "Actions not added up correctly."

    assert actions_total[0]["action"] == 0, actions_total[0]["action"]
    assert actions_total[0]["substations"] == [], actions_total[0]["substations"]
    assert actions_total[0]["lines"] == [], actions_total[0]["states"]


def test_create_all_unitary_pst_actions(test_grid_with_pst) -> None:
    """
    Test all possible PST actions.

    :param test_grid_with_pst: net fixture with a PST (see conftest.py)
    :type test_grid_with_pst: pandapowerNet

    Verifies:
    - Number of possible PST actions.
    """
    net = test_grid_with_pst

    pst_df = net.trafo[(net.trafo["vn_hv_kv"] == net.trafo["vn_lv_kv"]) &
                       (net.trafo["tap_step_degree"] > 0.0)]

    n_steps = 0
    for pst in pst_df.itertuples():
        steps = 1 + math.floor((pst.tap_max - pst.tap_min)/pst.tap_step_degree)
        n_steps += steps

    actions = create_all_unitary_pst_actions(net)
    assert len(actions) == int(n_steps)

    actions = create_unitary_pst_actions_and_donothing(net)
    assert len(actions) == int(n_steps) + 1


def test_add_actions_substation_line_pst(test_grid_with_pst) -> None:
    """
    Test the addition of substation switching, line switching and PST actions.

    :param test_grid_with_pst: net fixture with a PST (see conftest.py)
    :type test_grid_with_pst: pandapowerNet

    Verifies:
    - Total actions match the sum of substation, line and PST actions.
    - Action 0 (do-nothing action) behaves as expected.
    """
    net = test_grid_with_pst
    actions_total = add_actions_substation_line_pst(net)

    len_actions = len(create_unitary_substation_action(net))
    len_actions += len(_create_unitary_line_actions(net))
    len_actions += len(create_all_unitary_pst_actions(net))

    assert len(actions_total) == len_actions, "Actions not added up correctly."


    assert actions_total[0]["action"] == 0, actions_total[0]["action"]
    assert actions_total[0]["substations"] == [], actions_total[0]["substations"]
    assert actions_total[0]["lines"] == [], actions_total[0]["states"]
