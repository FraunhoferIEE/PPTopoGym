from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np
import pandapower.topology as top
import pandas as pd

from pandapower_env.action_space.substation_action_rules import (
    helper_generate_b_ary_numbers,
    passes_fully_connected_grid_rule,
    passes_islanded_elements_rule,
    passes_n_elements_rule,
    passes_two_bus_symmetry_rule,
)
from pandapower_env.substation.double_busbar_substation import (
    get_list_of_closed_and_open_substation_switches,
)

if TYPE_CHECKING:
    from pandapower.auxiliary import pandapowerNet


def calculate_open_and_closed_switches(
    net: pandapowerNet,
    dict_actions: list[dict[str, Any]] | list[defaultdict[str, Any]],
) -> None:
    """
    Add key-value pairs 'open_switches' and 'closed_switches' to the actions dictionary.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :param dict_actions: List of action dictionaries containing substations and states.
    :type dict_actions: list[dict[str, Any]] | list[defaultdict[str, Any]]
    """
    for action in dict_actions:
        if not isinstance(action, defaultdict):
            action.setdefault("open_switches", [])
            action.setdefault("closed_switches", [])
            action.setdefault("substations", [])
            action.setdefault("states", [])
        for i_sub, state in zip(action["substations"], action["states"]):
            closed_switches, open_switches = get_list_of_closed_and_open_substation_switches(net, i_sub, state)
            action["open_switches"].extend(open_switches)
            action["closed_switches"].extend(closed_switches)


def enforce_rules(
    net: pandapowerNet,
    dict_actions: list[defaultdict[str, Any]],
) -> list[defaultdict[str, Any]]:
    """
    Enforce all rules from substation_action_rules.py.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :param dict_actions: List of action dictionaries containing substations and states.
    :type dict_actions: list[defaultdict[str, Any]]
    :return: A filtered list of action dictionaries that satisfy all rules.
    :rtype: list[defaultdict[str, Any]]
    """
    return [action for action in dict_actions if verify_action(net, action)]


def verify_action(
    net: pandapowerNet,
    action: defaultdict[str, Any],
    fast_ctx: dict | None = None,
) -> bool:
    """
    Verify that the action is valid.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :param action: The action dictionary containing substations and states.
    :type action: defaultdict[str, Any]
    :param fast_ctx: Optional context (see :func:`verify_all_actions`) that lets the
        connectivity rule reuse a cached base graph instead of rebuilding it per action.
    :type fast_ctx: dict | None
    :return: True if the action is valid, False otherwise.
    :rtype: bool
    """
    substations = action["substations"]
    states = action["states"]
    for i_sub, bitset in zip(substations, states):
        sub = net.multi_bb_substation.loc[i_sub]

        if not passes_two_bus_symmetry_rule(bitset):
            return False
        if not passes_islanded_elements_rule(sub, bitset):
            return False
        if not passes_n_elements_rule(bitset):
            return False

    return passes_fully_connected_grid_rule(net, substations, states, fast_ctx=fast_ctx)

def create_actions_df(
    net: pandapowerNet,
    dict_actions: list[dict[str, Any]] | list[defaultdict[str, Any]],
) -> pd.DataFrame:
    """
    Create a DataFrame with the actions from the dict_actions list.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :param dict_actions: List of action dictionaries containing substations and states.
    :type dict_actions: list[dict[str, Any]] | list[defaultdict[str, Any]]
    :return: The DataFrame containing the actions.
    :rtype: pd.DataFrame
    """
    calculate_open_and_closed_switches(net, dict_actions)

    # ``d.get(key, [])`` over ``d``'s own keys can never miss, so this is just ``dict(d)``.
    df_actions = pd.DataFrame([dict(d) for d in dict_actions])

    # Actions that omit a column land as NaN and must become an empty list. Only the missing
    # cells are visited: ``DataFrame.map`` called the replacement once per *cell*, which is
    # n_actions x n_columns Python calls (~1.4M on case89).
    missing = df_actions.isna()
    for column in df_actions.columns:
        rows = np.flatnonzero(missing[column].to_numpy())
        if rows.size:
            values = df_actions[column].to_numpy()
            for row in rows:
                values[row] = []
            df_actions[column] = values

    df_actions.index.name = None
    return df_actions


def create_unitary_substation_action(net: pandapowerNet) -> list[defaultdict[str, Any]]:
    """
    Find all possibilities to switch 1 double busbar substation.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :return: A list of action dictionaries representing possible single substation actions.
    :rtype: list[defaultdict[str, Any]]
    """
    connection_lengths: list[int] = net.multi_bb_substation["connected_buses"].str.len()

    unitary_actions: list[defaultdict[str, Any]] = []
    # Generate all possible binary states for each substation.
    number_busbars: list[int] = net.multi_bb_substation["n_busbars_in_substation"].tolist()
    all_bus_connection_tuples = list(
        zip(net.multi_bb_substation.index, connection_lengths, number_busbars),
    )
    for isub, connections, n_busbars in all_bus_connection_tuples:
        actions = create_unitary_substation_action_one_subs(
            n_connections=connections,
            bus=isub,
            number_busbars=n_busbars,
        )
        unitary_actions.extend(actions)
    action_offset = 1
    for action_counter, entry in enumerate(unitary_actions):
        entry["action"] = action_counter + action_offset
    unitary_actions.insert(0, return_donothing_action())
    return unitary_actions


def create_unitary_substation_action_one_subs(
    n_connections: int,
    bus: int,
    number_busbars: int,
) -> list[defaultdict[str, Any]]:
    """
    Find all possibilities to switch 1 double busbar substation.

    :param n_connections: Number of connections to the substation.
    :type n_connections: int
    :return: A dictionary representing possible single substation actions.
    """
    unitary_actions: list[defaultdict[str, list[Any]]] = []
    # Generate all possible binary states for each substation (stored in a hex-str).
    # - `helper_generate_binary_numbers` generates all possible binary configurations for a substation.
    for state in helper_generate_b_ary_numbers(
        num_digits=n_connections,
        base=number_busbars,
    ):
        action_dict: defaultdict[str, Any] = defaultdict(list)
        action_dict["substations"].append(bus)
        action_dict["states"].append(state)
        unitary_actions.append(action_dict)
    return unitary_actions


def return_donothing_action() -> defaultdict[str, Any]:
    """
    Add a do-nothing-action to list of actions.

    :return: A dictionary representing a do-nothing action.
    :rtype: defaultdict[str, Any]
    """
    action_dict: defaultdict[str, Any] = defaultdict(list)
    action_dict["action"] = 0
    return action_dict


def _create_unitary_line_actions(net: pandapowerNet) -> list[defaultdict[str, Any]]:
    """
    Return actions for switching each line off or on.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :return: A list of action dictionaries representing line switching actions.
    :rtype: list[defaultdict[str, Any]]
    """
    n = len(net.line)
    line_actions = []
    for i in range(1, n + 1):
        action_description: defaultdict[str, Any] = defaultdict(list)
        action_description["action"] = i
        action_description["lines"] = [i - 1]
        action_description["disconnect_lines"] = [bool(1)]
        line_actions.append(action_description)

        action_description2: defaultdict[str, Any] = defaultdict(list)
        action_description2["action"] = i + n
        action_description2["lines"] = [i - 1]
        action_description2["disconnect_lines"] = [bool(0)]
        line_actions.append(action_description2)

    return line_actions


def create_unitary_line_actions_and_donothing(
    net: pandapowerNet,
) -> list[defaultdict[str, Any]]:
    """
    Return actions for switching each line off or on.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :return: A list of action dictionaries representing line switching actions.
    :rtype: list[defaultdict[str, Any]]
    """
    line_actions = _create_unitary_line_actions(net)
    line_actions.insert(0, return_donothing_action())

    return line_actions


def add_actions_substation_line_switching(
    net: pandapowerNet,
) -> list[defaultdict[str, Any]]:
    """
    Add all actions for busbar and line switching.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :return: A combined list of action dictionaries for substation and line switching.
    :rtype: list[defaultdict[str, Any]]
    """
    bus_actions = create_unitary_substation_action(net)
    line_actions = _create_unitary_line_actions(net)
    all_actions = bus_actions + line_actions

    for action_counter, entry in enumerate(all_actions):
        entry["action"] = action_counter

    return all_actions


def create_all_unitary_pst_actions(net: pandapowerNet) -> list[defaultdict[str, Any]]:
    """
    Return actions for changing the tap position of phase shift transformers.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :return: A list of action dictionaries representing line switching actions.
    :rtype: list[defaultdict[str, Any]]
    """
    pst_df = net.trafo[(net.trafo["vn_hv_kv"] == net.trafo["vn_lv_kv"]) &
                       (net.trafo["tap_step_degree"] > 0.0)]
    pst_actions = []
    n_action = 1
    for pst in pst_df.itertuples():
        step = pst.tap_step_degree
        tap_pos_range = np.arange(pst.tap_min, pst.tap_max + step/1000., step)
        for pos in tap_pos_range:
            action_description: defaultdict[str, Any] = defaultdict(list)
            action_description["action"] = n_action
            action_description["trafos"] = [pst.Index]
            action_description["tap_pos"] = [pos]
            pst_actions.append(action_description)
            n_action += 1

    return pst_actions


def create_unitary_pst_actions_and_donothing(
    net: pandapowerNet,
) -> list[defaultdict[str, Any]]:
    """
    Return actions for changing the tap position of phase shift transformers.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :return: A list of action dictionaries representing line switching actions.
    :rtype: list[defaultdict[str, Any]]
    """
    pst_actions = create_all_unitary_pst_actions(net)
    pst_actions.insert(0, return_donothing_action())

    return pst_actions


def add_actions_substation_line_pst(
    net: pandapowerNet,
) -> list[defaultdict[str, Any]]:
    """
    Add all actions for busbar switching, line switching and PST tap position changing.

    :param net: The pandapower network object.
    :type net: pandapowerNet
    :return: A combined list of all actions.
    :rtype: list[defaultdict[str, Any]]
    """
    bus_actions = create_unitary_substation_action(net)
    line_actions = _create_unitary_line_actions(net)
    pst_actions = create_all_unitary_pst_actions(net)
    all_actions = bus_actions + line_actions + pst_actions

    for action_counter, entry in enumerate(all_actions):
        entry["action"] = action_counter

    return all_actions

def verify_all_actions(net: pandapowerNet, actions: list[defaultdict[str, Any]]) -> list[defaultdict[str, Any]]:
    """Use all rules on the start action set.

    Builds an all-switches-closed base graph and a switch->edge map once and threads them
    through ``verify_action`` so the connectivity rule reuses the base graph per action
    instead of rebuilding the whole networkx graph for every one of (potentially many
    thousands of) actions.
    """
    net.switch["closed"] = True
    base_graph = top.create_nxgraph(net, respect_switches=True)
    switch_edges = {
        int(idx): (int(bus), int(element))
        for idx, bus, element in zip(
            net.switch.index, net.switch["bus"], net.switch["element"], strict=False,
        )
    }
    fast_ctx = {"graph": base_graph, "switch_edges": switch_edges}

    subset = [action for action in actions if verify_action(net, action, fast_ctx=fast_ctx)]
    for new_id, action_dict in enumerate(subset):  # 🌈 assign new sequential index
        action_dict["action"] = new_id
    return subset
