
import numpy as np
from pandapower import pandapowerNet


def find_bus_switch_tree(net: pandapowerNet, ibus: int, *, consider_open_switches: bool) -> list[int]:
    """
    Find all buses connected by bus-bus switches to ibus (useful for creating double-busbar substations).

    :param net: The pandapower network
    :type net: pandapowerNet
    :param ibus: index of the bus to consider
    :type ibus: int
    :param consider_open_switches: if the switch connecting two buses is open,
        do not consider them part of the same tree if set to True
    :type consider_open_switches: bool
    :return: a list of integers corresponding to the bus indices
    :rtype: list[int]
    """
    bus_switches = net.switch[(net.switch.et == "b")]

    current_bus_tree = {ibus}

    while True:
        # Find all switches that contain any switch in the current bus tree

        # print current_bus_tree,new_bus_tree
        switch_is_involved = np.any(bus_switches[["bus", "element"]].isin(current_bus_tree), axis=1)

        switch_is_closed = True
        if consider_open_switches:
            switch_is_closed = bus_switches["closed"]

        sel_switches = switch_is_involved & switch_is_closed
        new_bus_tree = set(bus_switches[sel_switches][["bus", "element"]].to_numpy().flatten().tolist())

        if len(new_bus_tree) == 0:
            return [ibus]

        if current_bus_tree == new_bus_tree:
            # make sure ibus is always in the beginning (will become bus0)
            new_bus_tree_list = sorted(new_bus_tree)
            new_bus_tree_list.pop(new_bus_tree_list.index(ibus))
            return [ibus, *new_bus_tree_list]

        current_bus_tree = new_bus_tree


def find_bus_switch_trees_from_list(
        net: pandapowerNet,
        buses: list[int] | None,
        *,
        consider_open_switches: bool,
        include_single_buses: bool,
        fail_on_overlap: bool=True) -> list[list[int]]:
    """
    Find groups of buses connected by bus-bus switches, starting from a list of buses.

    :param net: The pandapower network
    :type net: pandapowerNet

    :param consider_open_switches: if the switch connecting two buses is open,
        do not consider them part of the same tree if set to True
    :type consider_open_switches: bool
    :param include_single_buses: if a single bus is alone in a tree, report (True) or skip (False) that bus tree
    :type include_single_buses: bool
    :param fail_on_overlap: Throw an error if a bus was already used in another tree (to avoid user error)
    :type fail_on_overlap: bool
    :return: a list of all groups of buses connected by bus-bus switches
    :rtype: list[list[int]]
    """
    bus_trees = []
    used_buses = []

    if buses is None:
        buses = net.bus.index

    for ibus in buses:

        if ibus in used_buses:
            if fail_on_overlap:
                msg = f"A tree for bus {ibus} was requested, but it is already identified in another bus tree."
                raise ValueError(msg)
            continue

        bus_tree = find_bus_switch_tree(net, ibus, consider_open_switches=consider_open_switches)

        if len(bus_tree) > 1:
            bus_trees.append(bus_tree)
            used_buses.extend(bus_tree)

        elif include_single_buses:
            # No switch mentions the bus - it is standalone.
            bus_trees.append([ibus])

    return bus_trees


def find_all_bus_switch_trees(
        net: pandapowerNet,
        *,
        consider_open_switches: bool,
        include_single_buses: bool) -> list[list[int]]:
    return find_bus_switch_trees_from_list(
        net,
        None,
        consider_open_switches=consider_open_switches,
        include_single_buses=include_single_buses,
        fail_on_overlap=False,
    )
