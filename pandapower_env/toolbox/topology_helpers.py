from typing import TYPE_CHECKING

import numpy as np
from pandapower import pandapowerNet

if TYPE_CHECKING:
    import pandas as pd

def bus_switch_components(net: pandapowerNet, *, consider_open_switches: bool) -> dict[int, list[int]]:
    """
    Group every bus touched by a bus-bus switch into its connected component.

    This is the single traversal all bus-tree lookups in this module build on. It scans the
    bus-bus switch table once and unions the two endpoints of each relevant switch with a
    path-compressed union-find, which answers "which buses hang together via bus-bus switches"
    for the whole net in one pass instead of one fixpoint search per bus.

    Buses that no (relevant) bus-bus switch mentions are absent from the result - callers
    treat them as standalone.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param consider_open_switches: if True, an open switch does not join its two buses;
        if False, every bus-bus switch joins its buses regardless of its ``closed`` state
    :type consider_open_switches: bool
    :return: mapping of representative bus index -> sorted list of the buses in that component
    :rtype: dict[int, list[int]]
    """
    is_bus_switch = (net.switch["et"] == "b").to_numpy()
    if consider_open_switches:
        is_bus_switch &= net.switch["closed"].to_numpy().astype(bool)

    bus_a = net.switch["bus"].to_numpy()[is_bus_switch]
    bus_b = net.switch["element"].to_numpy()[is_bus_switch]

    parent: dict[int, int] = {}

    def find(bus: int) -> int:
        """Return the component representative of ``bus``, compressing the path on the way."""
        root = bus
        while parent[root] != root:
            root = parent[root]
        while parent[bus] != root:
            parent[bus], bus = root, parent[bus]
        return root

    for raw_a, raw_b in zip(bus_a.tolist(), bus_b.tolist(), strict=True):
        a, b = int(raw_a), int(raw_b)
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    components: dict[int, list[int]] = {}
    for bus in parent:
        components.setdefault(find(bus), []).append(bus)

    for member_buses in components.values():
        member_buses.sort()

    return components


def _component_of_bus(components: dict[int, list[int]], bus_to_root: dict[int, int], ibus: int) -> list[int]:
    """
    Return the bus tree containing ``ibus``, with ``ibus`` first (it becomes bus0).

    :param components: representative bus -> sorted member buses, from :func:`bus_switch_components`
    :type components: dict[int, list[int]]
    :param bus_to_root: member bus -> representative bus, the inverted ``components``
    :type bus_to_root: dict[int, int]
    :param ibus: the bus whose tree is requested
    :type ibus: int
    :return: ``[ibus]`` if no switch mentions the bus, else ``[ibus, *sorted(other members)]``
    :rtype: list[int]
    """
    root = bus_to_root.get(ibus)
    if root is None:
        return [ibus]
    return [ibus, *(bus for bus in components[root] if bus != ibus)]


def _invert_components(components: dict[int, list[int]]) -> dict[int, int]:
    """Build the member bus -> representative bus lookup for a component mapping."""
    return {bus: root for root, member_buses in components.items() for bus in member_buses}


def find_bus_switch_tree(net: pandapowerNet, ibus: int, *, consider_open_switches: bool) -> list[int]:
    """
    Find all buses connected by bus-bus switches to ibus (useful for creating double-busbar substations).

    Thin wrapper around :func:`bus_switch_components` so there is a single traversal
    algorithm in this module. Prefer :func:`find_bus_switch_trees_from_list` when several
    buses are queried against the same switch state - it shares one traversal across them.

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
    components = bus_switch_components(net, consider_open_switches=consider_open_switches)
    return _component_of_bus(components, _invert_components(components), ibus)


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
    used_buses: set[int] = set()

    if buses is None:
        buses = net.bus.index

    # One traversal for every requested bus, instead of one fixpoint search per bus.
    components = bus_switch_components(net, consider_open_switches=consider_open_switches)
    bus_to_root = _invert_components(components)

    for ibus in buses:

        if ibus in used_buses:
            if fail_on_overlap:
                msg = f"A tree for bus {ibus} was requested, but it is already identified in another bus tree."
                raise ValueError(msg)
            continue

        bus_tree = _component_of_bus(components, bus_to_root, ibus)

        if len(bus_tree) > 1:
            bus_trees.append(bus_tree)
            used_buses.update(bus_tree)

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

def _is_pst_in_substation(net: pandapowerNet, bus_id: int) -> bool:
    """Detect, if one trafo has both ends in the same substation."""
    # net.trafo lv_bus must be in the same connected_buses list as hv_bus
    trafo = net.trafo.loc[bus_id]
    hv_bus: int = trafo["hv_bus"]
    lv_bus: int = trafo["lv_bus"]

    # Early exit
    if hv_bus == lv_bus:
        return True
    substations: pd.DataFrame = net.multi_bb_substation
    for connected_buses in substations["connected_buses"]:
        # Convert to set for O(1) lookup instead of O(n) list lookup
        bus_set = set(connected_buses)
        # Check if both buses are in this substation
        if hv_bus in bus_set and lv_bus in bus_set:
            return True
    return False

def is_pst(net: pandapowerNet, bus_id: int) -> bool:
    """
    Decide whether a Trafo is a PST.

    Checking for the points:
    - trafo in one substation with both end-buses in the substation
        in own _is_PST_in_substation fct.
    - high&low voltage the same: net.trafo.vn_hv_kv == net.trafo.vn_lv_kv
    - tap_step_degree > 0
    - tab_changer_type: Not deterministically relevant
        if tap_changer_type != None, and none above match, it isn't a PST.
        if tap_changer_type=None, then these are Trafos without tab_change -> No PST.
    """
    row = net.trafo.loc[bus_id]
    if hasattr(net, "multi_bb_substation"): # slower function
        return _is_pst_in_substation(net, bus_id)
    if row.tap_changer_type is None:
        return False
    if row.vn_hv_kv != row.vn_lv_kv:
        return False
    if row.tap_step_degree is None or np.isnan(row.tap_step_degree) or row.tap_step_degree == 0: # noqa: SIM103
        return False
    return True
