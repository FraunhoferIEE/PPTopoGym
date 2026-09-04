"""
Helper functions for writing streamlined code that applies to e.g. both lines and trafos.

This should help shorten your code.
"""
from pandapower import pandapowerNet

from pandapower_env.toolbox.topology_helpers import bus_switch_components

ELEMENT_TYPES = ["line", "trafo", "ext_grid", "load", "gen", "sgen"]


# One row per element type: (from_bus-style column, to_bus-style column, short name).
# Elements with a single bus repeat it on both ends, which is what lets callers iterate over
# lines and trafos with the same code.
_ELEMENT_BUS_COLUMNS: dict[str, tuple[str, str, str]] = {
    "line": ("from_bus", "to_bus", "line"),
    "trafo": ("lv_bus", "hv_bus", "trafo"),
    "ext_grid": ("bus", "bus", "ext"),
    "gen": ("bus", "bus", "gen"),
    "sgen": ("bus", "bus", "sgen"),
    "load": ("bus", "bus", "load"),
}


def get_from_bus_str(element_type: str) -> str | None:
    """Return the "from_bus"-style column of an element type (e.g. ``lv_bus`` for a trafo).

    :param element_type: Which element to consider (e.g. "trafo", "line", etc.)
    :return: the column name, or None for an unknown element type.
    """
    entry = _ELEMENT_BUS_COLUMNS.get(element_type)
    return entry[0] if entry else None


def get_to_bus_str(element_type: str) -> str | None:
    """Return the "to_bus"-style column of an element type (e.g. ``hv_bus`` for a trafo).

    :param element_type: Which element to consider (e.g. "trafo", "line", etc.)
    :return: the column name, or None for an unknown element type.
    """
    entry = _ELEMENT_BUS_COLUMNS.get(element_type)
    return entry[1] if entry else None


def get_element_type_string(element_type: str) -> str | None:
    """Return the shortened name of an element type (e.g. ``ext`` for ``ext_grid``).

    Harmonizes the strings this package uses, so they can be changed centrally.

    :param element_type: Which element to consider (e.g. "trafo", "line", etc.)
    :return: the short name, or None for an unknown element type.
    """
    entry = _ELEMENT_BUS_COLUMNS.get(element_type)
    return entry[2] if entry else None


def harmonize_gen_voltage_setpoints(net: pandapowerNet, cache: dict | None = None) -> None:
    """
    Find electrically connected generators and average their voltage set points.

    The grouping of generators that share a point of common coupling depends only on the
    switch states (the generator buses are fixed), so it is expensive to recompute -- a
    bus-switch-tree traversal -- but stable between switching actions. When a ``cache``
    dict is provided, the grouping is memoised and rebuilt only when the bus-bus switch
    states change. The averaging itself depends on the per-timestep vm_pu setpoints and is
    always reapplied.

    :param net: The pandapower network (``net.gen.vm_pu`` is updated in place).
    :type net: pandapowerNet
    :param cache: Optional dict memoising the generator grouping across power flows.
        Runs uncached (recomputing the grouping every call) when ``None``.
    :type cache: dict | None
    """
    for pcc_generators in _pcc_generator_groups(net, cache):
        net.gen.loc[pcc_generators, "vm_pu"] = net.gen.loc[pcc_generators, "vm_pu"].mean()


def _pcc_generator_groups(net: pandapowerNet, cache: dict | None) -> list[list[int]]:
    """
    Return generator index groups (>=2 gens) that share a point of common coupling.

    Two layers keep this off the hot path. First, whether *any* two generators can share a
    PCC depends only on the generator buses and on which buses a bus-bus switch could ever
    join -- neither changes over an episode -- so a net where no such pair exists is
    answered with ``[]`` without touching the switch table (see :func:`_pcc_is_possible`).
    Second, for nets where it is possible, the grouping is memoised on the *bus-bus* switch
    states when ``cache`` is given, so element-switch writes do not invalidate it.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param cache: Optional memo dict, or ``None`` to recompute every call
    :type cache: dict | None
    :return: groups of ``net.gen`` indices sharing a point of common coupling
    :rtype: list[list[int]]
    """
    if cache is not None:
        # The switch table is static over an episode; re-derive the mask if it ever is not,
        # so a net that gains or loses switches cannot misindex a stale mask.
        bus_switch_rows = cache.get("bus_switch_rows")
        if bus_switch_rows is None or len(bus_switch_rows) != len(net.switch):
            bus_switch_rows = (net.switch["et"] == "b").to_numpy()
            cache["bus_switch_rows"] = bus_switch_rows
            cache["pcc_possible"] = _pcc_is_possible(net)
            cache.pop("switch_state_key", None)

        if not cache["pcc_possible"]:
            return []

        # Key on the bus-bus switches only: element switches cannot regroup generators.
        switch_state_key = net.switch["closed"].to_numpy()[bus_switch_rows].tobytes()
        if cache.get("switch_state_key") == switch_state_key:
            return cache["pcc_generator_groups"]
    elif not _pcc_is_possible(net):
        return []

    pcc_generator_groups = _group_generators_by_component(net)

    if cache is not None:
        cache["switch_state_key"] = switch_state_key
        cache["pcc_generator_groups"] = pcc_generator_groups
    return pcc_generator_groups


def _pcc_is_possible(net: pandapowerNet) -> bool:
    """
    Decide whether any two generators could ever share a point of common coupling.

    Answers the question for *all* switch states at once by ignoring ``closed`` entirely:
    if two generators share a bus outright, or fall in the same component when every
    bus-bus switch is treated as closed, then some switch state groups them. Otherwise no
    switch state ever can, and the grouping is permanently empty.

    Callers may cache this per net: topology actions only flip switch ``closed`` flags, so
    neither ``net.gen["bus"]`` nor the bus-bus switch wiring changes after the substations
    are built. Recompute it if a net is ever rewired in place.

    :param net: The pandapower network
    :type net: pandapowerNet
    :return: True if some switch state could put two generators on one PCC
    :rtype: bool
    """
    gen_buses = net.gen["bus"].to_numpy()
    min_generators_to_care = 2
    if len(gen_buses) < min_generators_to_care:
        return False

    # Two generators on the same bus share a PCC no matter how the switches stand.
    if len(set(gen_buses.tolist())) < len(gen_buses):
        return True

    # Otherwise only a bus-bus switch can join them - check the most permissive state.
    components = bus_switch_components(net, consider_open_switches=False)
    bus_to_root = {bus: root for root, member_buses in components.items() for bus in member_buses}
    reachable_roots = [bus_to_root[bus] for bus in gen_buses.tolist() if bus in bus_to_root]
    return len(set(reachable_roots)) < len(reachable_roots)


def _group_generators_by_component(net: pandapowerNet) -> list[list[int]]:
    """
    Group generator indices by the bus-switch component their bus currently sits in.

    Maps ``net.gen["bus"]`` through the component map rather than scanning ``net.gen`` once
    per component, so the cost is linear in the number of generators.

    :param net: The pandapower network
    :type net: pandapowerNet
    :return: groups of at least two ``net.gen`` indices that share a point of common coupling
    :rtype: list[list[int]]
    """
    components = bus_switch_components(net, consider_open_switches=True)
    bus_to_root = {bus: root for root, member_buses in components.items() for bus in member_buses}

    # Generators on a bus no closed switch touches are their own group, keyed by the bus itself.
    generators_by_pcc: dict[int, list[int]] = {}
    for gen_index, gen_bus in zip(net.gen.index.tolist(), net.gen["bus"].tolist(), strict=True):
        pcc = bus_to_root.get(gen_bus, gen_bus)
        generators_by_pcc.setdefault(pcc, []).append(gen_index)

    min_generators_to_care = 2
    return [generators for generators in generators_by_pcc.values() if len(generators) >= min_generators_to_care]
