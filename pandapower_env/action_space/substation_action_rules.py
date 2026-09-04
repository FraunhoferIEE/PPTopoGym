import itertools
from collections.abc import Generator

import networkx as nx
import pandapower.topology as top
import pandas as pd
from pandapower import pandapowerNet

from pandapower_env.substation.double_busbar_substation import (
    busbar_columns,
    get_list_of_closed_and_open_substation_switches,
)
from pandapower_env.substation.substation_bitsets import (
    binary_flag_busbar_n,
    create_binary_flag_from_list,
)

island_elements = ["ext_grid", "load", "gen", "sgen"]
bridge_elements = ["line", "trafo", "trafo<PST>"]


def passes_two_bus_symmetry_rule(hexset: str) -> bool:
    """
    Enforce the rule that the first element must be assigned to bus 0 -- even in the case of >2 buses.

    Assumption: The hexset here represents a hexadecimal of busbar assignments for elements.
    We do not want the first bit to be "0" to break symmetries.
    This is to break the symmetry e.g. 10110 <--> 01001, which is electrically identical.
    Be careful: 0b0011 (for a substation with 4 external elements) will sometimes show up as "11",
    omitting the leading zeros, but this bitset will fail regardless of whether it's
    represented as "0b0011" or "11" or "0b11".

    :param hexset: hexadecimal containing busbar assignments
    :type hexset: str
    :return: "True" if the bitset starts with a "0"
    :rtype: bool
    """
    return hexset.removeprefix("0x")[0] == "0"


def passes_islanded_elements_rule(sub: pd.Series, hexset: str) -> bool:
    """
    Enforce the rule that a busbar cannot contain an element (load, gen, etc.) and no connecting line or trafo.

    In this test, pst trafos are considered bridge elements, so an islanding
    can still occur, but this will be taken care of in a separate rule.

    :param sub: row in net.multi_bb_substation corresponding to a substation
    :type sub: pd.Series
    :param hexset: hexadecimal containing busbar assignment
    :type hexset: str
    :return: "True" if no gens/loads/etc. are alone on a busbar without a line/trafo.
    :rtype: bool
    """
    # Binary bitset containing a "1" for island (bridge) elements, and a "0" if otherwise.
    island_ele_bitset = create_binary_flag_from_list(
        sub.element_type, lambda x: x in island_elements,
    )
    bridge_ele_bitset = create_binary_flag_from_list(
        sub.element_type, lambda x: x in bridge_elements,
    )

    for busbar in range(sub.n_busbars_in_substation):
        # Convert e.g. a 0x33030 into the binary 0b11010
        busbar_bitset = binary_flag_busbar_n(hexset, busbar)

        island_ele_bitset_busbar = island_ele_bitset & busbar_bitset
        bridge_ele_bitset_busbar = bridge_ele_bitset & busbar_bitset

        # A generator/load cannot be alone on a busbar with no line/trafo connecting to the outside.
        if island_ele_bitset_busbar and not bridge_ele_bitset_busbar:
            return False

    return True


def passes_fully_connected_grid_rule(
    net: pandapowerNet,
    substations: list[int],
    states: list[str],
    fast_ctx: dict | None = None,
) -> bool:
    """
    Enforce the rule that actions cannot split the grid into two or more sub-grids.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param substations: list of substation indices in the net.multi_bb_substation DataFrame
    :type substations: list[int]
    :param states: List of bitsets containing busbar assignment for each substation in parameter "substations"
    :type states: list[int]
    :param fast_ctx: Optional context built once by :func:`verify_all_actions` to avoid
        rebuilding the whole networkx graph for every action. It holds an all-switches-
        closed base graph (``"graph"``) and a ``{switch_index: (bus, element)}`` map
        (``"switch_edges"``). When given, the per-action graph is derived from the base
        graph by removing one edge per opened (bus-bus) switch and the unused-busbar
        nodes -- which reproduces ``create_nxgraph(respect_switches=True)``'s
        connectivity exactly (connectivity depends only on the edge count per node pair).
    :type fast_ctx: dict | None
    :return: whether graph is fully connected
    :rtype: bool
    """
    local_busbar_columns = busbar_columns(net.multi_bb_substation)
    open_switches_all: list[int] = []
    ignore_busbars: list = []

    for i_sub, state in zip(substations, states):
        _, open_switches = get_list_of_closed_and_open_substation_switches(net, i_sub, state)
        open_switches_all.extend(open_switches)

        # Check for islanded busbars, and do not consider them when evaluating connectedness.
        busbars = net.multi_bb_substation.loc[i_sub, local_busbar_columns].dropna()
        for i_bb, busbar in enumerate(busbars):
            # If the busbar is not in the state, then it is allowed to be "islanded".
            if str(i_bb) not in state.removeprefix("0x"):
                ignore_busbars.append(busbar)

    if fast_ctx is None:
        # Fallback: rebuild the graph from the net's switch state (original behaviour).
        net.switch["closed"] = True
        if open_switches_all:
            net.switch.loc[open_switches_all, "closed"] = False
        _nxgraph = top.create_nxgraph(net, respect_switches=True, nogobuses=ignore_busbars)
        return nx.is_connected(_nxgraph)

    return _connected_via_base_graph(fast_ctx, open_switches_all, ignore_busbars)


def _connected_via_base_graph(
    fast_ctx: dict,
    open_switches: list[int],
    ignore_busbars: list,
) -> bool:
    """
    Check connectivity by reusing the cached all-switches-closed base graph.

    Temporarily removes this action's opened-switch edges and unused-busbar nodes from
    the base graph, tests connectivity, then restores it. ``is_connected`` is ~0.2ms
    while copying the whole graph costs more than rebuilding it, so we mutate in place
    and undo. Removing one edge per opened (bus-bus) switch matches
    ``create_nxgraph(respect_switches=True)`` for connectivity -- only the edge count per
    node pair matters -- and the base graph is restored exactly via the saved keys/data
    so subsequent actions start from the same state.
    """
    graph = fast_ctx["graph"]
    switch_edges = fast_ctx["switch_edges"]
    removed_edges: list = []
    removed_nodes: list = []
    try:
        for busbar in set(ignore_busbars):
            if graph.has_node(busbar):
                removed_nodes.append((busbar, list(graph.edges(busbar, keys=True, data=True))))
                graph.remove_node(busbar)
        for sw in open_switches:
            bus, element = switch_edges[int(sw)]
            if graph.has_edge(bus, element):
                key = next(iter(graph[bus][element]))
                removed_edges.append((bus, element, key, graph[bus][element][key]))
                graph.remove_edge(bus, element, key)
        return nx.is_connected(graph)
    finally:
        # Restore the base graph exactly (re-add edges first, then nodes + their edges).
        graph.add_edges_from(removed_edges)
        for busbar, incident in removed_nodes:
            graph.add_node(busbar)
            graph.add_edges_from(incident)


def passes_n_elements_rule(hexset: str) -> bool:
    """
    Enforce the rule that a busbar cannot contain only one element (e.g. 0x100000 or 0x111101).

    :param hexset: hexadecimal containing busbar assignment
    :type hexset: str
    :return: "True" if there are either 0 elements on a bus bar, or 2+ elements.
    :rtype: bool
    """
    hexset_list = [int(i_hex, 16) for i_hex in hexset.removeprefix("0x")]
    hexset_set = set(hexset_list)

    return all(hexset_list.count(i_busbar) != 1 for i_busbar in hexset_set)


def helper_generate_b_ary_numbers(
        num_digits: int,
        *,
        base: int = 2,
) -> Generator[str, None, None]:
    """
    Generate a list of b-ary numbers with num_digits digits.

    Workflow:
    - check input: 2 <= base <= 16
    - generate all possible b-ary numbers with num_digits digits:
        - use itertools.product to generate all combinations of digits
        - filter out numbers starting with 0
        - convert to hex-string

    :param num_digits: number of digits in the b-ary number
    :type num_digits: int
    :param base: base of the number system (default is 2 for binary)
    :type base: int
    :return: Generator of integers representing possibly legal busbar configurations
    :rtype: Generator[str, None, None]
    """
    if num_digits <= 0:
        msg = "num_digits must be a positive"
        raise ValueError(msg)

    if base < 2 or base > 16:  # noqa: PLR2004, max is hexadecimal
        msg = "Base must be between 2 and 16"
        raise ValueError(msg)

    hex_chars = "0123456789abcdef"

    def _hex_str_from_char_tuple(char_tuple: tuple[int, ...]) -> str:
        return "".join([hex_chars[d] for d in char_tuple])

    # E.g. (0, 1, 2) in the case of 3 busbars
    busbar_range = range(base)

    # The first busbar must be set to 0 due to symmetry rules
    # So for 3 busbars and 4 digits (elements), this is ( (0,), (0, 1, 2), (0, 1, 2), (0, 1, 2) )
    # and we will take a product over these.
    busbar_ranges = ((0,), *(busbar_range,) * (num_digits - 1))

    def _enforce_bus_order(the_tuple: tuple[int, ...]) -> bool:
        # The bus order should be 0, 1, 2, 3, 4, ...
        # In other words, we enforce a symmetry rule that the bus order is fixed to the above.
        # Otherwise, 0x0211 and 0x0122 are symmetric under (2 <-> 1)
        bus_order = list(range(base))  # [1, 0, *list(range(2, base))]
        first_instance = [the_tuple.index(a) if a in the_tuple else 16 for a in bus_order]
        return all(x <= y for x, y in zip(first_instance, first_instance[1:]))

    bus_assignment_tuples = filter(
        _enforce_bus_order,
        itertools.product(*busbar_ranges),
    )

    yield from map(
        _hex_str_from_char_tuple,
        bus_assignment_tuples,
    )

