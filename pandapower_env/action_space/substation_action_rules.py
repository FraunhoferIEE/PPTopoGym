import itertools
from collections.abc import Generator

import networkx as nx
import pandapower.topology as top
import pandas as pd
from pandapower import pandapowerNet

from pandapower_env.substation.multi_bb_substation import (
    get_list_of_open_substation_switches,
)
from pandapower_env.substation.substation_bitsets import (
    binary_flag_busbar_n,
    create_binary_flag_from_list,
)

island_elements = ["ext_grid", "load", "gen", "sgen"]
bridge_elements = ["line", "trafo", "trafo<PST>"]


def passes_two_bus_symmetry_rule(hexset: str) -> bool:
    """
    Enforce the rule that the first element must be assigned to bus 1 -- even in the case of >2 buses.

    Assumption: The hexset here represents a hexadecimal of busbar assignments for elements.
    We do not want the first bit to be "0" to break symmetries.
    This is to break the symmetry e.g. 10110 <--> 01001, which is electrically identical.
    Be careful: 0b0011 (for a substation with 4 external elements) will sometimes show up as "11",
    omitting the leading zeros, but this bitset will fail regardless of whether it's
    represented as "0b0011" or "11" or "0b11".

    :param sub: row in net.multi_bb_substation corresponding to a substation
    :type sub: pd.Series
    :param hexset: hexadecimal containing busbar assignments
    :type hexset: str
    :return: "True" if the bitset starts with a "1"
    :rtype: bool
    """
    return hexset.removeprefix("0x")[0] == "1"


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
) -> bool:
    """
    Enforce the rule that actions cannot split the grid into two or more sub-grids.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param substations: list of substation indices in the net.multi_bb_substation DataFrame
    :type substations: list[int]
    :param states: List of bitsets containing busbar assignment for each substation in parameter "substations"
    :type states: list[int]
    :return: whether graph is fully connected
    :rtype: bool
    """
    net.switch["closed"] = True

    for i_sub, state in zip(substations, states):
        open_switches = get_list_of_open_substation_switches(net, i_sub, state)
        net.switch.loc[open_switches, "closed"] = False

    _nxgraph = top.create_nxgraph(net, respect_switches=True)

    return nx.is_connected(_nxgraph)


def passes_n_elements_rule(hexset: str) -> bool:
    """
    Enforce the rule that a busbar cannot contain only one element (e.g. 0x100000 or 0x111101).

    :param sub: row in net.multi_bb_substation corresponding to a substation
    :type sub: pd.Series
    :param hexset: hexadecimal containing busbar assignment
    :type hexset: str
    :return: "True" if there are either 0 elements on a bus bar, or 2+ elements.
    :rtype: bool
    """
    hexset_list = [int(i_hex,16) for i_hex in hexset.removeprefix("0x")]
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
    :param number_steps: number of steps to generate
    :type number_steps: int | None
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

    # E.g. (2, 1, 0) in the case of 3 busbars
    busbar_range = range(base - 1, -1, -1)

    # The first busbar must be set to 1 due to symmetry rules
    # So for 3 busbars and 4 digits (elements), this is ( (1,), (2, 1, 0), (2, 1, 0), (2, 1, 0) )
    # and we will take a product over these.
    busbar_ranges = ((1,), *(busbar_range,) * (num_digits - 1))

    def _enforce_bus_order(the_tuple: tuple[int, ...]) -> bool:
        # The bus order should be 1, 0, 2, 3, 4, ...
        # In other words, we enforce a symmetry rule that the bus order is fixed to the above.
        # Otherwise, 0x1200 and 0x1022 are symmetric under (2 <-> 0)
        bus_order = [1, 0, *list(range(2, base))]
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


def count_n_instances(hexset: str, hexdigit: str) -> int:
    """
    Count the number of hexdigit (e.g. "B") in a bitset.

    :param bitset: The substation configuration bitset
    :type bitset: int
    :return: Number of ones counted
    :rtype: int
    """
    return hexset.removeprefix("0x").count(hexdigit)
