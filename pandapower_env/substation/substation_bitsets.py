"""
Module for imposing bus rules, given a hexset representation.

This module uses bitwise/hex logic to efficiently check for bus
configuration rule violations.
Bus configurations are represented as hex strings, e.g. "0x100111"
"""

from typing import Any, Callable


def fully_connected_hexset(n_bits: int) -> str:
    """
    Return a bitset representing a fully connected substation (e.g. all buses are connected to one another).

    This corresponds to a bitset of e.g. 0x1111111.
    The preferred bus is bus 1, even in multi-busbar substations, to be compatible with the enforcement
    of action rules.

    :param n_bits: number of bits for the bitset
    :type n_bits: int (non-negative)
    :return: hex string of length n
    :rtype: str (hexadecimal)
    """
    return "0x" + "1"*n_bits


def _hexset_check_n_busbar_violation(hexset: str, nbusbars: int) -> bool:
    """
    Check whether the hexset contains buses that do not exist (check if any hex digit is larger than nbusbars).

    :param hexset: The substation configuration hexset
    :type hexset: str
    :param nbusbars: The number of busbars in the substation (digits in the hex)
    :type nbusbars: int
    """
    # Check that each hex value (e.g. "B" is less than the number of busbars in the substation)
    return all(int(hex_i, 16) < nbusbars for hex_i in hexset.removeprefix("0x"))


def _hexset_check_n_elements(hexset: str, *, nbits: int, nbusbars: int) -> bool:
    """
    Check that the hex number does not exceed the maximum value according to the number of bits (elements) and busbars.

    :param hexset: The substation configuration hexset
    :type hexset: str
    :param nbits: The number of digits in the hexset (elements connected to the substation)
    :type nbits: int
    :param nbusbars: The number of busbars in the substation
    :type nbusbars: int
    """
    # There should be exactly n digits in the hex
    if len(hexset.removeprefix("0x")) != nbits:
        return False

    # max_hex is e.g. 0x333333 for a 4-busbar substation with 6 elements
    max_hex = "0x" + str(nbusbars-1)*nbits
    return int(hexset, 16) <= int(max_hex, 16)


def hexset_to_closed_switch_list(
        hexset: str,
        *,
        busbar: int,
        nbits: int,
        nbusbars: int) -> list[int]:
    """
    Convert a hexset e.g. 0x11010 to a list of booleans for whether the switch is open or closed.

    The conversion conventions are the following:
     - 0x11010 --> [True, True, False, True, False] for bus 1
     (in other words, "1" in the bitset represents assignment to bus 1.)

    Note that the hex number is read from left to right, i.e.
    element 0 corresponds to "1" in the example 0x11010. This is to conform
    to the convention that we read everything typically from left to right.

    :param hexset: The substation configuration hexset
    :type hexset: str
    :param busbar: Which busbar to consider (busbar 0 or busbar 1)
    :type busbar: int
    :param nbits: The number of digits in the hexset (elements connected to the substation)
    :type nbits: int
    :param nbusbars: The number of busbars in the substation
    :type nbusbars: int
    :return: list of booleans for whether the switches are closed.
    :rtype: list of bool
    """
    if not _hexset_check_n_busbar_violation(hexset, nbusbars):
        msg = "A bit in the flag is larger than number of busbars in the substation."
        raise RuntimeError(msg)

    if not _hexset_check_n_elements(hexset, nbits=nbits, nbusbars=nbusbars):
        msg = "Flag is larger than number of bits in substation."
        raise RuntimeError(msg)

    # Convert e.g. 0xAA0A1 to [T T F T F]
    return [int(hex_i, 16) == busbar for hex_i in hexset.removeprefix("0x")]


def binary_flag_busbar_n(hexset: str, busbar: int) -> int:
    """
    Find the elements set to a busbar, and return the results as a BINARY number.

    In other words, convert e.g. a 0x33030 into the binary 0b11010

    :param hexset: The substation configuration hexset
    :type hexset: str
    :return: Binary corresponding to the elements set to the specified busbar.
    :rtype: int
    """
    hexdigits = hexset.removeprefix("0x")
    return sum(0b1 << i for i, hex_i in enumerate(reversed(hexdigits)) if int(hex_i, 16) == busbar)


def create_binary_flag_from_list(
    _list: list[Any],
    condition_lambda: Callable[[Any], bool] = lambda x: bool(x),
) -> int:
    """
    Create a BINARY flag from a list of items, using some condition to determine what is "0" and "1".

    An example for double-busbar substations is to find which elements are lines or trafos using sub.element_type:

    create_flag_from_list(sub.element_type, lambda x: x in ["line","trafo"])

    will convert ["line", "line", "trafo", "gen", "load"] to 0b11100

    :param _list: List to consider (e.g. sub.element_type)
    :type _list: list[Any]
    :param condition_lambda: callable function to evaluate each list item into a boolean ("0" or "1")
    :type condition_lambda: Callable[[Any], bool]
    :return: bitset corresponding to the evaluated list.
    :rtype: int
    """
    # Important NOTE: "Reversed" because we read bitsets from left to right
    # to match the fact that we also read lists from left to right.
    return sum([0b1 << i for i, tp in enumerate(reversed(_list)) if condition_lambda(tp)])
