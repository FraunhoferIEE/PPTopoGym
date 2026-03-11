"""
File containing functions for manipulating double-busbar substations.

Double-busbar substations are stored as a descriptive pandapower-style Dataframe called `net.multi_bb_substation`.

A double-busbar substation is represented by a Dataframe with the following
columns:
 - bus_0: the busbar labeled "0"
 - bus_1: the busbar labeled "1"
 - connected_buses: the buses that connect to outside elements (lines, loads, etc.)
 - connected_elements: the index of each connected element in the net[element] Dataframe
 - element_type: the element type of the connected elements ('line', 'trafo', 'load', etc.)
 - b01_switch: the switch connecting bus_0 and bus_1
 - b0_switches: the bus-bus switches connecting bus_0 with the buses in the connected_buses list
 - b1_switches: the bus-bus switches connecting bus_1 with the buses in the connected_buses list

The columns `connected_buses`, `connected_elements`, and `element_type` have the same
length and order. For example, if element `i` is line 5 (net.line.loc[5]) connected to the
substation via bus 2 (net.bus.loc[2]), then this corresponds to
 - connected_buses[i] = 2
 - connected_elements[i] = 5
 - element_type[i] = 'line'

b0_switches and b1_switches also have the same order as `connected_buses`, etc. and correspond
to the switches connecting the `connected_buses` to bus_0 or bus_1 (respectively).
Switch b01_switch connects bus_0 and bus_1 together.
"""

import re

import pandas as pd
from pandapower import pandapowerNet

from pandapower_env.substation.substation_bitsets import (
    fully_connected_hexset,
    hexset_to_closed_switch_list,
)


def _bus_coupler_switch_columns(df_mbb: pd.DataFrame) -> list[str]:
    """
    Find all bus coupler columns (of the format b01_switch, b45_switch, etc.).

    :param df_mbb: The multi-busbar dataframe
    :type df_mbb: pd.DataFrame
    :return:
    :rtype: list[str]
    """
    # The ^ denotes the beginning of the string, $ the end of the string, and \d a number
    return [col for col in df_mbb.columns if re.match(r"^b\d+_switch$", col)]


def _element_switch_columns(df_mbb: pd.DataFrame) -> list[str]:
    """
    Find all switch columns (of the format b1_switches, b2_switches, etc.).

    :param df_mbb: The multi-busbar dataframe
    :type df_mbb: pd.DataFrame
    :return:
    :rtype: list[str]
    """
    # The ^ denotes the beginning of the string, $ the end of the string, and \d a number
    return [col for col in df_mbb.columns if re.match(r"^b\d+_switches$", col)]


def _busbar_columns(df_mbb: pd.DataFrame) -> list[str]:
    """
    Find all busbar columns (of the format bus_0, bus_1, etc.).

    :param df_mbb: The multi-busbar dataframe
    :type df_mbb: pd.DataFrame
    :return:
    :rtype: list[str]
    """
    # The ^ denotes the beginning of the string, $ the end of the string, and \d a number

    return [col for col in df_mbb.columns if re.match(r"^bus_\d+$", col)]


def reorder_substation_df(df_mbb: pd.DataFrame) -> list[str]:
    """
    Return the standard column order for the substation DataFrame.

    :param df_mbb: The multi-busbar dataframe
    :type df_mbb: pd.DataFrame
    :return:
    :rtype: list[str]
    """
    bus_couplers_cols = _bus_coupler_switch_columns(df_mbb)

    other_columns = [
        "connected_buses",
        "n_busbars_in_substation",
        "element_type",
        "connected_elements",
    ]

    col_order = _busbar_columns(df_mbb) + bus_couplers_cols + other_columns + _element_switch_columns(df_mbb)
    return df_mbb[col_order]


def get_all_substation_switches(net: pandapowerNet, i_sub: int) -> list[int]:
    """
    Get a list of all the substation switches in a given substation.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    :return: The index of each of the switches in the substation
    :rtype: list of int
    """
    sub = net.multi_bb_substation.loc[i_sub]

    ele_switches = sub[_element_switch_columns(net.multi_bb_substation)].dropna().sum()
    coupler_switches = sub[_bus_coupler_switch_columns(net.multi_bb_substation)].dropna().tolist()

    return ele_switches + coupler_switches


def is_fully_connected(net: pandapowerNet, i_sub: int, hexset: str) -> bool:
    """
    Check whether the hexset corresponds to a fully connected substation.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    :param hexset: hexset containing busbar assignment
    :type hexset: str | int (can be 0)
    """
    sub = net.multi_bb_substation.loc[i_sub]

    non_pst_elements = [x != "trafo_pst" for x in sub.element_type]
    hexset_nopst = [x for x, y in zip(hexset.removeprefix("0x"), non_pst_elements) if y]

    # Check whether they are all the same value
    return len(set(hexset_nopst)) == 1


def get_list_of_closed_and_open_substation_switches(
    net: pandapowerNet, i_sub: int, hexset: str,
) -> tuple[list[int], list[int]]:
    """
    Given a bitset corresponding to a busbar assignment in a substation, return the switches to be closed/opened.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    :param hexset: hexadecimal containing busbar assignment
    :type hexset: str
    :return tuple: List of closed (first index of tuple) and open (2nd index of tuple) switches
    :rtype: tuple[list[float], list[float]]
    """
    sub = net.multi_bb_substation.loc[i_sub]
    nbits = len(sub.connected_buses)
    nbusbars = sub.n_busbars_in_substation

    if is_fully_connected(net, i_sub, hexset):
        return get_all_substation_switches(net, i_sub), []

    open_switches = []
    closed_switches = []

    # Do not forget to open the switches between
    # bus 0 and bus 1 (etc.) if substation is not fully connected.
    coupler_switches = sub[_bus_coupler_switch_columns(net.multi_bb_substation)].dropna().tolist()
    open_switches.extend(coupler_switches)

    for ibusbar in range(nbusbars):
        _closedbits = hexset_to_closed_switch_list(hexset, busbar=ibusbar, nbits=nbits, nbusbars=nbusbars)

        sw_colname = f"b{ibusbar:d}_switches"
        _closed_switches = [switch for switch, is_closed in zip(sub[sw_colname], _closedbits) if is_closed]
        _open_switches = [switch for switch, is_closed in zip(sub[sw_colname], _closedbits) if not is_closed]

        closed_switches.extend(_closed_switches)
        open_switches.extend(_open_switches)

    # convert to int
    return [int(c) for c in closed_switches], [int(o) for o in open_switches]


def set_substation_switches(net: pandapowerNet, i_sub: int, hexset: str) -> None:
    """
    Set the network switches for a given substation according to the bitset specified.

    This operation will affect the outcome of the power flow calculation (run_pp).

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    :param hexset: binary bitset containing busbar assignment
    :type hexset: str
    """
    closed_list, open_list = get_list_of_closed_and_open_substation_switches(net, i_sub, hexset)

    net.switch.loc[closed_list, "closed"] = True
    net.switch.loc[open_list, "closed"] = False


def close_all_substation_switches(net: pandapowerNet, i_sub: int) -> None:
    """
    Close all the switches associated with a given substation (resulting in a fully-connected substation).

    This operation will affect the outcome of the power flow calculation (run_pp).

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param i_sub: index of the substation in the net.multi_bb_substation Dataframe
    :type i_sub: int
    """
    n_bits = len(net.multi_bb_substation.loc[i_sub, "connected_elements"])
    set_substation_switches(net, i_sub, fully_connected_hexset(n_bits))


def reset_all_substations(net: pandapowerNet) -> None:
    """
    Configure all substations to be fully connected.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    """
    for i_sub in net.multi_bb_substation.index:
        close_all_substation_switches(net, i_sub)


def select_element_indices(sub: pd.Series, el_type: str = "line") -> list[int]:
    """
    Select the element indices of the specified type (e.g. line, trafo, etc.).

    The [e.g. line] indices returned correspond to the `net.line` indices.

    :param sub: Series corresponding to a row in the net.multi_bb_substation Dataframe
    :type sub: pd.Series
    :param el_type: type of element ('line', 'trafo', 'gen', 'load', etc.)
    :type el_type: str
    :return: list of element indices for the corresponding net[el_type] Dataframe
    :rtype: list of int
    """
    return [
        sub.connected_elements[i]
        for i, tp in enumerate(sub.element_type)
        if tp == el_type
    ]
