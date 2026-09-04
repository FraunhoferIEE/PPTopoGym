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

import functools
import re

import pandas as pd
from pandapower import pandapowerNet

from pandapower_env.substation.substation_bitsets import (
    fully_connected_hexset,
    hexset_to_closed_switch_list,
)


@functools.lru_cache(maxsize=None)
def _classified_columns(columns: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Split substation-table column names into (busbar, bus-coupler switch, element switch).

    Cached on the column tuple because this is a pure function of the table's *layout*, which
    never changes while actions are being generated -- and it used to be re-derived with a
    regex over every column four times per candidate action, which dominated
    ``verify_all_actions`` on large grids.

    :param columns: the substation DataFrame's column names.
    :type columns: tuple[str, ...]
    :return: (``bus_N``, ``bN_switch``, ``bN_switches``) column-name tuples.
    :rtype: tuple[tuple[str, ...], ...]
    """
    # The ^ denotes the beginning of the string, $ the end of the string, and \d a number
    return (
        tuple(col for col in columns if re.match(r"^bus_\d+$", col)),
        tuple(col for col in columns if re.match(r"^b\d+_switch$", col)),
        tuple(col for col in columns if re.match(r"^b\d+_switches$", col)),
    )


def _bus_coupler_switch_columns(df_mbb: pd.DataFrame) -> list[str]:
    """Find all bus coupler columns (of the format b01_switch, b45_switch, etc.)."""
    return list(_classified_columns(tuple(df_mbb.columns))[1])


def _element_switch_columns(df_mbb: pd.DataFrame) -> list[str]:
    """Find all switch columns (of the format b1_switches, b2_switches, etc.)."""
    return list(_classified_columns(tuple(df_mbb.columns))[2])


def busbar_columns(df_mbb: pd.DataFrame) -> list[str]:
    """Find all busbar columns (of the format bus_0, bus_1, etc.)."""
    return list(_classified_columns(tuple(df_mbb.columns))[0])


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

    col_order = busbar_columns(df_mbb) + bus_couplers_cols + other_columns + _element_switch_columns(df_mbb)
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
    df_mbb = net.multi_bb_substation
    return _all_substation_switches_row(
        df_mbb.loc[i_sub], _element_switch_columns(df_mbb), _bus_coupler_switch_columns(df_mbb),
    )


def _all_substation_switches_row(
    sub: pd.Series, element_columns: list[str], coupler_columns: list[str],
) -> list[int]:
    """Every switch of one substation, given its already-fetched table row.

    Split from :func:`get_all_substation_switches` so callers that already hold the row (and
    the classified column names) do not pay for another ``.loc`` row build -- three of them
    used to happen per candidate action.
    """
    ele_switches = sub[element_columns].dropna().sum()
    coupler_switches = sub[coupler_columns].explode().dropna().tolist()
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
    return _is_fully_connected_row(net.multi_bb_substation.loc[i_sub], hexset)


def _is_fully_connected_row(sub: pd.Series, hexset: str) -> bool:
    """Whether a hexset leaves every non-PST element of one substation on the same busbar."""
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
    df_mbb = net.multi_bb_substation
    # One row build and one column classification for the whole call: this runs once per
    # candidate action during action-space generation, and used to fetch the row three times
    # (here, in is_fully_connected and in get_all_substation_switches).
    sub = df_mbb.loc[i_sub]
    coupler_columns = _bus_coupler_switch_columns(df_mbb)
    nbits = len(sub.connected_buses)
    nbusbars = sub.n_busbars_in_substation

    if _is_fully_connected_row(sub, hexset):
        return _all_substation_switches_row(sub, _element_switch_columns(df_mbb), coupler_columns), []

    open_switches = []
    closed_switches = []

    # Do not forget to open the switches between
    # bus 0 and bus 1 (etc.) if substation is not fully connected.
    # "explode" is to accommodate list of busbar couplers in one cell.
    coupler_switches = sub[coupler_columns].explode().dropna().tolist()
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
