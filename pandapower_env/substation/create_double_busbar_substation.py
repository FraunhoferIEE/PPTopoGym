"""
Module for creating pandapower-style double-busbar substations.

The creation starts from a simple pandapower network (e.g. networks where substations are
represented as a single bus). The information about the substations is stored in `net` under
`net.multi_bb_substation`. The necessary extra switches and buses required to emulate the
multi-busbar substation layout are created and added to the pandapower network.
"""

from __future__ import annotations

import logging
from itertools import combinations, repeat

import numpy as np
import pandapower as pp
import pandas as pd
from pandapower import pandapowerNet
from pandapower.create import _get_index_with_check

from pandapower_env.substation.double_busbar_substation import (
    _busbar_columns,
    reorder_substation_df,
)
from pandapower_env.toolbox.pandapower_tools import (
    ELEMENT_TYPES,
    get_element_type_string,
    get_from_bus_str,
    get_to_bus_str,
)
from pandapower_env.toolbox.topology_helpers import (
    find_all_bus_switch_trees,
    find_bus_switch_tree,
    find_bus_switch_trees_from_list,
)

logger = logging.getLogger(__name__)


def n_assignable_elements_in_bus(
    net: pandapowerNet,
    ibus: int,
    *,
    consider_bus_tree: bool = True,
    consider_open_switches: bool = False,
) -> int:
    """
    Determine how many elements can be assigned to a bus.

    In other words, determine how many lines + trafos + loads + gens + sgens are attached
    to the (simple) bus.

    :param net: The pandapower network (a simple network)
    :type net: pandapowerNet
    :param ibus: index of the bus in the network (net.bus Dataframe)
    :type ibus: int
    :param consider_bus_tree: Consider not only a single bus, but also buses attached to that bus via
        bus-to-bus switches.
    :type consider_bus_tree: bool
    :param consider_open_switches: if the switch connecting two buses is open,
        do not consider them part of the same tree if set to True
    :type consider_open_switches: bool
    :return n_total_elements: Number of total elements connected to the bus
    :rtype: int
    """
    buses = [ibus]

    if consider_bus_tree:
        buses = find_bus_switch_tree(net, ibus, consider_open_switches=consider_open_switches)

    n_total_elements = 0

    for el_type in ELEMENT_TYPES:
        to_bus = get_to_bus_str(el_type)
        from_bus = get_from_bus_str(el_type)

        # net.line, net.trafo, net.load, ... etc.
        df_element = getattr(net, el_type)

        # Count the number of elements referencing this bus
        n_elements = np.any(df_element[[to_bus, from_bus]].isin(buses), axis=1).sum()
        n_total_elements += n_elements

    return n_total_elements


def can_convert_to_n_busbar_substation(
    net: pandapowerNet,
    ibus: int,
    n: int = 2,
    *,
    consider_bus_tree: bool = True,
    consider_open_switches: bool = False,
) -> bool:
    """
    Check if the substation can be converted into a double-busbar station.

    To do this, it must contain at least 2n external elements (loads, gens, lines, etc.).
    Otherwise, Elements will be islanded every time.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param ibus: index of the bus in the network (net.bus Dataframe)
    :type ibus: int
    :param n: Number of busbars to consider
    :type n: int
    :param consider_bus_tree: Consider not only a single bus, but also buses attached to that bus via
        bus-to-bus switches.
    :type consider_bus_tree: bool
    :param consider_open_switches: if the switch connecting two buses is open,
        do not consider them part of the same tree if set to True
    :type consider_open_switches: bool
    :return: Whether a bus can be converted into a double-busbar substation
    :rtype: bool
    """
    n_total_elements = n_assignable_elements_in_bus(
        net,
        ibus,
        consider_bus_tree=consider_bus_tree,
        consider_open_switches=consider_open_switches,
    )

    min_n_elements = 2 * n
    if n_total_elements < min_n_elements:
        logger.debug("Bus %s has %s (<%s) elements, not busbarrable.", ibus, n_total_elements, min_n_elements)
        return False
    return True


def _find_bus2bus_switch(net: pandapowerNet, bus0: int, bus1: int) -> int:
    """
    Find the bus-bus switch connecting two buses.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param bus0: index of the "bus 0" busbar in the network (net.bus Dataframe)
    :type bus0: int
    :param bus1: index of the "bus 1" busbar in the network (net.bus Dataframe)
    :type bus1: int
    :return: The index of the existing switch connecting the two buses, corresponding to
        the net.switch Dataframe
    :rtype: int
    """
    sw = net.switch[net.switch["et"] == "b"]
    tmp = []
    tmp += sw[(sw["bus"] == bus0) & (sw["element"] == bus1)].index.tolist()
    tmp += sw[(sw["bus"] == bus1) & (sw["element"] == bus0)].index.tolist()
    return tmp[0]


def copy_bus(net: pandapowerNet, ibus: int, name: str) -> int:
    """
    Create a new bus with parameters identical to the copied bus, and add it to the network.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param ibus: index of the bus in the network (net.bus Dataframe)
    :type ibus: int
    :param name: name of the new bus
    :type name: str
    :return new_index: index of the new bus in the network (net.bus Dataframe)
    :rtype: int
    """
    # The line below creates a dataframe
    tmp_df = net.bus.loc[net.bus.index == ibus].copy()
    new_index = _get_index_with_check(net, "bus", None)
    tmp_df.index = [new_index]
    tmp_df["name"] = name
    net.bus = pd.concat([net.bus, tmp_df])
    # copy the geodata
    if hasattr(net, "bus_geodata"):
        tmp_df = net.bus_geodata.loc[net.bus_geodata.index == ibus].copy()
        tmp_df.index = [new_index]
        net.bus_geodata = pd.concat([net.bus_geodata, tmp_df])
    return new_index


def create_n_busbars(net: pandapowerNet, ibus: int, n: int) -> list[int]:
    """
    Create a list of new n-1 (NOT n) buses, each with the same parameters as the original bus.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param ibus: index of the bus in the network (net.bus Dataframe)
    :type ibus: int
    :param n: Number of new buses to create
    :type n: int
    :return: A list of new bus indices in the network (net.bus Dataframe)
    :rtype: list[int]
    """
    # Create a list of new bus indices (n-1 many)
    return [copy_bus(net, ibus, name=f"bus {i}") for i in range(n - 1)]


def create_n_busbar_switches(
    net: pandapowerNet,
    substation_buses: list[int],
) -> dict[str, int]:
    """
    Create a list of new bus-to-bus switches connecting the new buses.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param substation_buses: A list of new bus indices in the network (net.bus Dataframe)
    :type substation_buses: list[int]
    :return: A list of lists, each containing the indices of the new bus-to-bus switches
        connecting the new buses
    :rtype: list[list[int]]
    """
    # Create a list of new bus-to-bus switches
    new_switches_dict = {
        f"b{i}{j}_switch": pp.create_switch(
            net,
            bus=substation_buses[j],  # Target bus (higher index)
            element=substation_buses[i],  # Source bus (lower index)
            et="b",  # Element type (busbar)
            type="CB",  # Circuit breaker
            name=f"busbar switch, {i} to {j}",
        )
        for i, j in combinations(range(len(substation_buses)), 2)
    }
    return new_switches_dict  # noqa: RET504, makes code more readable


def _create_buses_and_switches_for_substation_elements(
    net: pandapowerNet,
    element_type: str,
    substation_dict: dict,
    buses: list[int],
    *,
    do_pst: bool = False,
) -> dict:
    """
    Create the buses and switches necessary to create the double-busbar substation.

    Returns the details of the new items, in a format ready for adding to the net.multi_bb_substation
    Dataframe.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param element_type: the desired element ('line','trafo','gen','load', etc.)
    :type element_type:
    :param substation_dict: The dictionary containing the details of the new elements
        (in a format compatible to be added to the net.multi_bb_substation)
    :type substation_dict: dict
    :param buses: A list of new bus indices in the network (net.bus Dataframe)
    :type buses: list[int]
    :return: The dictionary containing the details of the new elements (in a format
        compatible to be added to the net.multi_bb_substation)
    :rtype: dict
    """
    et = element_type

    from_bus_col = get_from_bus_str(element_type)
    to_bus_col = get_to_bus_str(element_type)
    bus_col = from_bus_col if (from_bus_col == to_bus_col) else None

    element_type_str = get_element_type_string(element_type)
    if element_type_str is None:  # this should never occur
        msg = f"Element type {element_type} is not supported for creating a substation."
        raise ValueError(msg)
    for j in range(len(buses)):
        substation_dict.setdefault(f"b{j}_switches", [])

    dataframe_keys = [
        "element_type",
        "connected_elements",
        "connected_buses",
        *[f"b{i}_switches" for i in range(len(buses))],
    ]

    for k in dataframe_keys:
        substation_dict[k] = substation_dict.get(k, [])

    # Find the elements that originate from the original bus
    bus0 = buses[0]  # the original bus

    df_element = net[et]
    if "trafo" in et and do_pst:
        df_element = df_element[df_element["tap_changer_type"] == "Ideal"]
    elif "trafo" in et:
        df_element = df_element[df_element["tap_changer_type"] != "Ideal"]

    if bus_col is not None:
        elements = df_element[df_element[bus_col] == bus0]
        # if bus_col == True, the following is not used in this case
        elements["bus_column"] = bus_col
    else:
        # We look at from and to separately to properly consider PSTs,
        # which have both "lv" and "hv" attached to bus0
        elements_from = df_element[df_element[from_bus_col] == bus0]
        elements_from["bus_column"] = from_bus_col
        elements_to = df_element[df_element[to_bus_col] == bus0]
        elements_to["bus_column"] = to_bus_col
        elements = pd.concat([elements_from, elements_to]).sort_index()

    # add element name
    for iel, row in elements.iterrows():  # iel == index of the element
        name = f"new bus between bus {bus0} and line {iel}"
        bus_for_el = copy_bus(net, bus0, name=name)

        # Add a string denoting PST in the case of trafo psts
        element_type_suffix = "<PST>" if do_pst else ""
        substation_dict["element_type"].append(element_type + element_type_suffix)
        substation_dict["connected_elements"].append(iel)
        substation_dict["connected_buses"].append(bus_for_el)

        switch_names = [f"bus {bus} to {element_type} {iel}" for bus in buses]

        bi_switches: list[int] = pp.create_switches(
            net,
            list(buses),
            list(repeat(bus_for_el, len(buses))),
            list(repeat("b", len(buses))),
            type=list(repeat("LBS", len(buses))),
            name=switch_names,
        )
        for i, switch in enumerate(bi_switches):
            substation_dict[f"b{i}_switches"].append(switch)

        # Reset the bus corresponding to the line
        # Be careful: PST trafos have both sides assigned to bus0!
        net[element_type].loc[iel, row.bus_column] = bus_for_el

    return substation_dict


def create_n_busbar_substation(
    net: pandapowerNet,
    ibus: int,  # index-bus
    *,
    n: int = 2,
) -> None:
    """
    Create the double-busbar substation for a bus (ibus) in the network.

    This will add the necessary switches and additional buses to create the structure, and modify the network.
    In general we refer to:
    - substation_buses: the buses in one soon-to-be-substation
    - elements: Everything connected to theses buses (lines, transformers, loads, etc.)

    Workflow to create a multi-busbar substation from one bus:
    1. Copy the original bus n-times -> We get b_0, ..., b_{n-1} as the busbars in the substation
    2. Add switches between all buses b_0, ..., b_{n-1} (bus-bus switches)
    3. Create one bus for each element (line, transformer, etc.) connected to the orig-bus
    4. Create switches between the new buses and the copied busbars b_0, ..., b_{n-1}
    5. Reassign the elements to the new buses


    :param net: The pandapower network
    :type net: pandapowerNet
    :param ibus: index of the "seed" bus in the network (net.bus Dataframe), will become "bus_0"
    :type ibus: int
    :param n: Number of busbars to create (default is 2 -> Downward compatible)
    :type n: int
    :return: None
    """
    max_bus_number = 16
    if n > max_bus_number:
        msg = f"Number of busbars exceeds maximum number of busbars ({max_bus_number})."
        raise ValueError(msg)
    if not can_convert_to_n_busbar_substation(net, ibus, n):
        # throw error, as this is most probable due to a user error
        elements_bus = n_assignable_elements_in_bus(net, ibus)
        msg = f"Number of elements at the bus {ibus} is too low ({elements_bus} < {2*n} needed)."
        raise ValueError(msg)

    if not hasattr(net, "multi_bb_substation"):
        net.multi_bb_substation = pd.DataFrame()
        index = 0
    else:
        # Check against existing buses in substations
        # (to prevent double-creating substations)
        columns = _busbar_columns(net.multi_bb_substation)
        existing_buses = pd.Series(net.multi_bb_substation[columns].to_numpy().ravel()).dropna().tolist()
        existing_buses += net.multi_bb_substation["connected_buses"].explode().tolist()
        if ibus in existing_buses:
            logger.debug("Bus belongs to an existing substation. Not proceeding.")
            return

        # Otherwise, get the new dbb substation index
        index = _get_index_with_check(net, "multi_bb_substation", index=None)
    bus0 = ibus  # rename variable ibus to bus0
    del ibus  # from now on, only bus0 is used for clarification
    substation_buses: list[int] = create_n_busbars(net, bus0, n)  # returns a list of new buses
    # add the bus0
    substation_buses = [bus0, *substation_buses]
    new_switches_dict = create_n_busbar_switches(net, substation_buses)
    sub_dict = {**{f"bus_{i}": substation_buses[i] for i in range(n)} }
    sub_dict.update(new_switches_dict)
    for el_type in ELEMENT_TYPES:
        _create_buses_and_switches_for_substation_elements(net, el_type, sub_dict, substation_buses, do_pst=False)
    # append PSTs at the end without re-ordering all elements
    _create_buses_and_switches_for_substation_elements(net, "trafo", sub_dict, substation_buses, do_pst=True)

    new_sub_entry = pd.Series(sub_dict).to_frame().T
    new_sub_entry = new_sub_entry.rename(index={0: index})
    new_sub_entry["n_busbars_in_substation"] = n

    # Convert some columns to Int64 (with a capital "I", which specifically allows for NA values).
    busbar_cols = [f"bus_{i}" for i in range(n)]
    busbar_switches = [f"b{i}{j}_switch" for i, j in combinations(range(n), 2)]
    cols_nan_int = busbar_cols + busbar_switches
    new_sub_entry[cols_nan_int] = new_sub_entry[cols_nan_int].astype("Int64")

    net.multi_bb_substation = pd.concat([net.multi_bb_substation, new_sub_entry])
    net.multi_bb_substation = reorder_substation_df(net.multi_bb_substation)

    # Possibly to be added in the future:
    # - identical_elements
    # - b0_nExternals
    # - b1_nExternals

    return


def _reassign_elements_to_bus0(
    net: pandapowerNet,
    bus0: int,
    bus_tree: list[int],
) -> None:
    """
    Assign all elements inside a "bus_tree" to bus0.

    This is required for the auto-doublebusbar code to work.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param bus0: index of the "bus 0" busbar in the network (net.bus Dataframe)
    :type bus0: int
    :param bus_tree: a list containing buses connected to bus0 via bus-to-bus switches
    :type bus_tree: list[int]
    """
    for el_type in ELEMENT_TYPES:
        # Reassign "from_bus" or "lv_bus", etc. to bus 0
        from_bus = get_from_bus_str(el_type)
        elements = net[el_type][net[el_type][from_bus].isin(bus_tree[1:])].index.tolist()
        net[el_type].loc[elements, from_bus] = bus0

        # Reassign "to_bus" or "hv_bus", etc. to bus 0
        to_bus = get_to_bus_str(el_type)
        elements = net[el_type][net[el_type][to_bus].isin(bus_tree[1:])].index.tolist()
        net[el_type].loc[elements, to_bus] = bus0

        # Reassign the switches as well (old bus --> bus0)
        switches = (net.switch["et"] == "l") & (net.switch["bus"].isin(bus_tree[1:]))
        if switches.sum():
            net.switch.loc[switches, "bus"] = bus0


def create_all_double_busbar_substations(net: pandapowerNet) -> None:
    """
    Create a double-busbar substation for all (eligible) buses in the network.

    :param net: The pandapower network
    :type net: pandapowerNet
    :return: None
    """
    _create_all_upto_n_busbar_substations(net, n=2)


def _create_all_upto_n_busbar_substations(net: pandapowerNet, n: int = 2) -> None:
    """
    Create all multi-busbar substations with upto n busbars and add them to the net.multi_bb_substation Dataframe.

    :param net: The pandapower network
    :type net: pandapowerNet
    """
    # Prepare the network for all possible substations to be created
    buses: list[int] = _prepare_buses_for_substation(net)

    for ibus in buses:
        # Now create the double-busbar substation dataframe
        n_iterate = n
        for i in range(n_iterate, 1, -1):
            # Create the double-busbar substation
            if not can_convert_to_n_busbar_substation(net, ibus, n=i, consider_bus_tree=True):
                msg = f"Bus {ibus} cannot be converted to a double-busbar substation, not enough connecting lines."
                logger.debug(msg)
                continue
            create_n_busbar_substation(net, ibus, n=i)
            break


def create_multi_bb_substations_from_list(
    net: pandapowerNet,
    bus_multiplicity_tuples: list[tuple[int, int]],
) -> None:
    """
    Create multi-busbar substations from a list of tuples.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param bus_multiplicity_tuples: A list of tuples, each containing the bus index and the number of busbars
        to create for that bus.
    :type bus_multiplicity_tuples: list[tuple[int, int]]
    """
    # Prepare the network for creating the substation (remove redundant buses)
    bus0_buses = [a[0] for a in bus_multiplicity_tuples]
    # The following line will fail if buses specified are on the same bus tree
    bus_trees = find_bus_switch_trees_from_list(net, bus0_buses,
                                                consider_open_switches=True, include_single_buses=True)
    _prepare_buses_for_substation(net, bus_trees=bus_trees, delete_unnecessary_buses=True)

    for ibus, n in bus_multiplicity_tuples:
        if not can_convert_to_n_busbar_substation(net, ibus, n=n, consider_bus_tree=True):
            msg = f"Bus {ibus} cannot be converted to {n}-multi-busbar substation, not enough connecting lines."
            raise RuntimeError(msg)
        create_n_busbar_substation(net, ibus, n=n)


def create_double_busbar_substation(net: pandapowerNet, bus: int) -> None:
    """
    Create a single double-busbar substation at the specified bus.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param bus: The bus where the substation should be created (will become bus0)
    :type bus: int
    """
    # We need to call this one, because it cleans up the unused buses (as opposed to create_n_busbar_substation)
    return create_multi_bb_substations_from_list(net, [(bus, 2)])


def create_3bb_with_pst_substation(net: pandapowerNet, bus: int, pst_config: dict | None = None) -> None:
    """
    Create a single 3bb substation with a PST.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param bus: The bus where the substation should be created (will become bus0)
    :type bus: int
    :param pst_config: Configuration dictionary for the PST trafo
    :type pst_config: dict | None
    """
    if pst_config is None:
        pst_config = {}
    # Set both ends of the pst to bus0. During the creation of the substation, switches will be added.
    pst_config["hv_bus"] = bus
    pst_config["lv_bus"] = bus
    add_pst_to_net(net, pst_config)

    # We need to call this one, because it cleans up the unused buses (as opposed to create_n_busbar_substation)
    return create_multi_bb_substations_from_list(net, [(bus, 3)])


def create_all_dbb_or_3bbwpst_substations(  # renamed from create_dbb_with_3_bb_for_pst_from_list
    net: pandapowerNet,
    buses_for_pst: list[int | np.integer],
    pst_config: dict | None = None,
) -> None:
    """
    Create three-busbar substations with a PST, or double-busbar substations according to user preference.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param buses_for_pst: The buses on which to put a
    :param pst_config: Configuration dictionary for the PST trafo
    :type pst_config: dict | None
    """
    buses: list[int] = _prepare_buses_for_substation(net)

    # Check to make sure that buses for pst are in the "buses" list
    if not all(pst_bus in buses for pst_bus in buses_for_pst):
        msg = "Buses indicated for PSTs are not in the list of eligible busbars (or may have been removed)."
        raise ValueError(msg)

    # add PSTs
    for bus in buses_for_pst:
        # create PST
        # Set both ends of the pst to bus0. During the creation of the substation, switches will be added.
        if pst_config is None:
            pst_config = {}
        pst_config["hv_bus"] = bus
        pst_config["lv_bus"] = bus
        add_pst_to_net(net, pst_config)

    # create substations
    for bus0 in buses:

        do_pst_substation = (bus0 in buses_for_pst)
        nbusbars = 3 if do_pst_substation else 2

        if not can_convert_to_n_busbar_substation(net, bus0, n=nbusbars, consider_bus_tree=True):
            msg = f"Bus {bus0} cannot be converted to a {nbusbars}-busbar substation, not enough elements."
            if do_pst_substation:
                # If we expected to make a 3-bus substation with a pst, then throw an error.
                raise RuntimeError(msg)
            logger.error(msg)
            continue

        create_n_busbar_substation(net, bus0, n=nbusbars)


def add_pst_to_net(net: pandapowerNet, user_pst_config: dict | None = None) -> np.integer:
    """
    Add a PST to the net.

    The user can give a config dict, which is filled with default values.
    :param net: The net to add the PST to.
    :type net: Pandapower net.
    :param user_pst_config: A dict with all specialties of the PST.
    :type user_pst_config: dict
    :return: the created index of the PST.
    """
    # The default PST configuration
    default_pst_config = {
        "name": "Querregelung PST",
        "std_type": None,
        "hv_bus": 1,
        "lv_bus": 1,
        "sn_mva": 500.0,
        "vn_hv_kv": 380.0,
        "vn_lv_kv": 380.0,
        "vk_percent": 12.0,
        "vkr_percent": 0.1,
        "pfe_kw": 0.0,
        "i0_percent": 0.0,
        "shift_degree": 10.0,
        "tap_side": "hv",
        "tap_neutral": 0,
        "tap_min": -30,
        "tap_max": 30,
        "tap_step_percent": np.nan,
        "tap_step_degree": 1.0,
        "tap_pos": 0,
        "tap_changer_type": "Ideal",
        "id_characteristic_table": None,
        "tap_dependency_table": False,
        "parallel": 1,
        "df": 1.0,
        "in_service": True,
    }

    # Start with defaults, then update with user values
    if user_pst_config is None:
        user_pst_config = {}

    pst_config = default_pst_config.copy()  # Don't modify the original
    pst_config.update(user_pst_config)

    # Create the PST
    pst_index = pp.create_transformer_from_parameters(net, **pst_config)
    return pst_index  # noqa: RET504


def _prepare_buses_for_substation(
    net: pandapowerNet,
    bus_trees: list[list[int]] | None = None,
    delete_unnecessary_buses: bool = True,  # noqa: FBT001, FBT002, clarity that something is deleted in this fct
) -> list[int]:
    """
    Check if a bus is in a bus tree.

    This is a helper function to be called, when checking for multi-buses to exist already.

    Bus-Trees have the functionality to detect bus-bus switches.
    These can occur not only in multi-bb-substations, but also for modeling circuit-brakers, etc.
    We clean up these bus-trees to only get the buses useable for modeling multi-busbar substations.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param bus_trees: trees connected via bus-bus switches (these will
        be collapsed and simplified into a single substation)
    :type bus_trees: list[list[int]
    :param delete_unnecessary_buses: If True, drop the unused buses in the tree (if more than 1 bus in a group)
    :type delete_unnecessary_buses: bool
    :return: True if the bus is in a tree, False otherwise
    :rtype: bool
    """
    if bus_trees is None:
        bus_trees = find_all_bus_switch_trees(
            net,
            consider_open_switches=True,
            include_single_buses=True,
        )
    all_buses = []
    for tree in bus_trees:
        if not can_convert_to_n_busbar_substation(net, tree[0], consider_bus_tree=True, n=2):
            # only check for dbb, rest is done in the function.
            logger.debug("Tree %s is not nbb-able.", tree)
            continue

        bus0 = tree[0]

        if len(tree) > 1:
            # Reassign all connected elements to bus 0
            _reassign_elements_to_bus0(net, bus0, tree)

        # Drop the unused buses in the tree (if more than 1 bus in a group)
        n_busbars = 1
        if len(tree) > n_busbars and delete_unnecessary_buses:
            net.bus = net.bus.drop(index=tree[1:])
        all_buses.append(bus0)
    return all_buses
