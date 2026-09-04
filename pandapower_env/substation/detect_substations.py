import itertools

import pandas as pd
from pandapower import pandapowerNet

from pandapower_env.substation.double_busbar_substation import (
    busbar_columns,
    reorder_substation_df,
)
from pandapower_env.toolbox.pandapower_tools import (
    ELEMENT_TYPES,
    get_from_bus_str,
    get_to_bus_str,
)


def get_element_terminals_df(net: pandapowerNet, et: str, *, do_pst: bool=False) -> pd.DataFrame:
    """
    Make a DataFrame with one entry per element terminal (line, trafo, load, sgen...).

    The "type" mimics that of the net.multi_bb_substation DataFrame.
    Columns are labeled to match eventual name in the substation DataFrame
    ["element_type", "connected_buses", "connected_elements" (index)]

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param et: Element type (trafo, load, etc.)
    :type et: str
    :param do_pst: Consider PSTs only (relevant only for trafos), or non-PSTs only
    :type do_pst: bool
    :return: DataFrame with one entry per element terminus
    :rtype: pd.DataFrame
    """
    df_all = pd.DataFrame()

    # Iterate over the set (e.g. "from_bus", "to_bus" for lines, "bus" for loads)
    terminals = {get_from_bus_str(et), get_to_bus_str(et)}
    for ft in terminals:
        df_element = net[et].copy()

        # Mimic what is done in the "create" code
        if "trafo" in et and do_pst:
            df_element = df_element[df_element["tap_changer_type"] == "Ideal"]
        elif "trafo" in et:
            df_element = df_element[df_element["tap_changer_type"] != "Ideal"]

        element_type_suffix = "<PST>" if do_pst else ""
        df_element["element_type"] = et + element_type_suffix
        df_element["connected_buses"] = df_element[ft]
        df_element["connected_elements"] = df_element.index
        df_element = df_element[["element_type", "connected_buses", "connected_elements"]]
        df_all = pd.concat([df_all, df_element])

    return df_all


def get_all_element_terminals_df(net: pandapowerNet) -> pd.DataFrame:
    """
    Get a DataFrame containing a list of all elements, including entries for "from" and "to" individually.

    Columns include ["element_type", "connected_buses", "connected_elements" (index)]

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :return: DataFrame of element terminuses (for further processing)
    :rtype: pd.DataFrame
    """
    df_elements = pd.DataFrame()
    for et in ELEMENT_TYPES:
        df_elements = pd.concat([df_elements, get_element_terminals_df(net, et, do_pst=False)])

    df_psts = get_element_terminals_df(net, "trafo", do_pst=True)

    return pd.concat([df_elements, df_psts])


def add_switch_info_to_elements_df(net: pandapowerNet, df_elements: pd.DataFrame) -> pd.DataFrame:
    """
    From the df_elements above, add connected switches.

    The other side of the switch is a candidate for a busbar.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param df_elements: The DataFrame resulting from get_all_element_terminals_df
    :type df_elements: pd.DataFrame
    :return: an element terminus dataframe with additional switch info attached (>1 switch per bus!)
    :rtype: pd.DataFrame
    """
    b_b_switches = net.switch.loc[net.switch["et"] == "b", ["bus", "element"]]
    b_b_switches = b_b_switches.rename(columns={"bus": "sw_bus"})
    b_b_switches["switches"] = b_b_switches.index

    common_columns = ["element_type", "connected_elements", "connected_buses", "switches"]

    # Using the bus connected to the element, find the corresponding switches (>1 switch per bus!)
    # Match element_buses with "element" in the switch DF
    df_ele_sw_1 = df_elements.merge(b_b_switches, how="inner", left_on="connected_buses", right_on="element")
    df_ele_sw_1 = df_ele_sw_1[[*common_columns, "sw_bus"]]
    df_ele_sw_1 = df_ele_sw_1.rename(columns={"sw_bus": "busbar"})

    # Match element buses with "bus" in the switch DF
    df_ele_sw_2 = df_elements.merge(b_b_switches, how="inner", left_on="connected_buses", right_on="sw_bus")
    df_ele_sw_2 = df_ele_sw_2[[*common_columns, "element"]]
    df_ele_sw_2 = df_ele_sw_2.rename(columns={"element": "busbar"})

    # Concatenate these two DFs to create a detailed switch DF
    # We assume the non-"element" switch must be the busbar
    return pd.concat([df_ele_sw_1, df_ele_sw_2])


def pivot_to_connected_buses(df_elements_plus_switches: pd.DataFrame) -> pd.DataFrame:
    """
    Group by "connected_buses", since busbars will have this in common.

    The multi_bb_substation is essentially finalized here.

    :param df_elements_plus_switches: the DataFrame result of add_switch_info_to_elements_df
    :type df_elements_plus_switches: pd.DataFrame
    :return: the newly constructed multi-busbar DataFrame
    :rtype: pd.DataFrame
    """
    # Group buses by (presumed) busbar. Each individual busbar has a row in the DF below
    df_busbars = df_elements_plus_switches.sort_values(by="connected_buses").groupby("busbar").agg(list)
    df_busbars["connected_buses_str"] = df_busbars["connected_buses"].apply(lambda x: ",".join([str(a) for a in x]))

    # Some preparation for the "pivot"
    df_busbars["idx"] = df_busbars.groupby("connected_buses_str").cumcount()
    df_busbars["busbar"] = df_busbars.index

    # Pivot Dataframe to assemble rows around the connected buses (which can therefore group the busbars)
    # Pivot DataFrame and treat the busbar information
    df_bbcols = df_busbars.pivot_table(index="connected_buses_str", columns="idx", values="busbar")
    df_bbcols.columns = [f"bus_{i}" for i in df_bbcols.columns]

    # Pivot DataFrame and treat the bus-to-element switch information
    df_swcols = df_busbars.pivot_table(index="connected_buses_str", columns="idx", values="switches", aggfunc="sum")
    df_swcols.columns = [f"b{i}_switches" for i in df_swcols.columns]

    # These columns are the same for each busbar (by construction)
    common_columns = ["connected_buses", "element_type", "connected_elements"]
    df_common = df_busbars.groupby("connected_buses_str").agg("first")[common_columns]

    # Join the columns together
    df_final = pd.concat([df_bbcols, df_common, df_swcols], axis=1).reset_index(drop=True)

    busbar_cols = busbar_columns(df_final)

    # Change to "Int64" which allows for NA values
    for col in busbar_cols:
        df_final[col] = df_final[col].astype("Int64")

    df_final["n_busbars_in_substation"] = (~df_final[busbar_cols].isna()).sum(axis=1)

    return df_final


def add_bus_coupler_columns(net: pandapowerNet, df_mbb: pd.DataFrame) -> pd.DataFrame:
    """
    Identify the bus couplers for each substation and add it to the DataFrame.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param df_mbb: The multi-busbar DataFrame formed from pivot_to_connected_buses
    :type df_mbb: pd.DataFrame
    :return: multi-busbar DataFrame with b01_switch (and similar) columns added.
    :rtype: pd.DataFrame
    """
    bus_sw = net.switch[net.switch["et"] == "b"]
    bus_sw["idx"] = bus_sw.index

    for col1, col2 in itertools.combinations(busbar_columns(df_mbb), 2):
        # extract bus integers from "bus_X"
        x = col1.split("_")[1]
        y = col2.split("_")[1]

        # Find e.g. the "b01_switch" and "b10_switch" and take whichever one is not NA
        xy_switch = df_mbb.merge(bus_sw, how="left", left_on=[col1, col2], right_on=["bus", "element"])["idx"]
        yx_switch = df_mbb.merge(bus_sw, how="left", left_on=[col2, col1], right_on=["bus", "element"])["idx"]
        df_mbb[f"b{x}{y}_switch"] = xy_switch.fillna(yx_switch).astype("Int64")

    return df_mbb


def detect_substations(net: pandapowerNet) -> pd.DataFrame:
    """
    Detect substations in a net using the switch configuration, and return a DataFrame.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :return: automatically-detected multi-busbar DataFrame
    :rtype: pd.DataFrame
    """
    df_elements = get_all_element_terminals_df(net)
    df_elements_plus_switches = add_switch_info_to_elements_df(net, df_elements)
    df_final = pivot_to_connected_buses(df_elements_plus_switches)
    df_final = add_bus_coupler_columns(net, df_final)

    return reorder_substation_df(df_final)
