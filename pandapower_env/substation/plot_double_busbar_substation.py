
from __future__ import annotations

import numpy as np
import pandapower as pp
import pandas as pd
from pandapower import pandapowerNet
from pandapower.plotting.plotting_toolbox import coords_from_node_geodata
from shapely.geometry import LineString

from pandapower_env.substation.double_busbar_substation import (
    is_fully_connected,
    select_element_indices,
)
from pandapower_env.substation.substation_bitsets import hexset_to_closed_switch_list
from pandapower_env.toolbox.plotting_helpers import (
    create_bus2bus_geodata,
    create_geo_column_from_xy,
    create_xy_columns_from_geo,
)


def create_double_busbar_plotting_net(net: pandapowerNet, isub: int) -> pandapowerNet:
    """
    Create a new pandapower network that can be used to visualize a *single double-busbar substation*.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param isub: index of the substation in the net.multi_bb_substation Dataframe
    :type isub: int
    :return: new pandapowerNet for plotting the structure of the selected substation.
    :rtype: pandapowerNet
    """
    #
    # This function creates a new network that can be used
    # to visualize a *single double-busbar substation*.
    #
    if not hasattr(net, "multi_bb_substation"):
        msg = "Network does not have a multi_bb_substation Dataframe."
        raise RuntimeError(msg)

    sub = net.multi_bb_substation.loc[isub]

    the_geodata = []
    _linetype = "NAYY 4x50 SE"

    intermediate_buses_geodata_x = [0] * len(sub.connected_buses)
    intermediate_buses_geodata_y = np.linspace(-1, 1, len(sub.connected_buses))

    tmpnet = pp.create_empty_network()

    bus0 = pp.create_bus(tmpnet, vn_kv=20., name=f"busbar 0 (bus {sub.bus_0})")
    the_geodata.append([-1.4, -0.8, None])
    bus1 = pp.create_bus(tmpnet, vn_kv=20., name=f"busbar 1 (bus {sub.bus_1})")
    the_geodata.append([-1.0, +0.8, None])

    pp.create_line(tmpnet, length_km=0.1, from_bus=bus0, to_bus=bus1, std_type=_linetype, name="b0-b1 switch")

    # Create the intermediate buses and the lines
    for i, bus in enumerate(sub.connected_buses):
        name = f"substation bus {bus} to {sub.element_type[i]} {sub.connected_elements[i]}"
        tmpbus = pp.create_bus(tmpnet, vn_kv=20., name=name)
        the_geodata.append([intermediate_buses_geodata_x[i], intermediate_buses_geodata_y[i], None])
        pp.create_line(tmpnet, length_km=0.1, from_bus=bus0, to_bus=tmpbus, std_type=_linetype, name="switch")
        pp.create_line(tmpnet, length_km=0.1, from_bus=bus1, to_bus=tmpbus, std_type=_linetype, name="switch")

    df_geodata = pd.DataFrame(the_geodata)
    df_geodata = df_geodata.rename(columns={0: "x", 1: "y", 2: "coords"})

    tmpnet.bus_geodata = df_geodata
    tmpnet.bus.geo = create_geo_column_from_xy(df_geodata)
    return tmpnet


def create_double_busbar_plotting_not_abbc(net: pandapowerNet, isub: int) -> pandapowerNet:
    """
    Create a new pandapower network that can be used to visualize a *single double-busbar substation*.

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param isub: index of the substation in the net.multi_bb_substation Dataframe
    :type isub: int
    :return: new pandapowerNet for plotting the structure of the selected substation.
    :rtype: pandapowerNet
    """
    #
    # This function creates a new network that can be used
    # to visualize a *single double-busbar substation*.
    #
    if not hasattr(net, "multi_bb_substation"):
        msg = "Network does not have a multi_bb_substation Dataframe."
        raise RuntimeError(msg)

    sub = net.multi_bb_substation.loc[isub]

    the_geodata = []
    _linetype = "NAYY 4x50 SE"

    intermediate_buses_geodata_x = [0] * len(sub.connected_buses)
    intermediate_buses_geodata_y = np.linspace(-1, 1, len(sub.connected_buses))

    tmpnet = pp.create_empty_network()

    bus0 = pp.create_bus(tmpnet, vn_kv=20.0, name=f"busbar 0 (bus {sub.bus_0})")
    the_geodata.append([-1.4, -0.8, None])
    bus1 = pp.create_bus(tmpnet, vn_kv=20.0, name=f"busbar 1 (bus {sub.bus_1})")
    the_geodata.append([-1.0, +0.8, None])

    # Create the intermediate buses and the lines
    # Create the intermediate buses and the lines
    for i, bus in enumerate(sub.connected_buses):
        name = (
            f"substation bus {bus} to {sub.element_type[i]} {sub.connected_elements[i]}"
        )
        tmpbus = pp.create_bus(tmpnet, vn_kv=20.0, name=name)
        the_geodata.append(
            [intermediate_buses_geodata_x[i], intermediate_buses_geodata_y[i], None],
        )

        # Assign alternating lines to busbars
        if i % 2 == 0:
            pp.create_line(
                tmpnet,
                length_km=0.1,
                from_bus=bus0,
                to_bus=tmpbus,
                std_type=_linetype,
                name="switch",
            )
        else:
            pp.create_line(
                tmpnet,
                length_km=0.1,
                from_bus=bus1,
                to_bus=tmpbus,
                std_type=_linetype,
                name="switch",
            )

    df_geodata = pd.DataFrame(the_geodata)
    df_geodata = df_geodata.rename(columns={0: "x", 1: "y", 2: "coords"})

    tmpnet.bus_geodata = df_geodata
    return tmpnet


def _incoming_linestring_stats(linestring_series: pd.Series) -> tuple[int, float]:
    """
    From a series of type LineString, find how many intersect, and the total line length.

    :param linestring_series: A pd.Series containing LineStrings.
    :type linestring_series: pd.Series of LineString
    :return: Number of intersections, and the total length of lines
    :rtype: [int, float]
    """
    total_len = 0
    n_intersections = 0
    for i, line1 in enumerate(linestring_series):
        total_len += LineString(line1).length
        for j, line2 in enumerate(linestring_series):
            if i >= j:
                continue
            if not line1.intersection(line2).is_empty:
                n_intersections += 1

    return n_intersections, total_len


def separate_substation_buses_visually(
        net: pandapowerNet,
        action_dict: dict,
        r_split_bus: float = 1.,
) -> tuple[pd.Series, pd.Series]:
    """
    Create a new bus_geodata that visually separates topologically disconnected parts of substations.

    This function does not modify the network - it only returns a new, temporary bus_geodata!

    :param net: The pandapower net (must have net.multi_bb_substation)
    :type net: pandapowerNet
    :param action_dict: Action dictionary containing keys-value pairs "substations": [list] and "states": [list].
    :type action_dict: dict
    :param r_split_bus: Radius of separation for the topologically split buses.
    :type r_split_bus: float
    :return bus_geo, line_geos: bus_geodata and line_geodata for representing the separated substations.
    :rtype: tuple[pd.Series, pd.Series]
    """
    return_bus_geodata = create_xy_columns_from_geo(net.bus.geo)

    for i_sub, state in zip(action_dict["substations"], action_dict["states"]):

        sub = net.multi_bb_substation.loc[i_sub]
        nbits = len(sub.connected_buses)

        if is_fully_connected(net, i_sub, state):
            continue

        # For now, we have only implemented up to 2 busbars in the plotting function.
        max_busbars_implemented = 2
        if sub.n_busbars_in_substation > max_busbars_implemented:
            continue

        b0_bitset = hexset_to_closed_switch_list(state, busbar=0, nbits=nbits, nbusbars=2)
        b1_bitset = hexset_to_closed_switch_list(state, busbar=1, nbits=nbits, nbusbars=2)

        b0_buses = [sub.connected_buses[i] for i in range(nbits) if b0_bitset[i]]
        b1_buses = [sub.connected_buses[i] for i in range(nbits) if b1_bitset[i]]

        buses_busbar0 = [sub.bus_0, *b0_buses]
        buses_busbar1 = [sub.bus_1, *b1_buses]

        all_buses = buses_busbar0 + buses_busbar1

        the_lines = select_element_indices(sub, "line")
        # Includes buses on the other side of the lines
        # Format is [l1_from,l1_to, l2_from,l2_to, ....]
        participating_buses = net.line.loc[the_lines, ["from_bus", "to_bus"]].to_numpy().flatten().tolist()

        the_trafos = select_element_indices(sub, "trafo")
        participating_buses_trafo = net.trafo.loc[the_trafos, ["lv_bus", "hv_bus"]].to_numpy().flatten().tolist()

        # Test different offset angles to find a good location.
        # In other words, separate the two buses and rotate them until we find a configuration
        # where not so many incoming lines are intersecting. The tie-breaker is to minimize the
        # sum of the distances of incoming lines.
        offset_df = pd.DataFrame(index=list(range(16)))
        offset_df["dist"] = 0.0
        offset_df["n_intersections"] = 0
        offset_df["angle"] = offset_df.index * 2 * np.pi / len(offset_df)

        # Currently this only works for two-busbar configurations.
        n_splits = 2

        # Testing different offsets
        for i_offset in offset_df.index:

            # Relevant buses include the other side of incoming lines / trafos
            relevant_buses = list(set(all_buses + participating_buses + participating_buses_trafo))
            tmp_bus_geodata = create_xy_columns_from_geo(net.bus.geo.loc[relevant_buses])

            for j, buses in enumerate([buses_busbar0, buses_busbar1]):
                r = r_split_bus
                angle = j * (2 * np.pi) / n_splits + offset_df.loc[i_offset, "angle"]
                tmp_bus_geodata.loc[buses, "x"] = tmp_bus_geodata.loc[buses, "x"] + r * np.cos(angle)
                tmp_bus_geodata.loc[buses, "y"] = tmp_bus_geodata.loc[buses, "y"] + r * np.sin(angle)

            df_line_geo = create_bus2bus_geodata(net,
                                                 net.line.loc[the_lines, "from_bus"],
                                                 net.line.loc[the_lines, "to_bus"],
                                                 bus_geodata=create_geo_column_from_xy(tmp_bus_geodata))
            df_line_geo["LineString"] = df_line_geo.apply(lambda a: LineString([[a["from_x"], a["from_y"]],
                                                                                [a["to_x"], a["to_y"]]]), axis=1)

            n_intersections, total_len = _incoming_linestring_stats(df_line_geo["LineString"])
            offset_df.loc[i_offset, "dist"] = total_len
            offset_df.loc[i_offset, "n_intersections"] = n_intersections

        # Pick the angle that led to the fewest intersections, with line distance the tie-breaker
        offset_angle = offset_df.sort_values(by=["n_intersections", "dist"]).iloc[0]["angle"]

        # Finally, use the best angle offset in the actual bus_geodata
        for j, buses in enumerate([buses_busbar0, buses_busbar1]):
            r = r_split_bus
            angle = j * (2 * np.pi) / n_splits + offset_angle
            return_bus_geodata.loc[buses, "x"] = return_bus_geodata.loc[buses, "x"] + r * np.cos(angle)
            return_bus_geodata.loc[buses, "y"] = return_bus_geodata.loc[buses, "y"] + r * np.sin(angle)

    bus_geo = create_geo_column_from_xy(return_bus_geodata)
    line_geos, line_index_successful = coords_from_node_geodata(net.line.index,
                                                                net.line["from_bus"],
                                                                net.line["to_bus"],
                                                                table_name="line",
                                                                node_geodata=bus_geo)

    if not (net.line.index == line_index_successful).all():
        msg = "Missing geodata caused separate_substation_buses_visually to fail."
        raise RuntimeError(msg)

    line_geos = pd.Series(line_geos, index=line_index_successful)

    return bus_geo, line_geos
