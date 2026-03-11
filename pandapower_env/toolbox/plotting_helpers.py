
import json
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pandapower import pandapowerNet
from pandapower.plotting.plotting_toolbox import _get_coords_from_geojson


def create_geo_column_from_xy(df_xy: pd.DataFrame) -> pd.Series:
    """
    Create a geojson string column from x and y values.

    :param df_xy: A DataFrame containing "x" and "y" columns for the bus locations.
    :type df_xy: pd.DataFrame
    :return: Bus geodata (geojson format) of geodata
    :rtype: pd.Series | None
    """
    return df_xy.apply(lambda row: json.dumps({"coordinates": [row["x"], row["y"]], "type": "Point"}), axis=1)


def create_xy_columns_from_geo(geo_column: pd.Series) -> pd.DataFrame:
    """
    Create "x" and "y" coordinates from a geojson-formatted series.

    Suitable for applying to the net.bus.geo column ("point" type).

    :param geo_column: Bus geodata (geojson format)
    :type geo_column: pd.Series | None
    :return: A DataFrame containing "x" and "y" columns for the bus locations.
    :rtype: pd.DataFrame
    """
    df_xy = pd.DataFrame()
    df_xy[["x", "y"]] = geo_column.apply(lambda g: _get_coords_from_geojson(g)).apply(pd.Series)
    return df_xy


def create_bus2bus_geodata(
        net: pandapowerNet,
        from_bus_indices: list | pd.Series,
        to_bus_indices: list | pd.Series,
        bus_geodata: pd.Series | None = None) -> pd.DataFrame:
    """
    Create a DataFrame containing useful reference values for plotting lines using bus_geodata.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param from_bus_indices: list or Series containing the "from" bus indices to use
    :type from_bus_indices: list | pd.Series
    :param to_bus_indices: list or Series containing the "from" bus indices to use
    :type to_bus_indices: list | pd.Series
    :param bus_geodata: Bus geodata (geojson format) to use, if not the default one.
    :type bus_geodata: pd.Series | None
    :return: A DataFrame containing useful positional information for e.g. plotting.
    :rtype: pd.DataFrame
    """
    # Need to move coordinates
    if bus_geodata is None:
        bus_geodata = net.bus.geo

    bus_geo_xy = create_xy_columns_from_geo(bus_geodata)

    # Add the from_bus geodata
    df_geodata_from = bus_geo_xy.loc[from_bus_indices, ["x", "y"]]
    df_geodata_from["x"] = df_geodata_from["x"].astype(float)  # need retype apparently
    df_geodata_from["y"] = df_geodata_from["y"].astype(float)  # need retype apparently
    df_geodata_from = df_geodata_from.reset_index()
    df_geodata_from = df_geodata_from.rename(columns={"index": "from_bus", "x": "from_x", "y": "from_y"})

    # Add the from_bus geodata
    df_geodata_to = bus_geo_xy.loc[to_bus_indices, ["x", "y"]]
    df_geodata_to["x"] = df_geodata_to["x"].astype(float)  # need retype apparently
    df_geodata_to["y"] = df_geodata_to["y"].astype(float)  # need retype apparently
    df_geodata_to = df_geodata_to.reset_index()
    df_geodata_to = df_geodata_to.rename(columns={"index": "to_bus", "x": "to_x", "y": "to_y"})

    df_b2b = pd.concat([df_geodata_from, df_geodata_to], axis=1)

    # Align the indices with the original line indices given, if applicable.
    if hasattr(from_bus_indices, "index"):
        df_b2b.index = from_bus_indices.index

    # Once the line and trafo info has been collected, derive some quantities
    df_b2b["x_len"] = df_b2b["to_x"]-df_b2b["from_x"]
    df_b2b["y_len"] = df_b2b["to_y"]-df_b2b["from_y"]
    # Trick to avoid undefined arctan below
    df_b2b.loc[df_b2b["x_len"] == 0, "x_len"] = 0.001*df_b2b["y_len"]
    df_b2b["distance"] = np.sqrt(np.square(df_b2b["x_len"])+np.square(df_b2b["y_len"]))
    df_b2b["theta_rad"] = np.arctan2(df_b2b["to_y"] - df_b2b["from_y"], df_b2b["to_x"]-df_b2b["from_x"])
    df_b2b["theta_deg"] = np.rad2deg(df_b2b["theta_rad"])
    df_b2b["sin_theta"] = df_b2b["y_len"]/df_b2b["distance"]
    df_b2b["cos_theta"] = df_b2b["x_len"]/df_b2b["distance"]
    # This theta will produce text that is always right-side up.
    df_b2b["theta_deg_text"] = np.rad2deg(np.arctan(df_b2b["y_len"]/df_b2b["x_len"]))
    df_b2b["x_midpoint"] = (df_b2b["to_x"]+df_b2b["from_x"])/2
    df_b2b["y_midpoint"] = (df_b2b["to_y"]+df_b2b["from_y"])/2

    return df_b2b


def calculate_externals_locations(net: pandapowerNet, offset: float) -> None:
    """
    Figure out where to locate externals in a pandapower network, for plotting.

    Call this function **before** creating substations or changing the topology of the grid.

    The code will create columns in e.g. net.load called "x" and "y" for the location of the element.
    It will create columns "bus_x" and "bus_y" for the corresponding bus (in order to draw a connecting line).
    It will create a column "plot_angle" indicating the angle between the two.

    :param net: The pandapower network (containing net.multi_bb_substation)
    :type net: pandapowerNet
    :param offset: Distance to offset the external items (loads, etc.) from the buses.
    :type offset: float
    """
    """
    This code needs to be cleaned up significantly, but it needs to be
    committed now to have the first working version.
    """
    net.sgen[["bus_x", "bus_y", "plot_angle"]] = 0., 0., 0.
    net.gen[["bus_x", "bus_y", "plot_angle"]] = 0., 0., 0.
    net.load[["bus_x", "bus_y", "plot_angle"]] = 0., 0., 0.

    line_b2b = create_bus2bus_geodata(net, net.line["from_bus"], net.line["to_bus"])
    trafo_b2b = create_bus2bus_geodata(net, net.trafo["lv_bus"], net.trafo["hv_bus"])
    all_b2b = pd.concat([line_b2b, trafo_b2b])

    # Recreate the old "x", "y" columns for calculation
    bus_geodata = create_xy_columns_from_geo(net.bus.geo)

    for bus in net.bus.index.tolist():

        lines_tmp_df = all_b2b[all_b2b[["from_bus", "to_bus"]].isin([bus]).any(axis=1)]

        lines_tmp_df["bus_angles"] = np.where(lines_tmp_df["from_bus"] == bus, lines_tmp_df["theta_rad"],
                                              np.pi + lines_tmp_df["theta_rad"])
        lines_tmp_df["bus_angles"] = lines_tmp_df["bus_angles"] % (2 * np.pi)
        lines_tmp_df["bus_angles_deg"] = np.rad2deg(lines_tmp_df["bus_angles"])
        lines_tmp_df = lines_tmp_df.sort_values(by="bus_angles")

        if not len(lines_tmp_df):
            continue

        existing_angles = lines_tmp_df["bus_angles"].tolist()

        # Calculate the angle differences (gaps) between consecutive points
        gaps = np.diff(np.concatenate((existing_angles, [existing_angles[0] + 2 * np.pi])))
        max_gap_index = np.argmax(gaps)

        # Determine how many externals we have
        n_externals = [(net.sgen["bus"] == bus).sum(),
                       (net.gen["bus"] == bus).sum(),
                       (net.load["bus"] == bus).sum()]
        n_all = sum(n_externals)
        new_angles = [existing_angles[max_gap_index] + (n + 1) * gaps[max_gap_index] / (n_all + 1) for n in
                      range(n_all)]

        i_angle = 0

        for i, el in enumerate(["sgen", "gen", "load"]):
            selection = net[el]["bus"] == bus
            plot_angles = new_angles[i_angle:i_angle + n_externals[i]]
            net[el].loc[selection, "plot_angle"] = plot_angles
            net[el].loc[selection, "bus_x"] = bus_geodata.loc[bus, "x"]
            net[el].loc[selection, "bus_y"] = bus_geodata.loc[bus, "y"]
            net[el].loc[selection, "x"] = bus_geodata.loc[bus, "x"] + offset * np.cos(plot_angles)
            net[el].loc[selection, "y"] = bus_geodata.loc[bus, "y"] + offset * np.sin(plot_angles)
            i_angle = n_externals[i]


def create_periphery_geodata(
        net: pandapowerNet,
        etype: str,
        bus_indices: list[int] | None = None,
        bus_geodata: pd.Series | None = None) -> pd.DataFrame:
    """
    Create a geodata Dataframe containing "x" and "y" values for the element, and "xbus" and "ybus" values for the bus.

    This is for the purpose of plotting lines connecting the element marker to the bus/substation.

    :param net: The pandapower network (containing net.multi_bb_substation)
    :type net: pandapowerNet
    :param etype: element type ("load", "sgen", "gen", etc.)
    :type etype: str
    :param bus_indices: Which buses to consider. If None, then all buses in the "bus" column of the element are used
    :type bus_indices: list[int] | None
    :param bus_geodata: Bus geodata (geojson format) to use, if not the default one.
    :type bus_geodata: pd.Series | None
    """
    if bus_geodata is None:
        bus_geodata = net.bus.geo

    bus_geo_xy = create_xy_columns_from_geo(bus_geodata)

    if bus_indices is None:
        bus_indices = net[etype]["bus"].tolist()

    df_element = net[etype]

    df_periphery_geo = df_element[df_element["bus"].isin(bus_indices)]
    df_periphery_geo = df_periphery_geo[["bus", "x", "y"]]

    df_bus_geo = bus_geo_xy.loc[df_periphery_geo["bus"].tolist(), ["x", "y"]]
    df_bus_geo.index = df_periphery_geo.index

    df_periphery_geo[["xbus", "ybus"]] = df_bus_geo

    return df_periphery_geo


def plot_peripherals(
        net: pandapowerNet,
        ax: Axes,
        etype: str,
        s: float = 100,
        bus_geodata: pd.Series | None = None) -> None:
    """
    Plot peripherals (loads, gens, etc.) for a particular element type.

    :param net: The pandapower network (containing net.multi_bb_substation)
    :type net: pandapowerNet
    :param ax: Matplotlib axis instance
    :type ax: Matplotlib.axes.Axes
    :param etype: element type ("load", "gen", etc.)
    :type etype: str
    :param s: size
    :type s: float
    :param bus_geodata: Bus geodata (geojson format) to use, if not the default one.
    :type bus_geodata: pd.Series | None
    """
    # Marker types / edge colors
    marker = {"gen": "o", "load": "^"}.get(etype)
    edgecolor = {"gen": "cornflowerblue", "load": "orange"}.get(etype)

    # Create and plot the lines connecting the periphery elements to the corresponding bus
    df_periphery = create_periphery_geodata(net, etype=etype, bus_geodata=bus_geodata)
    for row in df_periphery.itertuples():
        ax.plot([row.x, row.xbus], [row.y, row.ybus], marker="", color=edgecolor)

    # Plot the marker corresponding to the element
    ax.scatter(net[etype]["x"], net[etype]["y"],
               edgecolor=edgecolor, color="white", s=s, marker=marker, label=etype, zorder=100000)

    # If it is a generator type, add a tilde to make it look nice
    if etype == "gen":
        ax.scatter(net[etype]["x"], net[etype]["y"],
                   edgecolor=edgecolor, color=edgecolor, s=s*0.5, marker=r"$\sim$", label="gen", zorder=100000)


def plot_all_peripherals(
        net: pandapowerNet,
        ax: Axes,
        bus_geodata: pd.Series | None = None) -> None:
    """
    Plot the peripherals in the network ("gen", "load").

    :param net: The pandapower network (containing net.multi_bb_substation)
    :type net: pandapowerNet
    :param ax: Matplotlib axis instance
    :type ax: Matplotlib.axes.Axes
    :param bus_geodata: Bus geodata (geojson format) to use, if not the default one.
    :type bus_geodata: pd.Series | None
    """
    plot_peripherals(net, ax, "gen", s=150, bus_geodata=bus_geodata)
    plot_peripherals(net, ax, "load", bus_geodata=bus_geodata)


def label_buses(
        net: pandapowerNet,
        ax: Axes,
        bus_indices: list[int] | None = None,
        labels: list[str] | None = None,
        bus_geodata: pd.Series | None = None) -> None:
    """
    Label the buses.

    :param net: The pandapower network (containing net.multi_bb_substation)
    :type net: pandapowerNet
    :param ax: Matplotlib axis instance
    :type ax: Matplotlib.axes.Axes
    :param bus_indices: The bus indices to label
    :type bus_indices: list[int] | None
    :param labels: line labels (if None, bus indices are used)
    :type labels: list[int] | None
    :param bus_geodata: Bus geodata (geojson format) to use, if not the default one.
    :type bus_geodata: pd.Series | None
    """
    if bus_indices is None:
        bus_indices = net.bus.index.tolist()

    if labels is None:
        labels = [str(bus) for bus in bus_indices]

    if bus_geodata is None:
        bus_geodata = net.bus.geo

    bus_geo_xy = create_xy_columns_from_geo(bus_geodata)

    for b, label in zip(bus_indices, labels):
        ax.text(bus_geo_xy.loc[b, "x"], bus_geo_xy.loc[b, "y"], label, zorder=10000,
                ha="center", va="center")


def label_lines(
        net: pandapowerNet,
        ax: Axes,
        line_indices: list[int] | None = None,
        labels: list[str] | None = None,
        bus_geodata: pd.Series | None = None) -> None:
    """
    Label the lines.

    :param net: The pandapower network (containing net.multi_bb_substation)
    :type net: pandapowerNet
    :param ax: Matplotlib axis instance
    :type ax: Matplotlib.axes.Axes
    :param line_indices: line indices to label
    :type line_indices: list[int] | None
    :param labels: line labels (if None, line indices are used)
    :type labels: list[int] | None
    :param bus_geodata: Bus geodata (geojson format) to use, if not the default one.
    :type bus_geodata: pd.Series | None
    """
    if line_indices is None:
        line_indices = net.line.index.tolist()

    if labels is None:
        labels = [str(ln) for ln in line_indices]

    if bus_geodata is None:
        bus_geodata = net.bus.geo

    line_b2b = create_bus2bus_geodata(net,
                                      net.line.loc[line_indices, "from_bus"],
                                      net.line.loc[line_indices, "to_bus"],
                                      bus_geodata=bus_geodata)

    for ln, label, angle in zip(line_indices, labels, line_b2b["theta_deg_text"]):
        ax.text(line_b2b.loc[ln, "x_midpoint"], line_b2b.loc[ln, "y_midpoint"], label, zorder=10000,
                ha="center", va="center", rotation=angle)


def label_substations(
        net: pandapowerNet,
        ax: Axes,
        sub_indices: list[int] | None = None,
        labels: list[str] | None = None,
        bus_geodata: pd.Series | None = None) -> None:
    """
    Label the substations.

    :param net: The pandapower network (containing net.multi_bb_substation)
    :type net: pandapowerNet
    :param ax: Matplotlib axis instance
    :type ax: Matplotlib.axes.Axes
    :param sub_indices: The substation indices to label
    :type sub_indices: list[int] | None
    :param labels: line labels (if None, substation indices are used)
    :type labels: list[int] | None
    :param bus_geodata: Bus geodata (geojson format) to use, if not the default one.
    :type bus_geodata: pd.Series | None
    """
    if sub_indices is None:
        sub_indices = net.multi_bb_substation.index.tolist()

    if labels is None:
        labels = [str(i_sub) for i_sub in sub_indices]

    if bus_geodata is None:
        bus_geodata = net.bus.geo

    bus_geo_xy = create_xy_columns_from_geo(bus_geodata)

    kwargs: dict[str, Any] = {"zorder": 10000, "va": "center", "ha": "center", "fontweight": "bold"}

    for i_sub, label in zip(sub_indices, labels):
        bus0 = net.multi_bb_substation.loc[i_sub, "bus_0"]
        bus1 = net.multi_bb_substation.loc[i_sub, "bus_1"]
        b0_loc = bus_geo_xy.loc[bus0, ["x", "y"]].to_numpy().tolist()
        b1_loc = bus_geo_xy.loc[bus1, ["x", "y"]].to_numpy().tolist()

        if b0_loc == b1_loc:
            ax.text(b0_loc[0], b0_loc[1], label, **kwargs)

        else:
            ax.text(b0_loc[0], b0_loc[1], f"{label}.{0}", **kwargs)
            ax.text(b1_loc[0], b1_loc[1], f"{label}.{1}", **kwargs)
