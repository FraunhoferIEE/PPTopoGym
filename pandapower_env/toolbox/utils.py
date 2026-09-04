from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np
import pandapower as pp
import pandapower.contingency

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pandapower import pandapowerNet
logger = logging.getLogger(__name__)

def run_nminus1_powerflow(
    net: pandapowerNet,
    pf_type: str = "ac",
    use_ls2g: str | bool = "auto",
) -> None:
    """
    Run n-1 powerflows.

    :param net: a pandapower network
    :type net: pandapowerNet
    :param pf_type: the powerflow type, either 'ac' oder 'dc'
    :type pf_type: str
    :param use_ls2g: Whether lightsim2grid should be used as backend or not.
    :type use_ls2g: str | bool
    """
    if pf_type not in {"ac", "dc"}:
        msg = "pf_type must be 'ac' or 'dc'."
        raise ValueError(msg)

    if use_ls2g != "auto" and not isinstance(use_ls2g, bool):
        msg = "use_ls2g must be bool or 'auto'."
        raise ValueError(msg)

    nminus1_cases = {
        "line": {"index": net.line.index.to_numpy()},
        "trafo": {"index": net.trafo.index.to_numpy()},
        "trafo3w": {"index": net.trafo3w.index.to_numpy()},
    }

    if pf_type == "ac":
        pp.contingency.run_contingency(
            net=net,
            nminus1_cases=nminus1_cases,
            contingency_evaluation_function=pp.runpp,
            lightsim2grid=use_ls2g,
        )
        if (
            use_ls2g is not False
            and net._options["lightsim2grid"] is False  # noqa: SLF001
        ):  # we intend to look into what PP is doing
            logger.warning(
                "Warning: use_ls2g is %s, but lightsim2grid can't be used as backend.",
                use_ls2g,
            )

    if pf_type == "dc":
        pp.contingency.run_contingency(
            net=net,
            nminus1_cases=nminus1_cases,
            contingency_evaluation_function=pp.rundcpp,
            lightsim2grid=use_ls2g,
        )
        if use_ls2g:
            logger.warning(
                "Warning: use_ls2g is %s, but lightsim2grid can't be used for DC powerflow.",
                use_ls2g,
            )


def run_powerflow(
    net: pandapowerNet,
    pf_type: str = "ac",
    use_ls2g: str | bool = "auto",
) -> None:
    """Run the powerflow.

    :param net: a pandapower network
    :type net: pandapowerNet
    :param pf_type: the powerflow type, either 'ac' oder 'dc'
    :type pf_type: str
    :param use_ls2g: Whether lightsim2grid should be used as backend or not.
    :type use_ls2g: str | bool
    """
    if pf_type not in {"ac", "dc"}:
        msg = "pf_type must be 'ac' or 'dc'."
        raise ValueError(msg)

    if use_ls2g != "auto" and not isinstance(use_ls2g, bool):
        msg = "use_ls2g must be bool or 'auto'."
        raise ValueError(msg)

    if pf_type == "ac":
        pp.runpp(net, lightsim2grid=use_ls2g)
        if (
            use_ls2g is not False
            and net._options["lightsim2grid"] is False  # noqa: SLF001
        ):  # we intend to look into what PP is doing
            logger.warning(
                "Warning: use_ls2g is %s, but lightsim2grid can't be used as backend.",
                use_ls2g,
            )

    if pf_type == "dc":
        pp.rundcpp(net, lightsim2grid=use_ls2g)
        if use_ls2g:
            logger.warning(
                "Warning: use_ls2g is %s, but lightsim2grid can't be used for DC powerflow.",
                use_ls2g,
            )




def create_adjacency_matrix(net: pandapowerNet) -> np.ndarray:
    """
    Create and return the adjacency matrix (edge list) for a pandapower network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network used to generate the adjacency matrix.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (n_lines + n_trafo, 2) where each row is
        [from_bus, to_bus].
    """
    # Extract the edges from lines (from_bus and to_bus)
    line_edges = net.line[["from_bus", "to_bus"]].to_numpy(dtype=np.integer)



    # Extract the edges from transformers (high-voltage to low-voltage bus)
    trafo_edges = net.trafo[["hv_bus", "lv_bus"]].to_numpy(dtype=np.integer)

    # Concatenate the line and transformer edges into a single NumPy array.
    edges_to_concat: Sequence[np.ndarray] = [line_edges, trafo_edges]

    edges = np.concatenate(edges_to_concat, axis=0)
    if hasattr(net, "multi_bb_substation"):
        # Build mapping only for bus ids that appear in edges
        bus_map = _generate_origin_bus_map(net, edges)
        edges = _map_edge_buses(edges, bus_map)
    return edges

def _generate_origin_bus_map(net: pandapowerNet, bus_ids: np.ndarray) -> dict[int, int]:
    """
    Precompute a mapping from bus IDs to their original bus IDs for a given set of buses.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network used to resolve original bus IDs.
    bus_ids : np.ndarray
        Iterable of bus IDs to resolve. Only unique values are processed.

    Returns
    -------
    dict[int, int]
        A dictionary mapping each input bus ID to its resolved original bus ID,
        as determined by `_get_original_bus`.
    """
    bus_ids_set = np.unique(bus_ids)
    return {b: _get_original_bus(net, b) for b in bus_ids_set}

def _map_edge_buses(edges: np.ndarray, bus_map: dict[int, int]) -> np.ndarray:
    """
    Apply a {bus_id -> original_bus_id} mapping to an edge list.

    Parameters
    ----------
    edges : np.ndarray
        A NumPy array of shape (n_edges, 2) where each row is [from_bus, to_bus].
    bus_map : dict[int, int]
        Mapping from bus IDs to original bus IDs to be applied to both columns.

    Returns
    -------
    np.ndarray
        A NumPy array of shape (n_edges, 2) with bus IDs replaced according to `bus_map`.
    """
    flat = edges.reshape(-1)
    mapped = np.fromiter((bus_map.get(int(b), int(b)) for b in flat), dtype=np.integer)
    return mapped.reshape(edges.shape)

def _get_original_bus(net: pandapowerNet, bus_id: int) -> int: #noqa: C901
    if not hasattr(net, "multi_bb_substation"):
        return bus_id
    all_connected_buses = [int(bus) for sublist in net.multi_bb_substation["connected_buses"] for bus in sublist]
    if bus_id in all_connected_buses:
        for _ , row in net.multi_bb_substation.iterrows():
            for ind_bus, bus in enumerate(row["connected_buses"]):
                if bus == bus_id:
                    for count_switch in range(16):
                        str_sw = f"b{count_switch}_switches"
                        if str_sw not in net.multi_bb_substation.columns:
                            break
                        switch_id = row[str_sw][ind_bus]
                        if net.switch.loc[switch_id, "closed"]:
                            return row[f"bus_{count_switch}"]
    for _ , row in net.multi_bb_substation.iterrows():
        if bus_id == row["bus_1"]:
            return row["bus_0"]
    return bus_id
