"""
Observation module containing examples of how to use the pandapower net to create an observation.

Observations are expected to be highly customized, so the goal here is to demonstrate without
putting too many requirements on how the final implementation should look.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pandapower import pandapowerNet


def clipped_line_loading_observation_space(
    net: pandapowerNet,
    min_val: float | None = None,
    max_val: float | None = None,
) -> list[float]:
    """
    Calculate the line loading as a list corresponding to the net.list entries.

    This is a simple example meant to be used as a template for other observation space functions.

    :param net:
    :type net: pandapowerNet
    :param min_val:
    :type min_val: float | None
    :param max_val:
    :type max_val: float | None
    :return: clipped list of loading_percent from the pandapower line results of the power flow
    :rtype: list[float]
    """
    return np.clip(net.res_line["loading_percent"], min_val, max_val).tolist()


def line_overloading_number_total(
    net: pandapowerNet,
    threshold: float = 100.0,
) -> float:
    """
    Calculate the percentage of lines whose load is greater than or equal to a threshold.

    :param net: a pandapower net
    :type net: pandapowerNet
    :param threshold: the allowed maximum line loading in percent
    :type threshold: float
    :return: percentage of lines loaded above threshold
    :rtype: float
    """
    return np.sum(net.res_line.loading_percent >= threshold)


def line_overloading_percentage(net: pandapowerNet, threshold: float = 100.0) -> float:
    """
    Calculate the percentage of lines whose load is greater than or equal to a threshold.

    :param net: a pandapower net
    :type net: pandapowerNet
    :param threshold: the allowed maximum line loading in percent
    :type threshold: float
    :return: percentage of lines loaded above threshold
    :rtype: float
    """
    n_connected = np.sum(net.line.in_service)
    n_overloaded = np.sum(net.res_line.loading_percent >= threshold)

    return n_overloaded / n_connected


def line_specific_overloaded(
    net: pandapowerNet,
    threshold: float = 100.0,
) -> list[bool]:
    """
    Give as feedback all lines that are overloaded.

    :param net: a pandapower net
    :type net: pandapowerNet
    :param threshold: the allowed maximum line loading in percent
    :type threshold: float
    :return: percentage of lines loaded above threshold
    :rtype: float
    """
    in_service = net.line.in_service
    overloaded_lines = (net.res_line.loading_percent >= threshold) & in_service

    return overloaded_lines.tolist()


def nminus1_line_overloading_percentage(
    net: pandapowerNet,
    threshold: float = 100.0,
) -> float:
    """
    Calculate the n-1 cases in which the maximum line load is greater than or equal to a threshold.

    :param net: a pandapower net
    :type net: pandapowerNet
    :param threshold: the allowed maximum line loading in percent
    :type threshold: float
    :return: number of n-1 cases with line loading above defined treshold
    :rtype: float
    """
    if "max_loading_percent" not in net.res_line:
        msg = "Column max_loading_percent missing in net.res_line. Please run n-1 calculation analysis before."
        raise ValueError(msg)

    n_connected = np.sum(net.line.in_service)
    n_overloaded = np.sum(net.res_line.max_loading_percent > threshold)

    return n_overloaded / n_connected


def line_overloading_sum(net: pandapowerNet, threshold: float = 100.0) -> float:
    """
    Calculate the amount of lines whose load is greater than or equal to a threshold.

    :param net: a pandapower net
    :type net: pandapowerNet
    :param threshold: the allowed maximum line loading in percent
    :type threshold: float
    :return: number of lines loaded above threshold
    :rtype: float
    """
    return (np.sum(net.res_line.loading_percent >= threshold)).astype(float)


def nminus1_line_overloading_sum(net: pandapowerNet, threshold: float = 100.0) -> float:
    """
    Calculate the amount of lines whose load is greater than or equal to a threshold for N-1.

    :param net: a pandapower net
    :type net: pandapowerNet
    :param threshold: the allowed maximum line loading in percent
    :type threshold: float
    :return: percentage of lines loaded above threshold
    :rtype: float
    """
    if "max_loading_percent" not in net.res_line:
        msg = "Column max_loading_percent missing in net.res_line. Please run n-1 calculation analysis before."
        raise ValueError(msg)
    return (np.sum(net.res_line.max_loading_percent >= threshold)).astype(float)


def nminus1_bus_overloading_percentage(net: pandapowerNet) -> float:
    """
    Calculate the n-1 cases where the min and max bus voltages exceed the defined tresholds.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: number of n-1 cases with bus voltages out of limits
    :rtype: float
    """
    if "max_vm_pu" not in net.res_bus or "min_vm_pu" not in net.res_bus:
        msg = "Columns min_vm_pu & max_vm_pu missing in net.res_bus. Please run n-1 calculation before."
        raise ValueError(msg)

    n_connected = np.sum(net.bus.in_service)
    n_overloaded = np.sum(
        (net.res_bus.max_vm_pu > net.bus.max_vm_pu)
        | (net.res_bus.min_vm_pu < net.bus.min_vm_pu),
    )

    return n_overloaded / n_connected


def line_loading_mean(net: pandapowerNet) -> float:
    """
    Calculate the mean line loading over all lines in a net.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: mean line loading in percent
    :rtype: float
    """
    return np.mean(net.res_line.loading_percent)

def line_loading_var(net: pandapowerNet) -> float:
    """
    Calculate the variance line loading over all lines in a net.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: mean line loading in percent
    :rtype: float
    """
    return np.var(net.res_line.loading_percent)



def nminus1_line_loading_mean(net: pandapowerNet) -> float:
    """
    Calculate the mean line loading over all lines in a net based on N-1 calculation.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: mean line loading in percent
    :rtype: float
    """
    if "max_loading_percent" not in net.res_line:
        msg = "Column max_loading_percent missing in net.res_line. Please run n-1 calculation analysis before."
        raise ValueError(msg)
    return np.mean(net.res_line.max_loading_percent)


def open_busbar_coupler_percentage(net: pandapowerNet) -> float:
    """
    Count the open busbar couplers.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: number of open busbar couplers
    :rtype: float
    """
    bb_coupler_ids = net.multi_bb_substation.b01_switch.tolist()
    bb_couplers = net.switch.loc[bb_coupler_ids]
    return (np.sum(~bb_couplers.closed) / len(bb_coupler_ids)).astype(float)


def open_busbar_coupler_total(net: pandapowerNet) -> float:
    """
    Count the open busbar couplers.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: number of open busbar couplers
    :rtype: float
    """
    bb_coupler_ids = net.multi_bb_substation.b01_switch.tolist()
    bb_couplers = net.switch.loc[bb_coupler_ids]
    return (np.sum(~bb_couplers.closed)).astype(float)


def all_open_busbar_couplers(net: pandapowerNet) -> list[bool]:
    """Retrieve all open busbar couplers."""
    bb_coupler_ids = net.multi_bb_substation.b01_switch.tolist()
    bb_couplers = net.switch.loc[bb_coupler_ids]
    return (~bb_couplers.closed).tolist()


def line_power_losses_sum(net: pandapowerNet) -> float:
    """
    Calculate the sum of active power losses of all lines.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: sum of active power losses of all lines in MW
    :rtype: float
    """
    return np.sum(net.res_line.pl_mw)


def nminus1_line_loading_max(net: pandapowerNet) -> float:
    """
    Calculate the maximum loading over all lines for n-1.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: maximum line loading overall lines
    :rtype: float
    """
    if "max_loading_percent" not in net.res_line:
        msg = "Column max_loading_percent missing in net.res_line. Please run n-1 calculation analysis before."
        raise ValueError(msg)
    return np.nanmax(net.res_line.max_loading_percent)




def line_loading_max(net: pandapowerNet) -> float:
    """
    Calculate the maximum loading over all lines for n-0.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: maximum line loading overall lines
    :rtype: float
    """
    return net.res_line.loading_percent.max()


def overload_energy(net: pandapowerNet) -> float:
    """
    Calculate the total overload energy.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: total overload energy in MW
    :rtype: float
    """
    # First, we calculate the apparent power S flowing into and out of the lines
    net.res_line["S_from_mvar"] = np.sqrt(
        (net.res_line["p_from_mw"]) ** 2 + (net.res_line["q_from_mvar"]) ** 2,
    )
    net.res_line["S_to_mvar"] = np.sqrt(
        (net.res_line["p_to_mw"]) ** 2 + (net.res_line["q_to_mvar"]) ** 2,
    )

    # For each line, we take the maximum of S_from_mvar and S_to_mvar as the critical apparent power
    net.res_line["S_crit_mvar"] = net.res_line[["S_from_mvar", "S_to_mvar"]].max(axis=1)

    # We then calculate how much apparent power each line can handle
    for line in range(len(net.line)):
        max_vn_kv = np.max(
            [
                net.bus.loc[net.line.loc[line, "from_bus"], "vn_kv"],
                net.bus.loc[net.line.loc[line, "to_bus"], "vn_kv"],
            ],
        )
        net.res_line.loc[line, "S_100%_mvar"] = (
            (net.line.loc[line, "max_i_ka"]) * max_vn_kv * np.sqrt(3)
        )

    # The overload energy for each line results from substracting S_100%_mvar from S_crit_mvar
    net.res_line["overload_energy_mw"] = (
        net.res_line["S_crit_mvar"] - net.res_line["S_100%_mvar"]
    )

    # Finally, we sum up the values of each line to get the total overload energy for the grid
    return np.sum(net.res_line.overload_energy_mw)


def trafo_power_losses_sum(net: pandapowerNet) -> float:
    """
    Calculate the sum of active power losses of all transformers.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: sum of active power losses of all transformers in MW
    :rtype: float
    """
    if net.res_trafo.empty:
        return 0.0
    return float(np.nan_to_num(np.sum(net.res_trafo.pl_mw), nan=0.0))


def system_losses_sum(net: pandapowerNet) -> float:
    """
    Calculate total system losses (lines + transformers).

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: total system losses in MW
    :rtype: float
    """
    line_loss = line_power_losses_sum(net) if not net.res_line.empty else 0.0
    trafo_loss = trafo_power_losses_sum(net)
    return float(np.nan_to_num(line_loss + trafo_loss, nan=0.0))


def total_load_p(net: pandapowerNet) -> float:
    """
    Calculate total active power consumption from load results.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: total active load power in MW, 0.0 if no results
    :rtype: float
    """
    if not hasattr(net, "res_load") or net.res_load.empty:
        return 0.0
    return float(np.nan_to_num(net.res_load["p_mw"].sum(), nan=0.0))


def total_gen_p(net: pandapowerNet) -> float:
    """
    Calculate total active power generation from gen and sgen results.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: total active generation power in MW, 0.0 if no results
    :rtype: float
    """
    total = 0.0
    if hasattr(net, "res_gen") and not net.res_gen.empty:
        total += float(np.nan_to_num(net.res_gen["p_mw"].sum(), nan=0.0))
    if hasattr(net, "res_sgen") and not net.res_sgen.empty:
        total += float(np.nan_to_num(net.res_sgen["p_mw"].sum(), nan=0.0))
    return total


def has_gen_results(net: pandapowerNet) -> bool:
    """
    Check if generation results exist in the network.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: True if gen or sgen results exist
    :rtype: bool
    """
    res_gen_exists = hasattr(net, "res_gen") and not net.res_gen.empty
    res_sgen_exists = hasattr(net, "res_sgen") and not net.res_sgen.empty
    return res_gen_exists or res_sgen_exists


def has_load_results(net: pandapowerNet) -> bool:
    """
    Check if load results exist in the network.

    :param net: a pandapower net
    :type net: pandapowerNet
    :return: True if load results exist
    :rtype: bool
    """
    return hasattr(net, "res_load") and not net.res_load.empty
