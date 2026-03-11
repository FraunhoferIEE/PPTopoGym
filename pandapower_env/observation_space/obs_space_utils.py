
import numpy as np
import pandapower as pp


def aggregate_loads_to_buses(net: pp.pandapowerNet) -> np.ndarray:
    """Aggregate load values so that each bus receives its corresponding total load.

    Returns
    -------
    bus_loads_full (np.ndarray): An array of aggregated active load (p_mw)
                                     with length equal to the number of buses.
    """
    n_buses = len(net["bus"])
    bus_loads_full = np.zeros(n_buses, dtype=np.float32)
    # Iterate over each load and add its active power value to the corresponding bus.
    for _, load in net.load.iterrows():
        bus_idx = int(load["bus"])  # Ensure the index is an integer.
        bus_loads_full[bus_idx] += load["p_mw"]
    return bus_loads_full

def aggregate_generators_to_buses(net: pp.pandapowerNet) -> np.ndarray:
    """Aggregate generator values so that each bus receives its corresponding total generation.

    Returns
    -------
    bus_generators_full (np.ndarray): An array of aggregated generator active power (p_mw)
                                          with length equal to the number of buses.
    """
    n_buses = len(net["bus"])
    bus_generators_full = np.zeros(n_buses, dtype=np.float32)
    # Iterate over each generator and add its active power value to the corresponding bus.
    for _, gen in net.gen.iterrows():
        bus_idx = int(gen["bus"])
        bus_generators_full[bus_idx] += gen["p_mw"]
    return bus_generators_full
