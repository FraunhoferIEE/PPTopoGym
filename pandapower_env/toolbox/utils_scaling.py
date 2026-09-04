from __future__ import annotations

import logging
import typing

import numpy as np
import pandapower as pp
import pandapower.contingency
import pandas as pd

from pandapower_env.toolbox.utils import run_powerflow
from pandapower_env.toolbox.utils_profiles import get_first_sb_profiles, get_orig_profiles

logger = logging.getLogger(__name__)


def find_max_timestep(  # noqa: PLR0913
    df_profiles_load_p: pd.DataFrame,
    df_profiles_load_q: pd.DataFrame,
    df_profiles_sgen_p: pd.DataFrame,
    df_profiles_gen_p: pd.DataFrame,
    id_start: int = 0,
    id_end: int = 35136,  # 1 year with 366 days in 15min steps
) -> int:
    """
    Find the timestep with max. line loadings in ABBC topology.

    Parameters
    ----------
    df_profiles_load_p
    df_profiles_load_q
    df_profiles_sgen_p
    df_profiles_gen_p
    id_start
    id_end
    """
    all_profiles = pd.concat(
        [df_profiles_load_p, df_profiles_load_q, df_profiles_sgen_p, df_profiles_gen_p],
        axis=1,
    )
    all_profiles_sum = all_profiles[list(all_profiles.columns)].sum(axis=1)
    time_window = all_profiles_sum[id_start:id_end]
    return time_window.idxmax()

# ------------- Scaling with recursive function + helpers
def load_profile_timestep_into_net(net: pp.pandapowerNet, profiles: dict, index: int = 0) -> None:
    """
    Load profiles for a given timestep into the net.load, etc. Dataframes.

    :param index: The index of the desired timestep, in the timeseries Dataframe.
    :type index: int
    """
    # Replace the load p and q with the values stored in the profiles_load dataframe
    if len(net.load):
        net.load["p_mw"] = profiles["load_p"].loc[index].T.to_numpy()
        net.load["q_mvar"] = profiles["load_q"].loc[index].T.to_numpy()

    if len(net.sgen):
        net.sgen["p_mw"] = profiles["sgen_p"].loc[index].T.to_numpy()
        net.sgen["q_mvar"] = profiles["sgen_q"].loc[index].T.to_numpy()

    if len(net.gen):
        net.gen["p_mw"] = profiles["gen_p"].loc[index].T.to_numpy()
        net.gen["vm_pu"] = profiles["gen_vm"].loc[index].T.to_numpy()

def run_pf(net: pp.pandapowerNet) -> bool:
    """Run an AC power flow on ``net``, reporting convergence as a bool.

    Uses :func:`pandapower_env.toolbox.utils.run_powerflow` rather than calling
    ``pp.runpp`` directly. Roughly two thirds of a ``pp.runpp`` call is pandapower
    re-parsing its options, which is pure overhead when the same net is solved over and
    over -- as ``find_scaling_recursive`` does, once per recursion. ``run_powerflow``
    parses the options once, stores them on the net and then reuses them, which is
    ~10x faster across a scaling search while producing bit-identical results (it
    re-derives the ppc from the live net tables on every call, so the scaled ``p_mw``
    values are always picked up).

    :param net: The pandapower network to solve, mutated in place with the results.
    :return: ``True`` if the power flow converged, ``False`` if it did not.
    """
    try:
        run_powerflow(net)
    except pp.LoadflowNotConverged:
        return False
    return True


def target_loading(net: pp.pandapowerNet, loading_percent: int = 100) -> int:
    """Return the number of overloaded lines."""
    if not run_pf(net):
        return -10
    return sum(net.res_line["loading_percent"] > loading_percent)



def ensure_slack_gen(net: pp.pandapowerNet) -> None:
    """
    Ensure the first generator is a slack generator.

    Slack generators make a converging load flow more probable.

    Parameters
    ----------
    net: pandapower net
    """
    if not net.gen["slack"].any():
        net.gen.loc[0, "slack"] = True


def readjust_gen_values_for_convergence(net: pp.pandapowerNet) -> None:
    """
    Readjust values for powerflow convergence.

    Workflow:
    1. Balance load and generated power (active power)
        ensure loads == gens, with little difference
    2. Lower / enlarge the generated power accordingly.

    Parameters
    ----------
    net

    """
    sum_loads = net.load["p_mw"].sum()
    sum_gens = net.gen["p_mw"].sum() + (net.sgen["p_mw"].sum() if len(net.sgen) > 0 else 0)
    if sum_gens <= sum_loads * 1.04 and sum_gens >= sum_loads * 0.99:
        return

    scale_factor = (sum_loads / sum_gens) * 1.0

    net.gen["p_mw"] *= scale_factor
    net.gen.scenario_scaling *= scale_factor
    if len(net.sgen) > 0:
        net.sgen["p_mw"] *= scale_factor
        net.sgen.scenario_scaling *= scale_factor

def readjust_load_values_for_convergence(net: pp.pandapowerNet) -> None:
    """
    Readjust values for powerflow convergence.

    Workflow:
    1. Balance load and generated power (active power)
        ensure loads == gens, with little difference
    2. Lower / enlarge the generated power accordingly.

    Parameters
    ----------
    net

    """
    sum_loads = net.load["p_mw"].sum()
    sum_gens = net.gen["p_mw"].sum() + (net.sgen["p_mw"].sum() if len(net.sgen) > 0 else 0)
    if sum_gens <= sum_loads * 1.04 and sum_gens >= sum_loads * 0.99:
        return
    scale_factor = (sum_gens / sum_loads) * 1.0

    net.load["p_mw"] *= scale_factor
    #also save in scaling
    net.load.scenario_scaling *= scale_factor





def ensure_no_zero_values(net: pp.pandapowerNet) -> None:
    """
    Ensure, that no (s)gen / load has 0 production/consumption.

    Replace any 0 production/consumption with 1.
    This enables to distribute power production/consumption throughout all elements.
    ! Only use this for scaling, not for any real profiles one wants to adjust.

    Parameters
    ----------
    net
    """
    for elem in ("load", "gen", "sgen"):
        net[elem].loc[net[elem]["p_mw"] == 0, "p_mw"] = 1
    net.load.loc[net.load["q_mvar"] == 0, "q_mvar"] = 1


def redistribute_throughout_elements(net: pp.pandapowerNet) -> None:
    """
    Redistribute p_mw within each element type: load, gen, sgen.

    1. Compute difference between largest and 2nd-largest p_mw.
    2. Distribute this difference equally to the lower half of elements (sorted by p_mw).

    Parameters
    ----------
    net
    """
    element_types = ["load", "gen", "sgen"]

    for elem in element_types:
        df_elem = getattr(net, elem)
        if df_elem.empty:
            continue
            # Sort by p_mw ascending

        p_mw_values = df_elem["p_mw"].to_numpy()
        if len(p_mw_values) < 2:  # noqa: PLR2004
            continue  # Not enough elements to redistribute

        sorted_indices = np.argsort(p_mw_values)
        largest = p_mw_values[sorted_indices[-1]]
        mean = p_mw_values.mean()

        # Difference between largest and 2nd-largest
        diff = (largest - mean)*0.25 # the p_mw value we distribute
        diff_scale = 1 - (diff / largest if largest != 0 else 0)
        net[elem]["scenario_scaling"] *= diff_scale

        # Determine lower quarter of elements
        n = len(sorted_indices)
        lower_quarter_idx = df_elem.index[sorted_indices[:n//4]]
        lower_quarter_sum = p_mw_values[sorted_indices[:n//4]].sum()
        lower_scale = 1 + (diff / lower_quarter_sum if lower_quarter_sum != 0 else 0)
        # Apply scaling to lower quarter
        net[elem].loc[lower_quarter_idx, "scenario_scaling"] *= lower_scale

def adjust_values_w_scaling(
    net: pp.pandapowerNet,
    orig_profiles: dict[str, pd.DataFrame],
    run_powerflow: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """
    Adjust the scaling factors for the network components.

    1. take original values
    2. scale them with the column "scenario scaling"
    4. run pf (unless ``run_powerflow`` is False)

    Parameters
    ----------
    net
    orig_profiles: The profiles obtained initially.
    run_powerflow: If False, only set the scaled values without running a power flow.
        Callers that run their own power flow afterwards can skip the redundant one.
    """
    load_profile_timestep_into_net(net, orig_profiles)
    for key in ("load", "gen", "sgen"):
        net[key]["p_mw"] = net[key]["p_mw"].to_numpy() * net[key]["scenario_scaling"].to_numpy()
    if run_powerflow:
        run_pf(net) # local powerflow method, calling pp.runpp()

def scale_scenario_scaling(net: pp.pandapowerNet, value: float) -> None:
    """
    Scale the scaling factors for the network components.

    Parameters
    ----------
    net
    value
    """
    for key in ("load", "gen", "sgen"):
        net[key]["scenario_scaling"] *= value


def find_scaling_recursive(net: pp.pandapowerNet,  # noqa: C901, PLR0913
                           init_scaling: int = 1,
                           orig_profiles: None | dict = None,
                           max_percent: int = 90,
                           overloaded_lines: int = 2,
                           scale_gen: bool = True ) -> bool | typing.Callable:  # noqa: FBT001, FBT002
    """
    Call this function, until enough overloaded lines are found.

    ! This can lead to recursionError, so this function has to be adjusted.

    Parameters
    ----------
    net
    init_scaling: Hyperparamter, where to start scaling
    orig_profiles: orig profiles, to load and apply scaling to.
    max_percent: Minimal percentage, the lines should be scaled to.
    overloaded_lines: Number of lines which should be overloaded due to scaling.
    scale_gen: Boolean, whether generator should be equalized in energy production. Else, loads are re-adjusted.
    """
    if orig_profiles is None: # should happen, but in case, and to showcase the workflow
        logger.warning("No profiles given; falling back to the first Simbench profiles.")
        ensure_slack_gen(net)
        if not hasattr(net, "profiles"):
            get_first_sb_profiles(net, 2)
            for key, df in net.profiles.items():
                net.profiles[key] = df.replace(0, 1.0)
        orig_profiles = get_orig_profiles(net)
    elems = ("load", "gen", "sgen")
    for key in elems:
        if not hasattr(net[key], "scenario_scaling"):
            net[key]["scenario_scaling"] = init_scaling
    # Recursive part of the code -- exactly ONE power flow per call.
    #
    # Only ``scenario_scaling`` (and the ``scale_gen`` toggle) persists between calls;
    # p_mw and the power-flow results are always recomputed from
    # ``orig_profiles x scenario_scaling`` at the top of each call. The previous version
    # ran 3-4 power flows per recursion (inside the leading and trailing
    # ``adjust_values_w_scaling`` calls, and in the ">" branch) whose results were
    # immediately discarded -- only the single run_pf below actually decides
    # ``n_overloaded_lines``. Applying the scaling without those throwaway power flows
    # leaves the scenario_scaling trajectory (and therefore the final net) unchanged.
    adjust_values_w_scaling(net, orig_profiles, run_powerflow=False)
    if scale_gen:
        readjust_gen_values_for_convergence(net)
    else:
        readjust_load_values_for_convergence(net)
    converged = run_pf(net)
    n_overloaded_lines = (net.res_line["loading_percent"] > max_percent).sum()
    if not converged:
        redistribute_throughout_elements(net)
        scale_scenario_scaling(net, 0.9)
    elif n_overloaded_lines == overloaded_lines:
        return True
    elif n_overloaded_lines > overloaded_lines:
        redistribute_throughout_elements(net)
        scale_scenario_scaling(net, 0.95)
    elif n_overloaded_lines < overloaded_lines:
        scale_scenario_scaling(net, 1.2)
    return find_scaling_recursive(net,
                                  init_scaling=init_scaling,
                                  orig_profiles=orig_profiles,
                                  max_percent=max_percent,
                                  overloaded_lines=overloaded_lines,
                                  scale_gen = not scale_gen,
                                  )


def find_scaling_iterative(net: pp.pandapowerNet,
                           orig_profiles: dict,
                           max_percent: int = 90,
                           overloaded_lines: int = 2) -> dict[str, pd.Series] | None:
    """
    Scale the net iteratively, until exactly ``overloaded_lines`` lines are overloaded.

    Iterative counterpart of :func:`find_scaling_recursive`, which avoids its recursion limit.
    Each iteration applies the current ``scenario_scaling`` to the net, runs one power flow and
    then scales up or down depending on how many lines exceed ``max_percent`` loading.

    Parameters
    ----------
    net: The pandapower net to scale (mutated in place).
    orig_profiles: Original profiles, to load and apply the scaling to.
    max_percent: Loading percentage above which a line counts as overloaded.
    overloaded_lines: Number of lines which should be overloaded due to scaling.

    Returns
    -------
        Dict of the final scaled active powers (``load_p``, ``gen_p``, ``sgen_p``),
        or ``None`` if no scaling produced the requested number of overloaded lines.
    """
    # 1. Initialize scaling if not present
    for key in ("load", "gen", "sgen"):
        if "scenario_scaling" not in net[key].columns:
            net[key]["scenario_scaling"] = 1.0

    max_iterations = 50
    for i in range(max_iterations):
        # 2. Apply current scaling to the net
        adjust_values_w_scaling(net, orig_profiles)

        # 3. Check convergence and loading
        converged = run_pf(net)
        if not converged:
            # If it fails, scale down and redistribute to try and get a valid PF
            scale_scenario_scaling(net, 0.9)
            redistribute_throughout_elements(net)
            continue

        n_overloaded = (net.res_line["loading_percent"] > max_percent).sum()

        # 4. Success Condition
        if n_overloaded == overloaded_lines:
            logger.info("Found scaling with %s overloaded lines at iteration %s", overloaded_lines, i)
            # Return a dictionary of the FINAL scaled values
            return {
                "load_p": net.load.p_mw.copy(),
                "gen_p": net.gen.p_mw.copy(),
                "sgen_p": net.sgen.p_mw.copy(),
            }

        # 5. Adjustment Logic
        if n_overloaded < overloaded_lines:
            scale_scenario_scaling(net, 1.1) # Aggressive increase
        else:
            scale_scenario_scaling(net, 0.95) # Slight decrease

    return None # Or last best guess
