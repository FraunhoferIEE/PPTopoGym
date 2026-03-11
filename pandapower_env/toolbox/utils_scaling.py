from __future__ import annotations

import contextlib
import copy
import logging
import typing

import pandapower as pp
import pandapower.contingency
import pandas as pd

from pandapower_env.toolbox.utils import run_nminus1_powerflow
from pandapower_env.toolbox.utils_profiles import get_first_sb_profiles, get_orig_profiles


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
    try:
        pp.runpp(net)
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
    scale_factor = (sum_loads / sum_gens) * 1.0

    if sum_gens > (sum_loads*1.04) or (sum_gens < sum_loads*0.99):
        net.gen["p_mw"] *= scale_factor
        net.sgen["p_mw"] *= scale_factor
        #also save in scaling
        net.gen.scenario_scaling *= scale_factor
        net.sgen.scenario_scaling *= scale_factor

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
        df_sorted = df_elem.sort_values("p_mw").copy()

        # Difference between largest and 2nd-largest
        largest = df_sorted["p_mw"].iloc[-1]
        df_sorted.index[-1]
        mean = df_sorted["p_mw"].mean()
        diff = (largest - mean)*0.25 # the p_mw value we distribute
        diff_scale = diff / largest if largest != 0 else 0
        net[elem]["scenario_scaling"] *= diff_scale

        # Determine lower quarter of elements
        n = len(df_sorted)
        lower_quarter_idx = df_sorted.index[:n//4]
        lower_quarter_sum = df_sorted["p_mw"].iloc[:n//4].sum()
        lower_scale = diff / lower_quarter_sum if lower_quarter_sum != 0 else 0

        # Apply scaling to lower quarter
        net[elem].loc[lower_quarter_idx, "scenario_scaling"] *= lower_scale

def adjust_values_w_scaling(net: pp.pandapowerNet, orig_profiles: dict[str, pd.DataFrame]) -> None:
    """
    Adjust the scaling factors for the network components.

    1. take original values
    2. scale them with the column "scenario scaling"
    4. run pf

    Parameters
    ----------
    net
    orig_profiles: The profiles obtained initially.
    """
    load_profile_timestep_into_net(net, orig_profiles)
    for key in ("load", "gen", "sgen"):
        net[key]["p_mw"] *= net[key]["scenario_scaling"]
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


def find_scaling_recursive(net: pp.pandapowerNet,
                           init_scaling: int = 1,
                           orig_profiles: None | dict = None,
                           max_percent: int = 90,
                           overloaded_lines: int = 2 ) -> bool | typing.Callable:
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
    """
    if orig_profiles is None: # should happen, but in case, and to showcase the workflow
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
    # recursive part of the code
    load_profile_timestep_into_net(net, orig_profiles)
    adjust_values_w_scaling(net, orig_profiles)
    readjust_gen_values_for_convergence(net)
    run_pf(net)
    n_overloaded_lines = (net.res_line["loading_percent"] > max_percent).sum()
    if not run_pf(net):
        scale_scenario_scaling(net, 0.8)
    elif n_overloaded_lines == overloaded_lines:
        return True
    elif n_overloaded_lines > overloaded_lines:
        redistribute_throughout_elements(net)
        load_profile_timestep_into_net(net, orig_profiles)
        adjust_values_w_scaling(net, orig_profiles)
        scale_scenario_scaling(net, 0.95)
    elif n_overloaded_lines < overloaded_lines:
        scale_scenario_scaling(net, 1.1)
    adjust_values_w_scaling(net, orig_profiles)
    run_pf(net)
    return find_scaling_recursive(net,
                                  init_scaling=init_scaling,
                                  orig_profiles=orig_profiles,
                                  max_percent=max_percent,
                                  overloaded_lines=overloaded_lines,
                                  )




# ----------- Scaling with binary search # helper functions
def _run_pf_with_scaling(
    net: pp.pandapowerNet,
    scaling: float,
    min_percent_overload: float,
) -> int | None:
    """
    Set the same scaling for loads, generators, and static generators.

    Run the power flow, and return the number of lines with loading_percent >= min_percent_overload.

    :param net: The pandapower network.
    :type net: pp.pandapowerNet
    :param scaling: The scaling factor to apply.
    :type scaling: float
    :param min_percent_overload: The threshold loading percentage to consider a line overloaded.
    :type min_percent_overload: float
    :return: The number of overloaded lines if converged, or None if the power flow fails to converge.
    :rtype: int | None
    """
    net.load["scaling"] = scaling
    net.gen["scaling"] = scaling
    net.sgen["scaling"] = scaling
    if "res_line" not in net or net.res_line["loading_percent"].isna().any():
        pp.reset_results(net)
        return None
    try:
        pp.runpp(net)
    except pp.LoadflowNotConverged:
        pp.reset_results(net)
        return None
    return (net.res_line["loading_percent"] >= min_percent_overload).sum()


def _binary_search_threshold(  # noqa: PLR0913
    net: pp.pandapowerNet,
    lower: float,
    upper: float,
    target: int,
    min_percent_overload: float,
    max_search_iter: int = 50,
    tol: float = 1e-3,
) -> float:
    """
    Binary search to find the minimal scaling 's' within [lower, upper] such that f(s) >= target.

    :param net: The pandapower network.
    :type net: pp.pandapowerNet
    :param lower: Lower bound for the search interval.
    :type lower: float
    :param upper: Upper bound for the search interval.
    :type upper: float
    :param target: The target number of overloaded lines.
    :type target: int
    :param min_percent_overload: The threshold loading percentage.
    :type min_percent_overload: float
    :param max_search_iter: Maximum number of iterations for binary search.
    :type max_search_iter: int
    :param tol: Convergence tolerance.
    :type tol: float
    :return: The minimal scaling factor found.
    :rtype: float
    """
    for _ in range(max_search_iter):
        mid: float = (lower + upper) / 2.0
        val: int | None = _run_pf_with_scaling(net, mid, min_percent_overload)
        # Treat non-convergence as if overload is too high.
        if (
            val is None and upper - lower < tol * 10
        ):  # Early stopping if search space is too small
            return lower
        if val is None or val >= target:
            upper = mid
        else:
            lower = mid
        if upper - lower < tol:
            return mid
    return upper


def _lower_scaling_that_net_converges(net: pp.pandapowerNet, scaling: float) -> float:
    """
    Determine the lowest scaling factor such that the power flow converges.

    :param net: The pandapower network.
    :type net: pp.pandapowerNet
    :param scaling: The initial scaling factor to apply.
    :type scaling: float
    :return: The lowest scaling factor such that the power flow converges.
    :rtype: float
    """
    decay = 0.98
    minimal_scaling = 1e-5
    while scaling > minimal_scaling:
        # run N-1 powerflow and look that it converges
        with contextlib.suppress(pp.LoadflowNotConverged):
            run_nminus1_powerflow(net)
        if _run_pf_with_scaling(net, scaling, 100) is not None:
            return scaling
        scaling *= decay
        decay = (decay + 1) / 2
    msg = "Scaling factor too low, network may not be solvable."
    raise ValueError(msg)


def _bracket_search_bounds(
    net: pp.pandapowerNet,
    min_percent_overload: float,
    min_overloaded_lines: int,
) -> tuple[float, float]:
    """
    Determine search bounds for the scaling factor based on a base-case evaluation.

    :param net: The pandapower network.
    :type net: pp.pandapowerNet
    :param min_percent_overload: The threshold loading percentage.
    :type min_percent_overload: float
    :param min_overloaded_lines: Minimum required number of overloaded lines.
    :type min_overloaded_lines: int
    :return: A tuple (s_low, s_high) representing the lower and upper bounds for the scaling factor.
    :rtype: tuple[float, float]
    """
    min_s_low: float = 0.001
    max_s_high: float = 1e4
    s_low: float = 0.001
    s_high: float = 10.0
    base_val: int | None = _run_pf_with_scaling(net, 1.0, min_percent_overload)
    # check if the power flow converged
    if base_val is None:
        while s_low > min_s_low:
            s_low *= 0.5
            val: int | None = _run_pf_with_scaling(net, s_low, min_percent_overload)
            if val is not None:
                break
        else:
            msg = f"Scaling factor {round(s_low, 5)} exceeded search limits while decreasing."
            logging.debug(msg)
            return min_s_low, max_s_high

    elif base_val < min_overloaded_lines:
        while s_high < max_s_high:  # Limit the search
            s_high *= 2.0
            val = _run_pf_with_scaling(net, s_high, min_percent_overload)
            if val is not None and val >= min_overloaded_lines:
                return s_low, s_high
        msg = f"Scaling factor {s_high // 1} exceeded search limits while increasing."
        logging.debug(msg)
        return (min_s_low, max_s_high)  # s_low has already been successfully determined
    return s_low, s_high


def find_scaling_binarysearch(  # noqa: C901, PLR0915
    net: pp.pandapowerNet,
    min_percent_overload: float,
    min_overloaded_lines: int,
    max_search_iter: int = 200,
    tol: float = 1e-2,
) -> tuple[float, int, float]:
    """
    Find a scaling factor for loads, generators, and static generators such that line overloadings appear.

    :param net: The pandapower network.
    :type net: pp.pandapowerNet
    :param min_percent_overload: The threshold loading percentage for a line to be considered overloaded.
    :type min_percent_overload: float
    :param min_overloaded_lines: Minimum required number of overloaded lines.
    :type min_overloaded_lines: int
    :param max_search_iter: Maximum iterations for binary search.
    :type max_search_iter: int
    :param tol: Convergence tolerance.
    :type tol: float
    :return: A tuple (net, scaling_found, final_overloaded) where scaling_found is the scalig factor.
    :rtype: tuple[pp.pandapowerNet, float, int]
    """
    s_low, s_high = _bracket_search_bounds(
        net,
        min_percent_overload,
        min_overloaded_lines,
    )
    while _run_pf_with_scaling(net, s_high, min_percent_overload) is None:
        # If s_high doesn't work, reduce it.
        s_high *= 0.9
        if s_high < s_low:  # Ensure s_high stays above s_low
            msg = f"Failed to find a valid upper bound: s_high={s_high} is too close to s_low={s_low}."
            raise ValueError(msg)
    scaling_found: float = _binary_search_threshold(
        net,
        s_low,
        s_high,
        min_overloaded_lines,
        min_percent_overload,
        max_search_iter,
        tol,
    )
    final_overloaded: int | None = _run_pf_with_scaling(
        net,
        scaling_found,
        min_percent_overload,
    )
    while final_overloaded is None:
        msg = "Power flow did not converge with binary search - lowering the scaling factor."
        logging.debug(msg)
        scaling_found = _lower_scaling_that_net_converges(net, scaling_found)
        final_overloaded = _run_pf_with_scaling(
            net,
            scaling_found,
            min_percent_overload,
        )

    # Update the network with the found scaling factor.
    net.load["scaling"] = scaling_found
    net.gen["scaling"] = scaling_found
    net.sgen["scaling"] = scaling_found

    try:
        pp.runpp(net)
    except pp.LoadflowNotConverged as err:
        msg = "Final power flow did not converge at the determined scaling factor."
        raise ValueError(msg) from err

    # include scaling the lines
    # Step 2: Adjust line capacities if necessary
    line_scaling_low, line_scaling_high = 0.001, 3.0
    line_scaling_found = 1.0
    line_max = copy.deepcopy(net.line["max_i_ka"])
    iter_count = 0
    while not (
        min_overloaded_lines * 0.9 <= final_overloaded <= min_overloaded_lines * 1.1
    ):
        if final_overloaded < min_overloaded_lines:
            # Not enough overloaded lines → decrease line capacities
            line_scaling_high = line_scaling_found
            line_scaling_found = (line_scaling_low + line_scaling_found) / 2
        else:
            # Too many overloaded lines → increase line capacities
            line_scaling_low = line_scaling_found
            line_scaling_found = (line_scaling_high + line_scaling_found) / 2

        net.line["max_i_ka"] = line_max * line_scaling_found
        try:
            pp.runpp(net)
        except pp.LoadflowNotConverged:
            logging.debug("Power flow did not converge with line scaling.")
            if final_overloaded < min_overloaded_lines:
                line_scaling_high = line_scaling_found
            else:
                line_scaling_low = line_scaling_found
            line_scaling_found = (line_scaling_low + line_scaling_high) / 2
        final_overloaded = _run_pf_with_scaling(
            net,
            scaling_found,
            min_percent_overload,
        )
        if final_overloaded is None:
            msg = "Power flow did not converge with scaling."
            raise ValueError(msg)

        iter_count += 1
        if iter_count >= max_search_iter:
            logging.warning("Line scaling did not converge within max iterations.")
            break

    return scaling_found, int(final_overloaded), line_scaling_found
