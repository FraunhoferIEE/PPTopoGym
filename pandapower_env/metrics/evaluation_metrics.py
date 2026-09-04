from __future__ import annotations

from math import log
from typing import TYPE_CHECKING, Any, Callable, ClassVar

import numpy as np
from pandapower import LoadflowNotConverged

from pandapower_env.observation_space.pp_to_observation import (
    all_open_busbar_couplers,
    line_loading_max,
    line_loading_mean,
    line_loading_var,
    line_overloading_number_total,
    line_overloading_sum,
    line_specific_overloaded,
    nminus1_line_loading_max,
    open_busbar_coupler_total,
)
from pandapower_env.toolbox.utils import run_nminus1_powerflow

if TYPE_CHECKING:
    import pandas as pd

    from pandapower_env.metrics.metric_utils import MetricFn, StepData


class MetricRegistry:
    """
    Central registry + state container for all metrics.

    Class attributes are shared per Python process.
    """

    # metric name -> function
    METRICS: ClassVar[dict[str, MetricFn]] = {}

    # internal state cache
    CACHE: ClassVar[dict[str, dict[int, Any]]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[MetricFn], MetricFn]:
        def decorator(fn: MetricFn) -> MetricFn:
            cls.METRICS[name] = fn
            return fn
        return decorator

    @classmethod
    def add_metric(cls, name: str, fn: MetricFn, *, overwrite: bool = False) -> None:
        """Register a metric from outside (runtime registration)."""
        if (not overwrite) and (name in cls.METRICS):
            msg = f"Metric '{name}' already exists. Use overwrite=True."
            raise KeyError(msg)
        cls.METRICS[name] = fn

    @classmethod
    def cache_bucket(cls, bucket: str) -> dict[int, Any]:
        """Get a per-env cache bucket by name.

        External metrics can use their own bucket names.
        """
        return cls.CACHE.setdefault(bucket, {})

    @classmethod
    def reset(cls) -> None:
        """Clear all internal metric state."""
        cls.CACHE.clear()

    @classmethod
    def env_key(cls, step: StepData) -> int:
        """Stable key per env instance."""
        return int(step.info.get("_source_instance_id", id(step.env)))

# -----------------------------------------------------------------------------
# Loadflow performance metrics
# -----------------------------------------------------------------------------


@MetricRegistry.register("loading_improvement_optimization")
def loading_improvement_optimization(step: StepData) -> float:
    """Compute max line loading after optimization minus max line loading before optimization.

    Expects the environment to provide in ``step.info``:
        - "max_loading_percent_before"
        - "max_loading_percent_after"
    """
    info = step.info
    try:
        before = np.max(info["max_loading_percent_before"])
        after = np.max(info["max_loading_percent_after"])
    except KeyError:
        return float("nan")
    return float(after - before)


@MetricRegistry.register("overload_energy_difference_abs_mvah")
def overload_energy_difference_abs_mvah(step: StepData) -> float:
    """Calculate absolute change in overload energy (MVAh) before vs. after optimization.

    Uses:
        info["total_energy_overload_before"]
        info["total_energy_overload_after"]
    and scales with ``env.resolution`` (hours per step).
    """
    info = step.info
    before = info.get("total_energy_overload_before", np.nan)
    after = info.get("total_energy_overload_after", np.nan)

    if np.isnan(before) or np.isnan(after):
        return float("nan")

    return float((after - before) * step.env.resolution)


# -----------------------------------------------------------------------------
# Max loading / overload metrics
# -----------------------------------------------------------------------------


@MetricRegistry.register("max_line_loading_nminus0")
def max_line_loading_nminus0(step: StepData) -> float:
    """Calculate maximum line loading in the current N-0 network state."""
    net = step.env.net
    try:
        value = line_loading_max(net)
    except (LoadflowNotConverged, ValueError):
        value = np.nan
    return float(value)


def _ensure_nminus1(step: StepData) -> None:
    """Solve the N-1 contingency sweep for this timestep, at most once.

    Three registered metrics read the contingency columns of ``res_line``, and each used to
    call :func:`run_nminus1_powerflow` itself -- so selecting them together solved the entire
    contingency set two or three times per step for an identical answer (a sweep is ~400 ms on
    case30 and ~2.3 s on case89). The flag lives on the :class:`StepData`, which is exactly the
    per-timestep scope over which the result stays valid.

    :param step: the timestep context; ``step.nminus1_ready`` is set once the sweep has run.
    :type step: StepData
    """
    if step.nminus1_ready:
        return
    run_nminus1_powerflow(step.env.net)
    step.nminus1_ready = True


@MetricRegistry.register("max_line_loading_nminus1")
def max_line_loading_nminus1(step: StepData) -> float:
    """Calculate maximum line loading in the current N-1 network state."""
    net = step.env.net
    try:
        _ensure_nminus1(step)
        value = nminus1_line_loading_max(net)
    except (LoadflowNotConverged, ValueError):
        value = np.nan
    return float(value)


@MetricRegistry.register("max_line_loading_var_nminus0")
def max_line_loading_var_nminus0(step: StepData) -> float:
    """Calculate variance of line loadings in the current N-0 state.

    Variance indicates the spread/difference in line loadings.
    """
    net = step.env.net
    try:
        value = line_loading_var(net)
    except LoadflowNotConverged:
        value = np.nan
    return float(value)


@MetricRegistry.register("max_line_loading_mean_nminus0")
def max_line_loading_mean_nminus0(step: StepData) -> float:
    """Comupte mean line loading in the current N-0 state."""
    net = step.env.net
    try:
        value = line_loading_mean(net)
    except LoadflowNotConverged:
        value = np.nan
    return float(value)


@MetricRegistry.register("max_lines_overloaded_nminus0")
def max_lines_overloaded_nminus0(step: StepData) -> float:
    """Compute number of overloaded lines in N-0 for the current timestep.

    Aggregated statistics (e.g. max, mean, sum) over this column give
    episode-level overload measures.
    """
    net = step.env.net
    value = line_overloading_number_total(net)
    return float(value)


@MetricRegistry.register("max_lines_overloaded_nminus1")
def max_lines_overloaded_nminus1(step: StepData) -> float:
    """Compute number of overloaded lines in N-1 for the current timestep."""
    net = step.env.net
    threshold = 100.0
    try:
        _ensure_nminus1(step)
        overloaded = net.res_line["max_loading_percent"] > threshold
        value = float(np.sum(overloaded))
    except LoadflowNotConverged:
        # If the LF does not converge, consider all lines overloaded
        value = float(len(net.line))
    return value


@MetricRegistry.register("number_of_overloaded_nminus1_cases")
def number_of_overloaded_nminus1_cases(step: StepData) -> float:
    """Compute number of N-1 contingencies that lead to overloads.

    Requires res_line to contain:
        - "max_loading_percent"
        - "cause_element"
        - "cause_index"
    """
    net = step.env.net
    threshold_nminus1 = 100.0
    rl: pd.DataFrame = net.res_line

    required = {"max_loading_percent", "cause_element", "cause_index"}
    if not required.issubset(rl.columns):
        msg = "res_line missing contingency columns. Did you run run_nminus1_powerflow?"
        raise ValueError(msg)

    overloaded = rl[rl["max_loading_percent"] > threshold_nminus1]
    if overloaded.empty:
        return 0.0

    # count unique cause_index among overloaded lines
    return float(overloaded["cause_index"].dropna().nunique())


# -----------------------------------------------------------------------------
# Grid overload timestep metrics (per-step indicators)
# -----------------------------------------------------------------------------


@MetricRegistry.register("grid_timestep_overload")
def grid_timestep_overload(step: StepData) -> float:
    """Get the indicator if any line is overloaded in this timestep.

    A line is considered overloaded if max loading >= 100%.
    """
    net = step.env.net
    threshold = 100.0
    try:
        overloaded = line_loading_max(net) >= threshold
    except LoadflowNotConverged:
        overloaded = True
    return 1.0 if overloaded else 0.0


@MetricRegistry.register("grid_timestep_huge_overload")
def grid_timestep_huge_overload(step: StepData) -> float:
    """Indicate if any line has a *huge* overload in this timestep.

    A huge overload is defined as max loading >= 110%.
    """
    net = step.env.net
    threshold = 110.0
    try:
        overloaded = line_loading_max(net) >= threshold
    except LoadflowNotConverged:
        overloaded = True
    return 1.0 if overloaded else 0.0


@MetricRegistry.register("line_max_timestep_overload")
def line_max_timestep_overload(step: StepData) -> float:
    """Compute current number of overloaded lines (per timestep view).

    The original implementation tracked longest consecutive overload streaks
    per line via internal caches. In the new design we expose the per-step
    count; streak statistics can be computed from this time series.
    """
    net = step.env.net
    overloaded_flags = list(line_specific_overloaded(net))
    return float(np.sum(overloaded_flags))


# -----------------------------------------------------------------------------
# Topology & switching metrics
# -----------------------------------------------------------------------------


@MetricRegistry.register("open_busbar_couplers")
def open_busbar_couplers_metric(step: StepData) -> float:
    """Compute number of open busbar couplers in the current timestep."""
    net = step.env.net
    return float(open_busbar_coupler_total(net))


@MetricRegistry.register("open_busbar_couplers_window_last_8hr")
def open_busbar_couplers_window_last_8hr(step: StepData) -> float:
    """Compute number of changes in open busbar couplers within the last 8 hours."""
    env = step.env
    net = env.net

    key = MetricRegistry.env_key(step)

    history_by_env = MetricRegistry.cache_bucket("open_bc_history")
    changes_by_env = MetricRegistry.cache_bucket("open_bc_changes")

    history = history_by_env.setdefault(key, [])
    changes_hist = changes_by_env.setdefault(key, [])

    history.append(all_open_busbar_couplers(net))


    window_size = int(8 * 1.0 / env.resolution)
    window_size = max(window_size, 1)

    window = history[-window_size:]

    # Count changes inside the window
    changes = sum(1 for i in range(1, len(window)) if window[i] != window[i - 1])

    # Append and return cumulative max
    changes_hist.append(changes)
    return float(np.max(changes_hist)) if changes_hist else 0.0


@MetricRegistry.register("n_timesteps_substation_actions")
def n_timesteps_substation_actions(step: StepData) -> float:
    """Compute cumulative number of timesteps with substation switching."""
    env = step.env
    df_actions = env.df_actions

    # is the *current* action a substation action?
    action_row = df_actions.loc[df_actions["action"] == step.action]
    if action_row.empty:
        is_substation = False
    else:
        subs = action_row["substations"].iloc[0]
        is_substation = len(subs) > 0

    key = MetricRegistry.env_key(step)

    substation_steps_by_env = MetricRegistry.cache_bucket("substation_timesteps")
    prev = int(substation_steps_by_env.get(key, 0))
    current = prev + (1 if is_substation else 0)
    substation_steps_by_env[key] = current

    return float(current)


# -----------------------------------------------------------------------------
# Reward & action diagnostics
# -----------------------------------------------------------------------------


@MetricRegistry.register("average_reward")
def average_reward(step: StepData) -> float:
    """
    Compute per-step reward.

    The overall average reward can be computed as the mean of this
    metric's time series.
    """
    return float(step.reward)


@MetricRegistry.register("action_entropy")
def action_entropy(step: StepData) -> float:
    """Compute entropy of the empirical action distribution up to the current step.

    Uses info["prev_actions"], which is a LoggedArray of all past actions.
    """
    prev_actions = step.info.get("prev_actions")
    if prev_actions is None:
        return float("nan")

    actions = list(prev_actions)
    n = len(actions)
    if n <= 1:
        return 0.0

    # empirical probabilities
    unique, counts = np.unique(actions, return_counts=True)
    probs = counts.astype(float) / float(n)
    return -float(np.sum(probs * np.vectorize(log)(probs)))


# -----------------------------------------------------------------------------
# Overload energy
# -----------------------------------------------------------------------------


@MetricRegistry.register("sum_line_loading_nminus0")
def sum_line_loading_nminus0(step: StepData) -> float:
    """Calclate the sum of overload (above 100%) over all lines in N-0 for this step."""
    net = step.env.net
    value = line_overloading_sum(net)
    return float(value)


# -----------------------------------------------------------------------------
# Substation usage
# -----------------------------------------------------------------------------


@MetricRegistry.register("no_of_used_substations")
def no_of_used_substations(step: StepData) -> float:
    """Compute number of unique substations involved in actions so far.

    We reconstruct this from the logged action history stored in
    info["prev_actions"] and  env.df_actions.
    """
    prev_actions = step.info.get("prev_actions")
    if prev_actions is None:
        return float("nan")

    df_actions = step.env.df_actions
    used: set[int] = set()
    for act in prev_actions:
        row = df_actions.loc[df_actions["action"] == int(act)]
        if row.empty:
            continue
        subs = row["substations"].iloc[0]
        used.update(int(s) for s in subs)
    return float(len(used))


# -----------------------------------------------------------------------------
# Busbar phase angle / voltage differences before coupling
# -----------------------------------------------------------------------------


@MetricRegistry.register("bb_phase_angle_diff_before_coupling")
def bb_phase_angle_diff_before_coupling(step: StepData) -> float:
    """Compute maximum phase-angle difference between busbars that are about to be coupled."""
    return _max_busbar_spread_before_coupling(step, "va_degree")


@MetricRegistry.register("bb_voltage_diff_before_coupling")
def bb_voltage_diff_before_coupling(step: StepData) -> float:
    """Compute maximum voltage magnitude difference between busbars before coupling."""
    return _max_busbar_spread_before_coupling(step, "vm_pu")


def _max_busbar_spread_before_coupling(step: StepData, res_bus_column: str) -> float:
    """Largest spread of one ``res_bus`` quantity across busbars an action is about to couple.

    Closing a busbar coupler shorts the two busbars together, so a large phase-angle or voltage
    difference across the open coupler is what makes that switching operation stressful. The
    two registered metrics differ only in which ``res_bus`` column they read.

    :param step: the timestep context, whose action is inspected for couplings.
    :type step: StepData
    :param res_bus_column: the ``net.res_bus`` column to spread, ``va_degree`` or ``vm_pu``.
    :type res_bus_column: str
    :return: the largest max-minus-min across the coupled busbars; 0.0 if the action couples none.
    :rtype: float
    """
    net = step.env.net
    substations = action_is_coupling(step.action, step.env.df_actions)
    if not substations:
        return 0.0

    switch_columns = net.multi_bb_substation.filter(regex="_switch$").columns
    bus_columns = net.multi_bb_substation.filter(regex="bus_").columns

    spreads: list[list[float]] = []
    for sub in substations:
        for switch in net.multi_bb_substation.loc[sub, switch_columns]:
            if check_if_switch_if_closed(switch, net.switch):
                continue
            connected_buses = net.multi_bb_substation.loc[sub, bus_columns]
            spreads.append(_busbar_values(connected_buses.tolist(), net.res_bus, res_bus_column))

    if not spreads:
        return 0.0
    return max(float(np.max(values) - np.min(values)) for values in spreads)


# -----------------------------------------------------------------------------
# Helper functions (non-metrics)
# -----------------------------------------------------------------------------

def action_is_coupling(action: int | np.integer, df_actions: pd.DataFrame) -> list[int]:
    """Check if an action re-couples busbars and return affected substation indices."""
    states = df_actions.loc[df_actions["action"] == action, "states"].iloc[0]
    connected_substations: list[int] = []
    for index, state in enumerate(states):
        # all busbars equal -> they can be (re-)coupled
        if len(set(state)) == 1:
            substations = df_actions.loc[df_actions["action"] == action, "substations"].iloc[0]
            connected_substations.append(int(substations[index]))
    return connected_substations


def check_if_switch_if_closed(switch: int, switch_df: pd.DataFrame) -> bool:
    """Return True if the given switch is closed."""
    return bool(switch_df.loc[switch, "closed"])


def _busbar_values(bus_list: list[int], res_bus_df: pd.DataFrame, column: str) -> list[float]:
    """Return one ``res_bus`` column's value for each of the given buses."""
    return [float(res_bus_df.loc[bus, column]) for bus in bus_list]
