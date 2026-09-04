from __future__ import annotations

from itertools import groupby
from typing import Any

import numpy as np

from pandapower_env.metrics.metric_utils import (
    MetricBase,
    MetricContainer,
)


class AllMetrics(MetricContainer, MetricBase):
    """
    Summary of several metrics.

    All metrics are automatically descripted with FloatMetric.
    """

    # ---------- Peak Overload -------------
    def max_line_loading_nminus0(self, cache: dict[str, Any]) -> float:
        """Return the max line loading and store all intermediate values in cache."""
        from pandapower import LoadflowNotConverged

        from pandapower_env.observation_space.pp_to_observation import line_loading_max
        try:
            value = line_loading_max(self.net)
        except LoadflowNotConverged:
            value = np.nan
        cache.setdefault("max_line_loading_nminus0", [])
        cache["max_line_loading_nminus0"].append(value)
        return np.nanmax(cache["max_line_loading_nminus0"])

    def max_line_loading_nminus1(self, cache: dict[str, Any]) -> float:
        """Return the max line loading and store all intermediate values in cache."""
        from pandapower import LoadflowNotConverged

        from pandapower_env.observation_space.pp_to_observation import (
            nminus1_line_loading_max,
        )
        from pandapower_env.toolbox.utils import run_nminus1_powerflow
        try:
            run_nminus1_powerflow(self.net)
            value = nminus1_line_loading_max(self.net)
        except LoadflowNotConverged:
            value = np.nan
        cache.setdefault("max_line_loading_nminus1", [])
        cache["max_line_loading_nminus1"].append(value)
        return np.nanmax(cache["max_line_loading_nminus1"])

    def max_line_loading_var_nminus0(self, cache: dict[str, Any]) -> float:
        """
        Return the max variance and store all intermediate values in cache.

        Variance indicates the difference in line loadings.
        """
        from pandapower import LoadflowNotConverged

        from pandapower_env.observation_space.pp_to_observation import (
            line_loading_var,
        )
        cache.setdefault("max_line_loading_var_nminus0", [])
        try:
            value = line_loading_var(self.net)
        except LoadflowNotConverged:
            value = np.nan
        cache["max_line_loading_var_nminus0"].append(value)
        return np.nanmax(cache["max_line_loading_var_nminus0"])


    def max_line_loading_mean_nminus0(self, cache: dict[str, Any]) -> float:
        """Return the max line loading and store all intermediate values in cache."""
        from pandapower import LoadflowNotConverged

        from pandapower_env.observation_space.pp_to_observation import (
            line_loading_mean,
        )
        try:
            value = line_loading_mean(self.net)
        except LoadflowNotConverged:
            value = np.nan
        cache.setdefault("max_line_loading_mean_nminus0", [])
        cache["max_line_loading_mean_nminus0"].append(value)
        return np.nanmax(cache["max_line_loading_mean_nminus0"])

    def max_lines_overloaded_nminus0(self, cache: dict[str, Any]) -> float:
        """Return the max. number of lines overloaded at once."""
        from pandapower_env.observation_space.pp_to_observation import (
            line_overloading_number_total,
        )

        cache.setdefault("max_lines_overloaded_nminus0", [])
        value = line_overloading_number_total(self.net)
        cache["max_lines_overloaded_nminus0"].append(value)
        return float(np.max(cache["max_lines_overloaded_nminus0"]))


    # ------------ Overload duration ------------

    def grid_max_timestep_overload(self, cache: dict[str, Any]) -> float:
        """Return the max. number of timesteps any line in the grid was overloaded (consecutively)."""
        from itertools import groupby

        from pandapower_env.observation_space.pp_to_observation import line_loading_max

        cache.setdefault("grid_overloaded", [])
        cache.setdefault("longest_grid_overload", [])
        threshold_overload = 100
        value: bool = line_loading_max(self.net) >= threshold_overload
        cache["grid_overloaded"].append(value)
        group_list = [list(group) for key, group in groupby(cache["grid_overloaded"]) if key]
        max_length = 0 if len(group_list) == 0 else max(len(list(group)) for group in group_list)
        cache["longest_grid_overload"].append(max_length)
        return float(max_length) # latest appended value is automatically max-value.


    def line_max_timestep_overload(self, cache: dict[str, Any]) -> float:
        """Return the max. number of timesteps one line was overloaded consecutively."""
        from pandapower_env.observation_space.pp_to_observation import (
            line_specific_overloaded,  # boolean list for each line, if overloaded
        )
        def max_step(line: tuple) -> int:
            group_list = [list(group) for key, group in groupby(line) if key]
            return 0 if len(group_list) == 0 else max(len(list(group)) for group in group_list)

        cache.setdefault("highest_timestep_one_line_overloaded", [])
        cache.setdefault("_lines_overloaded", [])
        overloaded_lines = list(line_specific_overloaded(self.net))
        cache["_lines_overloaded"].append(overloaded_lines)
        all_line_values = zip(*cache["_lines_overloaded"])
        max_value = max(max_step(line) for line in all_line_values)
        cache["highest_timestep_one_line_overloaded"].append(max_value)
        return float(max_value) # already is the max of all floats



    # ------------- Topology Changes ---------------

    def open_busbar_couplers(self, cache: dict[str, Any]) -> float:
        """Return the max number of open busbar couplers."""
        from pandapower_env.observation_space.pp_to_observation import (
            open_busbar_coupler_total,
        )
        cache.setdefault("open_busbar_couplers", [])
        value = open_busbar_coupler_total(self.net)
        cache["open_busbar_couplers"].append(value)
        return float(np.max(cache["open_busbar_couplers"]))

    def open_busbar_couplers_window_32ts(self, cache: dict[str, Any]) -> float:
        """Return the number of open busbar couplers in 32 timesteps."""
        from pandapower_env.observation_space.pp_to_observation import (
            all_open_busbar_couplers,
        )

        cache.setdefault("open_busbar_couplers_total", [])
        cache.setdefault("changes_in_window", [])
        cache["open_busbar_couplers_total"].append(
            all_open_busbar_couplers(self.net),
        )
        window = cache["open_busbar_couplers_total"][-32:]
        changes = sum(1 for i in range(1, len(window)) if window[i] != window[i - 1])
        cache["changes_in_window"].append(changes)
        return float(np.max(cache["changes_in_window"]))

    # --------------- MISC ----------------------


    def average_reward(self, cache: dict[str, Any]) -> float:
        rewards = cache.setdefault("rewards", [])
        rewards.append(self.last_reward)
        return sum(rewards) / len(rewards)

    def action_entropy(self, cache: dict[str, Any]) -> float:
        from math import log

        actions = cache.setdefault("actions", [])
        actions.append(self.last_action)
        freq = {a: actions.count(a) for a in set(actions)}
        probs = [f / len(actions) for f in freq.values()]
        return -sum(p * log(p) for p in probs if p > 0)


    def sum_line_loading_nminus0(self, cache: dict[str, Any]) -> float:
        """Return the sum of all line loadings above 100."""
        from pandapower_env.observation_space.pp_to_observation import (
            line_overloading_sum,
        )

        cache.setdefault("sum_line_loading_nminus0", [])
        value = line_overloading_sum(self.net)
        cache["sum_line_loading_nminus0"].append(value)
        return np.sum(cache["sum_line_loading_nminus0"])




