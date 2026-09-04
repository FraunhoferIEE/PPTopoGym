from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.metrics.evaluation_metrics import MetricRegistry


@dataclass
class StepData:
    """
    Minimal per-timestep context for metrics.

    - env: PPTopoGym, from which you can get env.net, env.df_actions, etc.
    - action: last applied action (int)
    - reward: reward returned by env.step(...)
    - info: env.step(...) info dict (for KPIs)
    """

    t: int
    env: PPTopoGym
    action: int
    reward: float
    info: dict[str, Any]
    # Set once the N-1 contingency sweep has been solved for this timestep, so the several
    # metrics that read its columns share one sweep instead of each running their own.
    nminus1_ready: bool = False


# Metric: takes one StepData, returns a float
MetricFn = Callable[[StepData], float]


class EvaluateMetrics:
    """
    Minimal evaluator for a set of metrics.

    metrics: mapping name -> function(step: StepData) -> float

    Typical usage:

        from pandapower_env.metrics.metric_utils import EvaluateMetrics
        from pandapower_env.metrics.evaluation_metrics import METRICS

        evaluator = EvaluateMetrics(env_config, METRICS)
        df_steps, df_stats = evaluator.evaluate(
            actions=my_actions,
            metric_keys=["max_line_loading_nminus0", "grid_timestep_overload"],
            start_index=0,
        )
    """

    def __init__(
        self,
        env_config: dict,
        metrics: Mapping[str, MetricFn],
    ) -> None:
        env_config.pop("pf_type", None)
        self.env_config = dict(env_config)
        self._metrics: dict[str, MetricFn] = dict(metrics)

    def evaluate(
        self,
        actions: Sequence[int],
        metric_keys: Iterable[str] | None = None,
        start_index: int = 0,
        float_precision: int = 6,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run PPTopoGym with given actions and compute selected metrics.

        Returns
        -------
        df_steps : pd.DataFrame
            Rows = timesteps, columns = metric names.
        df_stats : pd.DataFrame
            Rows = aggregations ("mean", "std", "min", "max", "median"),
            columns = metric names.
        """
        if metric_keys is None:
            active_metrics: dict[str, MetricFn] = dict(self._metrics)
        else:
            keys = list(metric_keys)
            missing = [k for k in keys if k not in self._metrics]
            if missing:
                msg = f"Unknown metrics requested: {missing}"
                raise KeyError(msg)
            active_metrics = {k: self._metrics[k] for k in keys}

        if not active_metrics:
            msg = "No metrics selected for evaluation."
            raise ValueError(msg)

        MetricRegistry.reset()

        env = PPTopoGym(self.env_config)
        _, _ = env.reset(options={"index": start_index})

        values: dict[str, list[float]] = {name: [] for name in active_metrics}

        for t, action in enumerate(actions):
            obs, reward, terminated, truncated, info = env.step(int(action))
            info["_source_instance_id"] = id(env)
            step = StepData(
                t=t,
                env=env,
                action=int(action),
                reward=float(reward),
                info=info,
            )

            for name, fn in active_metrics.items():
                try:
                    v = fn(step)
                except Exception: # noqa: BLE001
                    v = np.nan
                values[name].append(float(v))

            if terminated or truncated:
                break

        # ---- per-step DataFrame ----
        df_steps = pd.DataFrame(values)
        df_steps.index.name = "timestep"
        if float_precision is not None:
            df_steps = df_steps.round(float_precision)

        # ---- aggregated stats DataFrame ----
        if df_steps.empty:
            return df_steps, pd.DataFrame()

        numeric_df = df_steps.astype(float)
        df_stats = numeric_df.agg(["mean", "std", "min", "max", "median"])
        df_stats.index.name = "aggregation"

        return df_steps, df_stats


