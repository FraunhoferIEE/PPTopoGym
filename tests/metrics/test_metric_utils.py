from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pandapower_env.metrics.evaluation_metrics import MetricRegistry
from pandapower_env.metrics.metric_utils import (
    EvaluateMetrics,
    MetricFn,
    StepData,
)


def test_evaluatemetrics_with_simple_metrics(env_config: dict[str, Any]) -> None:
    """Test evaluation metrics with simple custom metrics.

    EvaluateMetrics should:

    - run env with given actions,
    - call provided metric functions,
    - return per-step and aggregated DataFrames.
    """

    def metric_t(step: StepData) -> float:
        return float(step.t)

    def metric_reward(step: StepData) -> float:
        return float(step.reward)

    metrics: dict[str, MetricFn] = {
        "t": metric_t,
        "reward": metric_reward,
    }

    actions = [0, 0, 0]
    evaluator = EvaluateMetrics(env_config=env_config, metrics=metrics)

    df_steps, df_stats = evaluator.evaluate(actions=actions, start_index=0)

    # basic structure
    assert set(df_steps.columns) == {"t", "reward"}
    assert len(df_steps) >= 1  # at least one step before termination/truncation

    # aggregation frame
    assert set(df_stats.index) == {"mean", "std", "min", "max", "median"}
    assert set(df_stats.columns) == {"t", "reward"}

    # numeric dtypes
    assert df_steps.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all()
    assert df_stats.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all()


def test_evaluatemetrics_metric_keys_filtering(env_config: dict[str, Any]) -> None:
    """metric_keys argument should select a subset of metrics to evaluate."""

    def metric_a(step: StepData) -> float:  # noqa: ARG001
        return 1.0

    def metric_b(step: StepData) -> float:  # noqa: ARG001
        return 2.0

    metrics: dict[str, MetricFn] = {"a": metric_a, "b": metric_b}

    evaluator = EvaluateMetrics(env_config=env_config, metrics=metrics)

    actions = [0, 0]

    # single metric selected
    df_steps, df_stats = evaluator.evaluate(actions=actions, metric_keys=["a"])
    assert list(df_steps.columns) == ["a"]
    assert list(df_stats.columns) == ["a"]

    # subset with both metrics
    df_steps2, _ = evaluator.evaluate(actions=actions, metric_keys=["a", "b"])
    assert set(df_steps2.columns) == {"a", "b"}

    # unknown metric should raise KeyError
    with pytest.raises(KeyError):
        evaluator.evaluate(actions=actions, metric_keys=["does_not_exist"])


def test_evaluatemetrics_empty_metrics_raises(env_config: dict[str, Any]) -> None:
    """Passing an empty metrics dict should raise ValueError when evaluating."""
    evaluator = EvaluateMetrics(env_config=env_config, metrics={})
    with pytest.raises(ValueError): #noqa: PT011
        evaluator.evaluate(actions=[0, 0])


def test_evaluatemetrics_with_all_real_metrics(env_config: dict[str, Any]) -> None:
    """Run EvaluateMetrics with ALL metrics from evaluation_metrics.METRICS on a short action sequence.

    This ensures that each metric can be evaluated
    end-to-end without crashing and produces numeric/NaN output.
    """
    metrics = MetricRegistry.METRICS
    assert metrics, "METRICS must be a non-empty dict"

    evaluator = EvaluateMetrics(env_config=env_config, metrics=metrics)
    actions = [0, 0, 0, 0]

    df_steps, df_stats = evaluator.evaluate(
        actions=actions,
        metric_keys=None,  # None => all metrics
        start_index=0,
    )

    # all metrics must appear as columns
    assert set(df_steps.columns) == set(metrics.keys())
    # aggregation must cover same metrics
    assert set(df_stats.columns) == set(metrics.keys())
    # index of df_stats is our default aggregation set
    assert set(df_stats.index) == {"mean", "std", "min", "max", "median"}

    # all values should be numeric or NaN
    assert df_steps.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all()
    assert df_stats.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all()

def test_add_custom_metric_via_registry_and_runs_in_evaluator(env_config: dict[str, Any]) -> None:
    MetricRegistry.reset()

    def custom_counter(step: StepData) -> float:
        key = MetricRegistry.env_key(step)
        bucket = MetricRegistry.cache_bucket("custom_counter")
        bucket[key] = int(bucket.get(key, 0)) + 1
        return float(bucket[key])

    MetricRegistry.add_metric("custom_counter", custom_counter, overwrite=True)

    evaluator = EvaluateMetrics(env_config=env_config, metrics=MetricRegistry.METRICS)
    df_steps, df_stats = evaluator.evaluate(actions=[0, 0, 0], metric_keys=["custom_counter"], start_index=0)

    assert "custom_counter" in df_steps.columns
    vals = df_steps["custom_counter"].to_numpy(dtype=float)

    # should start at 1 and be monotonic increasing (episode may end early)
    assert vals[0] == 1.0
    assert np.all(np.diff(vals) >= 0.0)

    # cache bucket was created and populated
    assert "custom_counter" in MetricRegistry.CACHE
    assert any(v >= 1 for v in MetricRegistry.CACHE["custom_counter"].values())

    assert "custom_counter" in df_stats.columns


def test_add_metric_duplicate_name_requires_overwrite(env_config: dict[str, Any]) -> None:
    MetricRegistry.reset()

    def m1(step: StepData) -> float:  # noqa: ARG001
        return 1.0

    def m2(step: StepData) -> float:  # noqa: ARG001
        return 2.0

    MetricRegistry.add_metric("dup_metric", m1, overwrite=True)

    with pytest.raises(KeyError):
        MetricRegistry.add_metric("dup_metric", m2, overwrite=False)

    MetricRegistry.add_metric("dup_metric", m2, overwrite=True)

    evaluator = EvaluateMetrics(env_config=env_config, metrics=MetricRegistry.METRICS)
    df_steps, _ = evaluator.evaluate(actions=[0], metric_keys=["dup_metric"], start_index=0)

    assert np.isclose(float(df_steps["dup_metric"].iloc[0]),2.0)
