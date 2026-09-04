from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pandapower import LoadflowNotConverged

import pandapower_env.metrics.evaluation_metrics as em
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.metrics.evaluation_metrics import (
    MetricRegistry,
    _busbar_values,
    action_is_coupling,
    check_if_switch_if_closed,
)
from pandapower_env.metrics.metric_utils import StepData


def _raise_not_converged(*_args: object, **_kwargs: object) -> float:
    """Stand in for a power-flow helper that fails to converge."""
    msg = "simulated divergence"
    raise LoadflowNotConverged(msg)


def test_metrics_registry_structure() -> None:
    """METRICS should be a non-empty mapping with some known keys."""
    metrics = MetricRegistry.METRICS
    assert isinstance(metrics, dict)
    assert metrics, "METRICS dict must not be empty"

    expected_keys = {
        "average_reward",
        "loading_improvement_optimization",
        "max_line_loading_nminus0",
        "max_line_loading_nminus1",
        "grid_timestep_overload",
        "sum_line_loading_nminus0",
        "no_of_used_substations",
    }
    assert expected_keys.issubset(metrics.keys())


def test_average_reward_forwards_reward(env_config: dict[str, Any]) -> None:
    """average_reward metric should just forward the reward of the current step."""
    fn = MetricRegistry.METRICS["average_reward"]

    env = PPTopoGym(env_config)
    _, _ = env.reset(options={"index": 0})
    _, reward, _, _, info = env.step(0)

    step = StepData(
        t=0,
        env=env,
        action=0,
        reward=float(reward),
        info=info,
    )

    value = fn(step)
    assert value == pytest.approx(float(reward))


def test_overload_energy_difference_semantics(env_config: dict[str, Any]) -> None:
    """Test difference semantics.

    overload_energy_diffrence_abs_mvah should compute
    (total_energy_overload_after - total_energy_overload_before) * env.resolution
    when both keys are present.
    """
    fn = MetricRegistry.METRICS["overload_energy_difference_abs_mvah"]

    env = PPTopoGym(env_config)
    _, _ = env.reset(options={"index": 0})

    before = 10.0
    after = 4.0
    info = {
        "total_energy_overload_before": before,
        "total_energy_overload_after": after,
    }

    step = StepData(
        t=0,
        env=env,
        action=0,
        reward=0.0,
        info=info,
    )

    expected = (after - before) * env.resolution
    value = fn(step)
    assert value == pytest.approx(expected)


def test_overload_energy_difference_missing_keys_returns_nan(
    env_config: dict[str, Any],
) -> None:
    """overload_energy_difference_abs_mvah should return NaN if info keys are missing."""
    fn = MetricRegistry.METRICS["overload_energy_difference_abs_mvah"]

    env = PPTopoGym(env_config)
    _, _ = env.reset(options={"index": 0})

    step = StepData(
        t=0,
        env=env,
        action=0,
        reward=0.0,
        info={},  # no total_energy_overload_before/after
    )

    value = fn(step)
    assert np.isnan(value)


def test_stateful_metrics_use_and_reset_internal_caches(
    env_config: dict[str, Any],
) -> None:
    """
    Test that the stateful metrics using cached data.

      - open_busbar_couplers_window_last_8hr
      - n_timesteps_substation_actions

    Check that those metrics actually touch their internal caches, and that reset_metric_state()
    clears those caches.
    """
    # start from a clean state
    MetricRegistry.reset()
    assert MetricRegistry.CACHE == {}

    env = PPTopoGym(env_config)
    _, _ = env.reset(options={"index": 0})
    _, reward, _, _, info = env.step(0)

    # Ensure we have a stable env id key (should already be in info)
    key = int(info.get("_source_instance_id", id(env)))

    open_bc_fn = MetricRegistry.METRICS["open_busbar_couplers_window_last_8hr"]
    subst_fn = MetricRegistry.METRICS["n_timesteps_substation_actions"]

    step = StepData(
        t=0,
        env=env,
        action=0,
        reward=0.0,
        info=info,
    )

    # First calls should populate caches and return floats
    val_open_1 = open_bc_fn(step)
    val_subst_1 = subst_fn(step)

    assert isinstance(val_open_1, float)
    assert isinstance(val_subst_1, float)

    assert key in MetricRegistry.CACHE["open_bc_history"]
    assert key in MetricRegistry.CACHE["open_bc_changes"]
    assert key in MetricRegistry.CACHE["substation_timesteps"]

    # Second calls change history (values may or may not change, but shouldn't crash)
    val_open_2 = open_bc_fn(step)
    val_subst_2 = subst_fn(step)
    assert isinstance(val_open_2, float)
    assert isinstance(val_subst_2, float)

    # Finally, reset should clear all caches
    MetricRegistry.reset()
    assert MetricRegistry.CACHE == {}

def test_each_metric_callable_on_single_step(env_config)-> None:
    """Test that each metric function in METRICS can be called."""
    MetricRegistry.reset()
    env = PPTopoGym(env_config)
    _, _ = env.reset(options={"index": 0})
    _, reward, _, _, info = env.step(0)

    step = StepData(t=0, env=env, action=0, reward=float(reward), info=info)

    for name, fn in MetricRegistry.METRICS.items():
        try:
            val = fn(step)
        except ValueError:
            # number_of_overloaded_nminus1_cases may raise if no n-1 columns
            if name == "number_of_overloaded_nminus1_cases":
                continue
            raise
        assert isinstance(val, (int, float, np.integer, np.floating))



# ---------------------------------------------------------------------------
# Missing-input fallbacks
# ---------------------------------------------------------------------------


@pytest.fixture()
def stepped_env(env_config: dict[str, Any]) -> PPTopoGym:
    """Return an environment that has run one DoNothing step, so ``net.res_*`` is populated."""
    env = PPTopoGym(env_config)
    env.reset(options={"index": 0})
    env.step(0)
    return env


def _step(env: PPTopoGym, info: dict[str, Any] | None = None, action: int = 0) -> StepData:
    """Build a StepData around ``env`` with an arbitrary info payload."""
    return StepData(t=0, env=env, action=action, reward=0.0, info=info if info is not None else {})


def test_loading_improvement_returns_nan_without_before_after(stepped_env: PPTopoGym) -> None:
    """The metric is only defined for an optimization step; without the keys it abstains."""
    value = MetricRegistry.METRICS["loading_improvement_optimization"](_step(stepped_env))

    assert np.isnan(value)


def test_overload_energy_difference_returns_nan_without_keys(stepped_env: PPTopoGym) -> None:
    """Missing overload-energy keys default to NaN rather than to a fabricated zero."""
    value = MetricRegistry.METRICS["overload_energy_difference_abs_mvah"](_step(stepped_env))

    assert np.isnan(value)


@pytest.mark.parametrize("metric_name", ["action_entropy", "no_of_used_substations"])
def test_history_metrics_return_nan_without_prev_actions(metric_name: str, stepped_env: PPTopoGym) -> None:
    """Both history metrics need ``info['prev_actions']``; without it they report NaN."""
    assert np.isnan(MetricRegistry.METRICS[metric_name](_step(stepped_env)))


def test_action_entropy_of_a_single_action_is_zero(stepped_env: PPTopoGym) -> None:
    """One observation carries no distribution, hence zero entropy."""
    value = MetricRegistry.METRICS["action_entropy"](_step(stepped_env, {"prev_actions": [0]}))

    assert value == 0.0


def test_no_of_used_substations_skips_unknown_actions(stepped_env: PPTopoGym) -> None:
    """Action ids absent from ``df_actions`` are ignored instead of raising."""
    fn = MetricRegistry.METRICS["no_of_used_substations"]
    unknown_action = int(stepped_env.df_actions["action"].max()) + 99

    known = fn(_step(stepped_env, {"prev_actions": [0]}))
    with_unknown = fn(_step(stepped_env, {"prev_actions": [0, unknown_action]}))

    assert with_unknown == known


def test_n_timesteps_substation_actions_treats_unknown_action_as_non_substation(
    stepped_env: PPTopoGym,
) -> None:
    """An action id that is not in ``df_actions`` cannot be a substation switch."""
    unknown_action = int(stepped_env.df_actions["action"].max()) + 99

    value = MetricRegistry.METRICS["n_timesteps_substation_actions"](
        _step(stepped_env, action=unknown_action),
    )

    assert value == 0.0


# ---------------------------------------------------------------------------
# Non-convergence fallbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("metric_name", "patched_helper"),
    [
        ("max_line_loading_nminus0", "line_loading_max"),
        ("max_line_loading_var_nminus0", "line_loading_var"),
        ("max_line_loading_mean_nminus0", "line_loading_mean"),
        ("max_line_loading_nminus1", "nminus1_line_loading_max"),
    ],
)
def test_loading_metrics_report_nan_when_the_power_flow_diverges(
    metric_name: str,
    patched_helper: str,
    stepped_env: PPTopoGym,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A diverged power flow yields NaN, so the episode aggregate stays distinguishable from 0."""
    monkeypatch.setattr(em, patched_helper, _raise_not_converged)

    assert np.isnan(MetricRegistry.METRICS[metric_name](_step(stepped_env)))


@pytest.mark.parametrize("metric_name", ["grid_timestep_overload", "grid_timestep_huge_overload"])
def test_overload_indicators_assume_the_worst_when_the_power_flow_diverges(
    metric_name: str,
    stepped_env: PPTopoGym,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A grid that will not solve is treated as overloaded, never as healthy."""
    monkeypatch.setattr(em, "line_loading_max", _raise_not_converged)

    assert MetricRegistry.METRICS[metric_name](_step(stepped_env)) == 1.0


def test_max_lines_overloaded_nminus1_counts_all_lines_when_diverged(
    stepped_env: PPTopoGym,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If N-1 does not converge, every line is conservatively counted as overloaded."""
    monkeypatch.setattr(em, "run_nminus1_powerflow", _raise_not_converged)

    value = MetricRegistry.METRICS["max_lines_overloaded_nminus1"](_step(stepped_env))

    assert value == float(len(stepped_env.net.line))


# ---------------------------------------------------------------------------
# N-1 contingency counting
# ---------------------------------------------------------------------------


def test_number_of_overloaded_nminus1_cases_requires_contingency_columns(
    stepped_env: PPTopoGym,
) -> None:
    """Without the N-1 columns the metric is meaningless, so it raises instead of returning 0."""
    fn = MetricRegistry.METRICS["number_of_overloaded_nminus1_cases"]

    with pytest.raises(ValueError, match="res_line missing contingency columns"):
        fn(_step(stepped_env))


def test_number_of_overloaded_nminus1_cases_counts_unique_causes(stepped_env: PPTopoGym) -> None:
    """Overloads sharing a cause count once; a grid within limits scores zero."""
    fn = MetricRegistry.METRICS["number_of_overloaded_nminus1_cases"]
    net = stepped_env.net
    n_lines = len(net.line)

    net.res_line["cause_element"] = "line"
    net.res_line["cause_index"] = list(range(n_lines))
    net.res_line["max_loading_percent"] = 50.0
    assert fn(_step(stepped_env)) == 0.0

    net.res_line.loc[net.res_line.index[:3], "max_loading_percent"] = 150.0
    net.res_line.loc[net.res_line.index[:3], "cause_index"] = 4
    assert fn(_step(stepped_env)) == 1.0


# ---------------------------------------------------------------------------
# Busbar coupling metrics
# ---------------------------------------------------------------------------


def test_coupling_metrics_are_zero_for_a_non_coupling_action(stepped_env: PPTopoGym) -> None:
    """DoNothing couples nothing, so both busbar-difference metrics short-circuit to 0."""
    for metric_name in ("bb_phase_angle_diff_before_coupling", "bb_voltage_diff_before_coupling"):
        assert MetricRegistry.METRICS[metric_name](_step(stepped_env, action=0)) == 0.0


def _coupling_action(env: PPTopoGym) -> int:
    """Return an action whose state recouples at least one substation's busbars."""
    return next(
        int(action)
        for action in env.df_actions["action"]
        if action_is_coupling(action, env.df_actions)
    )


@pytest.mark.parametrize(
    ("metric_name", "res_bus_column"),
    [
        ("bb_phase_angle_diff_before_coupling", "va_degree"),
        ("bb_voltage_diff_before_coupling", "vm_pu"),
    ],
)
def test_coupling_metrics_measure_the_busbar_spread(
    metric_name: str,
    res_bus_column: str,
    stepped_env: PPTopoGym,
) -> None:
    """For a coupling action the metric is the spread across the substation's busbars."""
    env = stepped_env
    action = _coupling_action(env)

    substations = action_is_coupling(action, env.df_actions)
    cols_mbb_buses = env.net.multi_bb_substation.filter(regex="bus_").columns
    buses = env.net.multi_bb_substation.loc[substations[0], cols_mbb_buses].tolist()

    # Force a known spread across the busbars of the coupled substation, and make sure at
    # least one of its switches is open so the metric does not skip every switch.
    env.net.res_bus.loc[buses, res_bus_column] = np.linspace(1.0, 3.0, len(buses))
    cols_mbb_switch = env.net.multi_bb_substation.filter(regex="_switch$").columns
    env.net.switch.loc[env.net.multi_bb_substation.loc[substations[0], cols_mbb_switch], "closed"] = False

    value = MetricRegistry.METRICS[metric_name](_step(env, action=action))

    assert value == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Non-metric helpers
# ---------------------------------------------------------------------------


def test_check_if_switch_if_closed_reads_the_switch_table(stepped_env: PPTopoGym) -> None:
    """The helper is a plain lookup into ``net.switch['closed']``."""
    net = stepped_env.net
    i_switch = net.switch.index[0]

    net.switch.loc[i_switch, "closed"] = True
    assert check_if_switch_if_closed(i_switch, net.switch) is True

    net.switch.loc[i_switch, "closed"] = False
    assert check_if_switch_if_closed(i_switch, net.switch) is False


def test_busbar_lookups_return_values_in_the_requested_order(stepped_env: PPTopoGym) -> None:
    """Both lookups preserve the caller's bus order, which the spread calculation relies on."""
    net = stepped_env.net
    buses = list(net.res_bus.index[:3])

    for column in ("va_degree", "vm_pu"):
        assert _busbar_values(buses, net.res_bus, column) == [
            float(net.res_bus.loc[bus, column]) for bus in buses
        ]
