from __future__ import annotations

from typing import Any, Generator, cast
from unittest.mock import patch

import numpy as np
import pandapower as pp
from pandapower import LoadflowNotConverged

from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.metrics.evaluation_metrics import AllMetrics
from pandapower_env.metrics.metric_utils import EvaluateMetrics, FloatMetric


def test_allmetrics(env_config: dict) -> None:
    """
    Test the AllMetrics class.

    :param env_config: Environment configuration.
    :type env_config: dict
    """
    log_actions = [0, 0, 0]
    eval_ = EvaluateMetrics(env_config, log_actions)
    eval_.run()


def test_metric_functions_one_by_one(env_config: dict) -> None:
    """Test the average_reward method."""
    metrics = AllMetrics()
    reward = 10
    metrics.last_reward = reward
    assert (
        cast(FloatMetric, AllMetrics.average_reward).evaluate(metrics) == reward
    ), metrics.average_reward
    new_reward = 20
    metrics.last_reward = new_reward
    mean_reward = (reward + new_reward) / 2
    assert cast(FloatMetric, AllMetrics.average_reward).evaluate(metrics) == mean_reward
    env = PPTopoGym(env_config)
    metrics.net = env.net
    scaling = 0.1
    metrics.net.load["scaling"] = scaling
    metrics.net.gen["scaling"] = scaling
    metrics.net.sgen["scaling"] = scaling
    pp.runpp(metrics.net)
    metrics.last_action = 0
    assert cast(FloatMetric, AllMetrics.action_entropy).evaluate(metrics) == 0
    metrics.last_action = 1
    assert cast(FloatMetric, AllMetrics.action_entropy).evaluate(metrics) > 0
    assert cast(FloatMetric, AllMetrics.max_line_loading_nminus0).evaluate(metrics) > 0
    assert cast(FloatMetric, AllMetrics.max_line_loading_nminus1).evaluate(metrics) > 0
    assert (
        cast(FloatMetric, AllMetrics.max_line_loading_mean_nminus0).evaluate(
            metrics,
        )
        >= 0
    )
    assert cast(FloatMetric, AllMetrics.max_line_loading_var_nminus0).evaluate(metrics) > 0
    assert (
        cast(FloatMetric, AllMetrics.line_max_timestep_overload).evaluate(
            metrics,
        )
        >= 0
    )
    assert (
        cast(FloatMetric, AllMetrics.sum_line_loading_nminus0).evaluate(
            metrics,
        )
        >= 0
    )
    assert (
        cast(FloatMetric, AllMetrics.line_max_timestep_overload).evaluate(
            metrics,
        )
        >= 0
    )
    assert cast(FloatMetric, AllMetrics.grid_max_timestep_overload).evaluate(metrics) == 0
    assert cast(FloatMetric, AllMetrics.open_busbar_couplers).evaluate(metrics) == 0
    assert cast(FloatMetric, AllMetrics.open_busbar_couplers_window_32ts).evaluate(metrics) == 0
    assert cast(FloatMetric, AllMetrics.max_line_loading_var_nminus0).evaluate(metrics) > 0

# test single metrics if needed
def test_max_line_loading_nminus1() -> None:
    metrics = AllMetrics()
    with patch("pandapower_env.toolbox.utils.run_nminus1_powerflow", side_effect=LoadflowNotConverged):
        result = cast(FloatMetric, AllMetrics.max_line_loading_nminus1).evaluate(metrics) # cast(..) is only for mypy
        cache = cast(FloatMetric, AllMetrics.max_line_loading_nminus1).get_cache(metrics) # cast(..) is only for mypy
    assert np.isnan(result)
    # Check that np.nan was appended in cache
    assert "max_line_loading_nminus1" in cache, result
    assert np.isnan(cache["max_line_loading_nminus1"][-1])

def test_metric_line_max_timestep_overload(monkeypatch) -> None:
    # Mock the function
    def gen_mock() -> Generator[list[bool], Any, None]:
        yield [True, True]
        yield [True, False]
        yield [False, True]
        yield [True, False]


    gen = gen_mock()
    def mock_line_specific_overloaded(_) -> list[bool]:
        return next(gen)
    target_fct = "pandapower_env.observation_space.pp_to_observation.line_specific_overloaded"
    monkeypatch.setattr(target_fct, mock_line_specific_overloaded)
    metrics = AllMetrics()
    result = cast(FloatMetric, AllMetrics.line_max_timestep_overload).evaluate(metrics)
    correct_result = 1 # this is loaded into one long list in the function, as we only load one timestep
    assert result == correct_result, cast(FloatMetric, AllMetrics.line_max_timestep_overload).get_cache(metrics)
    result = cast(FloatMetric, AllMetrics.line_max_timestep_overload).evaluate(metrics)
    correct_result = 2  # this is loaded into one long list in the function, as we only load one timestep
    assert result == correct_result, cast(FloatMetric, AllMetrics.line_max_timestep_overload).get_cache(metrics)
    result = cast(FloatMetric, AllMetrics.line_max_timestep_overload).evaluate(metrics)
    correct_result = 2  # this is loaded into one long list in the function, as we only load one timestep
    assert result == correct_result, cast(FloatMetric, AllMetrics.line_max_timestep_overload).get_cache(metrics)
    result = cast(FloatMetric, AllMetrics.line_max_timestep_overload).evaluate(metrics)
    correct_result = 2  # this is loaded into one long list in the function, as we only load one timestep
    assert result == correct_result, cast(FloatMetric, AllMetrics.line_max_timestep_overload).get_cache(metrics)
