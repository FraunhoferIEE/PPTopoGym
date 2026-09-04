from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from pandapower_env.rlib_agents.callbacks import RayCallbacks

if TYPE_CHECKING:
    import pytest
    from ray.rllib.env.single_agent_episode import SingleAgentEpisode
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

MEAN_LOADING_OF_80_AND_100 = 90.0
WORST_CASE_LOADING_CAP = 200.0


class _DummyAlgorithm:
    """Mimics RLlib Algorithm just enough for the tests."""

    def __init__(self) -> None:
        self.logger = _DummyLogger()

class _DummyLogger:
    def __init__(self) -> None:
        self.records: list[str] = []

    def info(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if args:
            fmt: str = args[0]
            rest = args[1:]
            self.records.append(fmt % rest if rest else fmt)
        elif "msg" in kwargs:
            self.records.append(str(kwargs["msg"]))


def test_on_train_result_logs(caplog: pytest.LogCaptureFixture,
                              dummy_result: dict[str, Any]) -> None:
    """RayCallbacks should emit a detailed summary log line."""
    cb = RayCallbacks()
    algo = _DummyAlgorithm()

    # Ensure logger used in callback is captured
    target_logger = logging.getLogger("ray.train")
    target_logger.setLevel(logging.INFO)
    target_logger.addHandler(caplog.handler)
    target_logger.propagate = False

    cb.on_train_result(algorithm=algo,
                       metrics_logger={},
                       result=dummy_result)

    assert "iter=   1" in caplog.text
    assert "mean_ep_return=" in caplog.text
    assert "mean_step_reward=" in caplog.text


class _DummyEpisode:
    """Mimics an RLlib SingleAgentEpisode by returning a fixed list of info dicts."""

    def __init__(self, infos: list[dict[str, Any]]) -> None:
        self.infos = infos

    def get_infos(self) -> list[dict[str, Any]]:
        return self.infos


class _RecordingMetricsLogger:
    """Collects every ``log_value`` call so the test can assert on what was logged."""

    def __init__(self) -> None:
        self.logged: dict[tuple[str, ...], float] = {}
        self.reductions: dict[tuple[str, ...], str] = {}

    def log_value(self, key: tuple[str, ...], value: float, reduce: str = "mean") -> None:
        self.logged[key] = value
        self.reductions[key] = reduce


def test_on_episode_step_is_a_noop() -> None:
    """Per-step metrics are deliberately not collected; the hook must stay side-effect free."""
    logger_stub = _RecordingMetricsLogger()

    RayCallbacks().on_episode_step(
        episode=cast("SingleAgentEpisode", _DummyEpisode([])),
        metrics_logger=cast("MetricsLogger", logger_stub),
    )

    assert logger_stub.logged == {}


def test_on_episode_end_averages_valid_loadings() -> None:
    """NaN and missing loadings are dropped before averaging; a crash anywhere flags the episode."""
    infos: list[dict[str, Any]] = [
        {"loading_percent": 80.0, "crashed": False},
        {"loading_percent": float("nan"), "crashed": False},
        {"crashed": True},
        {"loading_percent": 100.0, "crashed": False},
    ]
    logger_stub = _RecordingMetricsLogger()

    RayCallbacks().on_episode_end(
        episode=cast("SingleAgentEpisode", _DummyEpisode(infos)),
        metrics_logger=cast("MetricsLogger", logger_stub),
    )

    assert logger_stub.logged[("custom_metrics", "mean_loading")] == MEAN_LOADING_OF_80_AND_100
    assert logger_stub.logged[("custom_metrics", "crash")] == 1.0


def test_on_episode_end_uses_worst_case_cap_without_loadings() -> None:
    """An episode that reported no usable loading falls back to the 200% worst-case cap."""
    logger_stub = _RecordingMetricsLogger()

    RayCallbacks().on_episode_end(
        episode=cast("SingleAgentEpisode", _DummyEpisode([{"crashed": False}])),
        metrics_logger=cast("MetricsLogger", logger_stub),
    )

    assert logger_stub.logged[("custom_metrics", "mean_loading")] == WORST_CASE_LOADING_CAP
    assert logger_stub.logged[("custom_metrics", "crash")] == 0.0


def test_on_train_result_waits_for_first_complete_episode(caplog: pytest.LogCaptureFixture) -> None:
    """Before any episode finishes, ``episode_len_mean`` is 0 and only a waiting note is logged."""
    target_logger = logging.getLogger("ray.train")
    target_logger.setLevel(logging.INFO)
    target_logger.addHandler(caplog.handler)
    target_logger.propagate = False

    result: dict[str, Any] = {
        "training_iteration": 7,
        "env_runners": {"episode_return_mean": 0.0, "episode_len_mean": 0},
    }
    RayCallbacks().on_train_result(algorithm=_DummyAlgorithm(), metrics_logger={}, result=result)

    assert "waiting for first complete episode" in caplog.text
    assert "custom_metrics" not in result
