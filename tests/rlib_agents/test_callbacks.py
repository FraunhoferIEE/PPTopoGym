from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pytest

from pandapower_env.rlib_agents.callbacks import RayCallbacks


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
