from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from ray.rllib.algorithms.callbacks import RLlibCallback

if TYPE_CHECKING:
    from ray.rllib.env.single_agent_episode import SingleAgentEpisode
    from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

logger = logging.getLogger("ray.train")

class RayCallbacks(RLlibCallback):
    """Custom callbacks for RLlib training, adding extra metrics."""

    def on_episode_step(self, *, episode: SingleAgentEpisode, metrics_logger: MetricsLogger, **kwargs) -> None: # noqa: ANN003
        pass  # No per-step metrics needed

    def on_episode_end(self,
                       *,
                       episode: SingleAgentEpisode,
                       metrics_logger: MetricsLogger,
                       **kwargs,  # noqa: ARG002, ANN003
                       ) -> None:
        infos = episode.get_infos()
        loads = [val for val in [i.get("loading_percent") for i in infos]
                 if val is not None and not np.isnan(val)]
        mean_load = float(np.mean(loads)) if loads else 200.0  # worst-case cap
        crashed = any(i.get("crashed") for i in infos)

        metrics_logger.log_value(("custom_metrics", "mean_loading"), mean_load, reduce="mean")
        metrics_logger.log_value(("custom_metrics", "crash"), 1.0 if crashed else 0.0, reduce="mean")


    def on_train_result(self, *, result: dict, **kwargs) -> None: #noqa: ARG002, ANN003
        it = result.get("training_iteration")
        runners: dict = result.get("env_runners", {})

        ep_ret: float = runners.get("episode_return_mean", 0.0)
        ep_len: int = runners.get("episode_len_mean", 0)

        cm_env: dict = runners.get("custom_metrics", {})

        mean_load = cm_env.get("mean_loading", 0.0)
        crash_rate = cm_env.get("crash", 0.0)


        if ep_ret is not None and ep_len:
            mean_step_reward = ep_ret / ep_len
            custom_metrics: dict = result.setdefault("custom_metrics", {})
            custom_metrics["mean_step_reward"] = mean_step_reward
            if mean_load is not None:
                custom_metrics["mean_loading"] = mean_load
            if crash_rate is not None:
                custom_metrics["crash_rate"] = crash_rate

            logger.info("[CB] iter=%4s | mean_ep_return=%s | mean_ep_len=%s | "
            "mean_step_reward=%s | mean_load=%s | crash_rate=%s", \
            it, ep_ret, ep_len, mean_step_reward, mean_load, crash_rate)
        else:
            logger.info("[CB] iter=%4s | waiting for first complete episode …", it)
