"""The parallel greedy agent must run flawlessly with N-1 feedback and many workers.

The greedy agent already parallelises across *actions* with loky; each worker runs the
*serial* N-1 internally. This guards that combination -- 20 workers, N-1 feedback, and the
"n-1 parallel" env flag set -- against crashes / deadlocks (no nested process pools).
"""

from __future__ import annotations

import copy

from pandapower_env.agents.benchmark_agents import GreedyAgent
from pandapower_env.environments.simulation_env import PPTopoGym

N_WORKERS = 20


def test_greedy_agent_nminus1_with_20_workers(env_config: dict) -> None:
    """A 20-worker greedy agent scoring on N-1 returns a valid action without deadlocking."""
    config = copy.deepcopy(env_config)
    config["nminus1"] = True
    # Set the env flag too, to prove the parallel-N-1 env config coexists with the parallel
    # greedy agent (greedy workers still use the serial N-1 -- no nested pools).
    config["n-1 parallel"] = True

    driver_env = PPTopoGym(config)
    observation, info = driver_env.reset(options={"index": 0})

    agent = GreedyAgent(
        driver_env.action_space,
        copy.deepcopy(config),
        feedback_type="nminus1",
        n_workers=N_WORKERS,
    )
    action = agent.act(observation, info)

    assert 0 <= int(action) < driver_env.action_space.n
