"""Time one greedy step -- a full sweep over the candidate actions -- on whichever branch is imported.

A greedy step is the workload this environment exists to serve and the one that shows the whole
stack at once: the action-space bookkeeping, the worker payloads, and one power flow per
candidate. It is also where the two branches differ most, so it is timed here rather than
inferred from the per-step numbers in ``bench_branch.py``.

Both branches expose the same ``GreedyAgent(action_space, env_config, feedback_type, n_workers,
dc_approximation, overload_threshold)`` constructor and the same ``act(observation, info)``, so
this script runs unmodified on either. What it reports:

    greedy.act.wN       median wall time of one ``act()`` call with N joblib workers
    greedy.per_action   that time divided by the number of candidates evaluated
    chosen              the action the sweep picked, and the score it picked it on

The chosen action is reported because a speedup that changes the decision is not a speedup.
Indices are only comparable between runs that used the same action list, i.e. within one branch.

Run with::

    python bench_greedy.py --grid grid_89.pkl --backend lightsim --workers 1,8 --json out.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from bench_branch import RESET_OPTIONS, expand_grid, grid_info, make_config, median_ms

from gymnasium import spaces

from pandapower_env.agents.benchmark_agents import GreedyAgent
from pandapower_env.environments.simulation_env import PPTopoGym


def time_greedy_step(config: dict, n_workers: int, repeat: int) -> dict:
    """Build a driver env and a greedy agent from one config, and time a single ``act()``.

    The agent owns a *separate* environment built from the same config, which is what
    ``state_from_info`` requires: it refuses to restore a state onto the instance that produced
    it. The driver env is reset to a fixed timestep so every branch sweeps the same grid state.

    :param config: the env config, already carrying the backend selection.
    :param n_workers: joblib workers for the sweep (1 runs in-process).
    :param repeat: timed repetitions to take the median over.
    :return: the median milliseconds, the chosen action and the number of candidates.
    """
    env = PPTopoGym(config)
    observation, _info = env.reset(options=RESET_OPTIONS)
    action_space = spaces.Discrete(len(config["action_space"]))
    agent = GreedyAgent(
        action_space=action_space,
        env_config=config,
        feedback_type="line_loadings",
        n_workers=n_workers,
    )
    info = env.state_to_info()

    chosen = int(agent.act(observation, info))
    elapsed = median_ms(lambda: agent.act(observation, info), repeat, warmup=1)
    return {
        "ms": elapsed,
        "chosen": chosen,
        "n_candidates": int(action_space.n),
        "per_action_ms": elapsed / int(action_space.n),
    }


def main() -> None:
    """Time a greedy step at each requested worker count and print / write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--backend", default="pandapower")
    parser.add_argument("--workers", default="1", help="comma-separated worker counts")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--actions-cache", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    with args.grid.open("rb") as handle:
        payload = pickle.load(handle)
    net, actions = expand_grid(payload, args.actions_cache)
    config = make_config(net, actions, payload["profiles"], nminus1=False)
    config["backend"] = args.backend

    build_start = time.perf_counter()
    PPTopoGym(config)
    build_ms = (time.perf_counter() - build_start) * 1000.0

    # Flat float results, so the same median-over-rounds reporting as bench_branch.py applies.
    results: dict[str, float] = {"env.build": build_ms}
    chosen: dict[str, int] = {}
    for n_workers in [int(w) for w in args.workers.split(",") if w.strip()]:
        record = time_greedy_step(config, n_workers, args.repeat)
        results[f"greedy.act.w{n_workers}"] = record["ms"]
        results[f"greedy.per_action.w{n_workers}"] = record["per_action_ms"]
        chosen[f"w{n_workers}"] = record["chosen"]
        print(  # noqa: T201
            f"{args.label:<22} w={n_workers:<3} {record['ms']:10.1f} ms  "
            f"({record['per_action_ms']:.3f} ms/action, {record['n_candidates']} candidates) "
            f"-> action {record['chosen']}",
        )

    payload_out = {
        "label": args.label, "backend": args.backend, "repeat": args.repeat,
        "chosen": chosen, "grid": grid_info(payload, net, actions), "results": results,
    }
    if args.json:
        args.json.write_text(json.dumps(payload_out, indent=2))


if __name__ == "__main__":
    main()
