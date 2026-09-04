"""Run greedy agents over many reset-and-play episodes, checking they stay deterministic.

Run with::

    python scripts/soak_greedy.py                          # case14, quick
    python scripts/soak_greedy.py --grids case14,case30 --episodes 5
    python scripts/soak_greedy.py --baseline soak.json     # write the action sequences
    python scripts/soak_greedy.py --check soak.json        # compare against them

Why this exists: the unit tests and the golden record both exercise short, fresh-state
paths. The failure modes a refactor of the caches, the worker payloads or the
save/restore machinery actually introduces -- state leaking across ``reset``, a stale
cached array surviving a topology change, a loky worker holding a net from a previous
``act()`` -- only appear under repetition. So this plays whole episodes, repeatedly, and
asserts three things that must hold no matter how the internals are rearranged:

1. **Same seed, same actions.** Two runs of the identical configuration must produce the
   identical action sequence.
2. **Serial == parallel.** ``n_workers=1`` and ``n_workers>1`` must agree exactly; a
   difference here is worker state leakage, which no single-process test can see.
3. **Memory is flat.** Resident set size per episode must not grow -- the profile-table
   cache is a ``WeakValueDictionary`` and the worker net caches are process globals, so a
   change to their keys can silently turn them into leaks.

:raises SystemExit: if determinism, serial/parallel agreement or a stored baseline fails.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

PAGE_SIZE = 4096


def rss_mb() -> float:
    """Resident set size of this process in MiB, read from ``/proc`` (Linux only)."""
    with Path("/proc/self/statm").open() as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * PAGE_SIZE / 1024 / 1024


def play_episodes(
    grid: str,
    config: dict,
    n_workers: int,
    feedback_type: str,
    episodes: int,
    steps: int,
) -> tuple[list[int], list[float], float]:
    """Play ``episodes`` greedy episodes and report the actions, per-episode RSS and rate.

    The agent owns its own environment (``BaseAgent`` builds one from the config), so the
    driver environment here is a genuinely different instance -- which is what lets the
    agent be handed ``state_to_info()`` and exercise the ``state_from_info`` action-log
    replay, the path the simulation API changes touch.

    :param grid: grid name, used only for log lines.
    :param config: the env config; deep-copied per agent so the two never share a net.
    :param n_workers: joblib workers for the greedy sweep (1 = serial).
    :param feedback_type: greedy scoring criterion, e.g. ``line_loadings`` or ``nminus1``.
    :param episodes: how many episodes to play.
    :param steps: steps per episode.
    :return: (flat action sequence, RSS in MiB after each episode, episodes per second).
    """
    from pandapower_env.agents.benchmark_agents import GreedyAgent
    from pandapower_env.environments.simulation_env import PPTopoGym

    action_space = spaces.Discrete(len(config["action_space"]))
    agent = GreedyAgent(
        action_space=action_space,
        env_config=copy.deepcopy(config),
        feedback_type=feedback_type,
        n_workers=n_workers,
        seed=1234,
    )
    env = PPTopoGym(copy.deepcopy(config))

    actions: list[int] = []
    rss_trace: list[float] = []
    start = time.perf_counter()
    # Clamp to the timeseries: the small case14 fixture carries only one episode's worth of
    # profile rows, so a plain episode*episode_length walks off the end.
    last_start = max(0, env.n_total_timesteps - steps - 1)
    for episode in range(episodes):
        obs, _ = env.reset(
            options={"index": min(episode * config["episode_length"], last_start)},
        )
        for _ in range(steps):
            action = int(agent.act(obs, env.state_to_info()))
            actions.append(action)
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                break
        rss_trace.append(rss_mb())
        print(f"    {grid} w={n_workers} {feedback_type} ep{episode}: rss {rss_trace[-1]:.1f} MiB", flush=True)
    elapsed = time.perf_counter() - start
    return actions, rss_trace, episodes / elapsed


def soak_grid(grid: str, episodes: int, steps: int, feedbacks: list[str]) -> dict[str, list[int]]:
    """Soak one grid across worker counts and feedback types; returns the action sequences.

    :raises SystemExit: on a determinism or serial/parallel mismatch, or on RSS growth.
    """
    from golden_record import build_config

    config = build_config(grid)
    sequences: dict[str, list[int]] = {}

    for feedback in feedbacks:
        per_workers: dict[int, list[int]] = {}
        for n_workers in (1, 2):
            actions, rss_trace, rate = play_episodes(
                grid, config, n_workers, feedback, episodes, steps,
            )
            per_workers[n_workers] = actions
            key = f"{grid}/{feedback}/workers{n_workers}"
            sequences[key] = actions
            print(f"  {key}: {rate:.2f} episodes/s, {len(actions)} actions", flush=True)

            # 1. same seed, same actions
            repeat, _, _ = play_episodes(grid, config, n_workers, feedback, episodes, steps)
            if repeat != actions:
                sys.exit(f"NON-DETERMINISTIC: {key} differed between two identical runs")

            # 3. flat memory
            if len(rss_trace) >= 3 and rss_trace[-1] > rss_trace[1] * 1.5:
                sys.exit(
                    f"MEMORY GROWTH: {key} rss {rss_trace[1]:.1f} -> {rss_trace[-1]:.1f} MiB",
                )

        # 2. serial == parallel
        if per_workers[1] != per_workers[2]:
            sys.exit(f"SERIAL != PARALLEL for {grid}/{feedback}: worker state leaked")
    return sequences


def main() -> None:
    """Parse arguments and run the soak, optionally against a stored action baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", default="case14")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--feedbacks", default="line_loadings")
    parser.add_argument("--baseline", type=Path, help="write the action sequences here")
    parser.add_argument("--check", type=Path, help="compare the action sequences against this")
    args = parser.parse_args()

    grids = [g.strip() for g in args.grids.split(",") if g.strip()]
    feedbacks = [f.strip() for f in args.feedbacks.split(",") if f.strip()]

    sequences: dict[str, list[int]] = {}
    for grid in grids:
        print(f"soaking {grid} ...", flush=True)
        sequences.update(soak_grid(grid, args.episodes, args.steps, feedbacks))

    if args.baseline:
        args.baseline.write_text(json.dumps(sequences, indent=1, sort_keys=True))
        print(f"wrote {args.baseline}")
    if args.check:
        stored = json.loads(args.check.read_text())
        if stored != sequences:
            differing = [k for k in stored if stored.get(k) != sequences.get(k)]
            sys.exit(f"ACTION SEQUENCES CHANGED: {differing}")
        print("action sequences match the baseline")
    print("soak OK")


if __name__ == "__main__":
    main()
