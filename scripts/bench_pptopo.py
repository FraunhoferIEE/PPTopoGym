"""Benchmark the PPTopoGym build and hot paths, as a stable A/B baseline.

Run with::

    poetry run python scripts/bench_pptopo.py                    # all suites
    poetry run python scripts/bench_pptopo.py --suite build      # one suite
    poetry run python scripts/bench_pptopo.py --json out.json    # machine-readable

This exists because timings on this box swing several percent with load, so a single
run compared against a number stored in a file is not evidence. Every suite here is
deterministic in *what work it does*, reports the median of ``--repeat`` runs, and
prints a stable machine-readable key per measurement so two runs can be diffed.

The suites deliberately cover the whole "build a PPTopoGym" path the project cares
about, not just ``step``:

- ``build``   -- ``config_case30()`` broken into named stages, plus ``PPTopoGym(config)``.
                 This is where most wall-clock time in a fresh process goes.
- ``step``    -- ``reset`` / DoNothing step / topology step on a built env.
- ``obs``     -- ``create_observation`` alone, the per-step observation cost.
- ``greedy``  -- one ``GreedyAgent.act`` over the full action space (serial).

Nothing here is imported by ``pandapower_env``; the shipped package carries no
benchmark dependency.

:raises SystemExit: if an unknown suite name is passed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from pandapower_env.environments.simulation_env import PPTopoGym

REPO_ROOT = Path(__file__).resolve().parent.parent


def _median_ms(fn: Callable[[], Any], repeat: int, warmup: int = 1) -> float:
    """Run ``fn`` and return the median wall time in milliseconds.

    A warm-up pass is discarded first: the first call through any pandapower path pays
    option parsing, imports and allocator growth that never recur, and including it
    would make a fast steady state look slow.

    :param fn: Zero-argument callable to time.
    :param repeat: How many timed repetitions to take the median over.
    :param warmup: How many untimed calls to run first.
    :return: Median duration in milliseconds.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def bench_build(repeat: int) -> dict[str, float]:
    """Time each stage of ``config_case30()`` plus the ``PPTopoGym`` constructor.

    Stages are timed individually rather than as one block so a regression can be
    attributed to the stage that caused it. Because the stages mutate ``net`` in
    sequence, the whole chain is re-run per repetition.

    :param repeat: Repetitions to take the median over.
    :return: Mapping of ``build.<stage>`` to median milliseconds.
    """
    from pandapower.networks import case30

    from pandapower_env.action_space.action_space import (
        add_actions_substation_line_switching,
        verify_all_actions,
    )
    from pandapower_env.environments.simulation_env import PPTopoGym
    from pandapower_env.substation.create_double_busbar_substation import (
        create_all_double_busbar_substations,
    )
    from pandapower_env.toolbox.utils_profiles import (
        create_simbench_data_from_profiles,
        get_first_sb_profiles,
        get_orig_profiles,
    )
    from pandapower_env.toolbox.utils_scaling import ensure_no_zero_values, find_scaling_recursive

    stage_samples: dict[str, list[float]] = {}

    def record(name: str, start: float) -> None:
        stage_samples.setdefault(name, []).append((time.perf_counter() - start) * 1000.0)

    for _ in range(repeat):
        t = time.perf_counter()
        net = case30()
        record("build.case30_load", t)

        t = time.perf_counter()
        get_first_sb_profiles(net, 2)
        record("build.get_first_sb_profiles", t)

        t = time.perf_counter()
        ensure_no_zero_values(net)
        for key, df in net.profiles.items():
            net.profiles[key] = df.replace(0.0, 1.0)
        record("build.profile_cleanup", t)

        t = time.perf_counter()
        orig_profiles = get_orig_profiles(net)
        record("build.get_orig_profiles", t)

        t = time.perf_counter()
        find_scaling_recursive(
            net, init_scaling=1, orig_profiles=orig_profiles, max_percent=40, overloaded_lines=3,
        )
        record("build.find_scaling_recursive", t)

        t = time.perf_counter()
        create_simbench_data_from_profiles(net, orig_profiles)
        record("build.create_simbench_data", t)

        t = time.perf_counter()
        create_all_double_busbar_substations(net)
        record("build.create_substations", t)

        t = time.perf_counter()
        actions = add_actions_substation_line_switching(net)
        record("build.add_actions", t)

        t = time.perf_counter()
        actions = verify_all_actions(net, actions)
        record("build.verify_all_actions", t)

        for eltype in ("gen", "sgen", "load"):
            if hasattr(net[eltype], "scenario_scaling"):
                del net[eltype]["scenario_scaling"]

        config = {
            "net": net,
            "n_episodes": 366,
            "episode_length": 96,
            "action_space": actions,
            "nminus1": False,
        }
        t = time.perf_counter()
        PPTopoGym(config)
        record("build.PPTopoGym_ctor", t)

    results = {name: statistics.median(vals) for name, vals in stage_samples.items()}
    results["build.TOTAL"] = sum(results.values())
    return results


def _built_env() -> PPTopoGym:
    """Build one ``config_case30`` environment for the hot-path suites.

    :return: A constructed ``PPTopoGym``.
    """
    from pandapower_env.data.example_configs import config_case30
    from pandapower_env.environments.simulation_env import PPTopoGym

    return PPTopoGym(config_case30())


def bench_step(repeat: int) -> dict[str, float]:
    """Time ``reset``, a DoNothing step, and a topology step on a built env.

    Each measurement resets to a fixed profile index first, so the timing does not
    depend on which random scenario ``reset`` would otherwise pick.

    :param repeat: Repetitions to take the median over.
    :return: Mapping of ``step.<name>`` to median milliseconds.
    """
    env = _built_env()
    topology_action = int(env.df_actions.index[1])

    def do_reset() -> None:
        env.reset(options={"index": 0})

    def do_donothing() -> None:
        env.reset(options={"index": 0})
        env.step(0)

    def do_topology() -> None:
        env.reset(options={"index": 0})
        env.step(topology_action)

    reset_ms = _median_ms(do_reset, repeat)
    return {
        "step.reset": reset_ms,
        # Subtract the reset each combined measurement includes, so the reported number
        # is the step itself rather than reset+step.
        "step.donothing": _median_ms(do_donothing, repeat) - reset_ms,
        "step.topology": _median_ms(do_topology, repeat) - reset_ms,
    }


def bench_obs(repeat: int) -> dict[str, float]:
    """Time ``create_observation`` alone on a settled env.

    :param repeat: Repetitions to take the median over.
    :return: Mapping with ``obs.create_observation`` in median milliseconds.
    """
    env = _built_env()
    env.reset(options={"index": 0})
    env.step(0)
    return {"obs.create_observation": _median_ms(env.create_observation, repeat)}


def bench_greedy(repeat: int) -> dict[str, float]:
    """Time one serial ``GreedyAgent.act`` sweep over the whole action space.

    Serial (``n_workers=1``) on purpose: the parallel path's speed depends on machine
    load and worker count, which makes it a poor regression signal. What is measured
    here is the per-action simulation cost the parallel path also pays.

    :param repeat: Repetitions to take the median over.
    :return: Mapping with ``greedy.act_full_sweep`` in median milliseconds.
    """
    from pandapower_env.agents.benchmark_agents import GreedyAgent

    env = _built_env()
    agent = GreedyAgent(env.action_space, env.orig_config, n_workers=1)
    observation, info = env.reset(options={"index": 0})

    def do_act() -> None:
        agent.act(observation, info)

    # Greedy is expensive; cap repetitions so the suite stays usable.
    return {"greedy.act_full_sweep": _median_ms(do_act, max(1, repeat // 2))}


SUITES: dict[str, Callable[[int], dict[str, float]]] = {
    "build": bench_build,
    "step": bench_step,
    "obs": bench_obs,
    "greedy": bench_greedy,
}


def main() -> None:
    """Parse arguments, run the requested suites, and print/write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        choices=[*SUITES, "all"],
        help="Suite(s) to run; repeatable. Default: all except greedy.",
    )
    parser.add_argument("--repeat", type=int, default=5, help="Timed repetitions per measurement.")
    parser.add_argument("--json", type=Path, help="Also write results as JSON here.")
    parser.add_argument("--label", default="", help="Free-text label recorded in the JSON output.")
    args = parser.parse_args()

    selected = args.suite or ["build", "step", "obs"]
    if "all" in selected:
        selected = list(SUITES)

    results: dict[str, float] = {}
    for name in selected:
        if name not in SUITES:
            sys.exit(f"unknown suite: {name}")
        print(f"running suite: {name} ...", flush=True)  # noqa: T201
        results.update(SUITES[name](args.repeat))

    print(f"\n{'measurement':<40} {'median ms':>12}")  # noqa: T201
    print("-" * 54)  # noqa: T201
    for key in sorted(results, key=lambda k: -results[k]):
        print(f"{key:<40} {results[key]:12.2f}")  # noqa: T201

    if args.json:
        args.json.write_text(
            json.dumps({"label": args.label, "repeat": args.repeat, "results": results}, indent=2),
        )
        print(f"\nwrote {args.json}")  # noqa: T201


if __name__ == "__main__":
    main()
