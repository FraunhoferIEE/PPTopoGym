"""Time the PPTopoGym hot paths for one grid, on whichever branch is on ``sys.path``.

The physical grid and the timeseries come from a pickle produced by ``build_grids.py``, so both
branches solve the identical network. The double-busbar expansion and the action list are built
here, by the branch under test, because the two ``multi_bb_substation`` schemas are not
interchangeable -- that modelling is part of what differs between the branches.

Two modes:

``--probe N``
    Report, for the first ``N`` substation actions, whether the step converges and what reward it
    scores. ``run_comparison.py`` intersects the two branches' probes to choose one switching
    action that is valid on *both*, because they do not agree on every bitset (on case30, actions
    2-4 converge on ``develop`` and island the grid on ``develop_muzero``).

``--action K``
    Time the hot paths, applying action ``K`` as the switching action. Measured, all as medians
    over ``--repeat`` runs:

        step.reset          reset(options={"index": 0})           (baseline for the step numbers)
        step.donothing      reset + step(0)     minus step.reset
        step.switching      reset + step(K)     minus step.reset
        nminus1.reset       reset on an env built with nminus1=True
        nminus1.donothing   reset + step(0) on that env, minus nminus1.reset
        nminus1.powerflow   one N-1 sweep over a solved net, through the env's configured backend
"""

from __future__ import annotations

import argparse
import json
import pickle
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import pandapower_env
from pandapower_env.action_space.action_space import add_actions_substation_line_switching
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.substation.create_double_busbar_substation import (
    create_all_double_busbar_substations,
)
from pandapower_env.toolbox.utils import run_powerflow

RESET_OPTIONS = {"index": 0}

# Action-space cap. Step cost does not depend on how many actions exist -- only which one is
# applied -- but case89 generates >150k actions, and building that DataFrame twice per run (once
# per env) would dominate the benchmark without saying anything about the step path.
MAX_ACTIONS = 500


def median_ms(fn: Callable[[], Any], repeat: int, warmup: int = 2) -> float:
    """Run ``fn`` repeatedly and return the median wall time in milliseconds.

    The warm-up passes are discarded: the first calls through pandapower pay option parsing, numba
    JIT and allocator growth that never recur, and including them would swamp the steady state this
    benchmark is about.

    :param fn: zero-argument callable to time.
    :param repeat: timed repetitions to take the median over.
    :param warmup: untimed calls to run first.
    :return: median duration in milliseconds.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def expand_grid(payload: dict, cache_path: Path | None) -> tuple:
    """Turn the shipped base net into a topology-control grid using the branch's own code.

    Generating case89's action list takes over a minute, and every process here would repeat it, so
    the (capped) list is cached to ``cache_path``. What is cached is the list as it comes out of
    ``add_actions_substation_line_switching`` -- before any ``create_actions_df`` call, which
    populates ``open_switches`` in place and on this branch appends rather than resets.

    :param payload: the unpickled grid payload.
    :param cache_path: where to cache the generated action list, or ``None`` to skip caching.
    :return: ``(net, action_list)``.
    """
    net = payload["net"]
    create_all_double_busbar_substations(net)
    if cache_path is not None and cache_path.exists():
        with cache_path.open("rb") as handle:
            return net, pickle.load(handle)

    actions = add_actions_substation_line_switching(net)[:MAX_ACTIONS]
    if cache_path is not None:
        with cache_path.open("wb") as handle:
            pickle.dump(actions, handle)
    return net, actions


def make_config(net, actions: list, profiles: dict, *, nminus1: bool, backend: str = "pandapower") -> dict:
    """Assemble the env config.

    The timeseries is passed as ``env_config["profiles"]`` because that is the one ingestion route
    both branches implement identically (``develop`` requires the key outright).

    :param net: the expanded net.
    :param actions: the (capped) action list.
    :param profiles: absolute per-timestep injections, keyed by element and column.
    :param nminus1: whether the env should run N-1 contingencies on every power flow.
    :param backend: which solver ``run_pf`` should use. Branches that predate the option simply
        never read the key, so passing it is safe on both sides of the comparison.
    :return: an env config dict.
    """
    return {
        "net": net,
        "profiles": profiles,
        "n_episodes": 366,
        "episode_length": 96,
        "action_space": actions,
        "nminus1": nminus1,
        "backend": backend,
    }


def probe_actions(env, limit: int) -> list[dict]:
    """Report convergence and reward for the first ``limit`` real substation actions.

    A crashed step returns early with ``worst_reward``, so timing one would measure the failure
    path rather than a topology change; and the two branches disagree about which bitsets are
    survivable. The reward is reported too, so the caller can check that an action both branches
    accept actually puts them in the same electrical state.

    :param env: a built ``PPTopoGym``.
    :param limit: how many substation actions to try.
    :return: one record per probed action.
    """
    records: list[dict] = []
    for idx in env.df_actions.index:
        if len(records) >= limit:
            break
        if idx == 0 or not len(env.df_actions.loc[idx, "open_switches"]):
            continue
        env.reset(options=RESET_OPTIONS)
        _, reward, *_ = env.step(int(idx))
        records.append({
            "action": int(idx),
            # bool() wraps the whole expression: `and` yields the numpy scalar on the right,
            # which json.dumps refuses.
            "converged": bool(env.net.converged and reward != env.worst_reward),
            "reward": float(reward),
        })
    return records


def bench_steps(env, switching_action: int, repeat: int) -> dict[str, float]:
    """Time reset, a DoNothing step and a switching step, without N-1.

    :param env: a built ``PPTopoGym`` with ``nminus1=False``.
    :param switching_action: index of the action to apply as the switching step.
    :param repeat: timed repetitions per measurement.
    :return: mapping of measurement key to median milliseconds.
    """
    reset_ms = median_ms(lambda: env.reset(options=RESET_OPTIONS), repeat)

    def donothing() -> None:
        env.reset(options=RESET_OPTIONS)
        env.step(0)

    def switching() -> None:
        env.reset(options=RESET_OPTIONS)
        env.step(switching_action)

    return {
        "step.reset": reset_ms,
        "step.donothing": median_ms(donothing, repeat) - reset_ms,
        "step.switching": median_ms(switching, repeat) - reset_ms,
    }


def bench_nminus1(net, actions: list, profiles: dict, repeat: int, backend: str) -> dict[str, float]:
    """Time an N-1 enabled env's reset/step and a bare contingency sweep.

    The bare sweep is measured on top of a solved N-0 state, which is what the env always hands to
    the contingency analysis.

    :param net: the expanded net.
    :param actions: the (capped) action list.
    :param profiles: the shipped absolute profiles.
    :param repeat: timed repetitions per measurement.
    :return: mapping of measurement key to median milliseconds.
    """
    env = PPTopoGym(make_config(net, actions, profiles, nminus1=True, backend=backend))
    reset_ms = median_ms(lambda: env.reset(options=RESET_OPTIONS), repeat)

    def donothing() -> None:
        env.reset(options=RESET_OPTIONS)
        env.step(0)

    donothing_ms = median_ms(donothing, repeat) - reset_ms

    env.reset(options=RESET_OPTIONS)
    run_powerflow(env.net)

    # Timed through the env, not through run_nminus1_powerflow directly: the sweep is what the
    # backend option now selects, so calling the pandapower entry point would quietly measure the
    # same solver for every variant. Branches without the option dispatch to the same place.
    return {
        "nminus1.reset": reset_ms,
        "nminus1.donothing": donothing_ms,
        "nminus1.powerflow": median_ms(env.run_pf, repeat),
    }


def grid_info(payload: dict, net, actions: list) -> dict:
    """Describe the grid as this branch expanded it, so the caller can check the two sides match.

    :param payload: the unpickled grid payload.
    :param net: the expanded net.
    :param actions: the (capped) action list.
    :return: a flat dict of integer grid statistics.
    """
    return {
        **{k: v for k, v in payload.items() if isinstance(v, int)},
        "n_substation": len(net.multi_bb_substation),
        "n_switch": len(net.switch),
        "n_bus_expanded": len(net.bus),
        "n_actions_used": len(actions),
    }


def main() -> None:
    """Probe or benchmark one grid and print / write the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--label", default="")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--actions-cache", type=Path)
    parser.add_argument("--probe", type=int, help="probe this many substation actions and exit")
    parser.add_argument("--action", type=int, help="action index to time as the switching step")
    parser.add_argument("--backend", default="pandapower")
    args = parser.parse_args()

    with args.grid.open("rb") as handle:
        payload = pickle.load(handle)

    net, actions = expand_grid(payload, args.actions_cache)
    profiles = payload["profiles"]
    env = PPTopoGym(make_config(net, actions, profiles, nminus1=False, backend=args.backend))
    info = grid_info(payload, net, actions)

    if args.probe:
        result = {"label": args.label, "grid": info, "probe": probe_actions(env, args.probe)}
    else:
        if args.action is None:
            parser.error("--action is required unless --probe is given")
        results = bench_steps(env, args.action, args.repeat)
        results.update(bench_nminus1(net, actions, profiles, args.repeat, args.backend))
        result = {"label": args.label, "grid": info, "repeat": args.repeat,
                  "backend": args.backend,
                  "switching_action": args.action, "results": results}
        for key, value in results.items():
            print(f"{key:<24} {value:10.3f} ms")  # noqa: T201

    print(f"# {args.label} {args.grid.name} package={pandapower_env.__file__}")  # noqa: T201
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
