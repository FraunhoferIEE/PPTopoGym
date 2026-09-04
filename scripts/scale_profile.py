"""Profile how the PPTopoGym build and hot paths scale with grid size.

Run with::

    poetry run python scripts/scale_profile.py                       # all grids
    poetry run python scripts/scale_profile.py --grid case118        # one grid
    poetry run python scripts/scale_profile.py --json out.json       # machine-readable

Everything measured so far in ``profiling/PERF_LEDGER.md`` is case30 (30 buses). Real
transmission grids are 1000s of buses, and the open question is not "how fast is a step"
but **which stage stops being tractable first**. The two candidates are structural, not
constant-factor:

- ``add_actions_substation_line_switching`` + ``verify_all_actions`` -- the action space
  is combinatorial in substation count, and every candidate action costs a power flow to
  verify. If this is superlinear it caps grid size regardless of step speed.
- N-1 -- the contingency sweep is one power flow per line, so it is inherently quadratic
  in the grid (lines x cost-per-solve, and cost-per-solve itself grows).

Each grid runs in its **own subprocess** (``--child``), because per-process caches -- the
numba JIT of pandapower's kernels, the Simbench profile cache -- would otherwise let a
later grid inherit an earlier grid's warm-up and fabricate a favourable scaling curve.
See "Measurement traps" in the ledger.

Nothing here is imported by ``pandapower_env``; the shipped package carries no benchmark
dependency.

:raises SystemExit: if an unknown grid name is passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import pandapower as pp

REPO_ROOT = Path(__file__).resolve().parent.parent

# Grid builders, smallest first. Each returns a fresh pandapower net.
GRID_BUILDERS: dict[str, str] = {
    "case30": "case30",
    "case89pegase": "case89pegase",
    "case118": "case118",
    "case300": "case300",
}

# Scaling-search parameters per grid. case30/case89 mirror the shipped example configs so
# the numbers stay comparable to the ledger; the new grids reuse the case89 settings,
# which are the ones tuned for a larger net.
SCALING_PARAMS: dict[str, dict[str, int]] = {
    "case30": {"init_scaling": 1, "max_percent": 40, "overloaded_lines": 3},
    "case89pegase": {"init_scaling": 100, "max_percent": 80, "overloaded_lines": 4},
    "case118": {"init_scaling": 100, "max_percent": 80, "overloaded_lines": 4},
    "case300": {"init_scaling": 100, "max_percent": 80, "overloaded_lines": 4},
}


@contextmanager
def _stage(results: dict[str, float], name: str) -> Generator[None, None, None]:
    """Time one named build stage into ``results`` (milliseconds).

    :param results: Dict the elapsed time is recorded into, keyed by ``name``.
    :param name: Stage label.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        results[name] = (time.perf_counter() - start) * 1000.0


def _median_ms(fn: Callable[[], Any], repeat: int, warmup: int = 1) -> float:
    """Run ``fn`` and return the median wall time in milliseconds.

    A warm-up pass is discarded: the first call through any pandapower path pays option
    parsing and allocator growth that never recur.

    :param fn: Zero-argument callable to time.
    :param repeat: Timed repetitions to take the median over.
    :param warmup: Untimed calls to run first.
    :return: Median duration in milliseconds.
    """
    import statistics

    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def _build_net(grid: str, stages: dict[str, float]) -> tuple[pp.pandapowerNet, list]:
    """Run the full config pipeline for ``grid``, timing each stage.

    Mirrors ``data/example_configs.config_case89`` so the measured pipeline is the one the
    package actually ships, not a reconstruction.

    :param grid: Key into :data:`GRID_BUILDERS`.
    :param stages: Dict that per-stage timings are recorded into.
    :return: The built net and its verified action list.
    """
    import pandapower.networks as pn

    from pandapower_env.action_space.action_space import (
        add_actions_substation_line_switching,
        verify_all_actions,
    )
    from pandapower_env.substation.create_double_busbar_substation import (
        create_all_double_busbar_substations,
    )
    from pandapower_env.toolbox.utils_profiles import (
        create_simbench_data_from_profiles,
        get_first_sb_profiles,
        get_orig_profiles,
    )
    from pandapower_env.toolbox.utils_scaling import ensure_no_zero_values, find_scaling_recursive

    params = SCALING_PARAMS[grid]

    with _stage(stages, "net_load"):
        net = getattr(pn, GRID_BUILDERS[grid])()

    with _stage(stages, "get_first_sb_profiles"):
        get_first_sb_profiles(net, 2)
        ensure_no_zero_values(net)
        for key, df in net.profiles.items():
            net.profiles[key] = df.replace(0.0, 1.0)

    orig_profiles = get_orig_profiles(net)

    with _stage(stages, "find_scaling_recursive"):
        find_scaling_recursive(
            net,
            init_scaling=params["init_scaling"],
            orig_profiles=orig_profiles,
            max_percent=params["max_percent"],
            overloaded_lines=params["overloaded_lines"],
        )

    with _stage(stages, "create_simbench_data"):
        create_simbench_data_from_profiles(net, orig_profiles)

    with _stage(stages, "create_double_busbars"):
        create_all_double_busbar_substations(net)

    with _stage(stages, "generate_actions"):
        actions = add_actions_substation_line_switching(net)
    stages["n_actions_generated"] = float(len(actions))

    with _stage(stages, "verify_all_actions"):
        actions = verify_all_actions(net, actions)
    stages["n_actions_verified"] = float(len(actions))

    for eltype in ("gen", "sgen", "load"):
        if hasattr(net[eltype], "scenario_scaling"):
            del net[eltype]["scenario_scaling"]

    return net, actions


def _solved_bus_count(net: pp.pandapowerNet) -> int:
    """Count the distinct buses the power flow actually solves for.

    Reads the ``_pd2ppc_lookups`` mapping left behind by the last power flow, which is
    where pandapower records the switch fusion. Returns 0 if no power flow has run.

    :param net: A net that has had at least one power flow run on it.
    :return: Number of distinct ppc nodes, or 0 if unavailable.
    """
    import numpy as np

    lookups = net.get("_pd2ppc_lookups") or {}
    bus_lookup = lookups.get("bus")
    if bus_lookup is None:
        return 0
    return int(len(np.unique(bus_lookup[bus_lookup >= 0])))


def _measure_grid(grid: str, repeat: int, *, with_n1: bool) -> dict[str, Any]:
    """Build ``grid`` end to end and measure build stages, step and N-1 costs.

    :param grid: Key into :data:`GRID_BUILDERS`.
    :param repeat: Repetitions for the steady-state (step / observation) measurements.
    :param with_n1: Whether to time a serial N-1 sweep, which is the slowest measurement.
    :return: A record of shape/timing measurements for this grid.
    """
    from pandapower_env.environments.simulation_env import PPTopoGym

    stages: dict[str, float] = {}
    net, actions = _build_net(grid, stages)

    config = {
        "net": net,
        "n_episodes": 366,
        "episode_length": 96,
        "action_space": actions,
        "nminus1": False,
    }

    with _stage(stages, "PPTopoGym_init"):
        env = PPTopoGym(config)

    # Reset first: _solved_bus_count reads the lookups a power flow leaves behind, and
    # PPTopoGym.reset runs one.
    env.reset(seed=0, options={"index": 0})

    shape = {
        "n_bus": int(len(net.bus)),
        # The solved system size, which is what actually drives power-flow cost. Double-
        # busbar substations add many auxiliary buses that are flagged in_service but fuse
        # away through closed switches, so in_service is a misleading proxy: on case30 all
        # 93 bus rows are in_service while the ppc has only 30 nodes.
        # env.net, not net: PPTopoGym deep-copies the net it is given, and only the env's
        # copy has had a power flow run on it.
        "n_ppc_bus": _solved_bus_count(env.net),
        "n_line": int(len(net.line)),
        "n_trafo": int(len(net.trafo)),
        "n_switch": int(len(net.switch)),
        "n_substations": int(len(net.multi_bb_substation)) if "multi_bb_substation" in net else 0,
        "n_actions": int(env.action_space.n),
    }

    env.reset(seed=0, options={"index": 0})
    hot: dict[str, float] = {}
    hot["reset_ms"] = _median_ms(lambda: env.reset(seed=0, options={"index": 0}), repeat)
    hot["step_donothing_ms"] = _median_ms(lambda: env.step(0), repeat)
    hot["create_observation_ms"] = _median_ms(env.create_observation, repeat)

    if with_n1:
        from pandapower_env.toolbox.utils import run_nminus1_powerflow

        env.reset(seed=0, options={"index": 0})
        hot["nminus1_serial_ms"] = _median_ms(
            lambda: run_nminus1_powerflow(env.net), repeat=1, warmup=1,
        )

    return {"grid": grid, "shape": shape, "stages": stages, "hot": hot}


def _run_child(grid: str, repeat: int, *, with_n1: bool) -> dict[str, Any]:
    """Measure one grid in a fresh subprocess and return its parsed record.

    Isolation is the point: per-process caches (numba JIT, the Simbench profile cache)
    would otherwise let a later grid inherit an earlier grid's warm-up.

    :param grid: Grid name to measure.
    :param repeat: Repetitions passed through to the child.
    :param with_n1: Whether the child should time N-1.
    :return: The child's measurement record, or an ``error`` record if it failed.
    """
    cmd = [
        sys.executable, str(Path(__file__).resolve()),
        "--child", grid, "--repeat", str(repeat),
    ]
    if with_n1:
        cmd.append("--n1")
    proc = subprocess.run(  # noqa: S603
        cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False,
    )
    marker = "===RESULT==="
    if proc.returncode != 0 or marker not in proc.stdout:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {
            "grid": grid,
            "error": tail[-1] if tail else f"exit {proc.returncode}",
            "returncode": proc.returncode,
        }
    return json.loads(proc.stdout.split(marker, 1)[1].strip())


def _print_report(records: list[dict[str, Any]]) -> None:
    """Print the scaling table: shape, build stages, and hot-path costs per grid.

    :param records: One measurement record per grid, smallest grid first.
    """
    ok = [r for r in records if "error" not in r]
    failed = [r for r in records if "error" in r]

    if ok:
        print("\n=== grid shape ===")  # noqa: T201
        cols = ["n_bus", "n_ppc_bus", "n_line", "n_substations", "n_actions"]
        print(f"{'grid':16s}" + "".join(f"{c:>18s}" for c in cols))  # noqa: T201
        for r in ok:
            print(f"{r['grid']:16s}" + "".join(f"{r['shape'][c]:>18d}" for c in cols))  # noqa: T201

        print("\n=== build stages (ms, cold process) ===")  # noqa: T201
        stage_names = [
            "net_load", "get_first_sb_profiles", "find_scaling_recursive",
            "create_simbench_data", "create_double_busbars", "generate_actions",
            "verify_all_actions", "PPTopoGym_init",
        ]
        print(f"{'stage':26s}" + "".join(f"{r['grid']:>16s}" for r in ok))  # noqa: T201
        for name in stage_names:
            row = "".join(f"{r['stages'].get(name, float('nan')):>16.1f}" for r in ok)
            print(f"{name:26s}{row}")  # noqa: T201
        total = "".join(
            f"{sum(v for k, v in r['stages'].items() if not k.startswith('n_')):>16.1f}"
            for r in ok
        )
        print(f"{'TOTAL':26s}{total}")  # noqa: T201

        print("\n=== action space ===")  # noqa: T201
        print(f"{'metric':26s}" + "".join(f"{r['grid']:>16s}" for r in ok))  # noqa: T201
        for name in ("n_actions_generated", "n_actions_verified"):
            row = "".join(f"{r['stages'].get(name, float('nan')):>16.0f}" for r in ok)
            print(f"{name:26s}{row}")  # noqa: T201

        print("\n=== hot paths (ms) ===")  # noqa: T201
        hot_names = ["reset_ms", "step_donothing_ms", "create_observation_ms", "nminus1_serial_ms"]
        print(f"{'metric':26s}" + "".join(f"{r['grid']:>16s}" for r in ok))  # noqa: T201
        for name in hot_names:
            if not any(name in r["hot"] for r in ok):
                continue
            row = "".join(f"{r['hot'].get(name, float('nan')):>16.2f}" for r in ok)
            print(f"{name:26s}{row}")  # noqa: T201

    for r in failed:
        print(f"\n!! {r['grid']}: FAILED -- {r['error']}")  # noqa: T201


def main() -> None:
    """Parse arguments and run the scaling profile, in-process or as a child."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", action="append", choices=list(GRID_BUILDERS))
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--n1", action="store_true", help="also time a serial N-1 sweep")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--child", type=str, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        import logging
        import warnings

        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)
        record = _measure_grid(args.child, args.repeat, with_n1=args.n1)
        print("===RESULT===")  # noqa: T201
        print(json.dumps(record))  # noqa: T201
        return

    grids = args.grid or list(GRID_BUILDERS)
    records = []
    for grid in grids:
        print(f"[scale_profile] measuring {grid} in a fresh process ...", flush=True)  # noqa: T201
        records.append(_run_child(grid, args.repeat, with_n1=args.n1))

    _print_report(records)

    if args.json:
        Path(args.json).write_text(json.dumps(records, indent=2))
        print(f"\nwrote {args.json}")  # noqa: T201


if __name__ == "__main__":
    main()
