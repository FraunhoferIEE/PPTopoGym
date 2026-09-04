"""Run a benchmark for every grid on every variant, alternating, and report the deltas.

``--bench step`` drives ``bench_branch.py`` (reset / step / N-1); ``--bench greedy`` drives
``bench_greedy.py`` (one full greedy sweep). Everything below -- isolated interpreters, rotated
order, median over rounds -- applies to both.

A *variant* is a checkout paired with a solver backend, written ``label=path`` or
``label=path:backend`` -- so ``develop`` and ``refactor`` and ``refactor+ls2g`` are three
variants, the last two sharing a checkout. The first variant given is the baseline everything
else is quoted against.

Each (variant, grid) measurement gets its **own interpreter**: pandapower_env carries per-process
caches (the Simbench profile cache, the shared profile tables, pandapower's parsed options) that
would otherwise leak from whichever side ran first into the second and fabricate a speedup.

In ``step`` mode, before timing anything, every variant is probed for which substation actions converge,
and the switching step is timed on the lowest-numbered action that converges on *all* of them and
scores the *same* reward on all -- i.e. one that leaves them in the same electrical state.
Without that check the comparison silently times a busbar split on one side against a crash-and-
return-early on the other.

The variant order is rotated per round and the median across rounds is reported, so a slow patch
of machine load lands on every side rather than on one.

Run with::

    python run_comparison.py --grids DIR --variant develop=PATH_TO_DEVELOP \\
        --variant refactor=. --variant refactor+ls2g=.:lightsim --rounds 3 --json OUT.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Which benchmark script each --bench mode drives. Both take the same --grid / --label / --json /
# --actions-cache / --backend interface and both report a flat {name: milliseconds} results dict,
# so the probing, alternation and median-over-rounds below serve either.
BENCHES = {
    "step": Path(__file__).resolve().parent / "bench_branch.py",
    "greedy": Path(__file__).resolve().parent / "bench_greedy.py",
}

# Deltas below this are reported as noise; timings on this box move several percent with load.
NOISE_THRESHOLD_PCT = 5.0

# Rewards are compared with a tolerance because the variants reach them through different float
# paths (a different solver, in the lightsim case); beyond this they are not in the same state.
REWARD_TOLERANCE = 1e-6


def parse_variant(spec: str) -> tuple[str, Path, str]:
    """Parse a ``label=path`` or ``label=path:backend`` variant specification.

    :param spec: the command-line string.
    :return: (label, package root, backend name).
    :raises SystemExit: if the specification has no ``=``.
    """
    if "=" not in spec:
        sys.exit(f"variant {spec!r} must look like label=path or label=path:backend")
    label, _, location = spec.partition("=")
    path, _, backend = location.partition(":")
    return label, Path(path).resolve(), backend or "pandapower"


def run_bench(bench: Path, package_root: Path, grid_path: Path, label: str, extra: list[str],
              cache_dir: Path) -> dict:
    """Run ``bench_branch.py`` in a fresh interpreter importing ``pandapower_env`` from one branch.

    ``PYTHONPATH`` and the working directory are both set to ``package_root``: ``sys.path[0]`` is
    the *script's* directory, so without ``PYTHONPATH`` the checkout the venv was installed from
    silently wins the ``pandapower_env`` import and both sides measure the same code.

    :param bench: the benchmark script to run.
    :param package_root: repo root (or worktree) whose ``pandapower_env`` should be imported.
    :param grid_path: pickle produced by ``build_grids.py``.
    :param label: variant label, also used to key the action cache.
    :param extra: extra arguments (``--probe N`` or ``--action K --repeat R``, plus ``--backend``).
    :param cache_dir: directory for the per-variant generated action lists.
    :return: the parsed JSON payload.
    :raises SystemExit: if the benchmark subprocess fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out_path = Path(handle.name)
    cache_path = cache_dir / f"actions_{label}_{grid_path.stem}.pkl"
    cmd = [sys.executable, str(bench), "--grid", str(grid_path), "--label", label,
           "--json", str(out_path), "--actions-cache", str(cache_path), *extra]
    result = subprocess.run(cmd, cwd=package_root, env={**os.environ, "PYTHONPATH": str(package_root)},
                            capture_output=True, text=True, check=False)
    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        sys.exit(f"benchmark failed for {label} on {grid_path.name}:\n{result.stdout}\n{result.stderr}")
    payload = json.loads(out_path.read_text())
    out_path.unlink(missing_ok=True)
    return payload


def choose_common_action(probes: dict[str, list[dict]]) -> int:
    """Pick the lowest action index that converges on every variant with the same reward.

    :param probes: variant label -> the probe records that variant reported.
    :return: the chosen action index.
    :raises SystemExit: if the variants share no such action.
    """
    by_variant = {label: {r["action"]: r for r in records} for label, records in probes.items()}
    labels = list(by_variant)
    shared = sorted(set.intersection(*(set(records) for records in by_variant.values())))
    for action in shared:
        records = [by_variant[label][action] for label in labels]
        if not all(r["converged"] for r in records):
            continue
        rewards = [r["reward"] for r in records]
        if max(rewards) - min(rewards) <= REWARD_TOLERANCE:
            return action
    sys.exit("no action converges on every variant with a matching reward")


def median_over_rounds(payloads: list[dict]) -> dict[str, float]:
    """Collapse several rounds of one (variant, grid) measurement into per-key medians.

    :param payloads: the per-round JSON payloads.
    :return: mapping of measurement key to median milliseconds.
    """
    keys = payloads[0]["results"].keys()
    return {key: statistics.median([p["results"][key] for p in payloads]) for key in keys}


def print_grid_report(stem: str, action: int, infos: dict[str, dict],
                      medians: dict[str, dict[str, float]]) -> None:
    """Print the per-grid comparison table, quoting every variant against the first.

    :param stem: grid file stem, e.g. ``grid_30``.
    :param action: the switching action every variant timed.
    :param infos: variant label -> the grid statistics that variant reported.
    :param medians: variant label -> measurement key -> median milliseconds.
    """
    labels = list(medians)
    baseline_label = labels[0]
    baseline_info = infos[baseline_label]
    mismatch = {
        label: {key: (baseline_info[key], info[key])
                for key in ("n_substation", "n_switch", "n_bus_expanded", "n_actions_used")
                if baseline_info[key] != info[key]}
        for label, info in infos.items()
    }
    print(f"\n=== {stem}: {baseline_info['n_bus']} buses, {baseline_info['n_line']} lines, "  # noqa: T201
          f"{baseline_info['n_load']} loads -> {baseline_info['n_bus_expanded']} buses / "
          f"{baseline_info['n_substation']} substations after expansion; switching action {action} ===")
    for label, differences in mismatch.items():
        if differences:
            print(f"  !! {label} expanded the grid differently: {differences}")  # noqa: T201

    header = f"{'measurement':<24}" + "".join(f"{label + ' ms':>18}" for label in labels)
    print(header)  # noqa: T201
    print("-" * len(header))  # noqa: T201
    for key in medians[baseline_label]:
        base = medians[baseline_label][key]
        cells = ""
        for label in labels:
            value = medians[label][key]
            if label == baseline_label:
                cells += f"{value:18.3f}"
            else:
                speedup = base / value if value else float("nan")
                cells += f"{value:11.3f}({speedup:4.1f}x)"
        print(f"{key:<24}{cells}")  # noqa: T201


def main() -> None:
    """Probe, benchmark every variant over every grid, and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grids", type=Path, required=True)
    parser.add_argument("--variant", action="append", required=True,
                        help="label=path or label=path:backend; the first is the baseline")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--probe", type=int, default=12)
    parser.add_argument("--bench", choices=sorted(BENCHES), default="step")
    parser.add_argument("--workers", default="1", help="greedy only: comma-separated worker counts")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    variants = [parse_variant(spec) for spec in args.variant]
    grid_paths = sorted(args.grids.glob("grid_*.pkl"), key=lambda p: int(p.stem.split("_")[1]))
    cache_dir = args.grids
    bench = BENCHES[args.bench]

    # The greedy benchmark sweeps the whole action space, so there is no single switching action
    # to agree on and no probe to run.
    chosen: dict[str, int] = {}
    if args.bench == "step":
        for grid_path in grid_paths:
            probes = {}
            for label, root, backend in variants:
                print(f"probing {label} on {grid_path.stem} ...", flush=True)  # noqa: T201
                probes[label] = run_bench(
                    bench, root, grid_path, label,
                    ["--probe", str(args.probe), "--backend", backend], cache_dir,
                )["probe"]
            chosen[grid_path.stem] = choose_common_action(probes)
            print(f"  -> switching action {chosen[grid_path.stem]}")  # noqa: T201

    collected: dict[tuple[str, str], list[dict]] = {}
    for round_index in range(args.rounds):
        # Rotate which variant goes first, so a warming machine does not always favour one.
        order = variants[round_index % len(variants):] + variants[:round_index % len(variants)]
        for grid_path in grid_paths:
            for label, root, backend in order:
                print(f"round {round_index + 1}/{args.rounds}: {label} on {grid_path.stem} ...",  # noqa: T201
                      flush=True)
                extra = (
                    ["--action", str(chosen[grid_path.stem]), "--repeat", str(args.repeat)]
                    if args.bench == "step"
                    else ["--workers", args.workers, "--repeat", str(args.repeat)]
                )
                payload = run_bench(
                    bench, root, grid_path, label, [*extra, "--backend", backend], cache_dir,
                )
                collected.setdefault((label, grid_path.stem), []).append(payload)

    report: dict = {"grids": {}, "rounds": args.rounds, "repeat": args.repeat,
                    "variants": [v[0] for v in variants]}
    for grid_path in grid_paths:
        stem = grid_path.stem
        medians = {label: median_over_rounds(collected[(label, stem)]) for label, _, _ in variants}
        infos = {label: collected[(label, stem)][0]["grid"] for label, _, _ in variants}
        report["grids"][stem] = {"switching_action": chosen.get(stem, -1),
                                 "info": infos, "medians": medians,
                                 "chosen": {label: collected[(label, stem)][0].get("chosen")
                                            for label, _, _ in variants}}
        print_grid_report(stem, chosen.get(stem, -1), infos, medians)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")  # noqa: T201


if __name__ == "__main__":
    main()
