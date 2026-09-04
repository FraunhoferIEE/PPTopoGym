"""Run ``bench_pptopo`` on the working tree and on a stashed baseline, back to back.

Run with::

    poetry run python scripts/ab_compare.py --suite build
    poetry run python scripts/ab_compare.py --suite build --suite step --repeat 7

Why this exists: timings on this machine move several percent with background load, so
"the number today vs. a number written in a file last week" cannot distinguish a real
speedup from noise. This runs BASELINE and CURRENT within seconds of each other in the
same conditions, alternating them, and reports the per-measurement delta.

Baseline and current are run in **alternating rounds** and reported as the median across
them. A single baseline-then-current pass hands the second side a warm OS page cache, which
here showed up as a uniform 10-18% "speedup" on build stages that had not been touched --
so always sanity-check that untouched measurements read as noise.

The baseline is the working tree with your uncommitted changes stashed
(``git stash push --keep-index`` semantics are deliberately NOT used -- the full
working tree is stashed, including staged changes, so the baseline is exactly HEAD
plus nothing). The stash is always restored, including on failure.

.. warning::
   This mutates the git working tree while running. It refuses to start if a stash
   operation would be ambiguous (e.g. an in-progress merge or rebase).

:raises SystemExit: if the repo is mid-merge/rebase, or if restoring the stash fails.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH = REPO_ROOT / "scripts" / "bench_pptopo.py"

# Deltas smaller than this are reported as noise rather than as a win or a regression.
NOISE_THRESHOLD_PCT = 3.0


def _git(*args: str) -> str:
    """Run a git command in the repo root and return its stdout.

    :param args: Git arguments, e.g. ``("stash", "list")``.
    :return: Captured stdout, stripped.
    :raises SystemExit: if git exits non-zero.
    """
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout.strip()


def _assert_safe_to_stash() -> None:
    """Refuse to run mid-merge or mid-rebase, where stashing is unsafe.

    :raises SystemExit: if a merge or rebase is in progress.
    """
    git_dir = Path(_git("rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = REPO_ROOT / git_dir
    for marker in ("MERGE_HEAD", "rebase-merge", "rebase-apply"):
        if (git_dir / marker).exists():
            sys.exit(f"refusing to stash: {marker} exists (merge/rebase in progress)")


def _run_bench(suites: list[str], repeat: int, label: str) -> dict[str, float]:
    """Run the benchmark script in a subprocess and return its results mapping.

    A subprocess is used so each side gets a cold interpreter: module-level caches in
    ``pandapower_env`` (profile tables, warm power-flow options) would otherwise leak
    from the first side into the second and flatter it.

    :param suites: Suite names to pass through.
    :param repeat: Repetitions per measurement.
    :param label: Label recorded in the JSON payload.
    :return: Mapping of measurement key to median milliseconds.
    :raises SystemExit: if the benchmark subprocess fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        out_path = Path(handle.name)
    cmd = [sys.executable, str(BENCH), "--repeat", str(repeat), "--json", str(out_path),
           "--label", label]
    for suite in suites:
        cmd += ["--suite", suite]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)  # noqa: S603
    if result.returncode != 0:
        out_path.unlink(missing_ok=True)
        sys.exit(f"benchmark failed for {label}:\n{result.stdout}\n{result.stderr}")
    payload = json.loads(out_path.read_text())
    out_path.unlink(missing_ok=True)
    return payload["results"]


def _median_across(rounds: list[dict[str, float]]) -> dict[str, float]:
    """Median of each measurement across rounds, so one loaded round cannot dominate.

    :param rounds: one results mapping per round.
    :return: mapping of measurement key to its median across the rounds that reported it.
    """
    keys = {key for round_results in rounds for key in round_results}
    return {
        key: statistics.median([r[key] for r in rounds if key in r])
        for key in keys
    }


def _report(baseline: dict[str, float], current: dict[str, float]) -> bool:
    """Print the per-measurement comparison and say whether anything regressed.

    :param baseline: Baseline results (HEAD, changes stashed).
    :param current: Current results (working tree).
    :return: ``True`` if no measurement regressed beyond the noise threshold.
    """
    keys = sorted(set(baseline) | set(current))
    print(f"\n{'measurement':<40} {'base ms':>10} {'curr ms':>10} {'delta':>10}  verdict")  # noqa: T201
    print("-" * 88)  # noqa: T201
    clean = True
    for key in keys:
        base = baseline.get(key)
        curr = current.get(key)
        if base is None or curr is None:
            print(f"{key:<40} {'-' if base is None else f'{base:10.2f}':>10} "  # noqa: T201
                  f"{'-' if curr is None else f'{curr:10.2f}':>10} {'n/a':>10}  missing on one side")
            continue
        pct = (curr - base) / base * 100.0 if base else 0.0
        if abs(pct) < NOISE_THRESHOLD_PCT:
            verdict = "noise"
        elif pct < 0:
            verdict = f"FASTER {-pct:.1f}%"
        else:
            verdict = f"SLOWER {pct:.1f}%"
            clean = False
        print(f"{key:<40} {base:10.2f} {curr:10.2f} {pct:9.1f}%  {verdict}")  # noqa: T201
    return clean


def _bench_baseline(suites: list[str], repeat: int) -> dict[str, float]:
    """Stash the working tree, benchmark HEAD, and always restore.

    :raises SystemExit: if the stash cannot be created or, critically, cannot be restored.
    """
    stash_before = _git("stash", "list")
    _git("stash", "push", "-u", "-m", "ab_compare-baseline")
    if _git("stash", "list") == stash_before:
        sys.exit("stash push did not create a stash; aborting without measuring.")
    try:
        return _run_bench(suites, repeat, "baseline(HEAD)")
    finally:
        restore = subprocess.run(  # noqa: S603
            ["git", "stash", "pop"],  # noqa: S607
            cwd=REPO_ROOT, capture_output=True, text=True, check=False,
        )
        if restore.returncode != 0:
            sys.exit(
                "CRITICAL: could not restore your changes with 'git stash pop'.\n"
                f"{restore.stderr}\nYour work is still in the stash: run 'git stash list'.",
            )


def main() -> None:
    """Benchmark HEAD against the working tree over alternating rounds, and report medians."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", action="append", help="Suite(s) to benchmark; repeatable.")
    parser.add_argument("--repeat", type=int, default=5, help="Repetitions per measurement.")
    parser.add_argument(
        "--rounds", type=int, default=3,
        help="Alternating baseline/current rounds; the median across rounds is reported.",
    )
    args = parser.parse_args()
    suites = args.suite or ["build", "step", "obs"]

    _assert_safe_to_stash()

    if not _git("status", "--porcelain"):
        sys.exit("working tree is clean: there is nothing to compare against HEAD.")

    # Alternate which side runs first. Running baseline-then-current once, as this script used
    # to, hands the second side a warm OS page cache and a warm Simbench CSV read -- which
    # showed up as a uniform 10-18% "speedup" on build stages that had not been touched at all.
    base_rounds: list[dict[str, float]] = []
    curr_rounds: list[dict[str, float]] = []
    for round_index in range(args.rounds):
        baseline_first = round_index % 2 == 0
        order = "baseline,current" if baseline_first else "current,baseline"
        print(f"round {round_index + 1}/{args.rounds} ({order}) ...", flush=True)  # noqa: T201
        if baseline_first:
            base_rounds.append(_bench_baseline(suites, args.repeat))
            curr_rounds.append(_run_bench(suites, args.repeat, "current(worktree)"))
        else:
            curr_rounds.append(_run_bench(suites, args.repeat, "current(worktree)"))
            base_rounds.append(_bench_baseline(suites, args.repeat))

    baseline = _median_across(base_rounds)
    current = _median_across(curr_rounds)
    clean = _report(baseline, current)
    print(f"\nmedian of {args.rounds} alternating rounds; noise threshold: +-{NOISE_THRESHOLD_PCT}%")  # noqa: T201
    print(  # noqa: T201
        "Sanity check: stages your change does not touch should read as noise. If they do not, "
        "the machine was loaded and the whole table is suspect.",
    )
    if not clean:
        print("at least one measurement regressed beyond the noise threshold.")  # noqa: T201


if __name__ == "__main__":
    main()
