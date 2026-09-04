"""Profile the PPTopoGym hot path with ``with-line-profiler``.

Run with::

    poetry run python scripts/profile_env.py

It builds the ``config_case30`` environment and profiles a fixed workload of 10
topology steps (substation switching actions, spread over different substations) plus
a few DoNothing steps, so the per-line cost of ``step`` / ``create_observation`` /
``run_pf`` is attributable.

Two layers of the tool are used together, because they answer different questions:

- ``lineprofiler.accounting`` records a *run* -- named phases with wall-clock spans
  and resource samples -- into ``run_dir``. That is where the trace, the text/JSON
  report and the timeline HTML come from, and it is what tells you how the run's time
  splits between building the env, stepping topology actions, and DoNothing steps.
- ``lineprofiler.LineProfiler`` attributes time *per source line* inside
  ``pandapower_env`` and renders it via ``to_html``.

Outputs land in ``profiling/`` (see ``OUTPUT_DIR``).

Profiling lives here, not in the core environment: nothing under ``pandapower_env/``
imports a profiler, so the shipped env carries no profiling overhead or dependency.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lineprofiler import LineProfiler
from lineprofiler import accounting as acc

from pandapower_env.data.example_configs import config_case30
from pandapower_env.environments.simulation_env import PPTopoGym

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "profiling"

# The ``lineprofiler`` console script, resolved next to the running interpreter so the
# script works under ``poetry run`` without relying on it being on PATH.
CLI = str(Path(sys.executable).parent / "lineprofiler")

# Only lines inside this package are attributed, so pandas/pandapower internals stay
# collapsed into the call that entered them.
PROJECT_FOLDER = Path(__file__).resolve().parent.parent / "pandapower_env"

N_TOPOLOGY_STEPS = 10
N_DONOTHING_STEPS = 3
START_INDEX = 0


def select_topology_actions(env: PPTopoGym, count: int) -> list[int]:
    """Pick ``count`` distinct topology actions, spread across different substations.

    Action 0 is DoNothing and is skipped. Actions are grouped by the substation they
    switch (the first entry of ``open_switches``) and drawn round-robin over those
    groups, so the profile covers several substations rather than hammering one.

    :param env: The environment whose ``df_actions`` supplies the action table.
    :param count: How many actions to return.
    :return: Action indices, at most ``count`` of them (fewer if the grid has fewer).
    """
    actions_by_substation: dict[object, list[int]] = {}
    for action in env.df_actions.index[1:]:
        switches = env.df_actions.loc[action, "open_switches"]
        substation = switches[0] if len(switches) else None
        actions_by_substation.setdefault(substation, []).append(int(action))

    selected: list[int] = []
    groups = list(actions_by_substation.values())
    depth = 0
    while len(selected) < count and any(depth < len(g) for g in groups):
        for group in groups:
            if depth < len(group):
                selected.append(group[depth])
                if len(selected) == count:
                    break
        depth += 1
    return selected


def run_workload(env: PPTopoGym, actions: list[int]) -> None:
    """Step the env through the topology actions, then a few DoNothing steps.

    Each topology action is applied from a freshly reset state so that a failed power
    flow on one action cannot terminate the rest of the workload. Every step is wrapped
    in an accounting phase, so the report attributes time per action kind.

    :param env: The environment to step.
    :param actions: Topology action indices to apply, one per reset.
    """
    for action in actions:
        with acc.phase("reset"):
            env.reset(options={"index": START_INDEX})
        with acc.phase("step_topology"):
            env.step(action)

    with acc.phase("reset"):
        env.reset(options={"index": START_INDEX})
    for _ in range(N_DONOTHING_STEPS):
        with acc.phase("step_donothing"):
            env.step(0)


def write_accounting_reports(run_dir: Path, output_dir: Path) -> None:
    """Render the recorded run into a text report, a JSON report, and a timeline.

    Uses the ``lineprofiler`` console script, which is the supported way to merge a run
    directory (the in-process buffers are flushed to ``run_dir`` by ``acc.stop()`` first).
    The package has no ``__main__``, so it cannot be invoked with ``python -m``.

    :param run_dir: The directory that ``acc.start`` recorded the run into.
    :param output_dir: Where the rendered report/trace files are written.
    """
    renders = [
        ("report", "text", output_dir / "env_report.txt"),
        ("report", "json", output_dir / "env_report.json"),
        ("trace", "html", output_dir / "env_trace.html"),
        ("trace", "json", output_dir / "env_trace.json"),
    ]
    for command, fmt, path in renders:
        result = subprocess.run(  # noqa: S603
            [CLI, command, str(run_dir), "--format", fmt, "-o", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            print(f"  ! {command}/{fmt} failed: {result.stderr.strip()}")  # noqa: T201


def main() -> None:
    """Build the env, profile the workload, and write trace/profile/HTML outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for the trace, profile, and HTML report.",
    )
    args = parser.parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / "run"

    # ``enabled``/``trace`` must be explicit: accounting is opt-in and otherwise stays a
    # no-op unless ``LINEPROFILER_PROFILE=1`` is exported, which would record nothing here.
    acc.start(run_dir=run_dir, role="profile_env", enabled=True, trace=True)

    print("Building config_case30 (slow: scaling + substations + action verification)...")  # noqa: T201
    with acc.phase("build_env"):
        env = PPTopoGym(config_case30())

    actions = select_topology_actions(env, N_TOPOLOGY_STEPS)
    print(f"Profiling {len(actions)} topology steps + {N_DONOTHING_STEPS} DoNothing steps")  # noqa: T201

    # Warm-up outside the profiler: the first call pays JIT, option parsing and imports.
    with acc.phase("warmup"):
        env.reset(options={"index": START_INDEX})
        env.step(0)

    profiler = LineProfiler(project_folder=PROJECT_FOLDER)
    with profiler:
        run_workload(env, actions)

    acc.stop()

    profiler.to_html(output_dir / "env_lines.html", title="PPTopoGym step profile")
    with (output_dir / "env_lines.txt").open("w") as handle:
        stdout, sys.stdout = sys.stdout, handle
        try:
            profiler.print_global_top_stats(top_n=40, min_time_us=0.1)
        finally:
            sys.stdout = stdout

    write_accounting_reports(run_dir, output_dir)
    print(f"Wrote trace, profile and HTML reports to {output_dir}")  # noqa: T201
    profiler.print_global_top_stats(top_n=15, min_time_us=1.0)


if __name__ == "__main__":
    main()
