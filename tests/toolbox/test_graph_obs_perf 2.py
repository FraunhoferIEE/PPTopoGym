"""Speed-regression guard for graph-observation extraction.

This test pins the performance of the hot ``create_observation`` path (which is
dominated by ``pandapower_env.toolbox.utils_graph_obs``) against a committed
baseline so that the functions+decorators rewrite cannot silently regress.

Workflow
--------
- The first run on a machine where ``graph_obs_baseline.json`` is absent *captures*
  the baseline (writes the file) and passes. This is how the pre-refactor baseline
  is recorded against the original class-based implementation.
- Subsequent runs compare the current median ``create_observation`` time against the
  stored baseline and fail if it exceeds ``baseline * TOLERANCE``.

The in-repo ``LineProfiler`` (``pandapower_env.toolbox.profiler``) is also run for
per-line attribution; its output is informational only.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pytest

from pandapower_env.data.example_configs import config_case30
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.toolbox.profiler import LineProfiler

BASELINE_PATH = Path(__file__).parent / "graph_obs_baseline.json"

# Generous tolerance: wall-clock timing in CI is noisy, we only want to catch real
# regressions (e.g. losing the caching), not 10% jitter.
TOLERANCE = 1.25

# Calls per timed repeat and number of repeats (median is used to dampen noise).
N_CALLS = 200
N_REPEATS = 7
WARMUP = 5


@pytest.fixture(scope="module")
def perf_env() -> PPTopoGym:
    """Return a case30 environment with power flow already run, fixed at index 0."""
    env = PPTopoGym(config_case30())
    env.reset(options={"index": 0})
    return env


def _median_us_per_call(env: PPTopoGym) -> float:
    """Median microseconds per ``create_observation`` call over several repeats."""
    for _ in range(WARMUP):
        env.create_observation()

    per_call_us: list[float] = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        for _ in range(N_CALLS):
            env.create_observation()
        elapsed = time.perf_counter() - start
        per_call_us.append(elapsed / N_CALLS * 1e6)
    return statistics.median(per_call_us)


def test_create_observation_speed(perf_env: PPTopoGym) -> None:
    """Median create_observation time must not regress beyond the stored baseline."""
    obs = perf_env.create_observation()
    median_us = _median_us_per_call(perf_env)

    if not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(
            json.dumps(
                {
                    "create_observation_us": median_us,
                    "n_obs_keys": len(obs),
                    "n_calls": N_CALLS,
                    "n_repeats": N_REPEATS,
                    "note": "Captured from the original class-based implementation.",
                },
                indent=2,
            ),
        )
        pytest.skip(
            f"Baseline captured ({median_us:.1f} us/call) -> {BASELINE_PATH.name}. "
            "Re-run to compare against it.",
        )

    baseline = json.loads(BASELINE_PATH.read_text())
    baseline_us = baseline["create_observation_us"]
    limit_us = baseline_us * TOLERANCE

    print(  # noqa: T201
        f"\ncreate_observation: current={median_us:.1f} us/call, "
        f"baseline={baseline_us:.1f} us/call, limit={limit_us:.1f} us/call",
    )

    assert median_us <= limit_us, (
        f"create_observation regressed: {median_us:.1f} us/call > "
        f"{limit_us:.1f} us/call (baseline {baseline_us:.1f} * {TOLERANCE})"
    )


def test_lineprofiler_attribution(perf_env: PPTopoGym) -> None:
    """Run the in-repo LineProfiler over the hot path (informational, never fails)."""
    profiler = LineProfiler(project_folder="pandapower_env")
    with profiler:
        for _ in range(20):
            perf_env.create_observation()
    profiler.print_global_top_stats(top_n=12, sort_by="time")


def test_no_redundant_recompute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Topology-only computations must run once per topology, not once per call."""
    import pandapower_env.toolbox.utils_graph_obs as ugo

    counts = {"canonical": 0, "edges": 0}
    orig_canonical = ugo._compute_pp_to_canonical
    orig_edges = ugo._collect_edges

    def counting_canonical(lookup_table):  # noqa: ANN202
        counts["canonical"] += 1
        return orig_canonical(lookup_table)

    def counting_edges(net):  # noqa: ANN202
        counts["edges"] += 1
        return orig_edges(net)

    monkeypatch.setattr(ugo, "_compute_pp_to_canonical", counting_canonical)
    monkeypatch.setattr(ugo, "_collect_edges", counting_edges)

    env = PPTopoGym(config_case30())
    env.reset(options={"index": 0})
    env.create_observation()  # warm the topology caches

    warm = dict(counts)
    for _ in range(15):
        env.create_observation()
    assert counts["canonical"] == warm["canonical"], "pp_to_canonical recomputed at fixed topology"
    assert counts["edges"] == warm["edges"], "edges recomputed at fixed topology"

    # A fresh power flow (new lookups identity) rebuilds each cache exactly ONCE per
    # create_observation -- not once per observation key (there are 30+ keys).
    id_before = id(env.net._pd2ppc_lookups)
    env.run_pf()
    id_after = id(env.net._pd2ppc_lookups)
    before = dict(counts)
    env.create_observation()
    if id_after != id_before:  # power flow produced a fresh lookups object
        assert counts["canonical"] - before["canonical"] == 1
        assert counts["edges"] - before["edges"] == 1
