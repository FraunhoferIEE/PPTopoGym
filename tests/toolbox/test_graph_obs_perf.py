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

For per-line attribution of this hot path, run ``scripts/profile_env.py`` (built on
``with-line-profiler``) rather than adding profiling to the test suite.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pytest

from pandapower_env.data.example_configs import config_case30
from pandapower_env.environments.simulation_env import PPTopoGym

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


def _coverage_is_tracing() -> bool:
    """Report whether coverage.py is currently instrumenting this interpreter.

    Line tracing roughly halves interpreter speed, which pushes ``create_observation``
    well past the wall-clock baseline recorded on an uninstrumented run. The baseline
    is only meaningful without tracing, so the timing test opts out under ``--cov``
    instead of reporting a regression that is not there.

    :return: ``True`` when a ``coverage`` measurement is active, ``False`` otherwise.
    """
    try:
        import coverage
    except ImportError:
        return False
    return coverage.Coverage.current() is not None


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
    if _coverage_is_tracing():
        pytest.skip("wall-clock baseline is not comparable under coverage instrumentation")

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


def _node_count_after(env: PPTopoGym, action: int) -> int:
    """Return the electrical node count after applying ``action`` from a fresh reset.

    :param env: the environment to step (left on the stepped state).
    :param action: the action index to apply.
    :return: ``n_nodes`` for the resulting topology.
    """
    import pandapower_env.toolbox.utils_graph_obs as ugo

    env.reset(options={"index": 0})
    env.step(action)
    return ugo.n_nodes(env.net, env._obs_cache)


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

    # A power flow allocates a fresh lookups object but does not change the topology, and the
    # caches are keyed on the bus-lookup *content*, so nothing is recomputed at all.
    env.run_pf()
    before = dict(counts)
    env.create_observation()
    assert counts["canonical"] == before["canonical"], "pp_to_canonical rebuilt after a plain power flow"
    assert counts["edges"] == before["edges"], "edges rebuilt after a plain power flow"

    # A real topology change must still rebuild each cache exactly ONCE per
    # create_observation -- not once per observation key (there are 30+ keys).
    nodes_before = ugo.n_nodes(env.net, env._obs_cache)
    splitting_action = next(
        action
        for action in env.df_actions.index[1:]
        if _node_count_after(env, int(action)) != nodes_before
    )
    # Re-apply from a clean baseline, and only then start counting: the search above already
    # stepped the env around and warmed the caches for those topologies. ``step`` builds an
    # observation internally, so the single rebuild is counted across the step, not after it.
    env.reset(options={"index": 0})
    env.step(0)
    env.create_observation()
    env.reset(options={"index": 0})
    before = dict(counts)
    env.step(int(splitting_action))
    env.create_observation()
    assert counts["canonical"] - before["canonical"] == 1
    assert counts["edges"] - before["edges"] == 1
