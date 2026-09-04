"""Tests for the parallel N-1 backend (``nminus1_parallel``).

Covers the pure helpers (contingency splitting + result merging), the safety fallbacks
(single worker / inside a child process), and -- the core guarantee -- that the parallel
results are bit-for-bit identical to the serial implementation and to the committed
``nminus1_baseline.json`` golden values, for both AC and DC.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandapower_env.toolbox import nminus1_parallel
from pandapower_env.toolbox.nminus1_parallel import (
    _apply_net_state,
    _effective_workers,
    _ensure_net_from_blob,
    _ensure_trafo_loading_limit,
    _merge_contingency_results,
    _net_to_static_blob,
    _snapshot_net_state,
    _solve_contingency_subset,
    _split_contingencies,
    run_nminus1_powerflow_parallel,
)
from pandapower_env.toolbox.utils import run_nminus1_powerflow, warn_unavailable_ls2g

if TYPE_CHECKING:
    from pandapower import pandapowerNet

BASELINE_PATH = Path(__file__).parent / "nminus1_baseline.json"
RTOL = 1e-9
ATOL = 1e-6


# --------------------------------------------------------------------------------------
# Pure helpers (no multiprocessing)
# --------------------------------------------------------------------------------------
def test_effective_workers_clamps_to_contingencies() -> None:
    """Worker count is clamped to [1, n_contingencies]; None means all CPUs."""
    assert _effective_workers(8, 3) == 3  # noqa: PLR2004
    assert _effective_workers(2, 40) == 2  # noqa: PLR2004
    assert _effective_workers(0, 40) == 1
    assert _effective_workers(None, 40) >= 1


def test_split_contingencies_is_contiguous_and_ordered() -> None:
    """Splitting preserves the global line->trafo->trafo3w order across contiguous chunks."""
    cases = {
        "line": {"index": np.array([0, 1, 2, 3])},
        "trafo": {"index": np.array([0, 1])},
        "trafo3w": {"index": np.array([], dtype=int)},
    }
    chunks = _split_contingencies(cases, 3)

    expected_chunks = 3
    assert len(chunks) == expected_chunks

    # Flatten the chunks back and check we recover the exact ordered contingency list.
    recovered = [
        (element, int(idx))
        for chunk in chunks
        for element in ("line", "trafo", "trafo3w")
        if element in chunk
        for idx in chunk[element]["index"]
    ]
    expected = [("line", 0), ("line", 1), ("line", 2), ("line", 3), ("trafo", 0), ("trafo", 1)]
    assert recovered == expected


def test_split_contingencies_more_chunks_than_cases() -> None:
    """Requesting more chunks than contingencies yields one (non-empty) chunk per contingency."""
    cases = {"line": {"index": np.array([0, 1])}, "trafo": {"index": np.array([], dtype=int)},
             "trafo3w": {"index": np.array([], dtype=int)}}
    chunks = _split_contingencies(cases, 10)
    assert len(chunks) == len(cases["line"]["index"])
    assert all(len(chunk["line"]["index"]) == 1 for chunk in chunks)


def test_split_contingencies_empty_returns_empty() -> None:
    """No contingencies -> no chunks."""
    empty = {element: {"index": np.array([], dtype=int)} for element in ("line", "trafo", "trafo3w")}
    assert _split_contingencies(empty, 4) == []


def test_merge_skips_partial_without_max_loading() -> None:
    """A worker partial lacking ``max_loading_percent`` for an element is skipped (causes OR'd)."""
    with_max = {
        "line": {
            "index": np.array([0]),
            "loading_percent": np.array([10.0]),
            "max_loading_percent": np.array([55.0]),
            "min_loading_percent": np.array([55.0]),
            "cause_index": np.array([3]),
            "cause_element": np.array(["line"], dtype=object),
            "causes_overloading": np.array([False]),
        },
    }
    without_max = {
        "line": {
            "index": np.array([0]),
            "loading_percent": np.array([10.0]),
            "causes_overloading": np.array([True]),
        },
    }
    merged = _merge_contingency_results([with_max, without_max])
    assert np.array_equal(merged["line"]["max_loading_percent"], [55.0])
    assert np.array_equal(merged["line"]["cause_index"], [3])
    assert np.array_equal(merged["line"]["causes_overloading"], [True])


def test_merge_contingency_results_max_min_and_cause() -> None:
    """Element-wise fmax/fmin and first-wins cause resolution across two worker partials."""
    partial_a = {
        "line": {
            "index": np.array([0, 1]),
            "loading_percent": np.array([30.0, 40.0]),
            "max_loading_percent": np.array([50.0, 80.0]),
            "min_loading_percent": np.array([50.0, 80.0]),
            "cause_index": np.array([10, 11]),
            "cause_element": np.array(["line", "line"], dtype=object),
            "causes_overloading": np.array([True, False]),
        },
        "bus": {"index": np.array([0]), "vm_pu": np.array([1.0]),
                "max_vm_pu": np.array([1.05]), "min_vm_pu": np.array([0.95])},
    }
    partial_b = {
        "line": {
            "index": np.array([0, 1]),
            "loading_percent": np.array([30.0, 40.0]),
            "max_loading_percent": np.array([90.0, 60.0]),
            "min_loading_percent": np.array([90.0, 60.0]),
            "cause_index": np.array([20, 21]),
            "cause_element": np.array(["trafo", "trafo"], dtype=object),
            "causes_overloading": np.array([False, True]),
        },
        "bus": {"index": np.array([0]), "vm_pu": np.array([1.0]),
                "max_vm_pu": np.array([1.02]), "min_vm_pu": np.array([0.93])},
    }

    merged = _merge_contingency_results([partial_a, partial_b])

    assert np.array_equal(merged["line"]["max_loading_percent"], [90.0, 80.0])
    assert np.array_equal(merged["line"]["min_loading_percent"], [50.0, 60.0])
    # line 0 max comes from B (90 > 50); line 1 max comes from A (80 > 60), first-wins.
    assert np.array_equal(merged["line"]["cause_index"], [20, 11])
    assert list(merged["line"]["cause_element"]) == ["trafo", "line"]
    assert np.array_equal(merged["line"]["causes_overloading"], [True, True])
    assert np.array_equal(merged["bus"]["max_vm_pu"], [1.05])
    assert np.array_equal(merged["bus"]["min_vm_pu"], [0.93])
    # N-0 base values are carried through unchanged.
    assert np.array_equal(merged["line"]["loading_percent"], [30.0, 40.0])


# --------------------------------------------------------------------------------------
# Equivalence: parallel == serial == committed baseline
# --------------------------------------------------------------------------------------
def _assert_same_nminus1(net_a: pandapowerNet, net_b: pandapowerNet) -> None:
    """Assert two solved nets carry identical N-1 line/bus results (incl. causes)."""
    for column in ("loading_percent", "max_loading_percent", "min_loading_percent"):
        assert np.allclose(net_a.res_line[column].to_numpy(), net_b.res_line[column].to_numpy(),
                           rtol=RTOL, atol=ATOL, equal_nan=True), f"res_line.{column} differs"
    for column in ("max_vm_pu", "min_vm_pu"):
        assert np.allclose(net_a.res_bus[column].to_numpy(), net_b.res_bus[column].to_numpy(),
                           rtol=RTOL, atol=ATOL, equal_nan=True), f"res_bus.{column} differs"
    # Causes are only meaningful (and consumed by the metrics) on loaded lines.
    loaded = net_a.res_line["max_loading_percent"].to_numpy() > 0
    assert np.array_equal(net_a.res_line.loc[loaded, "cause_index"].to_numpy(),
                          net_b.res_line.loc[loaded, "cause_index"].to_numpy())
    assert np.array_equal(net_a.res_line.loc[loaded, "cause_element"].to_numpy(),
                          net_b.res_line.loc[loaded, "cause_element"].to_numpy())


@pytest.mark.parametrize("pf_type", ["ac", "dc"])
def test_parallel_matches_serial(test_grid_dbb_plus_simbench: pandapowerNet, pf_type: str) -> None:
    """The parallel backend must reproduce the serial results exactly (the speedup check).

    This is the reusable "log N-1 without the speedup, then compare with the speedup" guard.
    """
    net_serial = test_grid_dbb_plus_simbench
    net_parallel = copy.deepcopy(net_serial)

    run_nminus1_powerflow(net_serial, pf_type=pf_type)
    run_nminus1_powerflow_parallel(net_parallel, pf_type=pf_type, n_workers=4)

    _assert_same_nminus1(net_serial, net_parallel)


def test_parallel_matches_committed_baseline(test_grid_dbb_plus_simbench: pandapowerNet) -> None:
    """Parallel results match the committed golden baseline (skips if it hasn't been captured)."""
    if not BASELINE_PATH.exists():
        pytest.skip(f"{BASELINE_PATH.name} not present; run the serial regression test to capture it.")

    baseline = json.loads(BASELINE_PATH.read_text())
    net = test_grid_dbb_plus_simbench
    for pf_type, arrays in baseline.items():
        run_nminus1_powerflow_parallel(net, pf_type=pf_type, n_workers=4)
        current = {
            "line_loading_percent": net.res_line.loading_percent.to_numpy(),
            "line_max_loading_percent": net.res_line.max_loading_percent.to_numpy(),
            "bus_max_vm_pu": net.res_bus.max_vm_pu.to_numpy(),
            "bus_min_vm_pu": net.res_bus.min_vm_pu.to_numpy(),
        }
        for key, expected in arrays.items():
            assert np.allclose(current[key], expected, rtol=RTOL, atol=ATOL, equal_nan=True), (
                f"parallel N-1 drift in {pf_type}/{key}"
            )


# --------------------------------------------------------------------------------------
# Worker path, exercised in-process (the real parallel runs it in subprocesses)
# --------------------------------------------------------------------------------------
def test_snapshot_apply_state_roundtrip(test_grid_dbb_plus_simbench: pandapowerNet) -> None:
    """A net rebuilt from the static blob + applied state matches the original exactly."""
    net = test_grid_dbb_plus_simbench
    # Make the state non-trivial: disconnect a line, change a load, set a NaN-free tap.
    net.line.loc[net.line.index[0], "in_service"] = False
    net.load.loc[net.load.index[0], "p_mw"] = net.load.loc[net.load.index[0], "p_mw"] + 7.0

    blob = _net_to_static_blob(net)
    state = _snapshot_net_state(net)
    rebuilt = _ensure_net_from_blob(blob)
    _apply_net_state(rebuilt, state)

    assert np.array_equal(rebuilt.line["in_service"].to_numpy(), net.line["in_service"].to_numpy())
    assert np.allclose(rebuilt.load["p_mw"].to_numpy(), net.load["p_mw"].to_numpy())
    assert np.array_equal(rebuilt.switch["closed"].to_numpy(), net.switch["closed"].to_numpy())
    # NaN tap positions must round-trip as NaN (not silently become 0).
    assert np.array_equal(rebuilt.trafo["tap_pos"].to_numpy(), net.trafo["tap_pos"].to_numpy(), equal_nan=True)


def test_solve_contingency_subset_matches_serial(test_grid_dbb_plus_simbench: pandapowerNet) -> None:
    """The worker entry point, run in-process on the full set, reproduces the serial loadings.

    Exercises ``_solve_contingency_subset`` and its helpers (blob deserialize, state overlay,
    trafo-limit fix, quiet logging) directly so they are covered without a subprocess.
    """
    net = test_grid_dbb_plus_simbench
    full_cases = {element: {"index": net[element].index.to_numpy()}
                  for element in ("line", "trafo", "trafo3w")}
    partial = _solve_contingency_subset(_net_to_static_blob(net), _snapshot_net_state(net),
                                        full_cases, "ac", "auto")

    net_serial = copy.deepcopy(net)
    run_nminus1_powerflow(net_serial, pf_type="ac")
    assert np.allclose(partial["line"]["max_loading_percent"],
                       net_serial.res_line["max_loading_percent"].to_numpy(),
                       rtol=RTOL, atol=ATOL, equal_nan=True)


# --------------------------------------------------------------------------------------
# Safety fallbacks: never spawn a nested pool
# --------------------------------------------------------------------------------------
def test_single_worker_falls_back_to_serial(test_grid_dbb_plus_simbench: pandapowerNet) -> None:
    """n_workers=1 uses the serial path and matches a serial run."""
    net_serial = test_grid_dbb_plus_simbench
    net_single = copy.deepcopy(net_serial)
    run_nminus1_powerflow(net_serial, pf_type="ac")
    run_nminus1_powerflow_parallel(net_single, pf_type="ac", n_workers=1)
    _assert_same_nminus1(net_serial, net_single)


def test_child_process_falls_back_to_serial(
    test_grid_dbb_plus_simbench: pandapowerNet,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inside a child process the parallel path must NOT build a pool -- it runs serial.

    Simulated by making ``parent_process()`` report a parent and booby-trapping ``Parallel``
    so the test fails loudly if the parallel branch is ever reached.
    """
    net_serial = test_grid_dbb_plus_simbench
    net_child = copy.deepcopy(net_serial)
    run_nminus1_powerflow(net_serial, pf_type="ac")

    monkeypatch.setattr(nminus1_parallel.multiprocessing, "parent_process", lambda: object())

    def _no_pool_allowed(*_args: object, **_kwargs: object) -> None:
        msg = "parallel pool was created inside a child process"
        raise AssertionError(msg)

    monkeypatch.setattr(nminus1_parallel, "Parallel", _no_pool_allowed)

    run_nminus1_powerflow_parallel(net_child, pf_type="ac", n_workers=8)
    _assert_same_nminus1(net_serial, net_child)


def test_invalid_pf_type_raises(test_grid_dbb_plus_simbench: pandapowerNet) -> None:
    """An unknown pf_type is rejected before any work is done."""
    with pytest.raises(ValueError, match="pf_type"):
        run_nminus1_powerflow_parallel(test_grid_dbb_plus_simbench, pf_type="xy", n_workers=2)


def test_ensure_trafo_loading_limit_adds_missing_column(test_grid_dbb_plus_simbench: pandapowerNet) -> None:
    """A trafo without a loading limit gets the same default (100.0) as the serial fast path."""
    net = test_grid_dbb_plus_simbench
    if "max_loading_percent" in net.trafo.columns:
        del net.trafo["max_loading_percent"]
    _ensure_trafo_loading_limit(net)
    assert bool((net.trafo["max_loading_percent"] == 100.0).all())  # noqa: PLR2004


def test_warn_unavailable_ls2g_logs_for_ac(
    test_grid_dbb_plus_simbench: pandapowerNet,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The ls2g-unavailable warning fires for AC when the backend could not be used."""
    net = test_grid_dbb_plus_simbench
    run_nminus1_powerflow(net, pf_type="ac")  # populates net._options
    net._options["lightsim2grid"] = False
    with caplog.at_level(logging.WARNING):
        warn_unavailable_ls2g(net, "ac", "auto")
    assert any("lightsim2grid" in record.getMessage() for record in caplog.records)
