"""Golden-value regression guard for the N-1 power-flow calculation.

This test pins the *results* of ``run_nminus1_powerflow`` (N-0 and N-1 line loadings
plus N-1 bus-voltage envelopes) against a committed baseline, so future changes to the
power-flow calc cannot silently alter the numbers the environment, agents, and metrics
depend on.

Workflow (mirrors ``tests/toolbox/test_graph_obs_perf.py``)
----------------------------------------------------------
- The first run on a machine where ``nminus1_baseline.json`` is absent *captures* the
  baseline (writes the file) and passes via ``pytest.skip``. Commit that file -- it is
  the snapshot taken from the reference implementation.
- Subsequent runs compare the current results against the stored baseline and fail on
  any drift beyond a tight numerical tolerance.

To regenerate the baseline intentionally (e.g. after a deliberate model change), delete
``nminus1_baseline.json`` and re-run.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandapower_env.toolbox.utils import run_nminus1_powerflow

if TYPE_CHECKING:
    from pandapower import pandapowerNet

BASELINE_PATH = Path(__file__).parent / "nminus1_baseline.json"

# Tight tolerance: the calc must stay numerically identical, this is not a noisy timing.
RTOL = 1e-9
ATOL = 1e-6


def _collect_nminus1_results(net: pandapowerNet) -> dict[str, list[float]]:
    """Snapshot the consumed/aggregated N-1 results of ``net`` as JSON-friendly lists.

    Captures exactly the columns downstream code reads: per-line N-0 ``loading_percent``
    and N-1 ``max_loading_percent``, and the per-bus N-1 voltage envelope
    (``max_vm_pu`` / ``min_vm_pu``). Kept as a standalone helper so the same snapshot
    shape can be reused when the baseline is regenerated.
    """
    return {
        "line_loading_percent": net.res_line.loading_percent.to_numpy().tolist(),
        "line_max_loading_percent": net.res_line.max_loading_percent.to_numpy().tolist(),
        "bus_max_vm_pu": net.res_bus.max_vm_pu.to_numpy().tolist(),
        "bus_min_vm_pu": net.res_bus.min_vm_pu.to_numpy().tolist(),
    }


def _snapshot_ac_dc(net: pandapowerNet) -> dict[str, dict[str, list[float]]]:
    """Run the AC and DC N-1 power flow on ``net`` and snapshot both."""
    snapshot: dict[str, dict[str, list[float]]] = {}
    for pf_type in ("ac", "dc"):
        run_nminus1_powerflow(net, pf_type=pf_type)
        snapshot[pf_type] = _collect_nminus1_results(net)
    return snapshot


def test_nminus1_results_match_baseline(test_grid_dbb_plus_simbench: pandapowerNet) -> None:
    """N-1 results on the deterministic test grid must match the committed baseline.

    :param test_grid_dbb_plus_simbench: deterministic 14-bus grid fixture (see conftest.py)
    :type test_grid_dbb_plus_simbench: pandapowerNet
    """
    current = _snapshot_ac_dc(test_grid_dbb_plus_simbench)

    if not BASELINE_PATH.exists():
        BASELINE_PATH.write_text(json.dumps(current, indent=2))
        pytest.skip(
            f"Baseline captured -> {BASELINE_PATH.name}. Re-run to compare against it.",
        )

    baseline = json.loads(BASELINE_PATH.read_text())
    for pf_type, arrays in baseline.items():
        for key, expected in arrays.items():
            actual = current[pf_type][key]
            assert np.allclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True), (
                f"N-1 result drift in {pf_type}/{key}: max|diff|="
                f"{np.nanmax(np.abs(np.asarray(actual) - np.asarray(expected))):.3e}"
            )


def test_nminus1_no_trafo_keyerror_spam(
    test_grid_dbb_plus_simbench: pandapowerNet,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A trafo without a max_loading_percent limit must not trigger swallowed KeyErrors.

    ``run_contingency`` logs (and swallows) a KeyError per contingency when a monitored
    element lacks ``max_loading_percent``. The fast path adds that column temporarily, so
    no such errors are logged and the net is left untouched.

    :param test_grid_dbb_plus_simbench: deterministic 14-bus grid fixture (see conftest.py)
    :type test_grid_dbb_plus_simbench: pandapowerNet
    """
    net = test_grid_dbb_plus_simbench
    assert len(net.trafo) > 0, "fixture must contain transformers for this test"
    del net.trafo["max_loading_percent"]

    with caplog.at_level(logging.ERROR):
        run_nminus1_powerflow(net, pf_type="ac")

    assert not any("causes" in record.getMessage() for record in caplog.records), (
        "run_contingency logged swallowed KeyErrors -- the temporary max_loading_percent "
        "column was not added before contingency evaluation"
    )
    assert "max_loading_percent" not in net.trafo.columns, (
        "temporary max_loading_percent column was not removed from net.trafo"
    )
    assert "max_loading_percent" in net.res_line.columns, "N-1 line results missing"
