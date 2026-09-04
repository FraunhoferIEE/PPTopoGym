"""Tests for the ``n-1-topk`` line-contingency filter (top-k%% of lines by apparent power).

Covers :func:`select_topk_line_contingencies` (the ranking / rounding / NaN handling) and the
``topk_percent`` plumbing through the serial (:func:`run_nminus1_powerflow`) and parallel
(:func:`run_nminus1_powerflow_parallel`) N-1 backends -- including that the parallel filter
stays identical to the serial one.
"""

from __future__ import annotations

import copy
import math

import numpy as np
import pandapower as pp
import pytest

from pandapower_env.toolbox.nminus1_parallel import run_nminus1_powerflow_parallel
from pandapower_env.toolbox.utils import run_nminus1_powerflow, select_topk_line_contingencies


def _expected_topk_lines(net: pp.pandapowerNet, topk_percent: float) -> np.ndarray:
    """Independently re-derive the kept line indices from apparent power ``S = |P + jQ|`` (MVA).

    Mirrors the production ranking (max of the from/to ends, NaN reactive power -> 0, all-NaN
    row -> ``-inf``) so a test can pin the metric (MVA, *not* ``loading_percent``) and the
    round-up rule without trusting the implementation under test.
    """
    res = net.res_line
    s_from = np.hypot(res["p_from_mw"].to_numpy(), np.nan_to_num(res["q_from_mvar"].to_numpy(), nan=0.0))
    s_to = np.hypot(res["p_to_mw"].to_numpy(), np.nan_to_num(res["q_to_mvar"].to_numpy(), nan=0.0))
    apparent_power = np.nan_to_num(np.fmax(s_from, s_to), nan=-np.inf)
    line_index = net.line.index.to_numpy()
    n_keep = math.ceil(len(line_index) * topk_percent / 100.0)
    kept = line_index[np.argsort(apparent_power, kind="stable")[::-1][:n_keep]]
    return np.sort(kept)


def test_topk_100_returns_all_lines(test_grid_dbb_plus_simbench: pp.pandapowerNet) -> None:
    """At 100%% (and above) every line is kept, in the original index order."""
    net = test_grid_dbb_plus_simbench
    pp.runpp(net)
    all_lines = net.line.index.to_numpy()
    assert np.array_equal(select_topk_line_contingencies(net, 100.0), all_lines)
    assert np.array_equal(select_topk_line_contingencies(net, 150.0), all_lines)


def test_topk_empty_net_returns_empty() -> None:
    """A net without lines returns an empty selection without touching ``res_line``."""
    net = pp.create_empty_network()
    result = select_topk_line_contingencies(net, 50.0)
    assert result.size == 0


@pytest.mark.parametrize("topk_percent", [0.0, -5.0])
def test_topk_nonpositive_raises(test_grid_dbb_plus_simbench: pp.pandapowerNet, topk_percent: float) -> None:
    """A non-positive percentage is rejected."""
    net = test_grid_dbb_plus_simbench
    pp.runpp(net)
    with pytest.raises(ValueError, match="topk_percent"):
        select_topk_line_contingencies(net, topk_percent)


@pytest.mark.parametrize("topk_percent", [1.0, 25.0, 50.0, 99.0])
def test_topk_rounding_and_ranking(test_grid_dbb_plus_simbench: pp.pandapowerNet, topk_percent: float) -> None:
    """The selection rounds up, stays sorted/subset, and matches the apparent-power ranking."""
    net = test_grid_dbb_plus_simbench
    pp.runpp(net)
    n_lines = len(net.line)
    result = select_topk_line_contingencies(net, topk_percent)

    assert len(result) == math.ceil(n_lines * topk_percent / 100.0)
    assert np.all(np.diff(result) > 0)  # strictly ascending
    assert set(result.tolist()).issubset(set(net.line.index.tolist()))
    assert np.array_equal(result, _expected_topk_lines(net, topk_percent))


def test_topk_rounds_up_to_single_most_loaded_line(test_grid_dbb_plus_simbench: pp.pandapowerNet) -> None:
    """A tiny percentage keeps exactly one line: the single highest apparent-power line."""
    net = test_grid_dbb_plus_simbench
    pp.runpp(net)
    result = select_topk_line_contingencies(net, 1.0)  # ceil(15 * 0.01) -> 1
    assert len(result) == 1
    assert np.array_equal(result, _expected_topk_lines(net, 1.0))


def test_topk_excludes_nan_power_lines(test_grid_dbb_plus_simbench: pp.pandapowerNet) -> None:
    """A line with NaN power flow ranks last (treated as -inf) and is dropped when filtering."""
    net = test_grid_dbb_plus_simbench
    pp.runpp(net)
    nan_line = int(select_topk_line_contingencies(net, 1.0)[0])  # would top the ranking otherwise
    net.res_line.loc[nan_line, ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]] = np.nan

    result = select_topk_line_contingencies(net, 50.0)  # keeps 8 of 15 lines
    assert nan_line not in result.tolist()


def test_serial_topk_loading_is_below_full(test_grid_dbb_plus_simbench: pp.pandapowerNet) -> None:
    """Filtering the contingency set can only lower each line's worst-case N-1 loading."""
    net = test_grid_dbb_plus_simbench
    run_nminus1_powerflow(net, topk_percent=100.0)
    full = net.res_line["max_loading_percent"].to_numpy().copy()

    run_nminus1_powerflow(net, topk_percent=50.0)
    filtered = net.res_line["max_loading_percent"].to_numpy()

    assert "max_loading_percent" in net.res_line.columns
    assert np.all(np.nan_to_num(filtered) <= np.nan_to_num(full) + 1e-6)


def test_parallel_topk_matches_serial(test_grid_dbb_plus_simbench: pp.pandapowerNet) -> None:
    """The genuine-parallel backend filters identically to the serial backend."""
    serial_net = test_grid_dbb_plus_simbench
    parallel_net = copy.deepcopy(serial_net)

    run_nminus1_powerflow(serial_net, topk_percent=50.0)
    run_nminus1_powerflow_parallel(parallel_net, n_workers=2, topk_percent=50.0)

    assert np.allclose(
        serial_net.res_line["max_loading_percent"].to_numpy(),
        parallel_net.res_line["max_loading_percent"].to_numpy(),
        rtol=1e-9,
        atol=1e-6,
        equal_nan=True,
    )


def test_parallel_topk_serial_fallback_matches(test_grid_dbb_plus_simbench: pp.pandapowerNet) -> None:
    """With a single worker the parallel entry point falls back to serial and still filters."""
    fallback_net = test_grid_dbb_plus_simbench
    serial_net = copy.deepcopy(fallback_net)

    run_nminus1_powerflow_parallel(fallback_net, n_workers=1, topk_percent=50.0)
    run_nminus1_powerflow(serial_net, topk_percent=50.0)

    assert np.allclose(
        fallback_net.res_line["max_loading_percent"].to_numpy(),
        serial_net.res_line["max_loading_percent"].to_numpy(),
        rtol=1e-9,
        atol=1e-6,
        equal_nan=True,
    )
