"""Tests for the N-1 sweep on the lightsim2grid backend.

Two things are pinned here. First, **routing**: an environment whose config says
``backend="lightsim"`` must run its N-1 contingencies through that backend -- always, including
when ``"n-1 parallel"`` is on -- while the default config must keep using pandapower unchanged.
Second, **parity**: the aggregates the sweep writes (``max_loading_percent`` and friends) have to
agree with ``run_nminus1_powerflow`` to solver tolerance, contingencies that island the grid
included, because those are what the observations and the greedy feedback read.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandapower_env.agents.greedy_worker import evaluate_actions
from pandapower_env.environments import gym_env_pp
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.toolbox import ls2g_backend
from pandapower_env.toolbox.ls2g_backend import LightsimBackend
from pandapower_env.toolbox.nminus1_parallel import _net_to_static_blob
from pandapower_env.toolbox.utils import run_nminus1_powerflow, run_powerflow

if TYPE_CHECKING:
    from pandapower.auxiliary import pandapowerNet

# The two paths assemble the same physics differently, so they agree to ~1e-11 rather than
# exactly; this leaves several orders of magnitude of headroom.
LOADING_TOLERANCE = 1e-6
RESET_INDEX = {"index": 12}
OVERLOAD_THRESHOLD_PERCENT = 100.0


def nminus1_env(reference: PPTopoGym, **overrides: object) -> PPTopoGym:
    """Build a second environment on the same grid, with N-1 on and the given config overrides."""
    config = dict(reference.orig_config)
    config["nminus1"] = True
    config.update(overrides)
    return PPTopoGym(config)


@pytest.mark.parametrize("parallel", [False, True])
def test_config_routes_nminus1_through_lightsim(simenv30, monkeypatch, parallel: bool) -> None:  # noqa: FBT001
    """``backend="lightsim"`` sends the sweep to the backend -- even with N-1 parallel asked for.

    The pandapower entry points are replaced by tripwires, so the test fails if any part of the
    sweep silently falls back to them.
    """
    def not_here(*_args: object, **_kwargs: object) -> None:
        msg = "the pandapower N-1 sweep must not run when the config selects the lightsim backend"
        raise AssertionError(msg)

    monkeypatch.setattr(gym_env_pp, "run_nminus1_powerflow", not_here)
    monkeypatch.setattr(gym_env_pp, "run_nminus1_powerflow_parallel", not_here)

    env = nminus1_env(simenv30, backend="lightsim", **{"n-1 parallel": parallel})
    env.reset(options=RESET_INDEX)
    env.step(0)

    assert env.net.converged
    assert "max_loading_percent" in env.net.res_line
    assert np.isfinite(env.net.res_line["max_loading_percent"].to_numpy()).any()


def test_default_backend_keeps_the_pandapower_sweep(simenv30, monkeypatch) -> None:
    """Without the config key nothing changes: the pandapower sweep still runs."""
    calls: list[str] = []
    original = gym_env_pp.run_nminus1_powerflow

    def record(
        net: pandapowerNet,
        pf_type: str = "ac",
        use_ls2g: str | bool = "auto",
        topk_percent: float = 100.0,
    ) -> None:
        calls.append("serial")
        original(net, pf_type, use_ls2g, topk_percent)

    monkeypatch.setattr(gym_env_pp, "run_nminus1_powerflow", record)

    env = nminus1_env(simenv30)
    env.reset(options=RESET_INDEX)
    env.step(0)

    assert calls, "the default backend must keep running pandapower's N-1"


@pytest.mark.parametrize("action", [0, 3, 5])
def test_nminus1_aggregates_match_pandapower(simenv30, action: int) -> None:
    """Tolerance parity on every column the observations read, split substations included."""
    reference = nminus1_env(simenv30)
    fast = nminus1_env(simenv30, backend="lightsim")

    reference.reset(options=RESET_INDEX)
    reference.step(action)
    if not reference.net.converged:
        pytest.skip(f"action {action} does not converge on the pandapower path")
    fast.reset(options=RESET_INDEX)
    fast.step(action)
    assert fast.net.converged, "the lightsim sweep diverged where pandapower converged"

    for table, columns in (
        ("res_line", ("max_loading_percent", "min_loading_percent")),
        ("res_bus", ("max_vm_pu", "min_vm_pu")),
    ):
        for column in columns:
            expected = reference.net[table][column].to_numpy()
            produced = fast.net[table][column].to_numpy()
            assert np.array_equal(np.isnan(expected), np.isnan(produced)), f"{table}.{column} NaNs"
            assert np.nanmax(np.abs(expected - produced)) < LOADING_TOLERANCE, f"{table}.{column}"

    # Decision parity: the same lines are reported N-1 overloaded, by the same contingencies.
    expected_overloads = reference.net.res_line["max_loading_percent"].to_numpy() > OVERLOAD_THRESHOLD_PERCENT
    produced_overloads = fast.net.res_line["max_loading_percent"].to_numpy() > OVERLOAD_THRESHOLD_PERCENT
    assert np.array_equal(expected_overloads, produced_overloads)
    assert np.array_equal(
        reference.net.res_line["causes_overloading"].to_numpy(),
        fast.net.res_line["causes_overloading"].to_numpy(),
    )


def test_islanding_contingency_is_solved_not_skipped(simenv30, monkeypatch) -> None:
    """A contingency that splits the grid is handed to pandapower rather than dropped.

    lightsim2grid has one slack bus, so it answers an islanding outage with an empty voltage
    vector. Skipping those cases would leave the sweep reporting a *lower* N-1 risk than the
    pandapower path does -- the failure this fallback exists to prevent.
    """
    fallbacks: list[bool] = []
    original = ls2g_backend._solve_with_pandapower

    def record(net: pandapowerNet) -> bool:
        fallbacks.append(True)
        return original(net)

    monkeypatch.setattr(ls2g_backend, "_solve_with_pandapower", record)

    env = nminus1_env(simenv30, backend="lightsim")
    env.reset(options=RESET_INDEX)
    env.step(0)

    assert fallbacks, "case30 has an islanding line outage; it must reach the pandapower fallback"


def test_topk_config_reaches_the_lightsim_sweep(simenv30) -> None:
    """``n-1-topk`` trims the contingency set here too, and can only lower the worst loading."""
    full = nminus1_env(simenv30, backend="lightsim")
    trimmed = nminus1_env(simenv30, backend="lightsim", **{"n-1-topk": 20.0})

    full.reset(options=RESET_INDEX)
    full.step(0)
    trimmed.reset(options=RESET_INDEX)
    trimmed.step(0)

    full_max = full.net.res_line["max_loading_percent"].to_numpy()
    trimmed_max = trimmed.net.res_line["max_loading_percent"].to_numpy()
    assert np.all(np.nan_to_num(trimmed_max) <= np.nan_to_num(full_max) + LOADING_TOLERANCE)
    # Fewer outages can only lower a line's worst case, and on case30 it demonstrably does for
    # some lines -- the *global* worst case survives, because the heaviest lines are exactly the
    # ones the filter keeps.
    assert np.any(np.nan_to_num(trimmed_max) < np.nan_to_num(full_max) - LOADING_TOLERANCE), \
        "a 20% contingency set must leave some line with a lower worst case"


def test_nminus1_matches_pandapower_on_a_grid_with_transformers(test_grid_dbb_plus_simbench) -> None:
    """Transformer outages are pushed into the model, not silently ignored.

    case30 has no transformers, so without this the ``deactivate_trafo`` half of the sweep would
    never be exercised -- and a contingency that is not applied looks like a perfectly healthy
    grid rather than an error.
    """
    net = test_grid_dbb_plus_simbench
    net.line["max_loading_percent"] = 100.0
    run_powerflow(net)

    reference = copy.deepcopy(net)
    run_nminus1_powerflow(reference)

    backend = LightsimBackend(net)
    assert backend.solve_nminus1(net)

    for table in ("res_line", "res_trafo"):
        for column in ("max_loading_percent", "min_loading_percent"):
            expected = reference[table][column].to_numpy()
            produced = net[table][column].to_numpy()
            assert np.nanmax(np.abs(expected - produced)) < LOADING_TOLERANCE, f"{table}.{column}"


def test_greedy_worker_scores_nminus1_through_lightsim(simenv30) -> None:
    """The greedy path honours the backend for its N-1 feedback, and scores the same actions."""
    env = simenv30
    env.reset(options=RESET_INDEX)
    blob = _net_to_static_blob(env.net)
    action_rows = [env.df_actions.loc[action].to_dict() for action in (0, 3, 5)]
    # The same arguments the greedy agent dispatches: without the topology snapshot the worker
    # scores each action on top of the previous one's switches, so the two calls would drift apart.
    state = {"base_topology": env.snapshot_topology(), "profile_slice": env.get_profile_slice(env.index)}

    expected = evaluate_actions(blob, action_rows, need_n1=True, backend="pandapower", **state)
    produced = evaluate_actions(blob, action_rows, need_n1=True, backend="lightsim", **state)

    for reference_result, fast_result in zip(expected, produced):
        assert reference_result["crashed"] == fast_result["crashed"]
        assert fast_result["nminus1"] == pytest.approx(reference_result["nminus1"], abs=1e-6)
