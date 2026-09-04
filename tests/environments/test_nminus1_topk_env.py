"""Environment-level tests for the ``n-1-topk`` config field.

Verifies the config plumbing onto ``PPTopoGym``, that ``run_pf`` honours the percentage, that
filtering can only lower the worst-case N-1 loading, and that serial and parallel N-1 agree
when ``n-1-topk`` is set.
"""

from __future__ import annotations

import copy

import numpy as np

from pandapower_env.environments.simulation_env import PPTopoGym


def _topk_max_loading(env_config: dict, topk_percent: float, *, parallel: bool = False) -> np.ndarray:
    """Build an env with N-1 + ``n-1-topk`` enabled, solve N-1 at index 0, return per-line worst loading."""
    config = copy.deepcopy(env_config)
    config["nminus1"] = True
    config["n-1-topk"] = topk_percent
    if parallel:
        config["n-1 parallel"] = True
        config["n-1 workers"] = 2
    env = PPTopoGym(config)
    env.reset(options={"index": 0})
    assert env.run_pf(nminus1=True) is True
    return env.net.res_line["max_loading_percent"].to_numpy()


def test_topk_config_default(env_config: dict) -> None:
    """Without the key, the env defaults to evaluating all lines (100%)."""
    assert PPTopoGym(env_config).nminus1_topk == 100.0  # noqa: PLR2004


def test_topk_config_is_read(env_config: dict) -> None:
    """The ``n-1-topk`` key is read onto the env."""
    config = copy.deepcopy(env_config)
    config["n-1-topk"] = 40.0
    assert PPTopoGym(config).nminus1_topk == 40.0  # noqa: PLR2004


def test_run_pf_with_topk_produces_max_loading(env_config: dict) -> None:
    """A topk-configured N-1 power flow converges and stores a real ``max_loading_percent``."""
    loading = _topk_max_loading(env_config, 50.0)  # helper asserts run_pf converged
    assert loading.size > 0
    assert np.isfinite(loading).any()


def test_topk_loading_is_below_full(env_config: dict) -> None:
    """Evaluating only the top 50%% of lines can only lower each line's worst-case loading."""
    full = _topk_max_loading(env_config, 100.0)
    filtered = _topk_max_loading(env_config, 50.0)
    assert np.all(np.nan_to_num(filtered) <= np.nan_to_num(full) + 1e-6)


def test_topk_serial_matches_parallel(env_config: dict) -> None:
    """Serial and parallel N-1 agree when ``n-1-topk`` filters the contingencies."""
    serial = _topk_max_loading(env_config, 50.0)
    parallel = _topk_max_loading(env_config, 50.0, parallel=True)
    assert np.allclose(serial, parallel, rtol=1e-9, atol=1e-6, equal_nan=True)
