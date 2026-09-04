"""Environment-level tests for the "n-1 parallel" config field.

Verifies the config plumbing, that a parallel-configured env produces the same N-1 results as
a serial one, and -- replicating the MuZero usage -- that an env with ``"n-1 parallel": True``
works flawlessly inside a ``spawn`` multiprocessing child (where it safely falls back to serial).
"""

from __future__ import annotations

import copy
import multiprocessing as mp

import numpy as np

from pandapower_env.environments.simulation_env import PPTopoGym


def _nminus1_max_loading(env: PPTopoGym) -> np.ndarray:
    """Reset to the (converging) first timestep and return the per-line N-1 worst loading."""
    env.reset(options={"index": 0})
    env.run_pf(nminus1=True)
    return env.net.res_line["max_loading_percent"].to_numpy()


def _nminus1_in_spawn_child(config: dict) -> list[float]:
    """Top-level worker: build a PPTopoGym from ``config`` and compute N-1 (run in a spawn child).

    Must be module-level so the ``spawn`` start method can import it. Inside the child,
    ``multiprocessing.parent_process()`` is set, so the parallel N-1 falls back to serial.
    """
    env = PPTopoGym(config)
    return _nminus1_max_loading(env).tolist()


def _parallel_config(env_config: dict, workers: int) -> dict:
    """Return a copy of ``env_config`` with N-1 enabled and run in parallel with ``workers`` workers."""
    config = copy.deepcopy(env_config)
    config["nminus1"] = True
    config["n-1 parallel"] = True
    config["n-1 workers"] = workers
    return config


def test_nminus1_parallel_config_defaults(env_config: dict) -> None:
    """Without the keys, the env defaults to serial N-1 (parallel off, workers unset)."""
    env = PPTopoGym(env_config)
    assert env.nminus1_parallel is False
    assert env.nminus1_workers is None


def test_nminus1_parallel_config_is_read(env_config: dict) -> None:
    """The "n-1 parallel" / "n-1 workers" keys are read onto the env."""
    env = PPTopoGym(_parallel_config(env_config, workers=2))
    assert env.nminus1_parallel is True
    assert env.nminus1_workers == 2  # noqa: PLR2004


def test_env_parallel_matches_serial(env_config: dict) -> None:
    """An env computing N-1 in parallel matches the same env computing N-1 serially."""
    serial_config = copy.deepcopy(env_config)
    serial_config["nminus1"] = True

    serial_loading = _nminus1_max_loading(PPTopoGym(serial_config))
    parallel_loading = _nminus1_max_loading(PPTopoGym(_parallel_config(env_config, workers=2)))

    assert np.allclose(serial_loading, parallel_loading, rtol=1e-9, atol=1e-6, equal_nan=True)


def test_env_with_parallel_flag_under_spawn(env_config: dict) -> None:
    """An env with parallel N-1 enabled runs flawlessly inside a spawn child (MuZero usage).

    The child must not deadlock or crash and must produce the serial results (the parallel
    backend falls back to serial inside a worker process to avoid nested pools).
    """
    serial_config = copy.deepcopy(env_config)
    serial_config["nminus1"] = True
    expected = _nminus1_max_loading(PPTopoGym(serial_config))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=1) as pool:
        from_child = pool.apply(_nminus1_in_spawn_child, (_parallel_config(env_config, workers=4),))

    assert np.allclose(from_child, expected, rtol=1e-9, atol=1e-6, equal_nan=True)
