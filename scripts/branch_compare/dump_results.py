"""Dump the physical results of a fixed scenario, so two branches / backends can be diffed.

``bench_branch.py`` answers "how fast"; this answers "and does it still say the same thing".
It plays a deterministic scenario -- a DoNothing walk over the first timesteps, then a set of
switching actions applied from timestep 0 -- and writes every number a consumer of this
environment actually reads: the reward, the convergence decision, line loadings, transformer
loadings and bus voltages.

Only quantities that mean the same thing on both branches are dumped. Bus *indices* do not:
``develop`` and this branch expand a substation into a different number of auxiliary buses, so
the comparison is restricted to the ``n_bus`` original buses, which both keep at their original
labels. Lines and transformers are never duplicated by the expansion, so those tables compare
row for row.

Run with::

    python dump_results.py --grid grid_30.pkl --backend pandapower --actions 5,7 --out a.npz
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from bench_branch import RESET_OPTIONS, expand_grid, make_config

from pandapower_env.environments.simulation_env import PPTopoGym


def scenario_record(env: PPTopoGym, n_bus: int, action: int, index: int) -> dict[str, np.ndarray]:
    """Reset to ``index``, apply ``action`` and capture everything the env exposes about it.

    :param env: the environment to drive.
    :param n_bus: number of original (pre-expansion) buses to keep from ``res_bus``.
    :param action: the action index to apply.
    :param index: the timeseries index to reset to.
    :return: named arrays, all float64 so a diff is a plain subtraction.
    """
    env.reset(options={**RESET_OPTIONS, "index": index})
    _obs, reward, terminated, _truncated, _info = env.step(action)
    net = env.net
    # step() reports non-convergence as `terminated` plus worst_reward; net.converged is the
    # flag the env itself set, and is what both branches expose identically.
    converged = bool(net.converged) and not terminated
    trafo_loading = (
        net.res_trafo["loading_percent"].to_numpy(dtype=float)
        if len(net.trafo) and not net.res_trafo.empty
        else np.zeros(0)
    )
    return {
        "reward": np.array([float(reward)]),
        "converged": np.array([float(converged)]),
        "line_loading": net.res_line["loading_percent"].to_numpy(dtype=float),
        "trafo_loading": trafo_loading,
        "bus_vm": net.res_bus["vm_pu"].to_numpy(dtype=float)[:n_bus],
        "bus_va": net.res_bus["va_degree"].to_numpy(dtype=float)[:n_bus],
    }


def main() -> None:
    """Play the scenario on one branch/backend and write the arrays to an ``.npz``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--backend", default="pandapower")
    parser.add_argument("--actions", default="", help="comma-separated switching actions")
    parser.add_argument("--steps", type=int, default=5, help="DoNothing timesteps to walk")
    parser.add_argument("--actions-cache", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with args.grid.open("rb") as handle:
        payload = pickle.load(handle)
    net, actions = expand_grid(payload, args.actions_cache)
    config = make_config(net, actions, payload["profiles"], nminus1=False)
    config["backend"] = args.backend
    env = PPTopoGym(config)

    record: dict[str, np.ndarray] = {}
    for index in range(args.steps):
        for key, value in scenario_record(env, payload["n_bus"], 0, index).items():
            record[f"donothing/t{index}/{key}"] = value
    for action in [int(a) for a in args.actions.split(",") if a.strip()]:
        for key, value in scenario_record(env, payload["n_bus"], action, 0).items():
            record[f"action{action}/{key}"] = value

    np.savez(args.out, **record)
    print(f"wrote {len(record)} arrays to {args.out}")  # noqa: T201


if __name__ == "__main__":
    main()
