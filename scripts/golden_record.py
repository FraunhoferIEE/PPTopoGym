"""Record and re-check a bit-for-bit fingerprint of everything PPTopoGym produces.

Run with::

    python scripts/golden_record.py --record            # write the baseline
    python scripts/golden_record.py --check             # compare against it
    python scripts/golden_record.py --record --grids case14,case30,case89

Why this exists: the refactor it guards deletes code, reorders work and changes *how many*
power flows run. The test suite proves the env still works; it does not prove the env still
produces the identical numbers. This walks every public surface -- build, episode, simulation
API, N-1, greedy -- and fingerprints every array, scalar and info value it yields, then
compares byte-for-byte. Floats are compared as raw bytes, so a NaN matches a NaN and a
one-ulp drift is a failure, which is exactly the contract "bit-identical" means.

The baseline is a JSON map of ``scenario key -> sha256 of the value's bytes``. Digests rather
than values keep it small and diffable, and the key names localise a mismatch to the exact
observation / step / grid that moved.

:raises SystemExit: if the baseline is missing on ``--check``, or if any digest differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_BASELINE = REPO_ROOT / "profiling" / "golden_record.json"

# Reset to a fixed timeseries row: `reset()` with no options picks a random scenario start,
# so every scenario here passes an explicit index (see the `reset` gotcha in CLAUDE.md).
# The case14 fixture carries only a 5-row profile table, so it has to start at 0.
RESET_INDICES = {"case14": 0}
DEFAULT_RESET_INDEX = 12
EPISODE_STEPS = 4


# --------------------------------------------------------------------------------------
# Digesting
# --------------------------------------------------------------------------------------


def digest(value: object) -> str:
    """Fingerprint any value this project produces, byte-exactly.

    Arrays contribute their dtype, shape and raw buffer, so ``NaN`` compares equal to ``NaN``
    and a one-ulp float change is visible. Containers recurse in a deterministic order.

    :param value: an array, scalar, string, DataFrame, or a nesting of those.
    :type value: object
    :return: a hex sha256 digest.
    :rtype: str
    """
    return hashlib.sha256(_to_bytes(value)).hexdigest()


def _to_bytes(value: object) -> bytes:
    """Serialize a value to bytes deterministically (helper for :func:`digest`)."""
    if value is None:
        return b"None"
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        if arr.dtype == object:
            return _to_bytes(list(arr.ravel())) + f"|shape{arr.shape}".encode()
        return f"{arr.dtype}|{arr.shape}|".encode() + arr.tobytes()
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return value.to_csv().encode()
    if isinstance(value, (bool, np.bool_)):
        return b"bool:1" if value else b"bool:0"
    if isinstance(value, (int, np.integer)):
        return f"int:{int(value)}".encode()
    if isinstance(value, (float, np.floating)):
        return b"float:" + np.float64(value).tobytes()
    if isinstance(value, (str, bytes)):
        return value if isinstance(value, bytes) else value.encode()
    if isinstance(value, dict):
        return b"{" + b",".join(
            _to_bytes(k) + b":" + _to_bytes(value[k]) for k in sorted(value, key=repr)
        ) + b"}"
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return b"[" + b",".join(_to_bytes(v) for v in items) + b"]"
    return repr(value).encode()


# --------------------------------------------------------------------------------------
# Scenario building
# --------------------------------------------------------------------------------------


def build_case14_config() -> dict:
    """Build the small 14-bus double-busbar config, mirroring the ``simenv`` test fixture.

    Self-contained so the record does not depend on pytest fixtures. The action list is the
    fixture's, deliberately including the actions that fail to converge (1, 2 and 4) -- the
    crash path writes `worst_reward`, an empty observation and a desynced action log, and is
    exactly the sort of thing a refactor breaks silently.

    :return: an ``env_config`` dict for :class:`PPTopoGym`.
    :rtype: dict
    """
    from pandapower.networks import case14

    from pandapower_env.substation.create_double_busbar_substation import (
        can_convert_to_n_busbar_substation,
        create_n_busbar_substation,
    )

    net = case14()
    net.gen["name"] = net.gen.index.to_series().apply(lambda x: f"Generator {x}")
    net.sgen["name"] = net.sgen.index.to_series().apply(lambda x: f"Static Generator {x}")
    net.load["name"] = net.load.index.to_series().apply(lambda x: f"Load {x}")

    n_gen, n_loads = len(net.gen), len(net.load)
    profile = [3.0, 3.0, 3.0, 3.0, 2.0]
    profile_end = [3.0, 3.0, 3.0, 3.0, 3.0]
    columns_list = [profile] * 2 * (n_loads - 1) + [profile_end] * 2
    df_loads = pd.DataFrame(columns_list).T
    df_loads.columns = [
        col for i in range(1, n_loads + 1) for col in (f"load {i}_pload", f"load {i}_qload")
    ]
    df_powerplants = pd.DataFrame(
        {f"profile{i + 1}": [5.0, 5.0, 5.0, 5.0, 5.0] for i in range(n_gen)},
    )
    net.profiles = {"load": df_loads, "powerplants": df_powerplants}

    for ibus in net.bus.index:
        if can_convert_to_n_busbar_substation(net, ibus):
            create_n_busbar_substation(net, ibus)

    dict_actions = [
        {"action": 0, "substations": [], "states": []},
        {"action": 1, "substations": [0], "states": ["0x110101"]},
        {
            "action": 2, "substations": [0, 1], "states": ["0x101101", "0x1100"],
            "lines": [5], "disconnect_lines": [True],
        },
        {"action": 3, "substations": [], "states": [], "lines": [5], "disconnect_lines": [False]},
        {"action": 4, "substations": [], "states": [], "lines": [2], "disconnect_lines": [True]},
    ]
    actions = [defaultdict(list, a) for a in dict_actions]
    actions.append(defaultdict(list, {"action": 5, "disconnect_lines": [1]}))

    return {
        "n_episodes": 10,
        "episode_length": 5,
        "net": net,
        "action_space": actions,
        "nminus1": False,
    }


def build_config(grid: str) -> dict:
    """Return the ``env_config`` for a named grid.

    :param grid: one of ``case14``, ``case30``, ``case89``.
    :type grid: str
    :raises SystemExit: if the grid name is unknown.
    :return: the config dict.
    :rtype: dict
    """
    if grid == "case14":
        return build_case14_config()
    from pandapower_env.data.example_configs import config_case30, config_case89

    if grid == "case30":
        return config_case30()
    if grid == "case89":
        return config_case89()
    sys.exit(f"unknown grid: {grid}")


# --------------------------------------------------------------------------------------
# Scenarios -- each fills `record` with `key -> digest`
# --------------------------------------------------------------------------------------


def record_build(record: dict[str, str], grid: str, config: dict) -> None:
    """Fingerprint the constructed grid and its action space."""
    net = config["net"]
    for table in ("bus", "line", "trafo", "switch", "load", "gen", "sgen", "ext_grid"):
        if table in net and len(net[table]):
            record[f"build/{grid}/net.{table}"] = digest(net[table])
    if "multi_bb_substation" in net:
        record[f"build/{grid}/net.multi_bb_substation"] = digest(net["multi_bb_substation"])
    record[f"build/{grid}/n_actions"] = digest(len(config["action_space"]))


def record_episode(
    record: dict[str, str], grid: str, env: Any, actions: list[int], index: int,
) -> None:
    """Fingerprint a full reset + fixed action sequence: obs, reward, flags and info."""
    obs, info = env.reset(options={"index": index})
    record[f"episode/{grid}/reset/info"] = digest(_clean_info(info))
    for key, value in obs.items():
        record[f"episode/{grid}/reset/obs/{key}"] = digest(value)

    for step_idx, action in enumerate(actions):
        obs, reward, terminated, truncated, info = env.step(action)
        prefix = f"episode/{grid}/step{step_idx}"
        record[f"{prefix}/action"] = digest(action)
        record[f"{prefix}/reward"] = digest(reward)
        record[f"{prefix}/terminated"] = digest(terminated)
        record[f"{prefix}/truncated"] = digest(truncated)
        record[f"{prefix}/info"] = digest(_clean_info(info))
        for key, value in obs.items():
            record[f"{prefix}/obs/{key}"] = digest(value)
        # These two counters advance on different schedules and neither is restored by
        # end_simulation / restore_state -- pin both explicitly (see CLAUDE.md).
        record[f"{prefix}/current_step"] = digest(env.current_step)
        record[f"{prefix}/episode_step_counter"] = digest(env.episode_step_counter)
        record[f"{prefix}/index"] = digest(env.index)
        if terminated or truncated:
            obs, info = env.reset(options={"index": index})


def record_simulation_api(
    record: dict[str, str], grid: str, env: Any, actions: list[int], index: int,
) -> None:
    """Fingerprint simulation / verify_action / save_state round trips and their restores."""
    env.reset(options={"index": index})
    for action in actions:
        outputs = env.simulation([action])
        for out_idx, out in enumerate(outputs):
            prefix = f"sim/{grid}/simulation{action}/out{out_idx}"
            record[f"{prefix}/reward"] = digest(out.reward)
            record[f"{prefix}/terminated"] = digest(out.terminated)
            record[f"{prefix}/info"] = digest(_clean_info(out.info))
            for key, value in out.observation.items():
                record[f"{prefix}/obs/{key}"] = digest(value)
        # The restore is the point: the grid must be back where it started. The two step
        # counters are pinned explicitly because end_simulation treats them differently from
        # each other (see CLAUDE.md) and a restore rewrite is exactly what would move them.
        record[f"sim/{grid}/simulation{action}/after/topology"] = digest(env.snapshot_topology())
        record[f"sim/{grid}/simulation{action}/after/index"] = digest(env.index)
        record[f"sim/{grid}/simulation{action}/after/current_step"] = digest(env.current_step)
        record[f"sim/{grid}/simulation{action}/after/episode_step"] = digest(env.episode_step_counter)
        record[f"sim/{grid}/simulation{action}/after/res_line"] = digest(env.net.res_line)
        record[f"sim/{grid}/simulation{action}/after/switch"] = digest(env.net.switch["closed"].to_numpy())
        record[f"sim/{grid}/verify{action}"] = digest(env.verify_action(action))

    state = env.save_state()
    env.step(actions[-1])
    env.restore_state(state, run_pf=True)
    record[f"sim/{grid}/save_restore/topology"] = digest(env.snapshot_topology())
    record[f"sim/{grid}/save_restore/index"] = digest(env.index)
    record[f"sim/{grid}/save_restore/res_line"] = digest(env.net.res_line)


def record_state_transfer(
    record: dict[str, str], grid: str, env: Any, env2: Any, index: int,
) -> None:
    """Fingerprint ``state_to_info`` -> ``state_from_info`` across two instances."""
    env.reset(options={"index": index})
    env.step(0)
    info = env.state_to_info()
    env2.state_from_info(info)
    record[f"state/{grid}/transfer/topology"] = digest(env2.snapshot_topology())
    record[f"state/{grid}/transfer/index"] = digest(env2.index)
    record[f"state/{grid}/transfer/res_line"] = digest(env2.net.res_line)


def record_nminus1(record: dict[str, str], grid: str, config: dict) -> None:
    """Fingerprint the serial and parallel N-1 sweeps at full and reduced top-k."""
    import copy

    from pandapower_env.toolbox.nminus1_parallel import run_nminus1_powerflow_parallel
    from pandapower_env.toolbox.utils import run_nminus1_powerflow

    for topk in (100.0, 50.0):
        for label, runner in (
            ("serial", lambda n, k=topk: run_nminus1_powerflow(n, topk_percent=k)),
            ("parallel", lambda n, k=topk: run_nminus1_powerflow_parallel(n, n_workers=2, topk_percent=k)),
        ):
            net = copy.deepcopy(config["net"])
            runner(net)
            prefix = f"n1/{grid}/topk{topk:g}/{label}"
            for col in ("max_loading_percent", "min_loading_percent", "cause_element", "cause_index"):
                if col in net.res_line:
                    record[f"{prefix}/res_line.{col}"] = digest(net.res_line[col].to_numpy())
            for col in ("max_vm_pu", "min_vm_pu"):
                if col in net.res_bus:
                    record[f"{prefix}/res_bus.{col}"] = digest(net.res_bus[col].to_numpy())


def record_greedy(record: dict[str, str], grid: str, config: dict, index: int) -> None:
    """Fingerprint the greedy agent's chosen action, serial and parallel."""
    import copy

    from gymnasium import spaces

    from pandapower_env.agents.benchmark_agents import GreedyAgent

    for n_workers in (1, 2):
        agent = GreedyAgent(
            action_space=spaces.Discrete(len(config["action_space"])),
            env_config=copy.deepcopy(config),
            n_workers=n_workers,
            seed=7,
        )
        obs, _ = agent.env.reset(options={"index": index})
        info = agent.env.state_to_info()
        # state_from_info refuses to run on the producing instance, so hand the agent an
        # empty info and let it score the state it already holds.
        action = agent.act(obs, {})
        record[f"greedy/{grid}/workers{n_workers}/action"] = digest(int(action))
        del info


def _clean_info(info: dict) -> dict:
    """Drop the parts of an info dict that are not reproducible across processes."""
    # _source_instance_id is id(self); prev_actions is a LoggedArray whose repr carries
    # object identity. Both are meaningless as a fingerprint.
    skip = {"_source_instance_id"}
    return {
        k: (np.asarray(v.to_numpy()) if hasattr(v, "to_numpy") else v)
        for k, v in info.items()
        if k not in skip
    }


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def collect(grids: list[str]) -> dict[str, str]:
    """Run every scenario on every grid and return the full digest map."""
    from pandapower_env.environments.simulation_env import PPTopoGym

    record: dict[str, str] = {}
    for grid in grids:
        print(f"  building {grid} ...", flush=True)
        config = build_config(grid)
        index = RESET_INDICES.get(grid, DEFAULT_RESET_INDEX)
        record_build(record, grid, config)

        n_actions = len(config["action_space"])
        actions = [a for a in (0, 1, 2, 3, 4, 5) if a < n_actions][:EPISODE_STEPS + 2]

        env = PPTopoGym(config)
        print(f"  {grid}: episode ...", flush=True)
        record_episode(record, grid, env, actions[:EPISODE_STEPS], index)
        print(f"  {grid}: simulation api ...", flush=True)
        record_simulation_api(record, grid, env, actions[:3], index)
        print(f"  {grid}: state transfer ...", flush=True)
        record_state_transfer(record, grid, env, PPTopoGym(env.orig_config), index)
        print(f"  {grid}: n-1 ...", flush=True)
        record_nminus1(record, grid, config)
        print(f"  {grid}: greedy ...", flush=True)
        record_greedy(record, grid, config, index)
    return record


def main() -> None:
    """Parse arguments, then record or check the golden record."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="write the baseline")
    parser.add_argument("--check", action="store_true", help="compare against the baseline")
    parser.add_argument("--grids", default="case14,case30", help="comma-separated grid names")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()

    if args.record == args.check:
        sys.exit("pass exactly one of --record / --check")

    grids = [g.strip() for g in args.grids.split(",") if g.strip()]
    print(f"grids: {', '.join(grids)}", flush=True)
    current = collect(grids)
    print(f"collected {len(current)} fingerprints", flush=True)

    if args.record:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(current, indent=1, sort_keys=True))
        print(f"wrote {args.baseline}")
        return

    if not args.baseline.exists():
        sys.exit(f"no baseline at {args.baseline} -- run --record first")
    baseline = json.loads(args.baseline.read_text())

    missing = sorted(set(baseline) - set(current))
    added = sorted(set(current) - set(baseline))
    changed = sorted(k for k in set(baseline) & set(current) if baseline[k] != current[k])

    for label, keys in (("MISSING", missing), ("ADDED", added), ("CHANGED", changed)):
        for key in keys[:40]:
            print(f"{label:8s} {key}")
        if len(keys) > 40:
            print(f"{label:8s} ... and {len(keys) - 40} more")

    if missing or added or changed:
        sys.exit(
            f"golden record MISMATCH: {len(changed)} changed, "
            f"{len(missing)} missing, {len(added)} added",
        )
    print(f"golden record OK ({len(current)} fingerprints identical)")


if __name__ == "__main__":
    main()
