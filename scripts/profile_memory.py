"""Measure the per-environment memory cost of ``PPTopoGym``.

Run with::

    poetry run python scripts/profile_memory.py [--n-envs 10]

Builds one ``config_case30`` and then N environments from it, reporting the resident memory
each additional environment costs and where that memory sits. This is the measurement that
drove the profile-sharing work: the Simbench timeseries used to be copied per environment
(live net + ``net_copy_from`` + ``_orig_config`` + the derived ``df_profiles_*``), which made
the multi-environment / vectorized setting expensive for data that is read-only after
``setup_profiles``.

Reference numbers on case30 (120-core linux box, 2026-08-25)::

    before sharing:  53.9 MB per env  (53.1 MB of it profile tables)
    after sharing :   0.7 MB per env

Profiling lives here, not in the core environment: nothing under ``pandapower_env/`` imports
a profiler, so the shipped env carries no profiling overhead.
"""

from __future__ import annotations

import argparse
import gc

import psutil

from pandapower_env.data.example_configs import config_case30
from pandapower_env.environments.simulation_env import PPTopoGym

PROFILE_TABLE_NAMES = (
    "df_profiles_load_p",
    "df_profiles_load_q",
    "df_profiles_sgen_p",
    "df_profiles_sgen_q",
    "df_profiles_gen_p",
    "df_profiles_gen_vm",
)


def resident_mb() -> float:
    """Resident set size in MB, after a full collection so freed objects are not counted."""
    gc.collect()
    return psutil.Process().memory_info().rss / 1e6


def dataframe_mb(frame: object) -> float:
    """Deep memory use of a DataFrame in MB (0.0 for anything without the accessor)."""
    usage = getattr(frame, "memory_usage", None)
    return float(usage(deep=True).sum()) / 1e6 if usage is not None else 0.0


def report_profile_footprint(env: PPTopoGym) -> None:
    """Print where one environment's profile memory sits, and whether it is shared."""
    derived = sum(dataframe_mb(getattr(env, name)) for name in PROFILE_TABLE_NAMES)
    raw_live = sum(dataframe_mb(df) for df in env.net.profiles.values())

    print(f"  derived df_profiles_*      : {derived:8.1f} MB")  # noqa: T201
    print(f"  net.profiles (raw)         : {raw_live:8.1f} MB")  # noqa: T201
    print(  # noqa: T201
        "  net_copy_from shares raw   : "
        f"{env.net_copy_from.profiles is env.net.profiles}",
    )
    print(  # noqa: T201
        "  _orig_config shares raw    : "
        f"{env._orig_config['net'].profiles is env.net.profiles}",  # noqa: SLF001
    )


def main() -> None:
    """Build N environments from one config and report the marginal cost of each."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-envs", type=int, default=10, help="How many environments to build.")
    args = parser.parse_args()

    print("Building config_case30 (slow: scaling + substations + action verification)...")  # noqa: T201
    config = config_case30()

    before = resident_mb()
    envs = [PPTopoGym(config) for _ in range(args.n_envs)]
    after = resident_mb()

    print(f"\n{args.n_envs} environments: +{after - before:.1f} MB total")  # noqa: T201
    print(f"  per environment          : {(after - before) / args.n_envs:8.2f} MB")  # noqa: T201

    print("\nProfile footprint of one environment:")  # noqa: T201
    report_profile_footprint(envs[0])

    shared = all(
        getattr(envs[0], name) is getattr(env, name)
        for env in envs[1:]
        for name in PROFILE_TABLE_NAMES
        if not getattr(envs[0], name).empty
    )
    print(f"\n  derived tables shared across envs: {shared}")  # noqa: T201


if __name__ == "__main__":
    main()
