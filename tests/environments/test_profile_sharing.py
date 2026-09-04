"""
Tests for the memory-sharing of profile tables between environments.

Building an environment used to copy the Simbench timeseries several times over (the live
net, the ``net_copy_from`` oracle and the stored ``_orig_config``), which dominated the
per-environment footprint and made the vectorized / multi-environment setting expensive.

``deepcopy_net_sharing_profiles`` and the ``_SHARED_PROFILE_TABLES`` cache let environments
built from the same inputs reference *one* set of profile tables. These tests pin the two
properties that makes safe:

1. **Sharing happens** -- the raw and derived profile tables are the same objects across
   environments, and the shared entry is released once the environments are gone.
2. **Nothing observable changes** -- a sharing environment produces exactly the same
   profiles, observations and rewards as one that built its own tables.
"""

from __future__ import annotations

import copy
import gc
import weakref

import numpy as np
import pytest

from pandapower_env.environments.gym_env_pp import deepcopy_net_sharing_profiles
from pandapower_env.environments.simulation_env import PPTopoGym

SENTINEL_P_MW = 12345.0  # a value no fixture grid uses, to prove copies stay independent

PROFILE_TABLES = (
    "df_profiles_load_p",
    "df_profiles_load_q",
    "df_profiles_sgen_p",
    "df_profiles_sgen_q",
    "df_profiles_gen_p",
    "df_profiles_gen_vm",
)


def test_deepcopy_net_sharing_profiles_shares_only_profiles(test_grid_dbb_plus_simbench) -> None:
    """The copy shares ``profiles`` but owns every element table it may mutate."""
    net = test_grid_dbb_plus_simbench
    clone = deepcopy_net_sharing_profiles(net)

    assert clone.profiles is net.profiles, "profiles must be shared, not copied"
    for table in ("load", "line", "switch", "bus"):
        assert clone[table] is not net[table], f"{table} must be an independent copy"

    # Mutating the clone's element tables must not reach the original.
    clone.load.loc[clone.load.index[0], "p_mw"] = SENTINEL_P_MW
    assert net.load["p_mw"].iloc[0] != SENTINEL_P_MW


def test_deepcopy_net_without_profiles_still_copies(test_grid) -> None:
    """A net carrying no profiles falls back to a plain deepcopy."""
    net = test_grid
    assert "profiles" not in net or not net.get("profiles")
    clone = deepcopy_net_sharing_profiles(net)
    assert clone is not net
    assert clone.bus is not net.bus


def test_envs_from_one_config_share_profile_tables(env_config) -> None:
    """Two envs built from the same config reference the same derived profile tables."""
    first = PPTopoGym(env_config)
    second = PPTopoGym(env_config)

    for name in PROFILE_TABLES:
        first_table, second_table = getattr(first, name), getattr(second, name)
        if first_table.empty:
            continue
        assert first_table is second_table, f"{name} should be shared between environments"

    assert first.net.profiles is second.net.profiles
    assert first.net_copy_from.profiles is first.net.profiles


def test_shared_profiles_do_not_change_values(env_config) -> None:
    """A sharing env has identical profile *values* to one that built its own tables."""
    reference = PPTopoGym(env_config)
    expected = {name: getattr(reference, name).copy() for name in PROFILE_TABLES}

    sharing = PPTopoGym(env_config)
    for name in PROFILE_TABLES:
        actual = getattr(sharing, name)
        assert actual.shape == expected[name].shape, name
        np.testing.assert_array_equal(actual.to_numpy(), expected[name].to_numpy(), err_msg=name)


def test_shared_profiles_do_not_change_steps(env_config) -> None:
    """Stepping two envs that share tables yields identical rewards and observations."""
    first = PPTopoGym(env_config)
    second = PPTopoGym(env_config)

    for env in (first, second):
        env.reset(options={"index": 0})

    for action in (0, 1, 0):
        obs_first, reward_first, *_ = first.step(action)
        obs_second, reward_second, *_ = second.step(action)
        assert reward_first == pytest.approx(reward_second)
        assert obs_first.keys() == obs_second.keys()
        for key in obs_first:
            np.testing.assert_allclose(
                np.asarray(obs_first[key], dtype=float),
                np.asarray(obs_second[key], dtype=float),
                err_msg=f"observation {key!r} diverged at action {action}",
            )


def test_env_with_different_base_values_does_not_share(env_config) -> None:
    """Scaling the net's base load changes the derived tables, so they must not be shared."""
    baseline = PPTopoGym(env_config)

    scaled_config = copy.deepcopy(env_config)
    scaled_config["net"].load["p_mw"] = scaled_config["net"].load["p_mw"] * 2.0
    scaled = PPTopoGym(scaled_config)

    assert scaled.df_profiles_load_p is not baseline.df_profiles_load_p
    np.testing.assert_allclose(
        scaled.df_profiles_load_p.to_numpy(),
        baseline.df_profiles_load_p.to_numpy() * 2.0,
    )


def test_shared_profile_entry_is_released(env_config) -> None:
    """The cache entry these envs share disappears once they are all collected.

    Tracked by weak reference to the entry itself rather than by the size of the cache,
    which other tests in the same session also populate.
    """
    config = copy.deepcopy(env_config)  # own net, so the fixture cannot keep the entry alive
    envs = [PPTopoGym(config) for _ in range(3)]

    shared = envs[0]._shared_profile_tables
    assert shared is not None, "environments should be sharing a profile-table entry"
    assert all(env._shared_profile_tables is shared for env in envs)
    entry_ref = weakref.ref(shared)

    del envs, config, shared
    gc.collect()
    assert entry_ref() is None, "the shared profile tables were not released"


def test_cache_entry_pins_the_profiles_its_key_names(env_config) -> None:
    """A live cache entry keeps the raw profile tables its key identifies by ``id()`` alive.

    The key is built from ``id()`` of the raw profile DataFrames. If those could be collected
    while the entry lived, CPython could reuse an ``id()`` for a different DataFrame and the
    cache would serve the wrong timeseries. The entry therefore holds them.
    """
    config = copy.deepcopy(env_config)
    env = PPTopoGym(config)

    entry = env._shared_profile_tables
    assert entry is not None
    assert entry.source_profiles is env.net.profiles, "the entry must pin its key's objects"

    # Even with every other reference to the config gone, the pinned tables stay alive.
    profile_ids = {id(df) for df in env.net.profiles.values()}
    del config
    gc.collect()
    assert {id(df) for df in entry.source_profiles.values()} == profile_ids


def test_orig_config_rebuilds_an_equivalent_env(env_config) -> None:
    """``orig_config`` still yields a config that builds an independent, equivalent env."""
    env = PPTopoGym(env_config)
    env.reset(options={"index": 0})
    expected_obs, expected_reward, *_ = env.step(1)

    clone = PPTopoGym(env.orig_config)
    clone.reset(options={"index": 0})
    actual_obs, actual_reward, *_ = clone.step(1)

    assert actual_reward == pytest.approx(expected_reward)
    for key in expected_obs:
        np.testing.assert_allclose(
            np.asarray(actual_obs[key], dtype=float),
            np.asarray(expected_obs[key], dtype=float),
            err_msg=f"observation {key!r} differs in the rebuilt env",
        )
    # The rebuilt env must own its grid: mutating it must not touch the source env.
    clone.net.load.loc[clone.net.load.index[0], "p_mw"] = SENTINEL_P_MW
    assert env.net.load["p_mw"].iloc[0] != SENTINEL_P_MW
