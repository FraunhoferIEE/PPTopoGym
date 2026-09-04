"""Tests for the config-sourced timeseries path (``env_config["profiles"]``).

An environment can take its timeseries from two places: ``net.profiles`` (Simbench-style,
per-unit shapes scaled by the net's base values) or ``env_config["profiles"]`` (per pandapower
element and column, already absolute). Both fill the same six derived ``df_profiles_*`` tables.

The tests below pin what makes that safe:

1. **The two routes agree.** They reach the same absolute numbers, which is what catches the one
   silent failure mode: multiplying the already-absolute config values by the base values again.
2. **A frozen config profile really freezes the grid.** Datasets that repeat one profile row for a
   whole episode rely on it; a config that was ignored in favour of the net's timeseries would run
   time-varying episodes and score them against constants.
3. **Nothing is dropped in silence.** An unsupported element/column, ragged frames or a missing
   timeseries raise rather than being skipped.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from pandapower_env.data.example_configs import config_case30
from pandapower_env.environments.gym_env_pp import _CONFIG_PROFILE_TARGETS
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.toolbox.utils_profiles import get_orig_profiles

EPISODE_LENGTH = 4


@pytest.fixture(scope="module")
def case30_config() -> dict:
    """Build a case30 config once for the module -- ``config_case30`` is expensive."""
    return config_case30()


def absolute_injection_profiles(net) -> dict[str, dict[str, pd.DataFrame]]:
    """Build the config-style timeseries dict for ``net``, the way its downstream users do.

    Mirrors the consumer-side helper that turns ``get_orig_profiles`` output into
    ``{element: {variable: DataFrame}}`` with the columns relabelled to the element index.
    Elements the net does not have are omitted, exactly as that helper omits them.
    """
    orig = get_orig_profiles(net)
    sources = {
        "load": {"p_mw": "load_p", "q_mvar": "load_q"},
        "gen": {"p_mw": "gen_p", "vm_pu": "gen_vm"},
        "sgen": {"p_mw": "sgen_p", "q_mvar": "sgen_q"},
    }
    profiles: dict[str, dict[str, pd.DataFrame]] = {}
    for element, variables in sources.items():
        if not len(net[element]):
            continue
        profiles[element] = {}
        for variable, orig_key in variables.items():
            relabelled = orig[orig_key].copy()
            relabelled.columns = net[element].index
            profiles[element][variable] = relabelled
    return profiles


def freeze_profiles(
    profiles: dict[str, dict[str, pd.DataFrame]],
    row: int,
    episode_length: int,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Repeat one profile row ``episode_length`` times, as the constant-episode datasets do."""
    return {
        element: {
            variable: pd.concat([profile_df.iloc[row:row + 1]] * episode_length, ignore_index=True)
            for variable, profile_df in variables.items()
        }
        for element, variables in profiles.items()
    }


def config_with_profiles(base_config: dict, profiles: dict, episode_length: int) -> dict:
    """Return ``base_config`` re-pointed at a config-supplied timeseries."""
    config = dict(base_config)
    config["profiles"] = profiles
    config["episode_length"] = episode_length
    return config


def values_only(profile_df: pd.DataFrame) -> pd.DataFrame:
    """Strip column labels and the row index, so two tables compare on their numbers alone."""
    return pd.DataFrame(profile_df.to_numpy())


def test_both_ingestion_paths_build_identical_profile_tables(case30_config) -> None:
    """The Simbench and the config route reach the same absolute per-timestep values.

    This is the test that catches a double multiplication: the config values are already scaled
    by the net's base ``p_mw``/``q_mvar``, so scaling them again the way ``setup_profiles`` scales
    the per-unit Simbench shapes would silently produce a different grid.
    """
    simbench_env = PPTopoGym(case30_config)
    config_env = PPTopoGym(
        config_with_profiles(
            case30_config,
            absolute_injection_profiles(case30_config["net"]),
            case30_config["episode_length"],
        ),
    )

    for table_name in PPTopoGym._PROFILE_TABLE_NAMES:
        pd.testing.assert_frame_equal(
            values_only(getattr(config_env, table_name)),
            values_only(getattr(simbench_env, table_name)),
            obj=table_name,
        )
    assert config_env.n_total_timesteps == simbench_env.n_total_timesteps


def test_constant_config_profiles_keep_the_grid_frozen(case30_config) -> None:
    """An episode built from one repeated profile row leaves the injections untouched throughout.

    The regression this ingestion path exists to prevent: with the config ignored, the env would
    walk the net's own timeseries and quietly run time-varying episodes.
    """
    profiles = freeze_profiles(absolute_injection_profiles(case30_config["net"]), 0, EPISODE_LENGTH)
    env = PPTopoGym(config_with_profiles(case30_config, profiles, EPISODE_LENGTH))

    env.reset(options={"index": 0})
    expected_load_p = env.net.load["p_mw"].to_numpy(copy=True)
    expected_gen_p = env.net.gen["p_mw"].to_numpy(copy=True)

    for _ in range(EPISODE_LENGTH):
        env.step(0)  # DoNothing: only the timeseries could move the injections
        np.testing.assert_array_equal(env.net.load["p_mw"].to_numpy(), expected_load_p)
        np.testing.assert_array_equal(env.net.gen["p_mw"].to_numpy(), expected_gen_p)


def test_episode_step_counter_starts_at_zero_and_counts_completed_steps(case30_config) -> None:
    """``episode_step_counter`` is 0 for the whole first step and 1 once it has finished."""
    profiles = freeze_profiles(absolute_injection_profiles(case30_config["net"]), 0, EPISODE_LENGTH)
    env = PPTopoGym(config_with_profiles(case30_config, profiles, EPISODE_LENGTH))

    seen: list[int] = []

    def record_counter() -> float:
        seen.append(env.episode_step_counter)
        return 0.0

    env.reward_function = record_counter

    env.reset(options={"index": 0})
    assert env.episode_step_counter == 0
    env.step(0)
    assert seen == [0], "the reward of the first step must see a counter of 0"
    assert env.episode_step_counter == 1
    env.step(0)
    assert seen == [0, 1]


def test_act_without_advancing_timeseries_leaves_the_index_alone(case30_config) -> None:
    """Acting without advancing scores the action but does not move the timeseries on."""
    profiles = freeze_profiles(absolute_injection_profiles(case30_config["net"]), 0, EPISODE_LENGTH)
    env = PPTopoGym(config_with_profiles(case30_config, profiles, EPISODE_LENGTH))

    env.reset(options={"index": 0})
    converged, reward = env.act_without_advancing_timeseries(0)

    assert converged is True
    assert env.index == 0
    assert env.episode_step_counter == 0
    assert reward == pytest.approx(env.calculate_reward())


def test_net_profiles_and_supported_action_types_are_exposed(case30_config) -> None:
    """Both ingestion paths expose the timeseries as ``{element: {variable: DataFrame}}``."""
    simbench_env = PPTopoGym(case30_config)
    config_env = PPTopoGym(
        config_with_profiles(
            case30_config,
            absolute_injection_profiles(case30_config["net"]),
            case30_config["episode_length"],
        ),
    )

    assert config_env.net_profiles.keys() == simbench_env.net_profiles.keys()
    for element, variables in config_env.net_profiles.items():
        assert variables.keys() == simbench_env.net_profiles[element].keys()
        for variable, profile_df in variables.items():
            assert profile_df is getattr(config_env, _CONFIG_PROFILE_TARGETS[(element, variable)]), \
                "net_profiles must be a view onto the derived tables, not a copy"

    # ``multi_bb_substation.state`` is narrowed away: the substation tables this branch builds
    # have no such column, and the configuration lives in ``switch.closed`` instead.
    assert config_env.supported_action_types == {
        "switch": ["closed"],
        "line": ["in_service"],
        "trafo": ["tap_pos"],
    }
    for element, columns in config_env.supported_action_types.items():
        for column in columns:
            assert column in config_env.net[element].columns, f"{element}.{column} must exist"


def test_config_profile_tables_are_shared_between_environments(case30_config) -> None:
    """Two envs built from one config-supplied timeseries reference the same derived tables."""
    config = config_with_profiles(
        case30_config,
        absolute_injection_profiles(case30_config["net"]),
        case30_config["episode_length"],
    )
    first = PPTopoGym(config)
    second = PPTopoGym(config)

    for table_name in PPTopoGym._PROFILE_TABLE_NAMES:
        assert getattr(first, table_name) is getattr(second, table_name), table_name
    assert second._shared_profile_tables is first._shared_profile_tables


def test_orig_config_shares_the_supplied_profile_frames(case30_config) -> None:
    """``orig_config`` must not duplicate the timeseries -- that is the memory win it protects."""
    profiles = absolute_injection_profiles(case30_config["net"])
    env = PPTopoGym(config_with_profiles(case30_config, profiles, case30_config["episode_length"]))

    stored = env.orig_config["profiles"]
    for element, variables in profiles.items():
        assert stored[element] is not variables, "the inner dicts must be copies"
        for variable, profile_df in variables.items():
            assert stored[element][variable] is profile_df, "the frames themselves must be shared"


def test_duplicated_profile_index_is_normalised(case30_config) -> None:
    """A timeseries assembled from repeated rows keeps a unique ``0..N-1`` index."""
    profiles = freeze_profiles(absolute_injection_profiles(case30_config["net"]), 0, EPISODE_LENGTH)
    duplicated = {
        element: {variable: profile_df.set_axis([0] * len(profile_df)) for variable, profile_df in variables.items()}
        for element, variables in profiles.items()
    }
    env = PPTopoGym(config_with_profiles(case30_config, duplicated, EPISODE_LENGTH))

    assert env.df_profiles_load_p.index.equals(pd.RangeIndex(EPISODE_LENGTH))
    env.reset(options={"index": 0})
    env.step(0)  # would raise on a duplicated label lookup


def test_unsupported_element_variable_is_rejected(case30_config) -> None:
    """A timeseries for a column this branch cannot vary raises instead of being ignored."""
    profiles = absolute_injection_profiles(case30_config["net"])
    profiles["line"] = {"in_service": pd.DataFrame(np.ones((len(next(iter(profiles["load"].values()))), 1)))}

    with pytest.raises(RuntimeError, match="line.in_service"):
        PPTopoGym(config_with_profiles(case30_config, profiles, case30_config["episode_length"]))


def test_ragged_profile_lengths_are_rejected(case30_config) -> None:
    """Frames with different row counts are a broken timeseries, not a partial one."""
    profiles = absolute_injection_profiles(case30_config["net"])
    profiles["load"]["q_mvar"] = profiles["load"]["q_mvar"].iloc[:-1]

    with pytest.raises(RuntimeError, match="same length"):
        PPTopoGym(config_with_profiles(case30_config, profiles, case30_config["episode_length"]))


def test_missing_required_timeseries_is_rejected(case30_config) -> None:
    """Every injection the net carries must be driven by a profile."""
    profiles = absolute_injection_profiles(case30_config["net"])
    del profiles["gen"]["p_mw"]

    with pytest.raises(RuntimeError, match="df_profiles_gen_p"):
        PPTopoGym(config_with_profiles(case30_config, profiles, case30_config["episode_length"]))


def test_absent_timeseries_names_both_sources(case30_config) -> None:
    """With neither ``net.profiles`` nor a config timeseries, the error names both."""
    config = dict(case30_config)
    config["net"] = copy.deepcopy(case30_config["net"])
    del config["net"]["profiles"]

    with pytest.raises(RuntimeError, match=r"env_config\['profiles'\].*net\.profiles"):
        PPTopoGym(config)
