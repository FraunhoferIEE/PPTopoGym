"""Pin the process-local Simbench profile cache: same values, no shared mutable state.

``deterministic_profiles`` reads the full Simbench profile library through a module-level
cache (``_all_simbench_profiles_cached``) because the underlying CSV read costs ~1.2 s and
is a pure function of the scenario index. That is only safe if two things hold:

1. Cached reads return the same values an uncached read would.
2. Callers get their own copies, so mutating one net's profiles cannot leak into the next
   net built in the same process. This is the failure the cache could plausibly introduce,
   so it is tested explicitly rather than assumed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pandapower.networks import case30

from pandapower_env.toolbox import utils_profiles
from pandapower_env.toolbox.utils_profiles import deterministic_profiles

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture()
def _clear_profile_cache() -> Iterator[None]:
    """Empty the cache before and after a test so ordering cannot affect the result."""
    utils_profiles._SIMBENCH_PROFILE_CACHE.clear()
    yield
    utils_profiles._SIMBENCH_PROFILE_CACHE.clear()


@pytest.mark.usefixtures("_clear_profile_cache")
def test_cached_read_matches_uncached() -> None:
    """A cache hit returns the same profile values as the initial cold read."""
    cold = deterministic_profiles(case30(), 2)
    warm = deterministic_profiles(case30(), 2)

    assert set(cold) == set(warm)
    for key in cold:
        pd.testing.assert_frame_equal(cold[key], warm[key])


@pytest.mark.usefixtures("_clear_profile_cache")
def test_caller_mutation_does_not_leak_into_next_build() -> None:
    """Mutating returned profiles in place must not affect a later call.

    This is the regression the cache would cause if it handed out views of the shared
    frames instead of copies.
    """
    first = deterministic_profiles(case30(), 2)
    # Column 0 is the Simbench ``time`` column (strings); pick the first numeric one.
    numeric_column = first["load"].select_dtypes("number").columns[0]
    original_value = first["load"].loc[0, numeric_column]

    first["load"].loc[0, numeric_column] = original_value + 12345.0

    second = deterministic_profiles(case30(), 2)
    assert second["load"].loc[0, numeric_column] == original_value


@pytest.mark.usefixtures("_clear_profile_cache")
def test_cache_is_populated_and_keyed_by_scenario() -> None:
    """The cache fills on first use and holds one entry per scenario index."""
    assert utils_profiles._SIMBENCH_PROFILE_CACHE == {}

    deterministic_profiles(case30(), 2)
    assert set(utils_profiles._SIMBENCH_PROFILE_CACHE) == {2}

    deterministic_profiles(case30(), 2)
    assert set(utils_profiles._SIMBENCH_PROFILE_CACHE) == {2}
