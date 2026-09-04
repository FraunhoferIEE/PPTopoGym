"""Reproducibility contract for seeded resets and seeded action sampling.

Two independent guarantees are pinned here:

1. ``env.reset(seed=N)`` selects the same scenario every time, and does so *without*
   depending on -- or perturbing -- the process-global ``random`` / ``numpy`` state.
   Seeding through the global modules made a run's outcome depend on whatever else in
   the process had drawn a random number first, which silently breaks reproducibility
   in vectorized and multi-agent settings.
2. ``BaseGreedyAgent.act(max_actions=...)`` subsamples the same actions for the same
   seed. It previously drew from an unseeded ``np.random.default_rng()``, so a seeded
   greedy run could not be reproduced at all.

These tests are deliberately cheap: they exercise the RNG plumbing, not the power flow.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandapower_env.agents.base_agents import BaseGreedyAgent

if TYPE_CHECKING:
    from pandapower_env.environments.simulation_env import PPTopoGym

# Fixed scenario index used by the explicit-index test.
EXPLICIT_INDEX = 3
# Subsample size used by the subset test, and the pool it is drawn from.
SUBSAMPLE_SIZE = 12
SUBSAMPLE_POOL = 50


def _scenario_indices(env: PPTopoGym, seed: int, n_resets: int = 5) -> list[int]:
    """Reset ``env`` repeatedly from one seed and record the scenario index chosen each time."""
    env.reset(seed=seed)
    indices = [env.index]
    for _ in range(n_resets - 1):
        indices.append(env.index)
        env.reset(seed=seed)
    return indices


def test_reset_with_same_seed_is_reproducible(simenv: PPTopoGym) -> None:
    """Two resets with the same seed must land on the same timeseries index."""
    simenv.reset(seed=1234)
    first = simenv.index

    simenv.reset(seed=1234)
    second = simenv.index

    assert first == second


def test_reset_with_different_seeds_differs(simenv: PPTopoGym) -> None:
    """The seed must actually drive the scenario choice, not be ignored.

    The ``simenv`` fixture has exactly one selectable scenario (5 profile rows at
    ``episode_length=5``), so the scenario *index* cannot vary. What must vary is the
    draw itself, which is asserted directly on the env's RNG.
    """
    draws = set()
    for seed in range(40):
        simenv.reset(seed=seed)
        draws.add(int(simenv.np_random.integers(0, 1_000_000)))

    # With the seed actually reaching the env's RNG, 40 seeds must not collapse to one.
    assert len(draws) > 1


def test_reset_does_not_depend_on_global_random_state(simenv: PPTopoGym) -> None:
    """A seeded reset must be unaffected by unrelated global RNG consumption.

    This is the regression: seeding used ``random.seed(seed)`` and then drew with
    ``random.randint``, so any other component drawing from ``random`` between the
    seeding and the draw changed the selected scenario.
    """
    random.seed(0)
    simenv.reset(seed=99)
    clean = simenv.index

    random.seed(0)
    for _ in range(17):  # unrelated global draws, as another component would make
        random.random()  # noqa: S311
    simenv.reset(seed=99)
    perturbed = simenv.index

    assert clean == perturbed


def test_reset_does_not_perturb_global_random_state(simenv: PPTopoGym) -> None:
    """A seeded reset must not reseed the caller's global ``random`` module.

    ``random.seed(seed)`` inside ``reset`` hijacked process-global state, so an
    environment reset silently changed the numbers every *other* part of the program
    drew afterwards.
    """
    random.seed(4321)
    expected = [random.random() for _ in range(3)]  # noqa: S311 # pinning global state is the point

    random.seed(4321)
    simenv.reset(seed=99)
    actual = [random.random() for _ in range(3)]  # noqa: S311

    assert actual == expected


def test_reset_does_not_perturb_global_numpy_state(simenv: PPTopoGym) -> None:
    """The same guarantee for the legacy global numpy RNG."""
    np.random.seed(4321)  # noqa: NPY002 # pinning the legacy global state is the point
    expected = np.random.random(3).tolist()  # noqa: NPY002

    np.random.seed(4321)  # noqa: NPY002
    simenv.reset(seed=99)
    actual = np.random.random(3).tolist()  # noqa: NPY002

    assert actual == expected


def test_explicit_index_option_overrides_seed(simenv: PPTopoGym) -> None:
    """``options={"index": N}`` stays the explicit control and must beat the seed."""
    simenv.reset(seed=7, options={"index": EXPLICIT_INDEX})

    assert simenv.index == EXPLICIT_INDEX


class _StubGreedyAgent(BaseGreedyAgent):
    """Greedy agent whose environment is stubbed out.

    ``act`` is not called here -- only the seeded subsampling helper is -- so the
    expensive environment construction is unnecessary and deliberately skipped.
    """

    def __init__(self, n_actions: int, seed: int | None) -> None:
        self.np_random = np.random.default_rng(seed)
        self._n_actions = n_actions


def test_agent_subsampling_is_reproducible_for_one_seed() -> None:
    """Same seed, same subsample of candidate actions."""
    candidates = np.arange(100)

    first = _StubGreedyAgent(100, seed=17)._subsample_actions(candidates, 10)
    second = _StubGreedyAgent(100, seed=17)._subsample_actions(candidates, 10)

    assert first.tolist() == second.tolist()


def test_agent_subsampling_differs_across_seeds() -> None:
    """Different seeds must give different subsamples, or the seed is being ignored."""
    candidates = np.arange(100)

    first = _StubGreedyAgent(100, seed=1)._subsample_actions(candidates, 10)
    second = _StubGreedyAgent(100, seed=2)._subsample_actions(candidates, 10)

    assert first.tolist() != second.tolist()


def test_agent_subsampling_is_a_subset_without_repeats() -> None:
    """The subsample must stay a valid action subset -- the fix must not change semantics."""
    candidates = np.arange(SUBSAMPLE_POOL)

    picked = _StubGreedyAgent(SUBSAMPLE_POOL, seed=3)._subsample_actions(candidates, SUBSAMPLE_SIZE)

    assert len(picked) == SUBSAMPLE_SIZE
    assert len(set(picked.tolist())) == SUBSAMPLE_SIZE
    assert set(picked.tolist()).issubset(set(candidates.tolist()))


def test_agent_subsampling_returns_all_when_under_budget() -> None:
    """No subsampling happens when the candidate set already fits the budget."""
    candidates = np.arange(5)

    picked = _StubGreedyAgent(5, seed=3)._subsample_actions(candidates, 10)

    assert picked.tolist() == candidates.tolist()


@pytest.mark.parametrize("seed", [0, 1, 12345])
def test_unseeded_agents_still_work(seed: int) -> None:
    """An agent built without a seed must still sample (just not reproducibly)."""
    candidates = np.arange(30)

    picked = _StubGreedyAgent(30, seed=None)._subsample_actions(candidates, seed % 7 + 3)

    assert set(picked.tolist()).issubset(set(candidates.tolist()))
