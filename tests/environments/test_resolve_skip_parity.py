"""``run_pf`` must skip a re-solve only when the results on the net already answer it.

The optimization these tests guard: a DoNothing step keeps ``net.converged`` True and changes
nothing about the grid, so re-running the power flow reproduces the results already there. The
risk it introduces is the opposite mistake -- serving *stale* results after something did change
-- so each test below pins one of the paths that must still force a solve.

``run_pf`` is counted rather than timed: whether the solver ran is the property under test, and a
timing assertion would be flaky on a loaded box.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandapower_env.environments import gym_env_pp

if TYPE_CHECKING:
    from pandapower_env.environments.simulation_env import PPTopoGym

RESET_INDEX = 12
RESULT_TABLES = ("res_bus", "res_line", "res_trafo")
# A step that changes the grid solves twice: once for the reward, once for the next observation.
SOLVES_PER_ACTING_STEP = 2


@pytest.fixture()
def solve_counter(monkeypatch) -> list[int]:
    """Count how often the environment actually reaches a solver, not how often ``run_pf`` is called.

    Patches the module-level ``run_powerflow`` that :meth:`BaseEnvPP.run_pf` delegates to, so the
    guard under test (which returns before it) is what the count measures.
    """
    calls = [0]
    original = gym_env_pp.run_powerflow

    def counting_run_powerflow(*args, **kwargs):  # noqa: ANN202
        calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(gym_env_pp, "run_powerflow", counting_run_powerflow)
    return calls


def result_snapshot(env: PPTopoGym) -> dict[str, np.ndarray]:
    """Copy every result table off the net, so a later solve cannot mutate the copy."""
    return {
        table: env.net[table].to_numpy(dtype=float, na_value=np.nan).copy()
        for table in RESULT_TABLES
        if len(env.net[table])
    }


def assert_results_identical(produced: dict[str, np.ndarray], expected: dict[str, np.ndarray]) -> None:
    """Assert two result snapshots match bit for bit, NaN positions included."""
    assert produced.keys() == expected.keys()
    for table, values in produced.items():
        np.testing.assert_array_equal(values, expected[table], err_msg=table)


def test_donothing_step_reuses_the_identical_solution(simenv30) -> None:
    """The results a skipped DoNothing solve leaves behind are the ones a solve would produce."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    env.step(0)
    reused = result_snapshot(env)

    # Defeat the guard and solve the very same state again: the reference the skip stands in for.
    env.net.converged = None
    assert env.run_pf()
    assert_results_identical(result_snapshot(env), reused)


def test_donothing_step_does_not_reach_the_solver(simenv30, solve_counter) -> None:
    """A DoNothing step re-uses the observation solve of the previous step."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    env.step(0)  # settles the net at (topology, index) with converged=True

    solve_counter[0] = 0
    env.step(0)
    # The step's own solve is skipped; the observation after the index advance still solves.
    assert solve_counter[0] == 1


def test_topology_action_still_solves(simenv30, solve_counter) -> None:
    """A real action clears ``converged``, so the guard must not swallow its solve."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    env.step(0)

    solve_counter[0] = 0
    env.step(7)
    assert solve_counter[0] == SOLVES_PER_ACTING_STEP, "action and observation solves must both run"


def test_profile_advance_forces_a_resolve(simenv30, solve_counter) -> None:
    """New injections invalidate the results even though the topology is untouched."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    assert env.run_pf()

    env.load_profile_timestep_into_net(RESET_INDEX + 1)
    solve_counter[0] = 0
    assert env.run_pf()
    assert solve_counter[0] == 1


def test_a_different_request_forces_a_resolve(simenv30, solve_counter) -> None:
    """The guard keys on the request, so a DC solve cannot be served from an AC result."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    assert env.run_pf(pf_type="ac")

    solve_counter[0] = 0
    assert env.run_pf(pf_type="dc")
    assert solve_counter[0] == 1
    # ... and the AC request is not served from the DC result either.
    assert env.run_pf(pf_type="ac")
    assert solve_counter[0] == SOLVES_PER_ACTING_STEP


def test_reset_forces_a_resolve(simenv30, solve_counter) -> None:
    """``reset`` restores the baseline topology, which must invalidate the marker."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    env.step(7)

    solve_counter[0] = 0
    env.reset(options={"index": RESET_INDEX})
    assert solve_counter[0] >= 1


def test_restore_state_forces_a_resolve(simenv30, solve_counter) -> None:
    """``save_state`` / ``restore_state`` move the grid without going through ``load_action``."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    state = env.save_state()
    env.step(7)

    env.restore_state(state)
    solve_counter[0] = 0
    assert env.run_pf()
    assert solve_counter[0] == 1
