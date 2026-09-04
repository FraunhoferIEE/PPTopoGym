"""Guards against ways the environment used to break or report the wrong thing.

Each test here pins behaviour that previously surfaced as a bare ``KeyError``, a silently
missing metric, or a discarded power flow. None of them changes the public API: the
observation space, ``step``'s return shape and the config keys are all unchanged.
"""

import numpy as np
import pytest

from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.observation_space.obs_space_utils import (
    build_info_observation_registry,
    build_observation_registry,
)

# Actions of the ``simenv`` fixture whose power flow converges; 1, 2 and 4 do not, and a
# crashed step terminates the episode without logging its action.
CONVERGING_ACTION = 3
OTHER_CONVERGING_ACTION = 5

# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------


def test_simulation_rejects_out_of_range_numpy_action(simenv) -> None:
    """A numpy integer out of range must raise ``ValueError``, like a plain ``int`` does.

    The bounds check used to be ``isinstance(action, int)``, which numpy integers fail, so an
    out-of-range ``np.int64`` skipped validation entirely and surfaced deep inside
    ``load_action`` as ``KeyError: np.int64(...)``. Agents return numpy integers
    (``action_space.sample()``, ``argmax``), so this was the common case, not the rare one.
    """
    simenv.reset(options={"index": 0})
    out_of_range = np.int64(len(simenv.df_actions) + 5)
    with pytest.raises(ValueError, match="Invalid action"):
        simenv.simulation(out_of_range)


def test_simulation_rejects_negative_action(simenv) -> None:
    """A negative index must raise instead of silently applying the last action row."""
    simenv.reset(options={"index": 0})
    with pytest.raises(ValueError, match="Invalid action"):
        simenv.simulation([-1])


def test_simulation_accepts_valid_numpy_action(simenv) -> None:
    """Validation must not reject the numpy integers agents legitimately hand in."""
    simenv.reset(options={"index": 0})
    outputs = simenv.simulation(np.int64(1))
    assert len(outputs) == 1


# ---------------------------------------------------------------------------
# End of the timeseries
# ---------------------------------------------------------------------------


def test_step_truncates_at_end_of_timeseries(simenv) -> None:
    """Stepping on the last profile row ends the episode instead of raising.

    ``step`` advanced ``self.index`` unconditionally and then looked the new row up by label,
    so a reset to a late ``options["index"]`` -- the documented way to select a scenario --
    walked off the end of the profile tables with ``KeyError: <n_timesteps>``.
    """
    last_index = simenv.n_total_timesteps - 1
    simenv.reset(options={"index": last_index})

    _, reward, terminated, truncated, _ = simenv.step(0)

    assert truncated, "the exhausted timeseries must truncate the episode"
    assert not terminated, "truncation is not a power flow failure"
    assert np.isfinite(reward), "the last row is still scored normally"
    assert simenv.index == last_index, "a truncated step must not advance past the last row"


# ---------------------------------------------------------------------------
# Info observations that are not observations
# ---------------------------------------------------------------------------


def test_info_only_aggregates_are_outside_the_observation_space(simenv) -> None:
    """The info-only aggregates must never widen the observation space.

    They are computed on request but deliberately excluded from the registry an environment
    defaults ``observation_keys`` to, because adding them would change the input dimension of
    every network already trained against this environment.
    """
    info_only = build_info_observation_registry()
    assert set(info_only) == {"total_energy_overload", "max_loading_percent"}
    assert not set(info_only) & set(build_observation_registry())
    assert not set(info_only) & set(simenv.observation_space.spaces)


def test_step_info_reports_overload_and_max_loading(simenv) -> None:
    """``info`` must carry the aggregates the evaluation metrics read.

    ``total_energy_overload`` and ``max_loading_percent`` are named by the default
    ``info_observations`` but had no registry entry, so ``create_observation`` silently dropped
    them and ``overload_energy_difference_abs_mvah`` / ``loading_improvement_optimization``
    returned NaN for every step ever evaluated.
    """
    simenv.reset(options={"index": 0})
    _, _, _, _, info = simenv.step(0)

    for key in ("total_energy_overload", "max_loading_percent"):
        for phase in ("before", "after"):
            name = f"{key}_{phase}"
            assert name in info, f"{name} missing from step info"
            assert np.all(np.isfinite(np.asarray(info[name], dtype=float)))


def test_create_observation_default_keys_exclude_info_aggregates(simenv) -> None:
    """A default ``create_observation()`` must return exactly the observation-space keys."""
    observation = simenv.create_observation()
    assert set(observation) == set(simenv.observation_space.spaces)


# ---------------------------------------------------------------------------
# end_simulation restores the same state, more cheaply
# ---------------------------------------------------------------------------


def test_simulation_restores_state_exactly(simenv) -> None:
    """``simulation`` must leave topology, index and results exactly as it found them.

    ``end_simulation`` no longer goes through the full ``reset``: it restores the state and
    replays the action log, skipping a power flow and an observation that described the
    pristine topology and were discarded one line later. This pins that the cheaper route
    lands on the same state.

    Actions 3 and 5 are used because 1, 2 and 4 do not converge on this grid; a crashed step
    terminates the episode without logging its action, which is a different code path.
    """
    simenv.reset(options={"index": 1})
    simenv.step(CONVERGING_ACTION)  # put a real action in the log, so the replay has work to do

    before = {
        "switches": simenv.net.switch["closed"].to_numpy().copy(),
        "lines": simenv.net.line["in_service"].to_numpy().copy(),
        "index": simenv.index,
        "current_step": simenv.current_step,
        "log_actions": np.array(simenv.log_actions, dtype=float),
        "loadings": simenv.net.res_line["loading_percent"].to_numpy().copy(),
    }

    simenv.simulation([OTHER_CONVERGING_ACTION])

    assert np.array_equal(simenv.net.switch["closed"].to_numpy(), before["switches"])
    assert np.array_equal(simenv.net.line["in_service"].to_numpy(), before["lines"])
    assert simenv.index == before["index"]
    assert simenv.current_step == before["current_step"]
    assert np.array_equal(
        np.array(simenv.log_actions, dtype=float), before["log_actions"], equal_nan=True,
    )
    assert np.allclose(simenv.net.res_line["loading_percent"].to_numpy(), before["loadings"])


def test_end_simulation_matches_reset_and_replay(simenv) -> None:
    """The cheap restore must agree with an explicit reset-and-replay on a second env."""
    reference_env = PPTopoGym(simenv.orig_config)

    simenv.reset(options={"index": 1})
    simenv.step(CONVERGING_ACTION)
    simenv.simulation([OTHER_CONVERGING_ACTION])

    # Reference: the state reached by replaying the same history from a clean reset.
    reference_env.reset(options={"index": 1})
    reference_env.load_action(CONVERGING_ACTION)
    reference_env.index = simenv.index
    reference_env.load_profile_timestep_into_net(reference_env.index)
    reference_env.run_pf()

    assert np.array_equal(
        simenv.net.switch["closed"].to_numpy(),
        reference_env.net.switch["closed"].to_numpy(),
    )
    assert np.array_equal(
        simenv.net.line["in_service"].to_numpy(),
        reference_env.net.line["in_service"].to_numpy(),
    )
    assert np.allclose(
        simenv.net.res_line["loading_percent"].to_numpy(),
        reference_env.net.res_line["loading_percent"].to_numpy(),
    )


def test_a_copied_env_drops_the_lightsim_backend_and_still_solves(simenv) -> None:
    """Copying or pickling an env that has solved on lightsim2grid must work, and agree.

    The backend wraps a C++ ``GridModel`` that survives neither: ``copy.deepcopy`` raised
    ``RuntimeError: Impossible to set the converter ls_to_orig``. The simulation API, the greedy
    agents and spawned workers all copy environments, so this broke them the moment the lightsim
    backend became the default in a consumer. ``BaseEnvPP.__getstate__`` drops the backend, which
    is derived state and is rebuilt lazily on the next solve.
    """
    import copy
    import pickle

    simenv.backend_name = "lightsim"
    simenv.reset(options={"index": 0})
    simenv.step(CONVERGING_ACTION)
    reference = float(np.nanmax(simenv.net.res_line["loading_percent"].to_numpy(dtype=float)))
    assert simenv._lightsim_backend is not None, "the env must have built a backend to copy"

    for clone in (copy.deepcopy(simenv), pickle.loads(pickle.dumps(simenv))):  # noqa: S301
        assert clone._lightsim_backend is None, "the C++ model must not be carried into the copy"
        clone.reset(options={"index": 0})
        clone.step(CONVERGING_ACTION)
        got = float(np.nanmax(clone.net.res_line["loading_percent"].to_numpy(dtype=float)))
        assert got == pytest.approx(reference, abs=1e-9), "the rebuilt backend must agree"
