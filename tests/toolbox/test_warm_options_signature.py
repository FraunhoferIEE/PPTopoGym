"""
Tests for the flat-start voltage signature that guards the warm runpp options.

``_init_vm_pu_signature`` is called on every ``run_powerflow``, so it is on the hottest path
in the package. It is computed with numpy instead of unpacking two pandas Series into a
Python list (~127 us -> ~10 us), which only helps if it keeps returning *exactly* the value
pandapower would derive -- the mean ``vm_pu`` of the in-service ``gen`` and ``ext_grid``
elements. A drift here would silently change the Newton-Raphson starting point, so these
tests pin the value against a direct reimplementation of that definition.
"""

from __future__ import annotations

import numpy as np
import pytest

from pandapower_env.toolbox.utils import _init_vm_pu_signature

# Fractions of elements switched out of service while fuzzing, chosen so both tables keep
# some in-service rows in essentially every draw.
GEN_OUT_OF_SERVICE_RATE = 0.3
EXT_GRID_OUT_OF_SERVICE_RATE = 0.2


def _mean_of_in_service_setpoints(net) -> float:
    """Return the definition the signature must match: mean vm_pu over in-service sources."""
    setpoints = [
        *net.gen.vm_pu[net.gen.in_service].to_numpy(),
        *net.ext_grid.vm_pu[net.ext_grid.in_service].to_numpy(),
    ]
    return float(np.mean(setpoints)) if setpoints else 1.0


def test_signature_matches_definition(test_grid_dbb_plus_simbench) -> None:
    """On an untouched grid the signature equals the plain mean of the setpoints."""
    net = test_grid_dbb_plus_simbench
    assert _init_vm_pu_signature(net) == pytest.approx(_mean_of_in_service_setpoints(net))


def test_signature_matches_over_random_states(test_grid_dbb_plus_simbench) -> None:
    """Varying setpoints and in-service flags never separates the two implementations."""
    net = test_grid_dbb_plus_simbench
    rng = np.random.default_rng(0)

    for _ in range(100):
        net.gen["vm_pu"] = rng.uniform(0.9, 1.1, len(net.gen))
        net.gen["in_service"] = rng.random(len(net.gen)) > GEN_OUT_OF_SERVICE_RATE
        net.ext_grid["in_service"] = rng.random(len(net.ext_grid)) > EXT_GRID_OUT_OF_SERVICE_RATE
        assert _init_vm_pu_signature(net) == pytest.approx(
            _mean_of_in_service_setpoints(net), abs=0.0, rel=0.0,
        )


def test_signature_is_one_without_in_service_sources(test_grid_dbb_plus_simbench) -> None:
    """With nothing in service the signature falls back to the flat 1.0 start."""
    net = test_grid_dbb_plus_simbench
    net.gen["in_service"] = False
    net.ext_grid["in_service"] = False
    assert _init_vm_pu_signature(net) == 1.0


def test_signature_tracks_a_changed_setpoint(test_grid_dbb_plus_simbench) -> None:
    """Changing a generator setpoint changes the signature, so warm options are re-parsed."""
    net = test_grid_dbb_plus_simbench
    if not len(net.gen):
        pytest.skip("grid has no gen elements to vary")

    before = _init_vm_pu_signature(net)
    net.gen.loc[net.gen.index[0], "vm_pu"] = net.gen["vm_pu"].iloc[0] + 0.05
    assert _init_vm_pu_signature(net) != before
