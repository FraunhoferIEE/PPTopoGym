"""Pin that the warm power flow in ``utils_scaling.run_pf`` matches plain ``pp.runpp``.

``run_pf`` reuses pandapower's parsed options instead of re-parsing them on every call
(~10x faster across a scaling search). That is only safe if it produces the same answer,
so these tests compare the warm path against a cold ``pp.runpp`` reference on both a
single solve and a full ``find_scaling_recursive`` search.

The scaling search is the important one: it is recursive and its branch decisions depend
on the power-flow result, so any divergence would compound into a different final net
rather than a small numeric difference.
"""

from __future__ import annotations

import numpy as np
import pandapower as pp
import pytest
from pandapower.networks import case30

from pandapower_env.toolbox import utils_scaling
from pandapower_env.toolbox.utils_profiles import get_first_sb_profiles, get_orig_profiles
from pandapower_env.toolbox.utils_scaling import ensure_no_zero_values, find_scaling_recursive, run_pf

# Results should be bit-identical, but compare with a tight tolerance rather than
# exact equality so the test reports a magnitude if a future change perturbs them.
RTOL = 1e-9
ATOL = 1e-12


def _cold_run_pf(net: pp.pandapowerNet) -> bool:
    """Solve ``net`` with a plain ``pp.runpp``, the reference with no option reuse.

    :param net: The network to solve, mutated in place.
    :return: ``True`` if the power flow converged.
    """
    try:
        pp.runpp(net)
    except pp.LoadflowNotConverged:
        return False
    return True


def _prepared_case30() -> tuple[pp.pandapowerNet, dict]:
    """Build a case30 net with profiles attached, ready for a scaling search.

    :return: The prepared net and its original (unscaled) profiles.
    """
    net = case30()
    get_first_sb_profiles(net, 2)
    ensure_no_zero_values(net)
    for key, df in net.profiles.items():
        net.profiles[key] = df.replace(0.0, 1.0)
    return net, get_orig_profiles(net)


def test_warm_run_pf_matches_cold_single_solve() -> None:
    """One warm solve gives the same line loadings as one cold ``pp.runpp``."""
    warm_net, _ = _prepared_case30()
    cold_net, _ = _prepared_case30()

    assert run_pf(warm_net) is True
    assert _cold_run_pf(cold_net) is True

    np.testing.assert_allclose(
        warm_net.res_line["loading_percent"].to_numpy(),
        cold_net.res_line["loading_percent"].to_numpy(),
        rtol=RTOL,
        atol=ATOL,
    )


def test_warm_run_pf_reports_non_convergence() -> None:
    """A diverging net still returns ``False`` rather than raising."""
    net, _ = _prepared_case30()
    net.load.p_mw = -10
    net.load.q_mvar = 0
    net.gen.p_mw = 10000
    assert run_pf(net) is False


@pytest.mark.parametrize("field", ["p_mw"])
def test_scaling_search_result_is_identical(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full ``find_scaling_recursive`` search converges to the same net either way.

    The search is run twice on identical inputs -- once with the warm ``run_pf`` and once
    with ``run_pf`` monkeypatched back to a cold ``pp.runpp`` -- and the resulting element
    powers and line loadings are compared.
    """
    warm_net, warm_profiles = _prepared_case30()
    find_scaling_recursive(
        warm_net, init_scaling=1, orig_profiles=warm_profiles, max_percent=40, overloaded_lines=3,
    )

    cold_net, cold_profiles = _prepared_case30()
    monkeypatch.setattr(utils_scaling, "run_pf", _cold_run_pf)
    find_scaling_recursive(
        cold_net, init_scaling=1, orig_profiles=cold_profiles, max_percent=40, overloaded_lines=3,
    )

    for element in ("load", "gen"):
        np.testing.assert_allclose(
            warm_net[element][field].to_numpy(),
            cold_net[element][field].to_numpy(),
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"{element}.{field} diverged between the warm and cold scaling search",
        )

    np.testing.assert_allclose(
        warm_net.res_line["loading_percent"].to_numpy(),
        cold_net.res_line["loading_percent"].to_numpy(),
        rtol=RTOL,
        atol=ATOL,
        err_msg="line loadings diverged between the warm and cold scaling search",
    )
