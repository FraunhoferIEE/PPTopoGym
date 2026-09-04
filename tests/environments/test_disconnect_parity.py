"""The positional islanding check must agree with the label-based one it replaced.

``_grid_is_disconnected`` decides whether a converged power flow is reported as a failure, so a
disagreement here does not shift a number -- it flips an episode between "scored" and "crashed".
The label path is kept in the environment as the fallback for non-aligned nets and is used here
as the oracle, the same arrangement ``test_load_action_parity`` uses for ``load_action``.
"""

from __future__ import annotations

import numpy as np
import pytest

RESET_INDEX = 12


def assert_both_paths_agree(env) -> bool:
    """Assert the fast and the label path give the same verdict, and return it."""
    fast = env._grid_is_disconnected()
    reference = env._grid_is_disconnected_by_label()
    assert fast == reference
    return fast


def test_connected_grid_agrees(simenv30) -> None:
    """The healthy case: every in-service bus is solved, both paths say connected."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    assert env.run_pf()
    assert assert_both_paths_agree(env) is False


def test_islanded_bus_is_detected_by_both_paths(simenv30) -> None:
    """A NaN voltage on an in-service bus must be seen by both paths."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    assert env.run_pf()

    in_service = env.net.bus.index[env.net.bus["in_service"]]
    env.net.res_bus.loc[in_service[0], "vm_pu"] = np.nan
    assert assert_both_paths_agree(env) is True


def test_nan_on_an_out_of_service_bus_is_ignored_by_both_paths(simenv30) -> None:
    """The auxiliary busbar buses are always NaN and must not count as an island."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    assert env.run_pf()

    out_of_service = env.net.bus.index[~env.net.bus["in_service"]]
    if not len(out_of_service):
        pytest.skip("this grid has no out-of-service buses")
    env.net.res_bus.loc[out_of_service[0], "vm_pu"] = np.nan
    assert assert_both_paths_agree(env) is False


@pytest.mark.parametrize("action", [1, 2, 4, 5, 7])
def test_paths_agree_over_actions(simenv30, action: int) -> None:
    """Real topology changes, including the ones that island a bus or fail to converge."""
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    env.step(action)
    if env.net.res_bus.empty:
        pytest.skip(f"action {action} left no results to check")
    assert_both_paths_agree(env)


def test_shuffled_res_bus_index_falls_back_to_the_label_path(simenv30) -> None:
    """A ``res_bus`` that is not row-aligned must still be answered by label, not by position.

    This is the case the fast path cannot handle: read positionally, a permuted ``res_bus`` pairs
    each bus with another bus's voltage. The guard has to notice and hand over.
    """
    env = simenv30
    env.reset(options={"index": RESET_INDEX})
    assert env.run_pf()

    in_service = env.net.bus.index[env.net.bus["in_service"]]
    out_of_service = env.net.bus.index[~env.net.bus["in_service"]]
    if not len(out_of_service):
        pytest.skip("this grid has no out-of-service buses")

    # Park the only NaN on an out-of-service bus, then reverse the row order. Positionally that
    # NaN now lines up with an in-service bus; by label it still does not.
    env.net.res_bus["vm_pu"] = 1.0
    env.net.res_bus.loc[out_of_service[0], "vm_pu"] = np.nan
    env.net["res_bus"] = env.net.res_bus.iloc[::-1]

    assert env._grid_is_disconnected() is False
    assert len(in_service) > 0
