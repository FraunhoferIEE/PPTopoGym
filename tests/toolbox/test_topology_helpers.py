"""Equivalence tests for the union-find bus-switch-tree traversal.

The fast implementation in :mod:`pandapower_env.toolbox.topology_helpers` replaced a fixpoint
pandas search that ran once per queried bus. These tests pin the new implementation to the old
one's output element-for-element (ordering included), because the traversal feeds substation
creation and the generator-PCC grouping, where a silent regrouping would change physics rather
than raise.
"""
from __future__ import annotations

import numpy as np
import pytest
from pandapower import pandapowerNet  # noqa: TCH002
from pandapower.networks import case14, case30, case118

from pandapower_env.substation.create_double_busbar_substation import (
    can_convert_to_n_busbar_substation,
    create_n_busbar_substation,
)
from pandapower_env.toolbox.topology_helpers import (
    _is_pst_in_substation,
    bus_switch_components,
    find_bus_switch_tree,
    find_bus_switch_trees_from_list,
    is_pst,
)


def _reference_find_bus_switch_tree(net: pandapowerNet, ibus: int, *, consider_open_switches: bool) -> list[int]:
    """Run the pre-union-find implementation, kept verbatim as the correctness oracle."""
    bus_switches = net.switch[(net.switch.et == "b")]

    current_bus_tree = {ibus}

    while True:
        switch_is_involved = np.any(bus_switches[["bus", "element"]].isin(current_bus_tree), axis=1)

        switch_is_closed = True
        if consider_open_switches:
            switch_is_closed = bus_switches["closed"]

        sel_switches = switch_is_involved & switch_is_closed
        new_bus_tree = set(bus_switches[sel_switches][["bus", "element"]].to_numpy().flatten().tolist())

        if len(new_bus_tree) == 0:
            return [ibus]

        if current_bus_tree == new_bus_tree:
            new_bus_tree_list = sorted(new_bus_tree)
            new_bus_tree_list.pop(new_bus_tree_list.index(ibus))
            return [ibus, *new_bus_tree_list]

        current_bus_tree = new_bus_tree


def _reference_find_bus_switch_trees_from_list(
    net: pandapowerNet,
    buses: list[int] | None,
    *,
    consider_open_switches: bool,
    include_single_buses: bool,
    fail_on_overlap: bool = True,
) -> list[list[int]]:
    """Run the pre-union-find list traversal, kept verbatim as the correctness oracle."""
    bus_trees = []
    used_buses: list[int] = []

    if buses is None:
        buses = net.bus.index

    for ibus in buses:
        if ibus in used_buses:
            if fail_on_overlap:
                msg = f"A tree for bus {ibus} was requested, but it is already identified in another bus tree."
                raise ValueError(msg)
            continue

        bus_tree = _reference_find_bus_switch_tree(net, ibus, consider_open_switches=consider_open_switches)

        if len(bus_tree) > 1:
            bus_trees.append(bus_tree)
            used_buses.extend(bus_tree)
        elif include_single_buses:
            bus_trees.append([ibus])

    return bus_trees


def _double_busbar_net(base_net_fn) -> pandapowerNet:
    """Split every convertible bus of a base grid into a double-busbar substation."""
    net = base_net_fn()
    for ibus in list(net.bus.index):
        if can_convert_to_n_busbar_substation(net, ibus):
            create_n_busbar_substation(net, ibus)
    return net


@pytest.fixture(scope="module")
def dbb_case30() -> pandapowerNet:
    """case30 with double-busbar substations: 93 buses, 116 bus-bus switches."""
    return _double_busbar_net(case30)


@pytest.fixture(scope="module")
def dbb_case118() -> pandapowerNet:
    """case118 with double-busbar substations: 556 buses, 811 bus-bus switches."""
    return _double_busbar_net(case118)


def _randomize_bus_switches(net: pandapowerNet, rng: np.random.Generator) -> None:
    """Flip the ``closed`` flags of the bus-bus switches to a fresh random state."""
    bus_switch_rows = net.switch.index[net.switch["et"] == "b"]
    net.switch.loc[bus_switch_rows, "closed"] = rng.random(len(bus_switch_rows)) < 0.5  # noqa: PLR2004


@pytest.mark.parametrize("net_fixture", ["dbb_case30", "dbb_case118"])
@pytest.mark.parametrize("consider_open_switches", [True, False])
@pytest.mark.parametrize("include_single_buses", [True, False])
def test_trees_from_list_match_reference_over_random_switch_states(
    net_fixture: str,
    consider_open_switches: bool,  # noqa: FBT001
    include_single_buses: bool,  # noqa: FBT001
    request: pytest.FixtureRequest,
) -> None:
    """Union-find output must equal the old traversal exactly, over many random switch states."""
    net = request.getfixturevalue(net_fixture)
    rng = np.random.default_rng(0)
    queried_buses = list(net.bus.index)

    n_trials = 40
    for _ in range(n_trials):
        _randomize_bus_switches(net, rng)

        expected = _reference_find_bus_switch_trees_from_list(
            net, queried_buses, consider_open_switches=consider_open_switches,
            include_single_buses=include_single_buses, fail_on_overlap=False)
        actual = find_bus_switch_trees_from_list(
            net, queried_buses, consider_open_switches=consider_open_switches,
            include_single_buses=include_single_buses, fail_on_overlap=False)

        assert actual == expected


def test_single_bus_tree_matches_reference(dbb_case30: pandapowerNet) -> None:
    """find_bus_switch_tree keeps the requested bus first and matches the old traversal."""
    net = dbb_case30
    rng = np.random.default_rng(1)

    n_trials = 20
    for _ in range(n_trials):
        _randomize_bus_switches(net, rng)
        for ibus in net.bus.index:
            expected = _reference_find_bus_switch_tree(net, ibus, consider_open_switches=True)
            actual = find_bus_switch_tree(net, ibus, consider_open_switches=True)
            assert actual == expected
            assert actual[0] == ibus


def test_fail_on_overlap_raises_like_reference(dbb_case30: pandapowerNet) -> None:
    """A bus already claimed by an earlier tree must still raise ValueError."""
    net = dbb_case30
    net.switch.loc[net.switch.index[net.switch["et"] == "b"], "closed"] = True

    # Find a component with at least two buses; querying both must trip the overlap guard.
    components = bus_switch_components(net, consider_open_switches=True)
    overlapping = next(buses for buses in components.values() if len(buses) > 1)

    with pytest.raises(ValueError, match="already identified in another bus tree"):
        find_bus_switch_trees_from_list(
            net, list(overlapping), consider_open_switches=True,
            include_single_buses=True, fail_on_overlap=True)


def test_components_cover_only_switched_buses(dbb_case30: pandapowerNet) -> None:
    """Buses no bus-bus switch mentions stay out of the component map and read as standalone."""
    net = dbb_case30
    net.switch.loc[net.switch.index[net.switch["et"] == "b"], "closed"] = False

    components = bus_switch_components(net, consider_open_switches=True)
    assert components == {}, "no closed bus-bus switch means no component"

    # ... and every bus then reports itself as its own tree.
    for ibus in list(net.bus.index)[:10]:
        assert find_bus_switch_tree(net, ibus, consider_open_switches=True) == [ibus]


# ---------------------------------------------------------------------------
# PST detection
# ---------------------------------------------------------------------------


def _plain_case14_trafo_net() -> pandapowerNet:
    """Return a case14 net with no ``multi_bb_substation`` table, so ``is_pst`` uses the rating path."""
    net = case14()
    assert not hasattr(net, "multi_bb_substation")
    return net


def test_is_pst_uses_substation_path_when_table_present(test_grid_with_pst: pandapowerNet) -> None:
    """A trafo with both ends inside one substation is a PST, whatever its ratings say."""
    net = test_grid_with_pst

    pst_trafos = [
        i_trafo
        for i_trafo in net.trafo.index
        if _is_pst_in_substation(net, i_trafo)
    ]
    assert pst_trafos, "the 3bb-with-PST fixture must contain at least one in-substation trafo"

    # is_pst must agree with the substation check while multi_bb_substation exists.
    for i_trafo in net.trafo.index:
        assert is_pst(net, i_trafo) == _is_pst_in_substation(net, i_trafo)


def test_is_pst_in_substation_true_for_self_loop(test_grid_with_pst: pandapowerNet) -> None:
    """A trafo whose hv and lv bus coincide short-circuits the substation scan."""
    net = test_grid_with_pst
    i_trafo = net.trafo.index[0]
    net.trafo.loc[i_trafo, "lv_bus"] = net.trafo.loc[i_trafo, "hv_bus"]

    assert _is_pst_in_substation(net, i_trafo) is True


def test_is_pst_in_substation_false_when_ends_are_in_different_substations(
    test_grid_with_pst: pandapowerNet,
) -> None:
    """Ends spread over two substations (or none) must not be reported as a PST."""
    net = test_grid_with_pst
    i_trafo = net.trafo.index[0]

    # Empty the substation table: no substation can then hold both ends.
    net.multi_bb_substation = net.multi_bb_substation.iloc[0:0]
    net.trafo.loc[i_trafo, "lv_bus"] = net.trafo.loc[i_trafo, "hv_bus"] + 1

    assert _is_pst_in_substation(net, i_trafo) is False


def test_is_pst_without_tap_changer_is_false() -> None:
    """Without a tap changer the trafo can't shift phase."""
    net = _plain_case14_trafo_net()
    i_trafo = net.trafo.index[0]
    net.trafo.loc[i_trafo, "tap_changer_type"] = None

    assert is_pst(net, i_trafo) is False


def test_is_pst_with_differing_nominal_voltages_is_false() -> None:
    """A voltage-transforming trafo is a regular transformer, not a PST."""
    net = _plain_case14_trafo_net()
    i_trafo = net.trafo.index[0]
    net.trafo.loc[i_trafo, "tap_changer_type"] = "Ideal"
    net.trafo.loc[i_trafo, "vn_hv_kv"] = 110.0
    net.trafo.loc[i_trafo, "vn_lv_kv"] = 20.0

    assert is_pst(net, i_trafo) is False


@pytest.mark.parametrize("tap_step_degree", [None, np.nan, 0.0])
def test_is_pst_without_angle_step_is_false(tap_step_degree: float | None) -> None:
    """A tap changer that only moves magnitude (no angle step) is not a PST."""
    net = _plain_case14_trafo_net()
    i_trafo = net.trafo.index[0]
    net.trafo.loc[i_trafo, "tap_changer_type"] = "Ideal"
    net.trafo.loc[i_trafo, "vn_lv_kv"] = net.trafo.loc[i_trafo, "vn_hv_kv"]
    net.trafo["tap_step_degree"] = net.trafo["tap_step_degree"].astype(object)
    net.trafo.loc[i_trafo, "tap_step_degree"] = tap_step_degree

    assert is_pst(net, i_trafo) is False


def test_is_pst_true_for_equal_voltages_with_angle_step() -> None:
    """Equal nominal voltages plus a non-zero angle step identify a PST."""
    net = _plain_case14_trafo_net()
    i_trafo = net.trafo.index[0]
    net.trafo.loc[i_trafo, "tap_changer_type"] = "Ideal"
    net.trafo.loc[i_trafo, "vn_lv_kv"] = net.trafo.loc[i_trafo, "vn_hv_kv"]
    net.trafo.loc[i_trafo, "tap_step_degree"] = 30.0

    assert is_pst(net, i_trafo) is True
