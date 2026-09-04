"""Tests for pandapower_tools helpers, focused on cached generator harmonization."""
from __future__ import annotations

import copy

import numpy as np
import pandapower as pp
import pytest
from pandapower import pandapowerNet
from pandapower.networks import case30

from pandapower_env.substation.create_double_busbar_substation import (
    can_convert_to_n_busbar_substation,
    create_n_busbar_substation,
)
from pandapower_env.toolbox.pandapower_tools import (
    _pcc_generator_groups,
    harmonize_gen_voltage_setpoints,
)
from pandapower_env.toolbox.topology_helpers import find_bus_switch_trees_from_list


def _reference_pcc_generator_groups(net: pandapowerNet) -> list[list[int]]:
    """Run the pre-optimization grouping, kept verbatim as the correctness oracle."""
    bus_groups = find_bus_switch_trees_from_list(net, net.gen["bus"], consider_open_switches=True,
                                                 include_single_buses=True, fail_on_overlap=False)
    min_generators_to_care = 2
    pcc_generator_groups = []
    for bus_group in bus_groups:
        pcc_generators = net.gen[net.gen["bus"].isin(bus_group)].index.tolist()
        if len(pcc_generators) >= min_generators_to_care:
            pcc_generator_groups.append(pcc_generators)
    return pcc_generator_groups


def _distinct_gen_setpoints(net: pandapowerNet) -> None:
    """Give every generator a distinct vm_pu so averaging is observable."""
    net.gen["vm_pu"] = np.linspace(0.95, 1.05, len(net.gen))


@pytest.fixture(scope="module")
def dbb_case30() -> pandapowerNet:
    """case30 with double-busbar substations: 93 buses, 5 generators."""
    net = case30()
    for ibus in list(net.bus.index):
        if can_convert_to_n_busbar_substation(net, ibus):
            create_n_busbar_substation(net, ibus)
    return net


@pytest.fixture()
def shared_pcc_net(dbb_case30: pandapowerNet) -> pandapowerNet:
    """Build a net where two generators really do share a point of common coupling.

    Both production grids return zero PCC groups, so the grouping path would otherwise be
    untested. Moving a second generator onto a bus that a closed bus-bus switch joins to the
    first generator's bus forces a real group.
    """
    net = copy.deepcopy(dbb_case30)
    first_gen, second_gen = net.gen.index[0], net.gen.index[1]

    # Put the second generator on a bus joined to the first generator's bus by a closed switch.
    gen_bus = int(net.gen.loc[first_gen, "bus"])
    joined = net.switch[(net.switch["et"] == "b") & (net.switch["bus"] == gen_bus)]
    if joined.empty:
        joined = net.switch[(net.switch["et"] == "b") & (net.switch["element"] == gen_bus)]
        partner_bus = int(joined.iloc[0]["bus"])
    else:
        partner_bus = int(joined.iloc[0]["element"])

    net.switch.loc[joined.index[0], "closed"] = True
    net.gen.loc[second_gen, "bus"] = partner_bus
    return net


def test_harmonize_cached_matches_uncached(test_grid_multi_bb_substations: pandapowerNet) -> None:
    """Cached harmonization must produce identical vm_pu to the uncached path."""
    net_uncached = test_grid_multi_bb_substations
    _distinct_gen_setpoints(net_uncached)
    net_cached = copy.deepcopy(net_uncached)

    harmonize_gen_voltage_setpoints(net_uncached)  # no cache
    cache: dict = {}
    harmonize_gen_voltage_setpoints(net_cached, cache)

    np.testing.assert_array_equal(
        net_uncached.gen["vm_pu"].to_numpy(),
        net_cached.gen["vm_pu"].to_numpy(),
    )


def test_harmonize_cache_reused_until_switch_changes(shared_pcc_net: pandapowerNet) -> None:
    """The grouping is memoised while switch states are stable and rebuilt when they change."""
    net = shared_pcc_net
    _distinct_gen_setpoints(net)
    cache: dict = {}

    harmonize_gen_voltage_setpoints(net, cache)
    first_groups = cache["pcc_generator_groups"]
    first_key = cache["switch_state_key"]

    # Second call with unchanged switches: same cached object, same key.
    harmonize_gen_voltage_setpoints(net, cache)
    assert cache["pcc_generator_groups"] is first_groups
    assert cache["switch_state_key"] == first_key

    # Flipping a bus-bus switch must invalidate the cache key.
    bus_switches = net.switch.index[net.switch["et"] == "b"]
    net.switch.loc[bus_switches[0], "closed"] = not net.switch.loc[bus_switches[0], "closed"]
    harmonize_gen_voltage_setpoints(net, cache)
    assert cache["switch_state_key"] != first_key


def test_shared_pcc_generators_are_averaged(shared_pcc_net: pandapowerNet) -> None:
    """On a grid where generators DO share a PCC, their setpoints must be averaged.

    This is the guard the production grids cannot provide: case30/case118 both yield zero
    groups, so without this test a broken grouping would be invisible end to end.
    """
    net = shared_pcc_net
    _distinct_gen_setpoints(net)

    groups = _pcc_generator_groups(net, None)
    assert groups, "fixture must produce at least one shared-PCC group"

    expected_means = {tuple(group): net.gen.loc[group, "vm_pu"].mean() for group in groups}

    harmonize_gen_voltage_setpoints(net)

    for group, expected_mean in expected_means.items():
        harmonized = net.gen.loc[list(group), "vm_pu"].to_numpy()
        np.testing.assert_allclose(harmonized, expected_mean)


def test_grouping_matches_reference_over_random_switch_states(shared_pcc_net: pandapowerNet) -> None:
    """The optimized grouping must equal the old traversal-based grouping, group for group."""
    net = shared_pcc_net
    rng = np.random.default_rng(0)
    bus_switch_rows = net.switch.index[net.switch["et"] == "b"]

    n_trials = 50
    for _ in range(n_trials):
        net.switch.loc[bus_switch_rows, "closed"] = rng.random(len(bus_switch_rows)) < 0.5  # noqa: PLR2004

        expected = sorted(sorted(group) for group in _reference_pcc_generator_groups(net))
        actual = sorted(sorted(group) for group in _pcc_generator_groups(net, None))

        assert actual == expected


def test_element_switch_write_does_not_invalidate_cache(shared_pcc_net: pandapowerNet) -> None:
    """Only bus-bus switches key the cache; toggling an element switch must not rebuild it."""
    net = shared_pcc_net
    _distinct_gen_setpoints(net)

    # This grid is all bus-bus switches; add a line switch so there is one to toggle.
    # Added before the first call, so the cache is built against the final switch table.
    line_switch = pp.create_switch(net, bus=int(net.line.loc[net.line.index[0], "from_bus"]),
                                   element=int(net.line.index[0]), et="l", closed=True)

    cache: dict = {}
    harmonize_gen_voltage_setpoints(net, cache)
    first_groups = cache["pcc_generator_groups"]

    net.switch.loc[line_switch, "closed"] = False

    harmonize_gen_voltage_setpoints(net, cache)
    assert cache["pcc_generator_groups"] is first_groups


def test_no_possible_pcc_short_circuits(dbb_case30: pandapowerNet) -> None:
    """case30's generators can never share a PCC, so the grouping is answered without traversal."""
    net = copy.deepcopy(dbb_case30)
    cache: dict = {}

    assert _pcc_generator_groups(net, cache) == []
    assert cache["pcc_possible"] is False

    # The early-out must hold for every switch state, matching the reference.
    rng = np.random.default_rng(2)
    bus_switch_rows = net.switch.index[net.switch["et"] == "b"]
    n_trials = 20
    for _ in range(n_trials):
        net.switch.loc[bus_switch_rows, "closed"] = rng.random(len(bus_switch_rows)) < 0.5  # noqa: PLR2004
        assert _reference_pcc_generator_groups(net) == []
        assert _pcc_generator_groups(net, cache) == []
