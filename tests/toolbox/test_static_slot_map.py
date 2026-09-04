"""Tests for the static node-slot map (``utils_graph_obs.static_slot_table`` / ``node_slot_map``).

Node indices are a ``np.unique`` renumbering of whichever buses are canonical right now, so they
shift as soon as a substation splits: row ``i`` is not the same bus before and after. A slot is the
stable alternative, derived from the grid's switch wiring rather than its switch state, so a
consumer can scatter node-aggregated values into a fixed-size array and have each row mean the same
electrical location in every step of every episode.

These tests pin the two properties that makes possible: the slot count is the same static bound the
environment already computes for ``static_obs_space``, and a busbar keeps its slot across every
topology that reaches it.
"""
from __future__ import annotations

import numpy as np
import pytest

from pandapower_env.data.example_configs import config_case30
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.toolbox.utils_graph_obs import (
    n_nodes,
    n_static_slots,
    node_slot_map,
    static_slot_table,
)


@pytest.fixture(scope="module")
def case30_env() -> PPTopoGym:
    """One case30 environment for the module: building it runs a scaling search and is not cheap."""
    return PPTopoGym(config_case30())


def _slot_map_after(env: PPTopoGym, action: int) -> np.ndarray:
    """Reset to the first timestep, apply one action and return the resulting slot map."""
    env.reset(options={"index": 0})
    env.step(action)
    return node_slot_map(env.net, env._obs_cache)


def test_slot_count_matches_the_environments_static_bound(case30_env: PPTopoGym) -> None:
    """One slot per base node plus one per extra busbar -- the same bound ``static_obs_space`` uses."""
    case30_env.reset(options={"index": 0})
    assert n_static_slots(case30_env.net, case30_env._obs_cache) == case30_env._compute_max_n_nodes()


def test_reset_topology_maps_each_node_to_itself(case30_env: PPTopoGym) -> None:
    """At reset the map is the identity over the active nodes, and ``-1`` beyond them."""
    case30_env.reset(options={"index": 0})
    slot_map = node_slot_map(case30_env.net, case30_env._obs_cache)
    node_count = n_nodes(case30_env.net, case30_env._obs_cache)

    assert np.array_equal(slot_map[:node_count], np.arange(node_count))
    assert (slot_map[node_count:] == -1).all()


def test_every_topology_keeps_the_map_fixed_length_and_collision_free(case30_env: PPTopoGym) -> None:
    """Whatever an action does to the grid, each active node still gets its own slot."""
    case30_env.reset(options={"index": 0})
    num_slots = n_static_slots(case30_env.net, case30_env._obs_cache)

    for action in range(int(case30_env.action_space.n)):
        slot_map = _slot_map_after(case30_env, action)
        node_count = n_nodes(case30_env.net, case30_env._obs_cache)
        active = slot_map[:node_count]

        assert len(slot_map) == num_slots
        assert (slot_map[node_count:] == -1).all()
        assert len(np.unique(active)) == len(active), f"slot collision on action {action}"
        assert (active >= 0).all()


def test_a_busbar_keeps_its_slot_across_different_splits(case30_env: PPTopoGym) -> None:
    """Two actions that split the same substation put its busbar on the same slot.

    This is what the dynamic node index cannot offer: it renumbers, so the same busbar appears at
    different rows depending on which other substations happen to be split.
    """
    case30_env.reset(options={"index": 0})
    base_node_count = n_nodes(case30_env.net, case30_env._obs_cache)

    actions_per_slot: dict[int, list[int]] = {}
    for action in range(1, int(case30_env.action_space.n)):
        slot_map = _slot_map_after(case30_env, action)
        node_count = n_nodes(case30_env.net, case30_env._obs_cache)
        for slot in slot_map[:node_count]:
            if slot >= base_node_count:
                actions_per_slot.setdefault(int(slot), []).append(action)

    assert actions_per_slot, "no action split a substation; the test grid may have changed"
    assert any(len(actions) > 1 for actions in actions_per_slot.values()), (
        "expected some busbar to be reachable by more than one action"
    )


def test_base_slots_are_unaffected_by_a_split(case30_env: PPTopoGym) -> None:
    """Splitting a substation adds a node without moving any of the others."""
    case30_env.reset(options={"index": 0})
    base_node_count = n_nodes(case30_env.net, case30_env._obs_cache)

    split_action = next(
        action
        for action in range(1, int(case30_env.action_space.n))
        if len(_slot_map_after(case30_env, action)) and
        n_nodes(case30_env.net, case30_env._obs_cache) > base_node_count
    )
    slot_map = _slot_map_after(case30_env, split_action)

    # The split-off node is appended; every earlier node still reports its own slot.
    assert np.array_equal(slot_map[:base_node_count], np.arange(base_node_count))
    assert slot_map[base_node_count] >= base_node_count


def test_slot_table_is_structural_and_cached(case30_env: PPTopoGym) -> None:
    """The bus -> slot table depends on wiring, not switch state, so it survives topology changes."""
    case30_env.reset(options={"index": 0})
    before, count_before = static_slot_table(case30_env.net, case30_env._obs_cache)

    case30_env.step(1)
    after, count_after = static_slot_table(case30_env.net, case30_env._obs_cache)

    assert count_before == count_after
    assert np.array_equal(before, after)
    assert after is before, "the structural table should be computed once, not per topology"
