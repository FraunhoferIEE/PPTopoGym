"""Graph-observation extraction for pandapower networks (functions + decorators).

The module turns a pandapower network into node-indexed observation arrays. Creating
the multi-busbars introduces many auxiliary buses that would blow up the observation
space; the internal ``net._pd2ppc_lookups`` tables are used to fold those auxiliary
buses back onto the canonical buses that actually carry the power flow.

Two layers
----------
1. **Standalone graph observation** -- :func:`create_adjacency_matrix` together with the
   three small ``numpy``-only helpers (:func:`_compute_pp_to_canonical`,
   :func:`_compute_node_index_map`, :func:`_collect_edges`) form a self-contained block.
   Call ``create_adjacency_matrix(net)`` with no cache and it just works -- no context
   manager, no caching decorators. Copy those four functions into any project where
   speed does not matter.

2. **Cached layer for this branch** -- a per-environment cache ``dict`` (see
   :func:`make_obs_cache`) plus the :func:`lookup_cached` decorator and the
   :func:`batch_observations` context manager keep repeated extraction fast by never
   recomputing anything that only depends on the (unchanged) topology. The topology is
   identified by the *content* of ``net._pd2ppc_lookups["bus"]``: pandapower replaces the
   lookups object on every power flow, but its content only changes when the grid really
   does, so comparing content (not identity) keeps the cache alive across timesteps.

Mapping vocabulary
------------------
``pp_to_canonical`` maps each pandapower bus index to its canonical (lowest-indexed)
representative inside its electrically connected group, e.g. ``[0,1,2,3,4,5,6] ->
[0,0,2,3,3,3,6]``. ``canonical_to_node_idx`` maps canonical bus ids to consecutive node
indices ``0..n_nodes-1`` (non-canonical entries are ``-1``).
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Generator, TypeVar

import numpy as np

from pandapower_env.toolbox.topology_helpers import bus_switch_components

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray
    from pandapower import pandapowerNet

logger = logging.getLogger(__name__)

T = TypeVar("T")

_BASE_TABLE_NAMES = ("load", "gen", "sgen", "bus", "line", "trafo")


# ============================================================================
# Standalone graph observation (copyable, numpy-only, no cache required).
# Copy create_adjacency_matrix + these three helpers to reuse elsewhere.
# ============================================================================
def _compute_pp_to_canonical(lookup_table: NDArray[np.int32]) -> NDArray[np.int32]:
    """Map each pp_bus to its canonical bus (index of its first occurrence)."""
    _, first_indices, inverse = np.unique(
        lookup_table,
        return_index=True,
        return_inverse=True,
    )
    return first_indices[inverse].astype(np.int32)


def _compute_node_index_map(
    pp_to_canonical: NDArray[np.int32],
) -> tuple[NDArray[np.int32], int]:
    """Build the canonical-bus -> consecutive-node-index map and the node count."""
    unique_canonical = np.unique(pp_to_canonical)
    n_nodes = len(unique_canonical)
    max_canonical = int(np.max(unique_canonical))
    canonical_to_node_idx = np.full(max_canonical + 1, -1, dtype=np.int32)
    canonical_to_node_idx[unique_canonical] = np.arange(n_nodes, dtype=np.int32)
    return canonical_to_node_idx, n_nodes


def _collect_edges(net: pandapowerNet) -> NDArray[np.int32]:
    """Collect raw (from, to) bus pairs for lines and transformers."""
    parts: list[NDArray[np.int32]] = []
    if len(net.line) > 0:
        parts.append(net.line[["from_bus", "to_bus"]].to_numpy(dtype=np.int32))
    if len(net.trafo) > 0:
        parts.append(net.trafo[["hv_bus", "lv_bus"]].to_numpy(dtype=np.int32))
    if not parts:
        return np.empty((0, 2), dtype=np.int32)
    return np.concatenate(parts, axis=0)


def create_adjacency_matrix(
    net: pandapowerNet,
    cache: dict | None = None,
) -> NDArray[np.int32]:
    """
    Build the adjacency matrix (edge list) with node indices.

    Standalone use (``cache=None``): everything is recomputed from scratch -- correct
    and dependency-free, but not cached. Pass an :func:`make_obs_cache` dict to reuse the
    topology mapping and edge list across calls (fast path for this branch).

    Parameters
    ----------
    net : pandapowerNet
        Network with ``_pd2ppc_lookups`` populated (run ``pandapower.runpp`` first).
    cache : dict | None
        Optional observation cache. When ``None`` the result is computed without caching.

    Returns
    -------
    NDArray[np.int32]
        Edge array of shape ``(n_edges, 2)`` expressed in node indices.
    """
    if cache is not None:
        return _cached_adjacency(net, cache)

    pp_to_canonical = _compute_pp_to_canonical(net._pd2ppc_lookups["bus"])  # noqa: SLF001
    canonical_to_node_idx, _ = _compute_node_index_map(pp_to_canonical)
    edges = _collect_edges(net)
    if len(edges) == 0:
        return edges
    return canonical_to_node_idx[pp_to_canonical[edges]]


# ============================================================================
# Cached layer: per-environment state dict + caching decorator + batch context.
# ============================================================================
def make_obs_cache() -> dict:
    """
    Create an empty observation cache for one environment.

    The cache holds everything that only depends on the topology, validated against the
    *content* of ``net._pd2ppc_lookups["bus"]`` (see :func:`_topology_fingerprint_changed`).
    Each environment owns its own cache, so parallel workers stay independent (no
    module-level global state).
    """
    return {
        # ``lookups_ref``/``lookup_obj_id`` track the currently active lookups object for the
        # table/bus sub-caches and for observability; they do NOT decide cache validity. The
        # per-consumer ``*_lookups_ref`` keys below hold a copy of the bus-lookup fingerprint.
        "lookup_obj_id": None,
        "lookups_ref": None,
        # Topology mapping (filled by _ensure_mapping).
        "pp_to_canonical": None,
        "canonical_to_node_idx": None,
        "n_nodes": None,
        # Per-table node indices: table_name -> NDArray, invalidated on topology change.
        "table_node_indices": {},
        "table_cache_lookups_ref": None,
        # Canonical bus index per node (for bus-level values).
        "canonical_bus_indices": None,
        "canonical_lookups_ref": None,
        # Bus -> static node slot, and the slot count. Structural, so it outlives topology changes.
        "static_slot_table": None,
        # Per-batch snapshot of table references + lengths (set by batch_observations).
        "batch": None,
    }


def _topology_fingerprint_changed(net: pandapowerNet, cache: dict, ref_key: str) -> bool:
    """Report whether the topology changed since ``cache[ref_key]`` was stored, and update it.

    The fingerprint is the *content* of ``net._pd2ppc_lookups["bus"]`` (the pandapower bus ->
    internal-bus lookup), compared with ``np.array_equal`` against a copy held in the cache.

    Content, not object identity: pandapower allocates a brand-new lookups object -- and a new
    ``"bus"`` array -- on **every** power flow, while the content only changes when the topology
    really does. Keying on identity therefore threw the whole topology cache away once per step
    and rebuilt it from scratch (~1.5 ms) for nothing. Comparing 'is the content the same' costs
    ~3 us against that, and holding our own copy also removes the ``id()``-recycling hazard that
    made the identity check need a strong reference in the first place.

    Everything derived from this cache (``pp_to_canonical``, ``canonical_to_node_idx``,
    ``n_nodes``, the adjacency, and the per-table node indices) is a function of this array plus
    the structural bus columns (``line.from_bus``, ``trafo.hv_bus``, ...). Those columns are grid
    definition and are never written by ``load_action``, ``load_profile_timestep_into_net`` or
    ``restore_topology`` -- which only touch ``switch.closed``, ``line.in_service`` and
    ``trafo.tap_pos`` -- so fingerprinting them as well would cost more than it protects.

    :param net: a pandapower network
    :type net: pandapowerNet
    :param cache: the observation cache holding the previous fingerprint
    :type cache: dict
    :param ref_key: cache key under which this consumer stores its fingerprint copy
    :type ref_key: str
    :return: True if the topology changed (the caller must rebuild), False if the cache is valid.
    :rtype: bool
    """
    current = net._pd2ppc_lookups["bus"]  # noqa: SLF001
    previous = cache.get(ref_key)
    if previous is not None and np.array_equal(previous, current):
        return False
    cache[ref_key] = current.copy()  # copy: pandapower reuses/reallocates this array
    return True


def lookup_cached(cache_key: str) -> Callable[[Callable[[pandapowerNet, dict], T]],
                                              Callable[[pandapowerNet, dict], T]]:
    """
    Memoize a ``func(net, cache)`` result, rebuilding when the topology changes.

    The result is stored in ``cache[cache_key]`` and reused while the topology fingerprint is
    unchanged -- see :func:`_topology_fingerprint_changed` for why validity is decided by the
    *content* of the bus lookup rather than by the identity of the lookups object. This is the
    functional replacement for the old ``BusMapper._ensure_cache_valid`` check.
    """
    ref_key = f"_{cache_key}_lookups_ref"

    def decorator(func: Callable[[pandapowerNet, dict], T]) -> Callable[[pandapowerNet, dict], T]:
        @functools.wraps(func)
        def wrapper(net: pandapowerNet, cache: dict) -> T:
            current = net._pd2ppc_lookups  # noqa: SLF001
            cache["lookups_ref"] = current  # active lookups for the table/bus sub-caches
            cache["lookup_obj_id"] = id(current)  # informational only (see test_cache_valid)
            if not _topology_fingerprint_changed(net, cache, ref_key) and cache.get(cache_key) is not None:
                return cache[cache_key]
            value = func(net, cache)
            cache[cache_key] = value
            return value

        return wrapper

    return decorator


@lookup_cached("mapping")
def _ensure_mapping(
    net: pandapowerNet,
    cache: dict,
) -> tuple[NDArray[np.int32], NDArray[np.int32], int]:
    """Build (and cache) the topology mapping; returns (pp_to_canonical, node_idx, n_nodes)."""
    pp_to_canonical = _compute_pp_to_canonical(net._pd2ppc_lookups["bus"])  # noqa: SLF001
    canonical_to_node_idx, n_nodes = _compute_node_index_map(pp_to_canonical)
    cache["pp_to_canonical"] = pp_to_canonical
    cache["canonical_to_node_idx"] = canonical_to_node_idx
    cache["n_nodes"] = n_nodes
    return pp_to_canonical, canonical_to_node_idx, n_nodes


@lookup_cached("edges")
def _cached_adjacency(net: pandapowerNet, cache: dict) -> NDArray[np.int32]:
    """Return the cached adjacency matrix (edge list in node indices)."""
    pp_to_canonical, canonical_to_node_idx, _ = _ensure_mapping(net, cache)
    edges = _collect_edges(net)
    if len(edges) == 0:
        return edges
    return canonical_to_node_idx[pp_to_canonical[edges]]


def _extra_busbar_buses(net: pandapowerNet) -> list[int]:
    """List the additional busbar buses of every multi-busbar substation, in table order.

    ``bus_0`` is the substation's original bus and already owns a base slot, so only
    ``bus_1 .. bus_{n-1}`` can add an electrical node when the substation splits.

    :param net: The pandapower network
    :type net: pandapowerNet
    :return: the extra busbar bus indices, substation-table order then busbar order
    :rtype: list[int]
    """
    substations = net.get("multi_bb_substation")
    if substations is None or not len(substations):
        return []
    extra_columns = sorted(
        (c for c in substations.columns if c.startswith("bus_") and c != "bus_0"),
        key=lambda c: int(c.removeprefix("bus_")),
    )
    return [int(bus) for column in extra_columns for bus in substations[column].dropna()]


def static_slot_table(net: pandapowerNet, cache: dict) -> tuple[NDArray[np.int32], int]:
    """Map every pandapower bus to a topology-independent node slot.

    The node indices of a graph observation are a ``np.unique`` renumbering of whichever buses
    are canonical *right now*, so they shift as soon as a substation splits: row ``i`` is not the
    same bus before and after. A slot is the stable alternative -- it is derived from the grid
    itself, never from the current switch state, so a bus keeps its slot for the life of the net.

    Slots are laid out as ``[base buses..., extra busbars...]``:

    * **base slots** -- one per component of the *fully fused* grid (every bus-bus switch treated
      as closed), which is exactly the reset topology's set of electrical nodes. Every bus of a
      component, auxiliary element buses included, maps to that component's slot.
    * **extra busbar slots** -- one per ``bus_1 .. bus_{n-1}`` of each multi-busbar substation, in
      substation-table order. These are the buses that become canonical when a substation splits.

    The total is therefore ``n_base + sum(n_busbars - 1)``, i.e. the same bound as
    ``PPTopoGym._compute_max_n_nodes``. The result depends only on the grid's structure (switch
    *wiring* and the substation table, never switch *state*), so it is computed once per net and
    kept across topology changes.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param cache: the observation cache (see :func:`make_obs_cache`)
    :type cache: dict
    :return: ``(bus_to_slot, n_slots)`` -- a slot per bus of ``net.bus``, and the slot count
    :rtype: tuple[NDArray[np.int32], int]
    """
    cached = cache.get("static_slot_table")
    if cached is not None:
        return cached

    fused_representative = np.arange(len(net.bus), dtype=np.int32)
    for members in bus_switch_components(net, consider_open_switches=False).values():
        fused_representative[members] = min(members)

    base_representatives = np.unique(fused_representative)
    slot_of_representative = np.full(len(net.bus), -1, dtype=np.int32)
    slot_of_representative[base_representatives] = np.arange(len(base_representatives), dtype=np.int32)
    bus_to_slot = slot_of_representative[fused_representative]

    extra_busbars = _extra_busbar_buses(net)
    for offset, busbar_bus in enumerate(extra_busbars):
        bus_to_slot[busbar_bus] = len(base_representatives) + offset

    n_slots = len(base_representatives) + len(extra_busbars)
    cache["static_slot_table"] = (bus_to_slot, n_slots)
    return bus_to_slot, n_slots


def n_static_slots(net: pandapowerNet, cache: dict) -> int:
    """Return the fixed number of node slots of this grid (see :func:`static_slot_table`)."""
    return static_slot_table(net, cache)[1]


@lookup_cached("node_slots")
def node_slot_map(net: pandapowerNet, cache: dict) -> NDArray[np.int32]:
    """Map the current topology's node indices to their static slots.

    Entry ``j`` is the slot of node ``j`` for ``j < n_nodes`` and ``-1`` for the trailing entries
    that this (less split) topology does not use. The length is always ``n_static_slots``, so the
    array itself is a fixed-shape observation whatever the topology does.

    This is what lets a consumer scatter node-aggregated observations into a fixed-size tensor and
    have row ``s`` mean the same electrical location in every step of every episode -- the
    property the variable-length node indices lack.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param cache: the observation cache (see :func:`make_obs_cache`)
    :type cache: dict
    :return: the node -> slot map, padded with ``-1`` to ``n_static_slots``
    :rtype: NDArray[np.int32]
    :raises ValueError: if two active nodes claim the same slot, which means the slot table does
        not describe this grid (e.g. a substation model this layout does not cover).
    """
    bus_to_slot, n_slots = static_slot_table(net, cache)
    canonical_bus_indices = _get_canonical_bus_indices(net, cache)

    slot_map = np.full(n_slots, -1, dtype=np.int32)
    active_slots = bus_to_slot[canonical_bus_indices]
    if len(np.unique(active_slots)) != len(active_slots):
        msg = (
            f"static slot collision: {len(active_slots)} active nodes claim "
            f"{len(np.unique(active_slots))} distinct slots"
        )
        raise ValueError(msg)
    slot_map[: len(active_slots)] = active_slots
    return slot_map


def n_nodes(net: pandapowerNet, cache: dict) -> int:
    """Return the number of unique nodes (canonical buses) for the current topology."""
    batch = cache.get("batch")
    if batch is not None:
        return int(batch["n_nodes"])
    return _ensure_mapping(net, cache)[2]


def map_to_node_index(
    net: pandapowerNet,
    cache: dict,
    bus_ids: NDArray[np.int_] | list[int],
) -> NDArray[np.int32]:
    """Map an array of pp_bus ids directly to node indices ``0..n_nodes-1``."""
    pp_to_canonical, canonical_to_node_idx, _ = _ensure_mapping(net, cache)
    if not isinstance(bus_ids, np.ndarray):
        bus_ids = np.asarray(bus_ids, dtype=np.int32)
    return canonical_to_node_idx[pp_to_canonical[bus_ids]]


@contextmanager
def batch_observations(net: pandapowerNet, cache: dict) -> Generator[None, None, None]:
    """
    Batch many observation extractions behind a single topology validation.

    On enter the topology mapping is built once and the base/``res_`` table references
    plus their lengths are snapshotted so individual ``get_observation`` calls avoid
    repeated attribute lookups. Reentrant: a nested ``with`` is a pass-through.

    On exit the per-batch snapshot and the injected profile tables are cleared (so the
    next timestep recomputes profile values), but the topology-keyed caches (mapping,
    edges, per-table node indices) are kept -- they survive until the topology changes.

    Example
    -------
    >>> cache = make_obs_cache()
    >>> with batch_observations(net, cache):
    ...     adj = create_adjacency_matrix(net, cache)
    ...     load_p = get_observation(net, cache, "load", "p_mw")
    ...     load_q = get_observation(net, cache, "load", "q_mvar")
    """
    if cache.get("batch") is not None:
        yield  # already batching -- just pass through
        return

    _, _, node_count = _ensure_mapping(net, cache)
    tables: dict = {name: getattr(net, name, None) for name in _BASE_TABLE_NAMES}
    table_lengths: dict[str, int] = {
        name: (len(tables[name]) if tables[name] is not None else 0)
        for name in _BASE_TABLE_NAMES
    }
    tables.update({
        f"res_{name}": getattr(net, f"res_{name}", None) for name in _BASE_TABLE_NAMES
    })
    cache["batch"] = {"n_nodes": node_count, "tables": tables, "table_lengths": table_lengths}
    try:
        yield
    finally:
        cache["batch"] = None


def _get_table_and_length(
    net: pandapowerNet,
    cache: dict,
    table_name: str,
) -> tuple[pd.DataFrame, int]:
    """Resolve a table reference and its element count (batch snapshot -> live)."""
    batch = cache.get("batch")
    if batch is not None and table_name in batch["tables"]:
        base_name = table_name.removeprefix("res_")
        return batch["tables"][table_name], batch["table_lengths"].get(base_name, 0)

    table = getattr(net, table_name)
    return table, len(table)


def _invalidate_table_node_cache(net: pandapowerNet, cache: dict) -> None:
    """Drop the per-table node-index cache if the topology fingerprint changed."""
    if _topology_fingerprint_changed(net, cache, "table_cache_lookups_ref"):
        cache["table_node_indices"] = {}


def _get_node_indices_for_table(
    net: pandapowerNet,
    cache: dict,
    table_name: str,
) -> NDArray[np.int32]:
    """Node index for each element of a table (cached; the bus column is timestep-stable)."""
    _invalidate_table_node_cache(net, cache)
    cached = cache["table_node_indices"].get(table_name)
    if cached is not None:
        return cached

    table, table_len = _get_table_and_length(net, cache, table_name)
    if table_len == 0:
        node_indices = np.array([], dtype=np.int32)
    else:
        if table_name in {"bus", "res_bus"}:
            bus_ids = table.index.to_numpy()
        elif "bus" in table.columns:
            bus_ids = table["bus"].to_numpy()
        else:
            msg = f"Table {table_name} has no bus index, maybe wrong Observation definition?"
            raise ValueError(msg)
        if not isinstance(bus_ids, np.ndarray):
            bus_ids = np.asarray(bus_ids, dtype=np.int32)
        elif bus_ids.dtype != np.int32:
            bus_ids = bus_ids.astype(np.int32)
        node_indices = map_to_node_index(net, cache, bus_ids)

    cache["table_node_indices"][table_name] = node_indices
    return node_indices


def _get_canonical_bus_indices(net: pandapowerNet, cache: dict) -> NDArray[np.int32]:
    """Index into the bus table of the canonical (first) bus of each node."""
    stale = _topology_fingerprint_changed(net, cache, "canonical_lookups_ref")
    if not stale and cache.get("canonical_bus_indices") is not None:
        return cache["canonical_bus_indices"]

    node_indices = _get_node_indices_for_table(net, cache, "bus")
    _, first_occurrence_idx = np.unique(node_indices, return_index=True)
    canonical_bus_indices = first_occurrence_idx.astype(np.int32)
    cache["canonical_bus_indices"] = canonical_bus_indices
    return canonical_bus_indices


def get_observation(
    net: pandapowerNet,
    cache: dict,
    table_name: str,
    column_name: str,
) -> NDArray[np.float32]:
    """
    Extract one column from a pandapower table as a node-indexed array.

    Multiple elements at the same node are summed (``gen``/``sgen`` are negated, matching
    generation sign convention). Bus-level columns take the value of the canonical bus.
    Line/trafo columns are returned per element (not aggregated to nodes).

    Parameters
    ----------
    net : pandapowerNet
        The network.
    cache : dict
        Observation cache from :func:`make_obs_cache`.
    table_name : str
        Table name, optionally ``res_``/``profile_`` prefixed (e.g. ``"load"``, ``"res_bus"``).
    column_name : str
        Column to extract (e.g. ``"p_mw"``).

    Returns
    -------
    NDArray[np.float32]
        Array of shape ``(n_nodes,)`` (or per-element for line/trafo tables).
    """
    batch = cache.get("batch")
    node_count = int(batch["n_nodes"]) if batch is not None else _ensure_mapping(net, cache)[2]

    table, table_len = _get_table_and_length(net, cache, table_name)
    if table_len == 0:
        if any(key in table_name for key in ("line", "trafo")):
            return np.array([], dtype=np.float32)
        return np.zeros(node_count, dtype=np.float32)

    # res_ tables carry values but no bus column; node indices come from the base table.
    table_base_name = table_name.removeprefix("res_")

    if table_base_name == "bus":
        canonical_indices = _get_canonical_bus_indices(net, cache)
        values = table[column_name].to_numpy()
        if not isinstance(values, np.ndarray):
            values = np.asarray(values, dtype=np.float32)
        elif values.dtype != np.float32:
            values = values.astype(np.float32)
        return values[canonical_indices]

    if table_base_name in {"line", "trafo"}:
        return table[column_name].to_numpy()

    indices = _get_node_indices_for_table(net, cache, table_base_name)
    values = table[column_name].to_numpy()
    if not isinstance(values, np.ndarray) or values.dtype != np.float32:
        values = np.asarray(values)

    result = np.bincount(indices, weights=values, minlength=node_count).astype(np.float32)
    if table_name in {"gen", "sgen"}:
        result = -result
    return result


def get_raw_observation(
    net: pandapowerNet,
    cache: dict,
    table_name: str,
    column_name: str,
) -> NDArray[np.float32]:
    """
    Extract one column from a pandapower table at raw (un-aggregated) table length.

    Unlike :func:`get_observation`, the values are returned per table element (one entry
    per row), without folding elements onto their electrical nodes. Used for the
    ``not fix_obs_space`` path where the observation keeps the native table length.

    Parameters
    ----------
    net : pandapowerNet
        The network.
    cache : dict
        Observation cache from :func:`make_obs_cache`.
    table_name : str
        Table name, optionally ``res_``/``profile_`` prefixed (e.g. ``"load"``, ``"res_bus"``).
    column_name : str
        Column to extract (e.g. ``"p_mw"``).

    Returns
    -------
    NDArray[np.float32]
        The raw column values (one entry per table element).
    """
    table, table_len = _get_table_and_length(net, cache, table_name)
    if table_len == 0:
        return np.array([], dtype=np.float32)
    return table[column_name].to_numpy()
