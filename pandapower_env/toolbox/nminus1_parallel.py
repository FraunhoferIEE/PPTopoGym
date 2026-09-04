"""Parallel N-1 line-loading calculation.

This is the *fast* sibling of :func:`pandapower_env.toolbox.utils.run_nminus1_powerflow`.
The serial version is kept untouched for safe-keeping and readability; this module only adds
a process-parallel backend with identical results.

Why this is safe to parallelise
--------------------------------
Each single-element contingency is an independent power flow, so the ``N`` contingencies can
be split across worker processes. Every worker rebuilds the grid once from a static JSON blob
(cached per process, exactly like the greedy agent's worker), overlays the *current* element
state, runs stock ``pandapower.contingency.run_contingency`` on its slice of the contingencies,
and returns the partial result dict. The parent merges those partials element-wise
(``np.fmax`` / ``np.fmin`` for loadings and bus voltages, first-wins for the overload cause).
Because the contingency list is split into *contiguous, ordered* chunks, the merge is
bit-for-bit identical to the serial result.

Nesting safety (spawn / greedy workers)
----------------------------------------
The environment is used inside ``spawn`` multiprocessing (MuZero) and inside the joblib/loky
parallelised greedy agent. To never create nested process pools, the public entry point falls
back to the serial implementation whenever it is already running inside a child process
(``multiprocessing.parent_process() is not None``) or whenever a single worker would suffice.
The speedup therefore applies in the *main* process (evaluation loops, notebooks); inside
worker processes the call degrades to the (correct) serial path.

Swappability
------------
The single public function writes exactly the same ``res_*`` columns as the serial version, so
either backend -- or a future neural-network surrogate for the N-1 line loadings -- can be
swapped in behind the same call site without touching the observation/busbar-lookup code.
"""

from __future__ import annotations

import contextlib
import io
import logging
import multiprocessing
import os
from typing import TYPE_CHECKING

import numpy as np
import pandapower as pp
import pandapower.contingency
from joblib import Parallel, delayed
from pandapower.powerflow import _powerflow

from pandapower_env.toolbox.utils import (
    run_nminus1_powerflow,
    select_topk_line_contingencies,
    warn_unavailable_ls2g,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandapower import pandapowerNet

logger = logging.getLogger(__name__)

# Element types whose contingencies are evaluated and whose loadings are monitored.
MONITORED_ELEMENTS = ("line", "trafo", "trafo3w")

# Process-local cache of the deserialized static net, keyed on the blob it was built from.
_WORKER_NET: pandapowerNet | None = None
_WORKER_KEY: int | None = None


def run_nminus1_powerflow_parallel(  # noqa: PLR0913
    net: pandapowerNet,
    pf_type: str = "ac",
    n_workers: int | None = None,
    use_ls2g: str | bool = "auto",
    static_blob: bytes | None = None,
    topk_percent: float = 100.0,
) -> None:
    """Run the N-1 power flow in parallel, writing the same results as the serial version.

    Splits the single-element contingencies across ``n_workers`` processes and merges the
    per-worker results into ``net`` (``res_line.max_loading_percent`` / ``min_loading_percent``
    / ``cause_element`` / ``cause_index`` and ``res_bus.max_vm_pu`` / ``min_vm_pu``), exactly
    as :func:`run_nminus1_powerflow` would. The N-0 ``loading_percent`` / ``vm_pu`` come from a
    single warm power flow on ``net`` in this (parent) process.

    Falls back to the serial implementation -- guaranteeing identical results with no process
    pool -- when only one worker would be used or when already inside a child process (so it is
    safe under ``spawn`` MuZero workers and the parallel greedy agent).

    :param net: a pandapower network (mutated in place with the N-1 results)
    :type net: pandapowerNet
    :param pf_type: the power-flow type, either 'ac' or 'dc'
    :type pf_type: str
    :param n_workers: number of worker processes; ``None`` uses ``os.cpu_count()``
    :type n_workers: int | None
    :param use_ls2g: whether lightsim2grid should be used as backend
    :type use_ls2g: str | bool
    :param static_blob: optional pre-serialized static net (the env caches one); built from
        ``net`` when omitted
    :type static_blob: bytes | None
    :param topk_percent: percentage of lines to evaluate as N-1 contingencies (default 100 = all);
        selected identically to the serial version (see :func:`select_topk_line_contingencies`)
    :type topk_percent: float
    :raises ValueError: if ``pf_type`` is not 'ac' or 'dc'
    """
    if pf_type not in {"ac", "dc"}:
        msg = "pf_type must be 'ac' or 'dc'."
        raise ValueError(msg)

    nminus1_cases = {element: {"index": net[element].index.to_numpy()} for element in MONITORED_ELEMENTS}
    n_contingencies = sum(len(case["index"]) for case in nminus1_cases.values())
    workers = _effective_workers(n_workers, n_contingencies)

    # Never nest process pools: inside a worker (greedy / spawn) or with a single worker the
    # serial implementation is used verbatim, so results (incl. the top-k filter) are identical.
    if workers <= 1 or multiprocessing.parent_process() is not None:
        run_nminus1_powerflow(net, pf_type=pf_type, use_ls2g=use_ls2g, topk_percent=topk_percent)
        return

    blob = static_blob if static_blob is not None else _net_to_static_blob(net)
    state = _snapshot_net_state(net)

    # The N-0 result that the env / agents read lives on the parent net; it also ranks the lines
    # for top-k contingency selection, so the selection is done once here and the workers receive
    # the already-filtered (still contiguous, ordered) line slice -- merge stays equal to serial.
    base_powerflow = pp.runpp if pf_type == "ac" else pp.rundcpp
    base_powerflow(net, lightsim2grid=use_ls2g)
    nminus1_cases["line"]["index"] = select_topk_line_contingencies(net, topk_percent)

    chunks = _split_contingencies(nminus1_cases, workers)
    partial_results = Parallel(n_jobs=workers, backend="loky")(
        delayed(_solve_contingency_subset)(blob, state, chunk, pf_type, use_ls2g) for chunk in chunks
    )
    merged_results = _merge_contingency_results(partial_results)
    _write_results_to_net(net, merged_results)
    warn_unavailable_ls2g(net, pf_type, use_ls2g)


def _effective_workers(n_workers: int | None, n_contingencies: int) -> int:
    """Clamp the requested worker count to ``[1, n_contingencies]`` (default: all CPUs)."""
    requested = n_workers if n_workers is not None else (os.cpu_count() or 1)
    return max(1, min(int(requested), n_contingencies))


def _split_contingencies(nminus1_cases: dict[str, dict], n_chunks: int) -> list[dict[str, dict]]:
    """Split the ordered contingency list into ``n_chunks`` contiguous per-element sub-cases.

    The contingencies are flattened in their global evaluation order (all lines, then trafos,
    then trafo3w), split into contiguous chunks, and regrouped into ``nminus1_cases``-shaped
    dicts. Contiguous + ordered chunks are what make the later merge identical to the serial
    result. Returns one sub-case dict per chunk; empty element types are omitted.

    :param nminus1_cases: ``{element: {"index": ndarray}}`` for the full contingency set
    :param n_chunks: number of chunks to produce (assumed ``>= 1``)
    :return: list of per-chunk ``{element: {"index": ndarray}}`` dicts
    """
    flat = [
        (element, int(index))
        for element in MONITORED_ELEMENTS
        for index in nminus1_cases.get(element, {}).get("index", [])
    ]
    if not flat:
        return []

    sub_cases: list[dict[str, dict]] = []
    for position_block in np.array_split(np.arange(len(flat)), min(n_chunks, len(flat))):
        grouped: dict[str, list[int]] = {}
        for position in position_block:
            element, index = flat[position]
            grouped.setdefault(element, []).append(index)
        sub_cases.append(
            {element: {"index": np.array(grouped[element], dtype=np.int64)}
             for element in MONITORED_ELEMENTS if element in grouped},
        )
    return sub_cases


def _merge_contingency_results(partials: list[dict[str, dict]]) -> dict[str, dict]:
    """Merge per-worker contingency result dicts into one global result.

    Loadings and bus voltages are combined with ``np.fmax`` / ``np.fmin`` (NaN-aware), the N-0
    base values are taken from the first worker (identical across workers), and the overload
    cause is resolved first-wins over the worker order -- reproducing the serial argmax exactly
    because the chunks are contiguous and ordered.

    :param partials: list of ``run_contingency`` result dicts, one per worker
    :return: one merged result dict in the same shape
    """
    base = partials[0]
    merged: dict[str, dict] = {}
    for element, base_result in base.items():
        result: dict = {"index": base_result["index"]}
        for base_only_key in ("loading_percent", "vm_pu"):  # N-0 values, identical per worker
            if base_only_key in base_result:
                result[base_only_key] = base_result[base_only_key]

        for key, combine in (
            ("max_loading_percent", np.fmax),
            ("min_loading_percent", np.fmin),
            ("max_vm_pu", np.fmax),
            ("min_vm_pu", np.fmin),
        ):
            combined = _combine_arrays(partials, element, key, combine)
            if combined is not None:
                result[key] = combined

        if element in MONITORED_ELEMENTS and any("max_loading_percent" in p[element] for p in partials):
            result.update(_merge_causes(partials, element, len(base_result["index"])))

        merged[element] = result
    return merged


def _combine_arrays(
    partials: list[dict[str, dict]],
    element: str,
    key: str,
    combine: np.ufunc,
) -> np.ndarray | None:
    """Reduce ``key`` across all partials that contain it via ``combine`` (e.g. ``np.fmax``)."""
    accumulator: np.ndarray | None = None
    for partial in partials:
        if key in partial[element]:
            values = partial[element][key].astype(np.float64)
            accumulator = values.copy() if accumulator is None else combine(accumulator, values, out=accumulator)
    return accumulator


def _merge_causes(partials: list[dict[str, dict]], element: str, n_elements: int) -> dict[str, np.ndarray]:
    """Resolve ``cause_element`` / ``cause_index`` / ``causes_overloading`` across workers.

    For each monitored element the cause is the contingency producing its highest N-1 loading;
    ties keep the earliest contingency. Iterating workers in chunk order with a strict ``>``
    comparison reproduces the serial first-wins behaviour exactly.
    """
    running_max = np.full(n_elements, -np.inf)
    cause_index = np.zeros(n_elements, dtype=np.int64)
    cause_element = np.empty(n_elements, dtype=object)
    causes_overloading = np.zeros(n_elements, dtype=bool)

    for partial in partials:
        element_result = partial[element]
        if "causes_overloading" in element_result:
            causes_overloading |= element_result["causes_overloading"]
        if "max_loading_percent" not in element_result:
            continue
        local_max = np.nan_to_num(element_result["max_loading_percent"], nan=-np.inf)
        improves = local_max > running_max
        cause_index = np.where(improves, element_result["cause_index"], cause_index)
        cause_element[improves] = element_result["cause_element"][improves]
        running_max = np.where(improves, local_max, running_max)

    return {"cause_index": cause_index, "cause_element": cause_element, "causes_overloading": causes_overloading}


def _write_results_to_net(net: pandapowerNet, merged: dict[str, dict]) -> None:
    """Write merged N-1 columns into ``net``'s ``res_*`` tables (mirrors ``run_contingency``).

    Columns already present (the N-0 ``loading_percent`` / ``vm_pu`` from the warm power flow)
    are left untouched; only the N-1 aggregates and cause columns are added.
    """
    for element, element_results in merged.items():
        index = element_results["index"]
        res_table = net[f"res_{element}"]
        for var, values in element_results.items():
            if var == "index" or var in res_table.columns.to_numpy():
                continue
            res_table.loc[index, var] = values


def _solve_contingency_subset(
    static_blob: bytes,
    state: dict[str, np.ndarray],
    sub_cases: dict[str, dict],
    pf_type: str,
    use_ls2g: str | bool,
) -> dict[str, dict]:
    """Worker entry point: solve one contingency subset and return its partial results.

    Rebuilds (cached) the static net from ``static_blob``, overlays the current ``state``,
    initialises the power-flow options once, and runs ``run_contingency`` over ``sub_cases``
    through the low-level ``_powerflow`` (no per-call option re-parsing). ``write_to_net`` is
    off -- only the result dict is returned to the parent for merging.
    """
    net = _ensure_net_from_blob(static_blob)
    _apply_net_state(net, state)
    _ensure_trafo_loading_limit(net)

    base_powerflow = pp.runpp if pf_type == "ac" else pp.rundcpp

    def evaluate_contingency(net: pandapowerNet, **_kwargs: object) -> None:
        _powerflow(net)

    with _quiet_pandapower():
        base_powerflow(net, lightsim2grid=use_ls2g)  # initialise options once
        return pp.contingency.run_contingency(
            net=net,
            nminus1_cases=sub_cases,
            contingency_evaluation_function=evaluate_contingency,
            write_to_net=False,
        )


def _ensure_net_from_blob(static_blob: bytes) -> pandapowerNet:
    """Build (or reuse) a process-local net from ``static_blob`` (cached by blob hash)."""
    global _WORKER_NET, _WORKER_KEY  # noqa: PLW0603
    key = hash(static_blob)
    if _WORKER_NET is None or key != _WORKER_KEY:
        with io.StringIO(static_blob.decode("utf-8")) as fp:
            _WORKER_NET = pp.from_json(fp)
        _WORKER_KEY = key
    return _WORKER_NET


def _snapshot_net_state(net: pandapowerNet) -> dict[str, np.ndarray]:
    """Snapshot the *current* mutable element state to overlay on the static base net.

    Captures the exact values (topology + setpoints), preserving ``NaN`` tap positions, so a
    worker net reconstructed from the static blob matches ``net`` bit-for-bit. This is
    deliberately value-based (not profile-index based) so it stays correct even if the
    line/bus loadings are later produced by a surrogate model.
    """
    state: dict[str, np.ndarray] = {}
    if len(net.switch):
        state["switch_closed"] = net.switch["closed"].to_numpy(dtype=bool)
    if len(net.line):
        state["line_in_service"] = net.line["in_service"].to_numpy(dtype=bool)
    for trafo in ("trafo", "trafo3w"):
        if len(net[trafo]):
            state[f"{trafo}_in_service"] = net[trafo]["in_service"].to_numpy(dtype=bool)
            state[f"{trafo}_tap_pos"] = net[trafo]["tap_pos"].to_numpy(dtype=np.float64)  # NaN preserved
    for element, columns in (("load", ("p_mw", "q_mvar")), ("sgen", ("p_mw", "q_mvar")), ("gen", ("p_mw", "vm_pu"))):
        if len(net[element]):
            for column in columns:
                state[f"{element}_{column}"] = net[element][column].to_numpy(dtype=np.float64)
    return state


def _apply_net_state(net: pandapowerNet, state: dict[str, np.ndarray]) -> None:
    """Overlay a :func:`_snapshot_net_state` snapshot onto ``net`` in place."""
    if "switch_closed" in state:
        net.switch["closed"] = state["switch_closed"]
    if "line_in_service" in state:
        net.line["in_service"] = state["line_in_service"]
    for trafo in ("trafo", "trafo3w"):
        if f"{trafo}_in_service" in state:
            net[trafo]["in_service"] = state[f"{trafo}_in_service"]
            net[trafo]["tap_pos"] = state[f"{trafo}_tap_pos"]
    for element, columns in (("load", ("p_mw", "q_mvar")), ("sgen", ("p_mw", "q_mvar")), ("gen", ("p_mw", "vm_pu"))):
        for column in columns:
            if f"{element}_{column}" in state:
                net[element][column] = state[f"{element}_{column}"]


def _ensure_trafo_loading_limit(net: pandapowerNet) -> None:
    """Give trafo / trafo3w a ``max_loading_percent`` so ``run_contingency`` doesn't KeyError.

    Uses the same default (100.0) as the serial fast path, so the swallowed-error spam is
    avoided and the ``causes_overloading`` results match the serial baseline exactly.
    """
    for element in ("trafo", "trafo3w"):
        if len(net[element]) and "max_loading_percent" not in net[element].columns:
            net[element]["max_loading_percent"] = 100.0


def _net_to_static_blob(net: pandapowerNet) -> bytes:
    """Serialize a *static* copy of ``net`` to JSON bytes (strips profiles + result tables).

    Mirrors ``PPTopoGym.dump_static_net_bytes`` for callers that don't have the env handy
    (e.g. tests): the heavy attributes are removed and restored in place to avoid a deepcopy.
    """
    saved_profiles = getattr(net, "profiles", None)
    had_profiles = hasattr(net, "profiles")
    if had_profiles:
        delattr(net, "profiles")
    saved_results = {}
    for res_key in ("res_bus", "res_line", "res_trafo", "res_trafo3w", "res_sgen", "res_load", "res_gen", "res_switch"):
        if hasattr(net, res_key):
            saved_results[res_key] = getattr(net, res_key)
            delattr(net, res_key)
    try:
        sink = io.StringIO()
        pp.to_json(net, sink)
        json_str = sink.getvalue()
    finally:
        if had_profiles:
            net.profiles = saved_profiles
        for res_key, value in saved_results.items():
            setattr(net, res_key, value)
    return json_str.encode("utf-8")


@contextlib.contextmanager
def _quiet_pandapower() -> Iterator[None]:
    """Silence pandapower's per-contingency convergence logging inside workers."""
    pp_logger = logging.getLogger("pandapower")
    previous_level = pp_logger.level
    pp_logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        pp_logger.setLevel(previous_level)

