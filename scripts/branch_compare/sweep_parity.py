"""Score every action on both backends of *this* branch and report where the answers differ.

``dump_results.py`` compares a handful of scenarios across checkouts. This walks the whole
capped action space on one checkout, so the question it answers is the sharper one: over every
topology the agent can reach, does swapping pandapower for lightsim2grid ever change what the
environment says?

Three kinds of disagreement are separated, because they matter differently:

1. **Decision differences** -- an action converges on one backend and not the other. Those change
   what an agent may do, so they are listed individually rather than aggregated.
2. **Ranking differences** -- the action a greedy sweep would pick. A numeric delta that never
   reorders the candidates costs nothing downstream.
3. **Magnitudes** -- worst absolute difference per quantity, over the actions both backends solved.

Both backends run in the same interpreter. That is deliberately wrong for *timing* (the second
one would inherit the first's warm caches -- see ``CLAUDE.md``) and exactly right for *values*:
it guarantees both are handed a bit-identical starting state.

``--nminus1`` switches the comparison to N-1 environments: the same three questions, asked of the
contingency aggregates the observations actually read (``max_loading_percent`` and friends) rather
than of the N-0 result. A sweep costs one full contingency analysis per action per backend, so
pair it with ``--limit``.

Run with::

    python sweep_parity.py --grid grid_89.pkl --actions-cache actions.pkl
    python sweep_parity.py --grid grid_89.pkl --actions-cache actions.pkl --nminus1 --limit 20
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from bench_branch import RESET_OPTIONS, expand_grid, make_config

from pandapower_env.environments.simulation_env import PPTopoGym

# Quantities pulled off the solved net for every action, as (label, extractor).
QUANTITIES = (
    ("line_loading", lambda net: net.res_line["loading_percent"].to_numpy(dtype=float)),
    ("bus_vm", lambda net: net.res_bus["vm_pu"].to_numpy(dtype=float)),
    ("bus_va", lambda net: net.res_bus["va_degree"].to_numpy(dtype=float)),
    ("trafo_loading", lambda net: (
        net.res_trafo["loading_percent"].to_numpy(dtype=float)
        if len(net.trafo) and not net.res_trafo.empty else np.zeros(0))),
)

# The extra quantities an N-1 environment produces: the aggregates over the contingency set, which
# is what every N-1 observation and the greedy "nminus1" feedback read.
NMINUS1_QUANTITIES = (
    ("n1_line_max", lambda net: net.res_line["max_loading_percent"].to_numpy(dtype=float)),
    ("n1_line_min", lambda net: net.res_line["min_loading_percent"].to_numpy(dtype=float)),
    ("n1_bus_vm_max", lambda net: net.res_bus["max_vm_pu"].to_numpy(dtype=float)),
    ("n1_bus_vm_min", lambda net: net.res_bus["min_vm_pu"].to_numpy(dtype=float)),
    ("n1_trafo_max", lambda net: (
        net.res_trafo["max_loading_percent"].to_numpy(dtype=float)
        if len(net.trafo) and not net.res_trafo.empty else np.zeros(0))),
)


def sweep(config: dict, actions: list[int], quantities: tuple) -> dict[int, dict]:
    """Apply every action from the same reset state and capture the solved grid.

    :param config: the env config, already carrying the backend selection.
    :param actions: action indices to sweep.
    :param quantities: the ``(label, extractor)`` pairs to record per action.
    :return: action index -> {"reward", "converged", plus every entry of ``quantities``}.
    """
    env = PPTopoGym(config)
    records: dict[int, dict] = {}
    for action in actions:
        env.reset(options=RESET_OPTIONS)
        _obs, reward, terminated, _truncated, _info = env.step(action)
        converged = bool(env.net.converged) and not terminated
        record: dict = {"reward": float(reward), "converged": converged}
        if converged:
            record.update({name: extract(env.net) for name, extract in quantities})
        records[action] = record
    return records


def report(left: dict[int, dict], right: dict[int, dict], labels: tuple[str, str],
           quantities: tuple) -> None:
    """Print the decision, ranking and magnitude comparison between two sweeps."""
    left_label, right_label = labels
    actions = sorted(left)

    disagreements = [a for a in actions if left[a]["converged"] != right[a]["converged"]]
    both = [a for a in actions if left[a]["converged"] and right[a]["converged"]]
    print(f"{len(actions)} actions swept; "  # noqa: T201
          f"{sum(r['converged'] for r in left.values())} converge on {left_label}, "
          f"{sum(r['converged'] for r in right.values())} on {right_label}")
    print(f"convergence decisions that differ: {len(disagreements)}"  # noqa: T201
          + (f" -> {disagreements}" if disagreements else ""))

    worst_reward = max((abs(left[a]["reward"] - right[a]["reward"]) for a in both), default=0.0)
    print(f"{'quantity':<16} {'worst abs diff':>16} {'worst rel diff':>16}")  # noqa: T201
    print("-" * 50)  # noqa: T201
    print(f"{'reward':<16} {worst_reward:16.3e} {'':>16}")  # noqa: T201
    for name, _extract in quantities:
        worst_abs, worst_rel = 0.0, 0.0
        for action in both:
            a, b = left[action][name], right[action][name]
            if a.size == 0:
                continue
            finite = np.isfinite(a) & np.isfinite(b)
            if not np.any(finite):
                continue
            delta = np.abs(a[finite] - b[finite])
            worst_abs = max(worst_abs, float(delta.max()))
            scale = np.maximum(np.abs(a[finite]), 1e-9)
            worst_rel = max(worst_rel, float((delta / scale).max()))
        print(f"{name:<16} {worst_abs:16.3e} {worst_rel:16.3e}")  # noqa: T201

    # A greedy sweep ranks on the worst line loading; agreeing on the pick is what matters.
    def best(records: dict[int, dict]) -> int:
        scored = [(np.nanmax(r["line_loading"]), a) for a, r in records.items() if r["converged"]]
        return min(scored)[1] if scored else -1

    left_best, right_best = best(left), best(right)
    verdict = "SAME" if left_best == right_best else "DIFFERENT"
    print(f"\ngreedy pick (min worst-line-loading): {left_label}={left_best} "  # noqa: T201
          f"{right_label}={right_best}  -> {verdict}")


def main() -> None:
    """Sweep the action space on both backends and print the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--actions-cache", type=Path)
    parser.add_argument("--limit", type=int, help="sweep only the first N actions")
    parser.add_argument("--nminus1", action="store_true",
                        help="compare the N-1 aggregates instead of the N-0 result")
    args = parser.parse_args()

    with args.grid.open("rb") as handle:
        payload = pickle.load(handle)
    net, action_list = expand_grid(payload, args.actions_cache)

    base_config = make_config(net, action_list, payload["profiles"], nminus1=args.nminus1)
    quantities = QUANTITIES + NMINUS1_QUANTITIES if args.nminus1 else QUANTITIES
    probe = PPTopoGym(dict(base_config))
    actions = list(probe.df_actions.index)[: args.limit] if args.limit else list(probe.df_actions.index)
    del probe

    sweeps = {
        backend: sweep({**base_config, "backend": backend}, actions, quantities)
        for backend in ("pandapower", "lightsim")
    }
    report(sweeps["pandapower"], sweeps["lightsim"], ("pandapower", "lightsim"), quantities)


if __name__ == "__main__":
    main()
