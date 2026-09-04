"""Build the benchmark grids ONCE and pickle them, so both branches measure the same physical net.

`develop` and `develop_muzero` have diverged in how a config is assembled (`find_scaling_iterative`
+ `apply_scaling_to_profiles` vs `find_scaling_recursive`, `verify_all_actions` only on the muzero
side), so calling each branch's own `config_caseXX()` would compare two *different* grids.

What is shipped is deliberately the state *before* the double-busbar expansion: the scaled net,
still plain pandapower tables, plus the absolute timeseries. The substation modelling itself is
branch-specific -- the two `multi_bb_substation` schemas are not interchangeable (`develop` looks
for a ``b0_1_switch`` column that this branch does not write) -- so each branch expands the same
base grid with its own code, which is part of what is being compared.

Run with::

    poetry run python build_grids.py --out DIR
"""

from __future__ import annotations

import argparse
import copy
import pickle
import time
from pathlib import Path

import pandapower as pp
import pandas as pd
from pandapower.networks import case14, case30, case89pegase

from pandapower_env.action_space.action_space import add_actions_substation_line_switching
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.substation.create_double_busbar_substation import (
    create_all_double_busbar_substations,
)
from pandapower_env.toolbox.utils_profiles import (
    create_simbench_data_from_profiles,
    get_first_sb_profiles,
    get_orig_profiles,
)
from pandapower_env.toolbox.utils_scaling import (
    ensure_no_zero_values,
    find_scaling_iterative,
    find_scaling_recursive,
)

# case -> (loader, init_scaling, max_percent, overloaded_lines); the case30/case89 numbers are the
# ones data/example_configs.py uses. case14 passes init_scaling=None because it cannot be congested
# by scaling at all (see derate_line_ratings) and takes the iterative path instead.
GRIDS = {
    14: (case14, None, 60, 3),
    30: (case30, 1, 40, 3),
    89: (case89pegase, 100, 80, 4),
}

# Base-case loading the case14 line ratings are derated to, so that scaling the load can congest
# the grid the way it does on case30/case89.
CASE14_BASE_LOADING = 0.6

# Number of timesteps kept in the shipped profiles. The full Simbench year is 35136 rows, which
# makes the pickles huge for no benefit: the benchmark only ever visits the first few indices.
N_TIMESTEPS = 1000


def derate_line_ratings(net) -> None:
    """Rewrite ``net.line.max_i_ka`` so the base case sits at :data:`CASE14_BASE_LOADING`.

    pandapower's ``case14`` ships 42 kA line ratings, which put the solved base case at ~1.5%
    loading: no amount of load scaling congests it, so ``find_scaling_recursive`` recurses until it
    hits the recursion limit and ``find_scaling_iterative`` walks all 50 iterations without ever
    seeing an overload. Derating each line to its own base-case current gives the 14-bus grid the
    same "a few lines near their limit" character case30 and case89 already have, which is what the
    reward path and the N-1 sweep need in order to do representative work.

    :param net: the net to derate, mutated in place.
    """
    pp.runpp(net)
    net.line["max_i_ka"] = (net.res_line["i_ka"] / CASE14_BASE_LOADING).clip(lower=1e-3)


def build_scaled_net(size: int):
    """Load a grid, attach Simbench profiles and scale it until it has overloaded lines.

    :param size: grid key in :data:`GRIDS` (14 / 30 / 89).
    :return: the scaled net, with ``net.profiles`` set and no substations created yet.
    """
    loader, init_scaling, max_percent, overloaded_lines = GRIDS[size]
    net = loader()
    if size == 14:
        derate_line_ratings(net)
    get_first_sb_profiles(net, 2)
    ensure_no_zero_values(net)
    for key, df in net.profiles.items():
        net.profiles[key] = df.replace(0.0, 1.0)
    orig_profiles = get_orig_profiles(net)
    if init_scaling is None:
        find_scaling_iterative(
            net, orig_profiles=orig_profiles,
            max_percent=max_percent, overloaded_lines=overloaded_lines,
        )
    else:
        find_scaling_recursive(
            net, init_scaling=init_scaling, orig_profiles=orig_profiles,
            max_percent=max_percent, overloaded_lines=overloaded_lines,
        )
    create_simbench_data_from_profiles(net, orig_profiles)

    for eltype in ("gen", "sgen", "load"):
        if "scenario_scaling" in net[eltype].columns:
            del net[eltype]["scenario_scaling"]

    # N-1 needs a loading limit on every branch; pandapower's case nets do not always carry one.
    for table in ("line", "trafo", "trafo3w"):
        if len(net[table]) and "max_loading_percent" not in net[table].columns:
            net[table]["max_loading_percent"] = 100.0
    return net


def extract_config_profiles(net) -> dict:
    """Derive the absolute {element: {variable: DataFrame}} profiles both branches can consume.

    Rather than re-deriving the per-unit -> absolute scaling by hand, this builds a throwaway
    ``PPTopoGym`` on a copy of the net down the ``net.profiles`` (Simbench) route and reads the six
    immutable ``df_profiles_*`` tables it produced. Shipping those means neither branch has to agree
    on how the scaling is computed -- both are handed the finished numbers, which is the one
    timeseries route (``env_config["profiles"]``) they implement identically.

    :param net: the scaled net from :func:`build_scaled_net`.
    :return: profiles keyed by pandapower element and column, labelled by element index.
    """
    scratch = copy.deepcopy(net)
    create_all_double_busbar_substations(scratch)
    actions = add_actions_substation_line_switching(scratch)
    env = PPTopoGym({
        "net": scratch, "n_episodes": 366, "episode_length": 96,
        "action_space": actions, "nminus1": False,
    })
    tables = {
        ("load", "p_mw"): env.df_profiles_load_p,
        ("load", "q_mvar"): env.df_profiles_load_q,
        ("sgen", "p_mw"): env.df_profiles_sgen_p,
        ("sgen", "q_mvar"): env.df_profiles_sgen_q,
        ("gen", "p_mw"): env.df_profiles_gen_p,
        ("gen", "vm_pu"): env.df_profiles_gen_vm,
    }
    profiles: dict[str, dict[str, pd.DataFrame]] = {}
    for (element, variable), table in tables.items():
        if not len(net[element]) or table.empty:
            continue
        trimmed = table.iloc[:N_TIMESTEPS].copy()
        trimmed.columns = net[element].index
        trimmed.index = range(len(trimmed))
        profiles.setdefault(element, {})[variable] = trimmed
    return profiles


def main() -> None:
    """Build every grid and write ``grid_<size>.pkl`` into the output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, action="append", choices=list(GRIDS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for size in args.size or list(GRIDS):
        start = time.perf_counter()
        net = build_scaled_net(size)
        profiles = extract_config_profiles(net)
        del net.profiles  # both branches are driven from the shipped `profiles` dict

        payload = {
            "size": size,
            "net": net,
            "profiles": profiles,
            "n_bus": len(net.bus),
            "n_line": len(net.line),
            "n_trafo": len(net.trafo),
            "n_load": len(net.load),
            "n_gen": len(net.gen) + len(net.sgen),
        }
        path = args.out / f"grid_{size}.pkl"
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        print(  # noqa: T201
            f"case{size}: {payload['n_bus']} buses, {payload['n_line']} lines, "
            f"{payload['n_trafo']} trafos, {payload['n_load']} loads "
            f"-> {path.name} ({path.stat().st_size / 1e6:.1f} MB, "
            f"{time.perf_counter() - start:.1f}s)",
        )


if __name__ == "__main__":
    main()
