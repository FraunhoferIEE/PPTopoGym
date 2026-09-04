from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import simbench

if TYPE_CHECKING:

    import pandapower as pp
    import pandapower.contingency
    from pandapower import pandapowerNet
logger = logging.getLogger(__name__)

# Process-local cache of the raw Simbench profile library, keyed on the scenario index.
# ``simbench.get_all_simbench_profiles`` re-reads several CSV files (~35136 x 618 values)
# on every call and takes ~1.2 s, while being a pure function of the scenario index. Every
# example config calls it once, so building two environments in one process paid it twice.
# Kept module-level (not on the net) because it is shared across nets and grids; spawned
# worker processes simply start with an empty cache.
_SIMBENCH_PROFILE_CACHE: dict[int, dict[str, pd.DataFrame]] = {}


def _all_simbench_profiles_cached(sb_index: int) -> dict[str, pd.DataFrame]:
    """Return the full Simbench profile library for ``sb_index``, reading it at most once.

    The cached frames are the *shared* originals and must never be handed to callers
    directly -- :func:`deterministic_profiles` slices and copies them, so a caller that
    mutates its profiles in place cannot corrupt what the next build sees.

    :param sb_index: The Simbench scenario index (0 low, 1 med, 2 high, ...).
    :return: The cached dict of raw profile DataFrames, keyed by Simbench table name.
    """
    if sb_index not in _SIMBENCH_PROFILE_CACHE:
        _SIMBENCH_PROFILE_CACHE[sb_index] = simbench.get_all_simbench_profiles(sb_index)
    return _SIMBENCH_PROFILE_CACHE[sb_index]


def deterministic_profiles(net: pp.pandapowerNet, sb_index: int = 0) -> dict[str, pd.DataFrame]:
    """
    Return the first profiles from simbench and create DFs, consecutively for laod, gen, sgen.

    Conversion Simbench <-> Pandapower:
    load : load, renewables: static generator, powerplants: generator

    Parameters
    ----------
    net: pp net
    sb_index: 0 low, 1 med, 2 high, 3 high renewable, 4 high load, low renewable, 5 low load, high renewable

    Returns
    -------
    det_profiles : a dict of dataframes in Simbench style, usable later for PPTopoGym.

    """
    all_profiles = _all_simbench_profiles_cached(sb_index)
    net.gen["name"] = net.gen.index.to_series().apply(lambda x: f"Generator {x}")
    net.sgen["name"] = net.sgen.index.to_series().apply(
        lambda x: f"Static Generator {x}",
    )
    net.load["name"] = net.load.index.to_series().apply(lambda x: f"Load {x}")
    n_loads = len(net.load)*2 + 1 if len(net.load) > 0 else 0
    n_gens = len(net.gen) + 1 if len(net.gen) > 0 else 0
    n_sgens = len(net.sgen) + 1 if len(net.sgen) > 0 else 0
    # ``.copy()`` is required, not defensive style: the frames come from the shared
    # process-local cache, so returning views would let one net's in-place edit leak into
    # every net built afterwards. Copying the narrow slice costs ~2 ms against the ~1.2 s
    # the cache saves.
    det_profiles = {}
    det_profiles["load"] = all_profiles["load"].iloc[:, :n_loads].copy()
    det_profiles["renewables"] = all_profiles["renewables"].iloc[:, :n_sgens].copy()
    det_profiles["powerplants"] = all_profiles["powerplants"].iloc[:, :n_gens].copy()
    return det_profiles

def _add_column_names(net: pp.pandapowerNet) -> None:
    """
    Add profile names to load, gen, sgen.

    In PPTopoGym, the load, gen, sgen need profile names, to match the column-names of the profiles.

    Parameters
    ----------
    net: pandapowerNet
    """
    column_names = {}
    # For load profiles
    for sb_name in ["load", "renewables", "powerplants"]:
        pp_name = {"load": "load", "renewables":"sgen", "powerplants": "gen"}
        column_names[pp_name[sb_name]] = net.profiles[sb_name].columns[1:].to_series().apply(
                lambda x: x.replace("_pload", "").replace("_qload", ""),
                    ).unique()
    # add column names to load, gen, sgen:
    for sth in ["load", "gen", "sgen"]:
        net[sth]["profile"] = column_names[sth]

def get_first_sb_profiles(net: pp.pandapowerNet, sb_index: int =0) -> dict[str,  pd.DataFrame]:
    """
    Get the first set of profiles from the simbench profiles.

    Parameters
    ----------
    net: pandapowerNet
    sb_index: 0 low, 1 med, 2 high, 3 high renewable, 4 high load, low renewable, 5 low load, high renewable
    """
    net.profiles = deterministic_profiles(net, sb_index)
    _add_column_names(net)
    return net.profiles

def get_orig_profiles(net: pp.pandapowerNet) -> dict[str, pd.DataFrame]:
    """
    Create original profiles for loads, generators, and static generators.

    This creates initial absolute values for profiles, as in ssb.get_absolute_values(...).

    Parameters
    ----------
    net: pandapowerNet

    Returns
    -------
    orig_profiles: Dataframes for all injection profiles.
    """
    # Load profiles - vectorized
    load_p_profiles = pd.DataFrame(
        {f"{ld['profile']}_pload": net.profiles["load"][f"{ld['profile']}_pload"] for _, ld in net.load.iterrows()},
    )
    load_q_profiles = pd.DataFrame(
        {f"{ld['profile']}_qload": net.profiles["load"][f"{ld['profile']}_qload"] for _, ld in net.load.iterrows()},
    )

    df_profiles_load_p = load_p_profiles @ np.diag(net.load.p_mw.to_numpy())
    df_profiles_load_q = load_q_profiles @ np.diag(net.load.q_mvar.to_numpy())
    df_profiles_load_p.columns = net.load["name"].to_numpy()
    df_profiles_load_q.columns = net.load["name"].to_numpy()

    # Sgen profiles - vectorized
    sgen_profiles = pd.DataFrame(
        {sgen["profile"]: net.profiles["renewables"][sgen["profile"]] for _, sgen in net.sgen.iterrows()},
    )
    df_profiles_sgen_p = sgen_profiles @ np.diag(net.sgen.p_mw.values)
    df_profiles_sgen_p.columns = net.sgen["name"].to_numpy()
    df_profiles_sgen_q = pd.DataFrame(0.0, index=sgen_profiles.index, columns=net.sgen["name"].to_numpy())

    # Gen profiles - vectorized
    gen_profiles = pd.DataFrame(
        {gen["profile"]: net.profiles["powerplants"][gen["profile"]] for _, gen in net.gen.iterrows()},
    )
    df_profiles_gen_p = gen_profiles @ np.diag(net.gen.p_mw.to_numpy())
    df_profiles_gen_p.columns = net.gen["name"].to_numpy()
    df_profiles_gen_vm = pd.DataFrame(np.tile(net.gen.vm_pu.values, (len(gen_profiles), 1)),
                                    index=gen_profiles.index, columns=net.gen["name"].to_numpy())
    orig_profiles = {}
    orig_profiles["gen_p"] = df_profiles_gen_p
    orig_profiles["gen_vm"] = df_profiles_gen_vm
    orig_profiles["sgen_p"] = df_profiles_sgen_p
    orig_profiles["sgen_q"] = df_profiles_sgen_q
    orig_profiles["load_p"] = df_profiles_load_p
    orig_profiles["load_q"] = df_profiles_load_q

    return orig_profiles

def scale_profiles(net: pp.pandapowerNet, profiles: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Scale profiles by an internal column.

    This is only used internally in the scaling_recursive function.

    Parameters
    ----------
    net: pp.pandapowerNet
    profiles: dict, representing the profiles to be scaled.

    Returns
    -------
    profiles: dict, profiles to be scaled


    """
    profiles["load_p"] = pd.DataFrame(
        profiles["load_p"].to_numpy() @ np.diag(net.load["scenario_scaling"]),
        index=profiles["load_p"].index,
        columns=profiles["load_p"].columns,
    )
    profiles["sgen_p"] = pd.DataFrame(
        profiles["sgen_p"].to_numpy() @ np.diag(net.sgen["scenario_scaling"]),
        index=profiles["sgen_p"].index,
        columns=profiles["sgen_p"].columns,
    )
    profiles["gen_p"] = pd.DataFrame(
        profiles["gen_p"].to_numpy() @ np.diag(net.gen["scenario_scaling"]),
        index=profiles["gen_p"].index,
        columns=profiles["gen_p"].columns,
    )
    return profiles


def create_simbench_data_from_profiles(net: pp.pandapowerNet,
                                       profiles: dict[str, pd.DataFrame],
                                       ) -> dict[str, pd.DataFrame]:
    """
    Create a simulation benchmark dataset from the given profiles.

    Create Data, such that PPTopoGym can read this similar to Simbench input.
    Then also include the correct profiles names into net.gen, sgen, load DFs.

    Parameters
    ----------
    net: pp.pandapowerNet
    profiles: dict, representing the profiles from get_orig_profiles

    Returns
    -------
    profiles: dict[str, pd.DataFrame]: profiles in simbench form

    """
    # store time
    time_col = net.profiles["load"]["time"]

    # set profile-values to 1
    net.load["p_mw"] = 1.0
    net.sgen["p_mw"] = 1.0
    net.gen["p_mw"] = 1.0
    net.load["q_mvar"] = 1.0

    # del net.profiles
    simbench_data = {}
    df_load = pd.DataFrame()
    df_load["time"] = time_col
    # loads is a combination of profiles[load_p] and profiles[load_q],
    # alternating, with time-frame in the beginning, name them .._pload, ...qload
    for col in profiles["load_p"].columns:
        df_load[f"{col}_pload"] = profiles["load_p"][col]
        df_load[f"{col}_qload"] = profiles["load_q"][col]
    # gen
    df_gen = pd.DataFrame()
    df_gen["time"] = time_col
    for col in profiles["gen_p"].columns:
        df_gen[col] = profiles["gen_p"][col]
    df_sgen = pd.DataFrame()
    df_sgen["time"] = time_col
    for col in profiles["sgen_p"].columns:
        df_sgen[col] = profiles["sgen_p"][col]
    simbench_data = {"load": df_load, "powerplants": df_gen, "renewables": df_sgen}
    net.profiles = simbench_data

    # rename net.load.profiles, etc.
    net.load.profile = profiles["load_p"].columns
    net.sgen.profile = profiles["sgen_p"].columns
    net.gen.profile = profiles["gen_p"].columns

    return simbench_data



def add_random_profiles(  # noqa: PLR0913
    net: pp.pandapowerNet,
    pv_percentage: float = 0.25,
    wp_percentage: float = 0.25,
    bm_percentage: float = 0.25,
    hy_percentage: float = 0.25,
    seed: int | None = None,
) -> None:
    """
    Randomly assigns load and generation profiles to elements in the given pandapower network.

    This function ensures that the renewable generation profiles (PV, wind power, biomass, and hydro)
    are assigned according to the specified percentage distributions. The total percentage is normalized
    if necessary, and the profiles are assigned randomly using a given seed.

    :param net: The pandapower network to which the profiles will be assigned.
    :type net: pp.pandapowerNet
    :param seed: The seed for the random number generator to ensure reproducibility.
    :type seed: int
    :param pv_percentage: The percentage of static generators assigned to PV profiles.
    :type pv_percentage: float
    :param wp_percentage: The percentage of static generators assigned to wind power (WP) profiles.
    :type wp_percentage: float
    :param bm_percentage: The percentage of static generators assigned to biomass (BM) profiles.
    :type bm_percentage: float
    :param hy_percentage: The percentage of static generators assigned to hydro (Hydro) profiles.
    :type hy_percentage: float

    :return: None. The function modifies the `net` object in place.
    :rtype: None
    """
    if not hasattr(net, "profiles") or not net.profiles:
        net.profiles = simbench.get_all_simbench_profiles(0)
        # Make sure loads, generators each have unique names
        net.gen["name"] = net.gen.index.to_series().apply(lambda x: f"Generator {x}")
        net.sgen["name"] = net.sgen.index.to_series().apply(
            lambda x: f"Static Generator {x}",
        )
        net.load["name"] = net.load.index.to_series().apply(lambda x: f"Load {x}")
    renewables = get_profile_names(net.profiles["renewables"]).tolist()
    renewables = list(
        filter(lambda word: word.startswith(("PV", "WP", "BM", "Hydro")), renewables),
    )
    # ensure that percentages sum up to 1:
    total = pv_percentage + wp_percentage + bm_percentage + hy_percentage
    pv_percentage /= total
    wp_percentage /= total
    bm_percentage /= total
    hy_percentage /= total
    rng = (
        np.random.default_rng(seed=seed)
        if seed is not None
        else np.random.default_rng()
    )

    # For the static generators we only want solar, wind, bio mass and hydro power right now
    net.load["profile"] = rng.choice(
        get_profile_names(net.profiles["load"]),
        len(net.load),
    )
    net.gen["profile"] = rng.choice(
        get_profile_names(net.profiles["powerplants"]),
        len(net.gen),
    )

    # Choose renewable assets by percentage:
    pv_amount = int(pv_percentage * len(net.sgen))
    wp_amount = int(wp_percentage * len(net.sgen))
    bm_amount = int(bm_percentage * len(net.sgen))
    hy_amount = int(hy_percentage * len(net.sgen))
    # ensure, that this is in total the length we want to have
    difference = len(net.sgen) - (pv_amount + wp_amount + bm_amount + hy_amount)
    if difference != 0:
        amounts = {
            "pv_amount": pv_amount,
            "wp_amount": wp_amount,
            "bm_amount": bm_amount,
            "hy_amount": hy_amount,
        }
        keys = list(amounts.keys())
        index = 0

        while difference > 0:
            amounts[keys[index]] += 1
            difference -= 1
            index = (index + 1) % len(keys)
        while difference < 0:
            amounts[keys[index]] += 1
            difference += 1
            index = (index + 1) % len(keys)
        pv_amount, wp_amount, bm_amount, hy_amount = amounts.values()
    random.seed(seed)
    pv_list = random.choices(  # noqa: S311
        list(filter(lambda word: word.startswith("PV"), renewables)),
        k=pv_amount,
    )
    wp_list = random.choices(  # noqa: S311
        list(filter(lambda word: word.startswith("WP"), renewables)),
        k=wp_amount,
    )
    bm_list = random.choices(  # noqa: S311
        list(filter(lambda word: word.startswith("BM"), renewables)),
        k=bm_amount,
    )
    hy_list = random.choices(  # noqa: S311
        list(filter(lambda word: word.startswith("Hydro"), renewables)),
        k=hy_amount,
    )
    sgens = pv_list + wp_list + bm_list + hy_list
    random.shuffle(sgens)
    net.sgen["profile"] = sgens

    # Make sure loads, generators each have unique names
    net.gen["name"] = net.gen.index.to_series().apply(lambda x: f"Generator {x}")
    net.sgen["name"] = net.sgen.index.to_series().apply(
        lambda x: f"Static Generator {x}",
    )
    net.load["name"] = net.load.index.to_series().apply(lambda x: f"Load {x}")


def get_scenario_profiles(  # noqa: C901, PLR0912, PLR0913
    net: pandapowerNet,
    window: pd.Timedelta | str = "D",
    wp: str | None = None,
    pv: str | None = None,
    bm: str | None = None,
    hy: str | None = None,
) -> list[int]:
    """
    Find start indices of time windows in the simbench profiles which match the desired scenario.

    :param net: The pandapower network
    :type net: pandapowerNet
    :param window: time window to search for, e.g. 'D', 'h', '2D'
    :type window: str | None
    :param wp: wind power scenario, either None or one of ['low', 'medium', 'high']
    :type wp: str | None
    :param pv: solar energy scenario, either None or one of ['low', 'medium', 'high']
    :type pv: str | None
    :param bm: bio mass scenario, either None or one of ['low', 'medium', 'high']
    :type bm: str | None
    :param hy: hydro energy scenario, either None or one of ['low', 'medium', 'high']
    :type hy: str | None
    :return: list of start indices for relevant time windows
    :rtype: list[int]
    :return start_indices: list of start indices for relevant time windows
    :rtype: list[int]
    :return df_total: Dataframe containing total power generation for each renewable type per timestep
    :rtype: pd.DataFrame
    :return df_resampled: Resampled df_total dataframe: Aggregated per time window
    :rtype: pd.DataFrame
    """
    # Get list of static generators used in the net
    profiles = ["time", *net.sgen["profile"].to_list()]

    # Create a subset dataframe with only the relevant columns
    df_subset = net.profiles["renewables"][profiles]
    df_subset["time"] = pd.to_datetime(df_subset["time"], format="%d.%m.%Y %H:%M")

    # Multiply profile values with p_mw of the corresponding assets
    for c in range(1, len(df_subset.columns), 1):
        df_subset.iloc[:, [c]] = df_subset.iloc[:, [c]] * net.sgen["p_mw"].iloc[c - 1]

    # Get list of column names for each relevant energy source
    wp_cols = [x for x in profiles if x.startswith("WP")]
    pv_cols = [x for x in profiles if x.startswith("PV")]
    bm_cols = [x for x in profiles if x.startswith("BM")]
    hy_cols = [x for x in profiles if x.startswith("Hydro")]

    # Check if a scenario is set for which there is no corresponding asset in the grid
    if wp and not wp_cols:
        logger.warning(
            "Warning: WP scenario set to %s but no WP exists in the grid. Ignoring.",
            wp,
        )
        wp = None
    if pv and not pv_cols:
        logger.warning(
            "Warning: PV scenario set to %s but no PV exists in the grid. Ignoring.",
            pv,
        )
        pv = None
    if bm and not bm_cols:
        logger.warning(
            "Warning: BM scenario set to %s but no BM exists in the grid. Ignoring.",
            bm,
        )
        bm = None
    if hy and not hy_cols:
        logger.warning(
            "Warning: HY scenario set to %s but no HY exists in the grid. Ignoring.",
            hy,
        )
        hy = None

    # Summarize all assets of each type in new columns
    for cols, name in [
        (wp_cols, "WP_total"),
        (pv_cols, "PV_total"),
        (bm_cols, "BM_total"),
        (hy_cols, "HY_total"),
    ]:
        if cols:
            df_subset[name] = df_subset[cols].sum(axis=1)

    # Only keep time and total columns
    cols_total = ["time"] + [x for x in df_subset if x.endswith("_total")]
    df_subset = df_subset[cols_total]
    df_subset.copy()

    # Resample dataframe by defined rule and normalize it
    df_mean = df_subset.resample(rule=window, on="time").mean()
    df_mean.copy()

    # Calculate quantiles for each column to define bounds for low, medium, high values
    bounds = {
        col: (
            df_mean[col].min(),
            df_mean[col].quantile(0.25),
            df_mean[col].quantile(0.75),
            df_mean[col].max(),
        )
        for col in df_mean.columns
    }

    # Default range is from 0 to 3 (no filtering)
    mapping = {"low": (0, 1), "medium": (1, 2), "high": (2, 3)}
    wp1, wp2 = (0, 3) if wp is None else mapping.get(wp, (0, 3))
    pv1, pv2 = (0, 3) if pv is None else mapping.get(pv, (0, 3))
    bm1, bm2 = (0, 3) if bm is None else mapping.get(bm, (0, 3))
    hy1, hy2 = (0, 3) if hy is None else mapping.get(hy, (0, 3))

    # Apply filtering
    if wp:
        df_mean = df_mean[
            df_mean["WP_total"].between(
                bounds["WP_total"][wp1],
                bounds["WP_total"][wp2],
            )
        ]
    if pv:
        df_mean = df_mean[
            df_mean["PV_total"].between(
                bounds["PV_total"][pv1],
                bounds["PV_total"][pv2],
            )
        ]
    if bm:
        df_mean = df_mean[
            df_mean["BM_total"].between(
                bounds["BM_total"][bm1],
                bounds["BM_total"][bm2],
            )
        ]
    if hy:
        df_mean = df_mean[
            df_mean["HY_total"].between(
                bounds["HY_total"][hy1],
                bounds["HY_total"][hy2],
            )
        ]

    # Find start indices of suitable time windows
    start_indices = []
    for d in df_mean.index.to_numpy():
        date = pd.to_datetime(str(d)).strftime("%d.%m.%Y %H:%M")
        index = net.profiles["renewables"]["time"][
            net.profiles["renewables"]["time"] == date
        ].index[0]
        start_indices.append(index)

    if not start_indices:
        msg = "No matches found for chosen scenario."
        raise ValueError(msg)

    return start_indices


def get_profile_names(df_profile: pd.DataFrame) -> np.ndarray:
    """
    Get the names of each of the profiles offered by Simbench.

    :param df_profile: The net.profiles['load'], net.profiles['powerplants'] or net.profiles['renewables'] dataframe
    :type df_profile: pd.DataFrame
    :return: an array containing the names of the Simbench profiles
    :rtype: np.ndarray
    """
    return (
        df_profile.columns[1:]
        .to_series()
        .apply(lambda x: x.replace("_pload", "").replace("_qload", ""))
        .unique()
    )


def setup_profiles(
    net: pp.pandapowerNet,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate power profile DataFrames for loads, (static) generators.

    This function extracts and scales the active (P) and reactive (Q) power profiles for loads,
    as well as active power profiles for static generators and power plants.

    :param net: The pandapower network containing load, generator, and profile data.
    :type net: pp.pandapowerNet

    :return: A tuple containing four DataFrames:
             - df_profiles_load_p: Active power profiles for loads.
             - df_profiles_load_q: Reactive power profiles for loads.
             - df_profiles_sgen_p: Active power profiles for static generators.
             - df_profiles_gen_p: Active power profiles for conventional generators.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    df_profiles_load_p = pd.DataFrame(
        {
            ld["name"]: ld.p_mw * net.profiles["load"]["{}_pload".format(ld["profile"])]
            for _, ld in net.load.iterrows()
        },
    )

    df_profiles_load_q = pd.DataFrame(
        {
            ld["name"]: ld.q_mvar
            * net.profiles["load"]["{}_qload".format(ld["profile"])]
            for _, ld in net.load.iterrows()
        },
    )

    df_profiles_sgen_p = pd.DataFrame(
        {
            sgen["name"]: sgen.p_mw * net.profiles["renewables"][sgen["profile"]]
            for _, sgen in net.sgen.iterrows()
        },
    )

    df_profiles_gen_p = pd.DataFrame(
        {
            gen["name"]: gen.p_mw * net.profiles["powerplants"][gen["profile"]]
            for _, gen in net.gen.iterrows()
        },
    )

    return df_profiles_load_p, df_profiles_load_q, df_profiles_sgen_p, df_profiles_gen_p


def make_constant_profiles(net: pandapowerNet, row_index: int, num_copies: int) -> None:
    """
    Duplicate a single row from all profile DataFrames in the network.

    Parameters
    ----------
    net : pandapowerNet
        The pandapower network object containing profiles
    row_index : int
        Index of the row to duplicate
    num_copies : int
        Number of times to duplicate the row

    Raises
    ------
    ValueError
        If the network object does not have a 'profiles' attribute
    """
    if not hasattr(net, "profiles"):
        msg = "The given network object does not have a 'profiles' attribute."
        raise ValueError(msg)

    for key in net.profiles:
        row_to_duplicate = net.profiles[key].iloc[row_index:row_index+1]
        result = pd.concat([row_to_duplicate] * num_copies, ignore_index=True)
        net.profiles[key] = result
