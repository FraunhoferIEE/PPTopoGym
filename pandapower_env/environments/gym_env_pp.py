from __future__ import annotations

import copy
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, MutableMapping, SupportsFloat, TypeVar
from weakref import WeakValueDictionary

import gymnasium as gym
import numpy as np
import pandapower as pp
import pandapower.contingency
import pandas as pd

from pandapower_env.toolbox.nminus1_parallel import run_nminus1_powerflow_parallel
from pandapower_env.toolbox.utils import run_nminus1_powerflow, run_powerflow

if TYPE_CHECKING:
    from pandapower_env.toolbox.ls2g_backend import LightsimBackend

ObsType = TypeVar("ObsType")

logger = logging.getLogger(__name__)


class _ProfileTables(dict):
    """The derived ``df_profiles_*`` tables of one environment.

    A plain ``dict`` subclass, so it can be weak-referenced by :data:`_SHARED_PROFILE_TABLES`.
    Each environment that adopts these tables keeps a strong reference to *this* object, which
    is what decides how long the shared entry lives.

    ``source_profiles`` pins the raw profile tables whose ``id()`` values make up this entry's
    cache key. Without it a key could outlive the objects it names, and CPython could hand the
    same ``id()`` to a *different* DataFrame -- a false cache hit serving the wrong timeseries.
    Holding them makes that impossible: a live key always names live objects.
    """

    source_profiles: dict[str, Any] | None = None


# Net key naming the ``(pf_type, use_ls2g, nminus1)`` request the results currently on the net were
# solved for. Stored on the net so it shares the exact lifetime of ``net.converged``, which is what
# says those results are still valid -- see :meth:`BaseEnvPP.run_pf`.
_SOLVED_REQUEST_KEY = "_ppenv_solved_for"


# The ``(element, variable)`` pairs a config-supplied ``profiles`` dict may carry, mapped to the
# derived table each one fills. This is deliberately the injection-only subset: varying arbitrary
# net columns (``line.in_service``, ``trafo.tap_pos``, ``multi_bb_substation.state``) over time is
# a UCTE feature that this branch does not carry, so anything outside this map is rejected rather
# than silently dropped (see BaseEnvPP.setup_profiles_from_config).
_CONFIG_PROFILE_TARGETS: dict[tuple[str, str], str] = {
    ("load", "p_mw"): "df_profiles_load_p",
    ("load", "q_mvar"): "df_profiles_load_q",
    ("sgen", "p_mw"): "df_profiles_sgen_p",
    ("sgen", "q_mvar"): "df_profiles_sgen_q",
    ("gen", "p_mw"): "df_profiles_gen_p",
    ("gen", "vm_pu"): "df_profiles_gen_vm",
}

# Elements and columns an agent action may change, before narrowing to a given net (see
# _supported_action_types). ``multi_bb_substation.state`` is listed because grids produced by
# the UCTE-style tooling carry that column; the substation tables this branch builds do not,
# and their state lives entirely in ``switch.closed``.
_SUPPORTED_ACTION_TYPES: dict[str, tuple[str, ...]] = {
    "switch": ("closed",),
    "line": ("in_service",),
    "multi_bb_substation": ("state",),
    "trafo": ("tap_pos",),
}

# Derived profile tables shared between environments built from the same profiles and base
# values (see BaseEnvPP._reuse_shared_profile_tables). Entries vanish once the last
# environment referencing them is collected, so this never grows into a leak. It is a
# same-process cache, hence empty inside each spawned worker.
_SHARED_PROFILE_TABLES: MutableMapping[tuple, _ProfileTables] = WeakValueDictionary()


def deepcopy_net_sharing_profiles(net: pp.pandapowerNet) -> pp.pandapowerNet:
    """Deep-copy a pandapower net but share its ``profiles`` tables with the original.

    Everything an environment mutates -- the element tables, switches, results -- is deep
    copied as usual. Only ``net.profiles`` is shared, because it is *read-only* after
    construction: :meth:`BaseEnvPP.setup_profiles` reads it once to build the immutable
    ``df_profiles_*`` tables and nothing in the package writes to it afterwards.

    This matters in the multi-environment setting. The Simbench profile tables dominate an
    environment's footprint (~53 of ~54 MB per case30 env, held once for the live net, once
    for ``net_copy_from`` and once for the config), so copying them per environment is what
    makes a vectorized run expensive. Sharing them makes N environments cost roughly one
    copy of the timeseries instead of N.

    The caller must not mutate ``net.profiles`` in place afterwards -- doing so would be
    visible to every net sharing it. Replace the dict (``net.profiles = {...}``) instead,
    which rebinds only that net.

    :param net: the network to copy
    :type net: pp.pandapowerNet
    :return: a deep copy whose ``profiles`` entry is the very same object as the original's
    :rtype: pp.pandapowerNet
    """
    profiles = net.get("profiles")
    if profiles is None:
        return copy.deepcopy(net)

    # deepcopy(memo) is told the profiles object is "already copied" to itself, so the copy
    # references it instead of duplicating it -- without mutating `net` even temporarily
    # (which would be unsafe if another thread/env reads it concurrently).
    memo = {id(profiles): profiles}
    return copy.deepcopy(net, memo)


def _supported_action_types(net: pp.pandapowerNet) -> dict[str, list[str]]:
    """Narrow :data:`_SUPPORTED_ACTION_TYPES` to the elements and columns ``net`` actually has.

    Callers snapshot and restore exactly these columns around a simulated action, so naming a
    column the net does not carry turns every such snapshot into a ``KeyError``. Narrowing keeps
    the declaration honest per grid: the double-busbar tables this branch builds have no
    ``state`` column, and dropping it loses nothing because the substation configuration is
    fully described by ``switch.closed``, which stays in the list.

    :param net: the network whose element tables decide what survives.
    :type net: pp.pandapowerNet
    :return: the supported ``{element: [column, ...]}``, without empty entries.
    :rtype: dict[str, list[str]]
    """
    supported: dict[str, list[str]] = {}
    for element, columns in _SUPPORTED_ACTION_TYPES.items():
        if element not in net:
            continue
        present = [column for column in columns if column in net[element].columns]
        if present:
            supported[element] = present
    return supported


def _positionally_indexed(profile_df: pd.DataFrame) -> pd.DataFrame:
    """Return ``profile_df`` with a plain ``0..N-1`` row index.

    ``load_profile_timestep_into_net`` looks a timestep up by *label* (``df.loc[index]``), so a
    profile whose index is not ``0..N-1`` -- e.g. one assembled by concatenating duplicated rows --
    would either raise or, worse, return several rows for one timestep. Frames that already carry
    the right index are returned unchanged, so the common case costs one index comparison and no
    copy.

    :param profile_df: a timeseries table, one row per timestep.
    :type profile_df: pd.DataFrame
    :return: the same frame, or a re-indexed copy of it.
    :rtype: pd.DataFrame
    """
    if profile_df.index.equals(pd.RangeIndex(len(profile_df))):
        return profile_df
    return profile_df.reset_index(drop=True)


class BaseEnvPP(gym.Env, ABC):
    """
    Prototype for a simple electrical grid RL environment.

    This is a base class which will not work correctly on its own, but
    must be used in a derived class that implements load_action,
    create_observation, and calculate_reward.

    The specific behavior of the environment is described in the step()
    function (which can also be rewritten in the derived class).
    """

    def __init__(self, env_config: dict[Any, Any]) -> None:
        """
        Initialize the gym_env_pp environment.

        Required configuration keys in env_config:
        - net: pp.pandapowerNet or path to pandapower network file
        - n_episodes: Number of episodes to run
        - episode_length: the length (number of timesteps) of a single episode.
        - info_intermediate_obs: observations for the direct feedback of topology change.
            default: bus voltages, line loadings

        :param env_config: environment configuration
        :type env_config: dict
        """
        super().__init__()

        # The number of timesteps prepared
        self.n_episodes = env_config["n_episodes"]

        # Episode length (number of timesteps per episode)
        self.episode_length = env_config["episode_length"]
        self.current_step = 0
        # Steps *completed* in the current episode. Distinct from ``current_step``, which step()
        # raises before the action is applied: this one only advances once a step has produced a
        # reward, so a reward function can test ``episode_step_counter == 0`` for "first step of
        # the episode" -- which is what the potential-based rewards gate their carry-over on.
        self.episode_step_counter = 0

        # If n-1 powerflows should be calculated
        self.nminus1 = env_config.get("nminus1", False)
        # If the n-1 powerflows should be run in parallel (process pool). Falls back to serial
        # inside child processes (spawn / greedy workers). "n-1 workers" None -> all CPUs.
        self.nminus1_parallel = env_config.get("n-1 parallel", False) # 16 for case30, 32 for case89
        self.nminus1_workers = env_config.get("n-1 workers")
        # Percentage of lines evaluated as N-1 contingencies (100 = all lines). Below 100, only
        # the top-k% most loaded lines are switched off (see select_topk_line_contingencies).
        self.nminus1_topk = env_config.get("n-1-topk", 100.0)

        # The current timeseries index of the episode
        self.index = 0

        self.resolution = env_config.get("resolution", 1.)  # timestep resolution in hours
        # info to store the intermediate values
        self.info_intermediate_obs = env_config.get("info_observations", [
            "bus_voltage_magnitude",
            "bus_voltage_angle",
            "line_loadings",
            "total_energy_overload",
            "max_loading_percent",
        ]) # default: bus voltages, line loadings


        # Set up the network grid (you can also use set_grid to do this.)
        # Deep-copy a net passed by object so each env owns its grid: multiple envs built
        # from the same config (e.g. an env and an agent's internal env) must not share a
        # mutable net. Otherwise the in-place reset of one env mutates the other's load
        # setpoints, and setup_profiles -- which scales the *live* load.p_mw -- compounds
        # the scaling on every subsequent env, eventually diverging the power flow.
        if isinstance(env_config["net"], pp.pandapowerNet):
            self.net = deepcopy_net_sharing_profiles(env_config["net"])
        else:
            self.net = pp.from_json(env_config["net_file"])
        self.net.converged = None  # initially, no powerflow has been run
        # Pristine oracle (deep=True reset fallback). The profile tables are shared rather
        # than copied: they are read-only after setup_profiles, and this copy exists only to
        # restore topology/setpoints. On case30 that is ~19.5 MB per environment saved.
        self.net_copy_from = deepcopy_net_sharing_profiles(self.net)

        # Baseline topology, captured once from the pristine net (== net_copy_from here).
        # reset() restores these in place instead of deepcopying the whole net; power
        # setpoints are re-derived by load_profile_timestep_into_net and res_* are
        # recomputed by the power flow, so only topology needs to be restored.
        self._baseline_topology: dict[str, np.ndarray] = self._capture_topology()


        # Where the timeseries comes from. A config-supplied ``profiles`` dict wins over
        # ``net.profiles``: it carries *absolute* per-element values keyed by pandapower element
        # and column (see setup_profiles_from_config), which is what the frozen-episode datasets
        # need, whereas ``net.profiles`` is the Simbench per-unit route scaled by the net's base
        # values. Both end at the same derived ``df_profiles_*`` tables.
        self._config_profiles: dict[str, dict[str, pd.DataFrame]] | None = env_config.get("profiles")
        if self._config_profiles is None and "profiles" not in self.net:
            msg = (
                "Error - no timeseries for the environment: neither env_config['profiles'] nor "
                "net.profiles is set. Stopping."
            )
            raise RuntimeError(msg)

        # The elements and columns an agent action -- or a restore around one -- may change,
        # narrowed to the ones this net actually carries.
        self.supported_action_types: dict[str, list[str]] = _supported_action_types(self.net)

        # Set when the derived profile tables are shared with other envs (see
        # _reuse_shared_profile_tables); holds the shared entry alive for this env's lifetime.
        self._shared_profile_tables: _ProfileTables | None = None

        self.df_profiles_load_p = pd.DataFrame()
        self.df_profiles_load_q = pd.DataFrame()
        self.df_profiles_sgen_p = pd.DataFrame()
        self.df_profiles_sgen_q = pd.DataFrame()
        self.df_profiles_gen_p = pd.DataFrame()
        self.df_profiles_gen_vm = pd.DataFrame()

        # Cache of the per-timestep profile rows (numpy arrays), keyed on self.index.
        # The profile tables are immutable after setup_profiles, so every profile
        # observation/aggregate at one timestep reads the same rows; this avoids a pandas
        # .loc label lookup per observation key (and per create_observation while the
        # index is unchanged). Consumed by PPTopoGym._current_profile_rows.
        self._profile_rows_index: int | None = None
        self._profile_rows_cache: dict[tuple[str, str], np.ndarray] = {}

        # Build the derived profile tables from whichever source is live
        if not self._reuse_shared_profile_tables():
            if self._config_profiles is not None:
                self.setup_profiles_from_config(self._config_profiles)
            else:
                self.setup_profiles()
            self._publish_shared_profile_tables()

        self.pf_type = env_config.get("pf_type", "ac")

        # Which solver run_pf uses. "pandapower" is the default because it is the one that is
        # bit-identical to every stored result in this repository; "lightsim" re-expresses the
        # switched topology as per-element bus assignments and is ~14x faster, at ~1e-8 agreement
        # (see toolbox/ls2g_backend.py). It covers the N-1 sweep too, so one key selects the
        # solver for both.
        self.backend_name = env_config.get("backend", "pandapower")
        self._lightsim_backend: LightsimBackend | None = None

        # test if the episode length is smaller than the number of timesteps
        if self.episode_length > self.n_total_timesteps:
            msg = (
                f"Episode length {self.episode_length} is larger than "
                f"number of timesteps {self.n_total_timesteps} in the profiles."
            )
            raise RuntimeError(msg)

        self.worst_reward = env_config.get("worst_reward", -1000.)

        # tracking
        self.episode_index = 0
        self.cache: dict = {} # a dictionary, to externally store information across different steps

        # global variables
        self.clip_max_loading = env_config.get("clip_max_loading", 200.0)

    # Names of the derived, immutable per-timestep profile tables built by setup_profiles.
    _PROFILE_TABLE_NAMES = (
        "df_profiles_load_p", "df_profiles_load_q",
        "df_profiles_sgen_p", "df_profiles_sgen_q",
        "df_profiles_gen_p", "df_profiles_gen_vm",
    )

    @property
    def n_total_timesteps(self) -> int:
        """Number of timeseries rows the profile tables share.

        Read across all six derived tables rather than off ``df_profiles_load_p`` alone, so a net
        without loads still reports the real horizon instead of 0.

        :return: the number of timesteps available to reset into and step through.
        :rtype: int
        """
        return max(len(getattr(self, name)) for name in self._PROFILE_TABLE_NAMES)

    @property
    def net_profiles(self) -> dict[str, dict[str, pd.DataFrame]]:
        """The live timeseries as ``{element: {variable: DataFrame}}``, whatever built it.

        A read-only *view* onto the derived ``df_profiles_*`` tables -- the very same objects, not
        copies -- so a caller that needs to know which element columns vary over time (a grid-state
        snapshot taken around a simulated action, say) gets the same answer whether the environment
        was built from ``net.profiles`` or from ``env_config["profiles"]``.

        :return: absolute per-timestep values, keyed by pandapower element and column.
        :rtype: dict[str, dict[str, pd.DataFrame]]
        """
        view: dict[str, dict[str, pd.DataFrame]] = {}
        for (element, variable), table_name in _CONFIG_PROFILE_TARGETS.items():
            table = getattr(self, table_name)
            if len(self.net[element]) and len(table):
                view.setdefault(element, {})[variable] = table
        return view

    def _base_element_values(self) -> tuple[bytes, ...]:
        """Serialize the net's base injection values, the second half of the sharing key."""
        return tuple(
            self.net[table][column].to_numpy().tobytes()
            for table, column in (
                ("load", "p_mw"), ("load", "q_mvar"),
                ("sgen", "p_mw"), ("sgen", "q_mvar"),
                ("gen", "p_mw"), ("gen", "vm_pu"),
            )
        )

    def _profile_source(self) -> dict[str, Any] | None:
        """Return the raw timeseries dict the derived tables were built from, or None.

        Either the Simbench ``{name: DataFrame}`` tables off the net, or the config's
        ``{element: {variable: DataFrame}}`` dict -- whichever this environment ingested.
        """
        if self._config_profiles is not None:
            return self._config_profiles
        return self.net.get("profiles")

    def _shared_profile_key(self) -> tuple | None:
        """Identify the derived profile tables this environment would produce, for cross-env sharing.

        Both ingestion paths are pure functions of their raw frames and of the net's base element
        values, so two environments whose key matches would build byte-identical ``df_profiles_*``
        tables. The key combines the *identity* of the raw frames (cheap, and exact now that nets
        share them -- see :func:`deepcopy_net_sharing_profiles`) with the base values, which are
        small. The base values matter on the config path too: they fill the tables a config may
        omit (``gen.vm_pu`` is tiled from the net, ``sgen.q_mvar`` sized from it).

        The two paths get disjoint key namespaces, so a Simbench entry can never be served to a
        config-sourced environment or the other way round.

        :return: a hashable key, or None if there is no timeseries to share.
        :rtype: tuple | None
        """
        if self._config_profiles is not None:
            frame_ids = tuple(sorted(
                (element, variable, id(df))
                for element, variables in self._config_profiles.items()
                for variable, df in variables.items()
            ))
            return ("config", frame_ids, self._base_element_values())
        profiles = self.net.get("profiles")
        if not profiles:
            return None
        return (tuple(sorted((name, id(df)) for name, df in profiles.items())), self._base_element_values())

    def _reuse_shared_profile_tables(self) -> bool:
        """Adopt already-built profile tables from another env with the same inputs.

        The tables are immutable after construction (only ever read, row by row, by
        :meth:`load_profile_timestep_into_net` and the profile observations), so several
        environments can reference the same objects. This is what makes a vectorized run cheap:
        building N environments from one config costs one set of profile tables, not N.

        :return: True if tables were adopted and ``setup_profiles`` can be skipped.
        :rtype: bool
        """
        key = self._shared_profile_key()
        if key is None:
            return False
        cached = _SHARED_PROFILE_TABLES.get(key)
        if cached is None:
            return False
        # Strong reference: the shared entry lives exactly as long as an env still uses it.
        self._shared_profile_tables = cached
        for name in self._PROFILE_TABLE_NAMES:
            setattr(self, name, cached[name])
        if self.n_episodes <= 0:  # the side effect the ingestion would have applied
            self.n_episodes = self.n_total_timesteps
        return True

    def _publish_shared_profile_tables(self) -> None:
        """Offer the freshly built profile tables for reuse by environments with equal inputs."""
        key = self._shared_profile_key()
        if key is None:
            return
        tables = _ProfileTables(
            (name, getattr(self, name)) for name in self._PROFILE_TABLE_NAMES
        )
        # Pin the raw profile tables the key identifies by id(), so no id can be recycled
        # while this entry is reachable (see _ProfileTables).
        tables.source_profiles = self._profile_source()
        self._shared_profile_tables = tables  # keeps the weakly-held entry alive
        _SHARED_PROFILE_TABLES[key] = tables

    def setup_profiles(self) -> None: #noqa: PLR0915, PLR0912, C901
        """
        Configure profiles for load, gen, etc. timeseries.

        The code expects that the pandapowerNet contains a key-value pair "profiles"
        containing a dictionary:
        net.profiles = {
            'load': [Dataframe], #load.p_mw AND load.q_mvar
            'renewables': [Dataframe], # sgen.p_mw
            'powerplants': [Dataframe], # gen.p_mw
            'gen_vm': [Dataframe], # gen.vm_pu (voltage PER UNIT, scaling the bus-voltages)
            'sgen_q': [Dataframe], # sgen.q_mvar
            }
        (This profile format matches the conventions used in simbench.)

        The dataframes hold the timeseries, each with a unique [column] name.
        For loads, timeseries must have column names "NAME_pload" and "NAME_qload" for active
        and reactive power, respectively.

        Then, the workflow is:
        1. extract the dfs for load_p, load_q, gen_p, gen_vm, sgen_p, sgen_q
        2. Make the DFs match the length of the number of corresponding net-elements
        3. Multiply the profile-values with the current load.p_mw, load.p_mvar, ... values
        4. Store the results in (then unchangeable!) profile dataframes.

        There must be one profile column per net-element.
        """
        # get number of elements:
        n_load = len(self.net.load)
        n_gen = len(self.net.gen)
        n_sgen = len(self.net.sgen)
        # get length of profiles:
        len_profiles = 0
        if "load" in self.net.profiles:
            len_profiles = len(self.net.profiles["load"])
        if "powerplants" in self.net.profiles:
            len_pp = len(self.net.profiles["powerplants"])
            if (len_profiles > 0) and (len_pp > 0) and (len_pp != len_profiles):
                msg = "The profiles have different number of timesteps."
                raise RuntimeError(msg)
            len_profiles = max(len_profiles, len_pp)
        if "renewables" in self.net.profiles:
            len_renewables = len(self.net.profiles["renewables"])
            if (len_profiles > 0) and (len_renewables > 0) and (len_renewables != len_profiles):
                msg = "The profiles have different number of timesteps."
                raise RuntimeError(msg)
            len_profiles = max(len_renewables, len_profiles)
        if "gen_vm" in self.net.profiles:
            len_genvm = len(self.net.profiles["gen_vm"])
            if (len_profiles > 0) and (len_genvm > 0) and (len_genvm != len_profiles):
                msg = "The profiles have different number of timesteps."
                raise RuntimeError(msg)
            len_profiles = max(len_profiles, len_genvm)
        if "sgen_q" in self.net.profiles:
            len_sgenq = len(self.net.profiles["sgen_q"])
            if (len_profiles > 0) and (len_sgenq > 0) and (len_sgenq != len_profiles):
                msg = "The profiles have different number of timesteps."
                raise RuntimeError(msg)
            len_profiles = max(len_profiles, len_sgenq)
        if len_profiles == 0:
            return # no profiles to add




        #extract load_p, load_q dataframes:
        if n_load > 0:
            if "load" not in self.net.profiles:
                msg = "No profiles for pp net.load defined (DF 'load' storing NAME_pload, NAME_qload columns.)."
                raise RuntimeError(msg)
            n_needed_columns = 2*n_load +1 if "time" in self.net.profiles["load"].columns else 2*n_load
            if self.net.profiles["load"].shape[1] < n_needed_columns: # has one time column
                msg = "load does not have enough profiles for all load elements in pp net."
                raise RuntimeError(msg)
            df_load_p = self.net.profiles["load"].filter(regex="_pload$").iloc[:, :n_load]
            #extract load_q dataframe:
            df_load_q = self.net.profiles["load"].filter(regex="_qload$").iloc[:, :n_load]
        # extract sgen_p dataframe:
        if n_sgen > 0:
            if "renewables" not in self.net.profiles:
                msg = "No profiles for pp net.sgen defined ('renewables' for sgen_p, 'sgen_q' for sgen_q.)."
                raise RuntimeError(msg)
            df_sgen_p = self.net.profiles["renewables"].drop(columns="time", errors="ignore").iloc[:, :n_sgen]
            # Optional: extract sgen_q from net.profiles["sgen_q]
            if "sgen_q" in self.net.profiles:
                if self.net.profiles["sgen_q"].shape[1] < n_sgen:
                    msg = "sgen_q does not have enough profiles for all sgen elements in pp net."
                    raise RuntimeError(msg)
                df_sgen_q = self.net.profiles["sgen_q"].drop(columns="time", errors="ignore").iloc[:, :n_sgen]
            # else-case handled below

        if n_gen > 0:
            if "powerplants" not in self.net.profiles:
                msg = "No profiles for pp net.gen defined ('powerplants' for gen_p)."
                raise RuntimeError(msg)
            if self.net.profiles["powerplants"].shape[1] < n_gen:
                msg = "The DF renewables does not have enough profiles for all gen elements in pp net."
                raise RuntimeError(msg)
            # extract gen_p dataframe:
            df_gen_p = self.net.profiles["powerplants"].drop(columns="time", errors="ignore").iloc[:, :n_gen]

            if "gen_vm" in self.net.profiles:
                if self.net.profiles["gen_vm"].shape[1] < n_gen:
                    msg = "gen_vm does not have enough profiles for all gen elements in pp net."
                    raise RuntimeError(msg)
                df_gen_vm = self.net.profiles["gen_vm"].drop(columns="time", errors="ignore").iloc[:, :n_gen]
            # else-case handled below


        # build dataframes:
        if n_load > 0:
            self.df_profiles_load_p = df_load_p @ np.diag(self.net.load.p_mw.to_numpy())
            self.df_profiles_load_q = df_load_q @ np.diag(self.net.load.q_mvar.to_numpy())
        if n_gen > 0:
            self.df_profiles_gen_p = df_gen_p @ np.diag(self.net.gen.p_mw.to_numpy())
        if n_sgen > 0:
            self.df_profiles_sgen_p = df_sgen_p @ np.diag(self.net.sgen.p_mw.to_numpy())

        #optional dataframes:
        if "sgen_q" in self.net.profiles:
            self.df_profiles_sgen_q = df_sgen_q @ np.diag(self.net.sgen.q_mvar.to_numpy())
        else: # set to 0
            self.df_profiles_sgen_q = pd.DataFrame(
                np.zeros((len_profiles, n_sgen)),
            )
        if "gen_vm" in self.net.profiles:
            self.df_profiles_gen_vm = df_gen_vm @ np.diag(self.net.gen.vm_pu.to_numpy())
        else: # set constantly to gen.vm_pu value (as this seldomly changes in applications)
            vm_values = self.net.gen.vm_pu.to_numpy()
            self.df_profiles_gen_vm = pd.DataFrame(
                np.tile(vm_values, (len_profiles, 1)),  # shape (n_rows, n_cols)
                columns=[f"gen {i + 1}_vm" for i in range(len(vm_values))],  # valid column names
            )


        if self.n_episodes <= 0:
            self.n_episodes = len(self.df_profiles_load_p)

    def setup_profiles_from_config(self, profiles: dict[str, dict[str, pd.DataFrame]]) -> None:
        """Fill the derived ``df_profiles_*`` tables from a config-supplied timeseries dict.

        The dict is keyed by pandapower element and column::

            profiles = {
                "load": {"p_mw": [DataFrame], "q_mvar": [DataFrame]},
                "gen":  {"p_mw": [DataFrame], "vm_pu": [DataFrame]},
                "sgen": {"p_mw": [DataFrame], "q_mvar": [DataFrame]},
            }

        with one column per element of that table and one row per timestep.

        **These values are absolute and are assigned, never multiplied.** This is the one way to
        break this path silently: :meth:`setup_profiles` reaches the same numbers from *per-unit*
        Simbench shapes by scaling them with the net's base ``p_mw`` / ``q_mvar`` / ``vm_pu``, so
        re-using its ``@ np.diag(...)`` here would scale already-scaled values a second time and
        quietly hand the agent a different grid. The two routes end at the same tables precisely
        because only one of them multiplies.

        Only the six injection pairs in :data:`_CONFIG_PROFILE_TARGETS` are supported. Varying any
        other net column over time is a UCTE data-model feature that this branch does not carry, so
        such an entry raises instead of being dropped -- a silently ignored timeseries is exactly
        the failure this ingestion path exists to remove.

        :param profiles: absolute per-timestep values, keyed by element and column.
        :type profiles: dict[str, dict[str, pd.DataFrame]]
        :raises RuntimeError: if an element/variable pair is unsupported, the supplied frames do
            not all have the same number of rows, or a table the net needs is missing.
        """
        frames: dict[str, pd.DataFrame] = {}
        row_counts: set[int] = set()
        for element, variables in profiles.items():
            for variable, profile_df in variables.items():
                table_name = _CONFIG_PROFILE_TARGETS.get((element, variable))
                if table_name is None:
                    msg = (
                        f"Timeseries for '{element}.{variable}' is not supported on this branch: "
                        f"only {sorted(_CONFIG_PROFILE_TARGETS)} can be varied over time. Varying "
                        f"other net columns is a UCTE data-model feature that was not migrated."
                    )
                    raise RuntimeError(msg)
                frames[table_name] = _positionally_indexed(profile_df)
                row_counts.add(len(profile_df))

        if len(row_counts) > 1:
            msg = f"Network profiles do not have the same length: found {sorted(row_counts)} rows."
            raise RuntimeError(msg)
        n_rows = row_counts.pop() if row_counts else 0

        missing = self._missing_required_profile_tables(frames)
        if missing:
            msg = (
                f"env_config['profiles'] has no timeseries for {missing}, but the net has those "
                f"elements. Every injection the net carries must be driven by a profile."
            )
            raise RuntimeError(msg)
        frames.update(self._default_profile_tables(frames, n_rows))

        for table_name, profile_df in frames.items():
            setattr(self, table_name, profile_df)

        if self.n_episodes <= 0:
            self.n_episodes = n_rows

    def _missing_required_profile_tables(self, frames: dict[str, pd.DataFrame]) -> list[str]:
        """Name the tables a config must supply for the elements this net actually has.

        ``sgen.q_mvar`` and ``gen.vm_pu`` are left out: :meth:`setup_profiles` defaults them too,
        and :meth:`_default_profile_tables` reproduces those defaults here.

        :param frames: the tables built from the config so far, keyed by attribute name.
        :type frames: dict[str, pd.DataFrame]
        :return: attribute names of the required-but-absent tables, empty when complete.
        :rtype: list[str]
        """
        required = (
            ("load", "df_profiles_load_p"), ("load", "df_profiles_load_q"),
            ("sgen", "df_profiles_sgen_p"), ("gen", "df_profiles_gen_p"),
        )
        return [name for element, name in required if len(self.net[element]) and name not in frames]

    def _default_profile_tables(self, frames: dict[str, pd.DataFrame], n_rows: int) -> dict[str, pd.DataFrame]:
        """Build the optional tables a config may omit, exactly as :meth:`setup_profiles` does.

        ``sgen.q_mvar`` defaults to zero and ``gen.vm_pu`` to the net's setpoint held constant, so
        that both ingestion paths always end with the same six tables populated -- including the
        zero-column shapes :meth:`setup_profiles` produces for an element the net does not have.

        :param frames: the tables built from the config so far, keyed by attribute name.
        :type frames: dict[str, pd.DataFrame]
        :param n_rows: number of timesteps the defaults must span.
        :type n_rows: int
        :return: the defaulted tables, keyed by attribute name.
        :rtype: dict[str, pd.DataFrame]
        """
        defaults: dict[str, pd.DataFrame] = {}
        if "df_profiles_sgen_q" not in frames:
            defaults["df_profiles_sgen_q"] = pd.DataFrame(np.zeros((n_rows, len(self.net.sgen))))
        if "df_profiles_gen_vm" not in frames:
            vm_values = self.net.gen.vm_pu.to_numpy()
            defaults["df_profiles_gen_vm"] = pd.DataFrame(
                np.tile(vm_values, (n_rows, 1)),
                columns=[f"gen {i + 1}_vm" for i in range(len(vm_values))],
            )
        return defaults

    def load_profile_timestep_into_net(self, index: int) -> None:
        """
        Load profiles for a given timestep into the net.load, etc. Dataframes.

        Replace the load p and q with the values stored in the profiles_load dataframe

        :param index: The index of the desired timestep, in the timeseries Dataframe.
        :type index: int
        """
        load_count = len(self.net.load)
        sgen_count = len(self.net.sgen)
        gen_count = len(self.net.gen)

        if load_count:
            profile_row_p = self.df_profiles_load_p.loc[index]
            profile_row_q = self.df_profiles_load_q.loc[index]
            self.net.load["p_mw"].to_numpy()[:] = profile_row_p.to_numpy()
            self.net.load["q_mvar"].to_numpy()[:] = profile_row_q.to_numpy()

        if sgen_count:
            profile_row_p = self.df_profiles_sgen_p.loc[index]
            profile_row_q = self.df_profiles_sgen_q.loc[index]
            self.net.sgen["p_mw"].to_numpy()[:] = profile_row_p.to_numpy()
            self.net.sgen["q_mvar"].to_numpy()[:] = profile_row_q.to_numpy()

        if gen_count:
            profile_row_p = self.df_profiles_gen_p.loc[index]
            profile_row_vm = self.df_profiles_gen_vm.loc[index]
            self.net.gen["p_mw"].to_numpy()[:] = profile_row_p.to_numpy()
            self.net.gen["vm_pu"].to_numpy()[:] = profile_row_vm.to_numpy()

        self.net.converged = None  # reset converged flag

    def __getstate__(self) -> dict:
        """Drop the lightsim2grid backend when the env is copied or pickled.

        The backend wraps a C++ ``GridModel``, which does not survive either: ``copy.deepcopy``
        raises ``RuntimeError: Impossible to set the converter ls_to_orig``. That matters well
        beyond tests -- the simulation API, the greedy agents and the AlphaZero serializer all
        deep-copy environments, and spawned workers pickle them.

        Dropping it is safe because it is *derived* state: ``_solve_with_lightsim`` rebuilds it
        from the net on the next solve whenever it is None. The copy is therefore identical in
        behaviour, and pays one mirror-net construction the first time it solves.

        :return: The instance dictionary with the backend cleared.
        :rtype: dict
        """
        state = self.__dict__.copy()
        state["_lightsim_backend"] = None
        return state

    def run_pf(
        self,
        pf_type: str = "ac",
        use_ls2g: bool | str = "auto",
        nminus1: bool | None = None,
    ) -> bool:
        """
        Run the powerflow. Return True if successful, False if not.

        Returns immediately when the results already on the net answer this exact request.
        ``net.converged`` is the "results are current" flag: every path that changes what a
        power flow would produce clears it to ``None`` -- :meth:`load_action` (for a real
        action), :meth:`load_profile_timestep_into_net` and :meth:`_restore_baseline_net`.
        The one action that does *not* is DoNothing, which deliberately keeps the flag, so a
        DoNothing step used to re-solve a grid whose topology, injections and profile index
        were all unchanged. The request tuple is compared as well because ``pf_type`` differs
        between callers (``reset`` passes ``self.pf_type``, ``step`` takes the default), and an
        N-1 sweep writes columns a plain solve does not.

        :param pf_type: the powerflow type, either 'ac' oder 'dc'
        :type pf_type: str
        :param use_ls2g: Whether lightsim2grid should be used as backend or not.
        :type use_ls2g: bool | str
        :param nminus1: Whether to run the n-1 powerflow or not.
        :type nminus1: bool | None
        :return: True if the powerflow converged, False if not.
        :rtype: bool
        """
        if nminus1 is None:
            nminus1 = self.nminus1
        request = (pf_type, use_ls2g, nminus1)
        if self.net.converged is True and self.net.get(_SOLVED_REQUEST_KEY) == request:
            return True
        try:
            if nminus1:
                if not self._run_nminus1(pf_type):
                    self.net.converged = False
                    return False
            elif self.backend_name == "lightsim":
                if not self._solve_with_lightsim():
                    self.net.converged = False
                    return False
            else:
                run_powerflow(self.net, pf_type, use_ls2g)
        except pp.LoadflowNotConverged:
            self.net.converged = False
            return False
        if self._grid_is_disconnected():
            self.net.converged = False
            return False
        self.net.converged = True
        self.net[_SOLVED_REQUEST_KEY] = request
        return True

    def _run_nminus1(self, pf_type: str) -> bool:
        """Run the N-1 sweep on whichever backend the config selected.

        Three backends write the same ``res_*`` columns: the lightsim2grid sweep, the
        process-parallel pandapower one, and the serial pandapower one. ``backend="lightsim"``
        wins over ``"n-1 parallel"`` -- the lightsim sweep is already a single-process path, and
        splitting it across workers would only re-introduce the solver it replaces.

        :param pf_type: the power-flow type, either 'ac' or 'dc'.
        :type pf_type: str
        :return: True if the sweep produced a result; False if the power flow diverged.
        :rtype: bool
        :raises RuntimeError: if the sweep ran but left no ``max_loading_percent`` behind, which
            every N-1 observation reads.
        """
        if self.backend_name == "lightsim":
            if not self._solve_with_lightsim(nminus1=True):
                return False
        elif self.nminus1_parallel:
            static_blob = getattr(self, "static_net_blob", None)
            if static_blob is None and hasattr(self, "dump_static_net_bytes"):
                static_blob = self.dump_static_net_bytes()
            run_nminus1_powerflow_parallel(
                self.net, pf_type, self.nminus1_workers, static_blob=static_blob,
                topk_percent=self.nminus1_topk,
            )
        else:
            run_nminus1_powerflow(self.net, pf_type, topk_percent=self.nminus1_topk)

        if "max_loading_percent" not in self.net.res_line:
            msg = "N-1 analysis ran but didn't store max_loading_percent"
            raise RuntimeError(msg)
        return True

    def _solve_with_lightsim(self, *, nminus1: bool = False) -> bool:
        """Solve the current net through the lightsim2grid backend, building it on first use.

        The backend is built lazily rather than in ``__init__`` so that an environment which
        never runs a power flow (and every environment on the default backend) pays nothing for
        the mirror-net construction.

        :param nminus1: run the N-1 sweep (writing ``res_line.max_loading_percent`` and the rest)
            instead of a single power flow. Honours the env's ``n-1-topk`` setting, exactly as
            the pandapower N-1 backends do.
        :type nminus1: bool
        :return: True if the power flow converged.
        :rtype: bool
        """
        if self._lightsim_backend is None:
            from pandapower_env.toolbox.ls2g_backend import LightsimBackend
            self._lightsim_backend = LightsimBackend(self.net)
        if nminus1:
            return self._lightsim_backend.solve_nminus1(self.net, topk_percent=self.nminus1_topk)
        return self._lightsim_backend.solve(self.net)

    def _grid_is_disconnected(self) -> bool:
        """
        Check whether the last power flow left part of the grid unsupplied.

        pandapower does not raise on a split grid: it silently drops the islanded
        buses from the solved system and still reports convergence, leaving their
        ``res_bus`` entries as NaN. Treating that as a valid result would hand the
        agent a reward for a grid where loads are no longer served, so this check
        turns it into the same failure as a non-converged power flow (see
        :meth:`step`, which then terminates the episode).

        Only in-service buses are inspected, because the double-busbar substations
        add many permanently out-of-service auxiliary buses that are always NaN.

        pandapower writes ``res_bus`` with ``net.bus``'s own index, so the in-service flags line
        up with the voltages row for row and the check is a numpy mask (~10 us). Confirming that
        alignment is far cheaper than the label ``reindex`` it replaces (~151 us), which pandas
        pays in full even when both indexes are identical. This runs once per power flow, so a
        net indexed some other way still gets the same answer from
        :meth:`_grid_is_disconnected_by_label`.

        :return: True if any in-service bus has no power flow result.
        :rtype: bool
        """
        res_bus = self.net.res_bus
        if not res_bus.index.equals(self.net.bus.index):
            return self._grid_is_disconnected_by_label()
        in_service = self.net.bus["in_service"].to_numpy()
        return bool(np.isnan(res_bus["vm_pu"].to_numpy()[in_service]).any())

    def _grid_is_disconnected_by_label(self) -> bool:
        """Answer :meth:`_grid_is_disconnected` through a label lookup (fallback path).

        Used when ``res_bus`` is not row-aligned with ``net.bus``, and kept as the correctness
        reference the fast path is tested against (``tests/environments/test_disconnect_parity.py``).

        :return: True if any in-service bus has no power flow result.
        :rtype: bool
        """
        in_service_buses = self.net.bus.index[self.net.bus["in_service"]]
        voltages = self.net.res_bus["vm_pu"].reindex(in_service_buses)
        return bool(voltages.isna().any())



    def act_without_advancing_timeseries(self, action: int | np.integer) -> tuple[bool, SupportsFloat]:
        """Apply an action and score it, without moving the timeseries on.

        This is the first half of :meth:`step` -- apply the action, solve the power flow, take the
        reward -- stopping short of advancing ``self.index`` and of building an observation. Paired
        with a state restore it lets a search evaluate many actions from one grid state without a
        reset (and its network deep copy) per action.

        :param action: the action to apply, as :meth:`load_action` understands it.
        :type action: int | np.integer
        :return: whether the power flow converged, and the reward (``worst_reward`` if it did not).
        :rtype: tuple[bool, SupportsFloat]
        """
        self.load_action(action)
        converged = self.run_pf()
        return converged, self.calculate_reward() if converged else self.worst_reward

    def step(
        self,
        action: int | np.integer,
    ) -> tuple[dict | ObsType | list[float], SupportsFloat, bool, bool, dict]:
        """
        Perform the "step" action in the environment.

        In this implementation, the action is performed and the reward is calculated based
        on the observation that the agent sees. In other words, the action is performed
        "instantaneously" without changing anything else. After the reward is calculated,
        the timeseries moves forward one step and the next observation is calculated, *with*
        the selected action still in place.

        :param action: to be defined by the user in the derived class.
        :type action: TBD
        :return: Gymnasium-style output (observation, reward, terminated, truncated, info)
        :rtype: tuple[observation, reward, episode_over, truncated, info]
        """
        self.current_step += 1
        self.load_action(action)

        # Run the powerflow
        self.run_pf()
        info = self.initialize_info()
        info.update(self.observation_to_info(metric_key="before"))
        if self.net.converged is False: # should actually verify action and do DoNothing instead
            #if action is 0, then skip the next line
            #else set action to 0 and call this function again

            terminated = True
            truncated = False
            logger.warning("Net did not converge.")
            info.update({"message": "network did not converge (next step).",
                 "crashed": True,
                 })
            return (
                self.create_observation(),
                self.worst_reward,
                terminated,
                truncated,
                info,
            )
        terminated = False
        truncated = self.current_step >= self.episode_length
        # For now, because one timestep = one episode, the episode is always
        # immediately over
        reward = self.calculate_reward()
        # Only now, after the reward has been taken: a reward function reads this counter to tell
        # the first step of an episode from a later one.
        self.episode_step_counter += 1
        info.update({
            "message": "step completed successfully.",
            "crashed": False,
        })
        # The timeseries is finite. Once the last row has been scored there is no next row to
        # advance to, so the episode ends here -- which is what ``truncated`` means. Without this
        # the step walked off the end of the profile tables and raised a bare
        # ``KeyError: <index>`` from ``load_profile_timestep_into_net``. Only reachable when a
        # caller resets to an explicit late ``options["index"]``; the default random scenario
        # start always leaves a full episode of rows.
        if not truncated and self.index + 1 >= self.n_total_timesteps:
            logger.warning(
                "Timeseries exhausted at index %s of %s; truncating the episode.",
                self.index, self.n_total_timesteps,
            )
            truncated = True
        if truncated:
            observation = self.create_observation()
            return observation, reward, terminated, truncated, info
        self.index += 1
        self.load_profile_timestep_into_net(self.index)
        info.update(self.observation_to_info(metric_key="after"))
        observation = self.create_observation()
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment."""
        txt = "Current timestep index: {}"
        logger.info(txt.format(self.index))

    def _capture_topology(self) -> dict[str, np.ndarray]:
        """
        Capture switch/line/trafo topology arrays (native dtype, NaN-preserving).

        Inverse of :meth:`restore_topology`. Used for the reset baseline and for
        the cheap state snapshots (``save_state``); native dtypes preserve NaN
        tap positions so a snapshot round-trips byte-for-byte.
        """
        topo: dict[str, np.ndarray] = {}
        if len(self.net.switch):
            topo["switch_closed"] = self.net.switch["closed"].to_numpy(copy=True)
        if len(self.net.line):
            topo["line_in_service"] = self.net.line["in_service"].to_numpy(copy=True)
        if len(self.net.trafo):
            topo["trafo_tap_pos"] = self.net.trafo["tap_pos"].to_numpy(copy=True)
        return topo

    def restore_topology(self, topo: dict[str, np.ndarray]) -> None:
        """Set switch/line/trafo topology arrays in place (inverse of ``_capture_topology``)."""
        if "switch_closed" in topo:
            self.net.switch["closed"] = topo["switch_closed"].astype(bool)
        if "line_in_service" in topo:
            self.net.line["in_service"] = topo["line_in_service"].astype(bool)
        if "trafo_tap_pos" in topo:
            self.net.trafo["tap_pos"] = topo["trafo_tap_pos"]  # keep NaN for non-tap trafos

    def _restore_baseline_net(self) -> None:
        """
        Restore the network to its baseline state in place (no deepcopy).

        Only the topology arrays captured in __init__ are restored; power
        setpoints are re-derived by load_profile_timestep_into_net and res_*
        are cleared here (to match the empty-results net_copy_from) and
        recomputed by the next power flow.
        """
        self.restore_topology(self._baseline_topology)
        pp.reset_results(self.net)  # match empty-res net_copy_from
        self.net.converged = None

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict, dict[str, Any]]:
        """
        Reset the environment.

        Picks a random scenario start unless ``options["index"]`` names one explicitly.

        Seeding goes through ``gym.Env``'s per-instance ``self.np_random`` rather than the
        process-global ``random`` module. That matters for reproducibility: the global
        module is shared with every other component in the process, so seeding it here
        made the chosen scenario depend on whatever else had drawn a number since -- and
        silently reseeded the caller's RNG as a side effect. Vectorized envs, which reset
        several instances in one process, could not be reproduced at all.

        :param seed: seed for this environment's RNG. Note this seeds the scenario
            *choice*; to reset to a specific timeseries index pass ``options["index"]``.
        :type seed: int | None
        :param options: optional dict. ``options["index"]`` sets the scenario
            start; ``options["deep"]=True`` forces the legacy deepcopy restore
            (kept as a parity oracle / escape hatch).
        :type options: dict | None
        :return: observation and [empty] info dict
        :rtype: tuple[observation type, dict]
        """
        # Seeds self.np_random when seed is not None; a no-op otherwise, so an unseeded
        # reset keeps drawing from the RNG it already had.
        super().reset(seed=seed)

        if options is not None and options.get("deep"):
            self.net = copy.deepcopy(self.net_copy_from)  # legacy escape hatch / parity oracle
        else:
            self._restore_baseline_net()
        self.current_step = 0
        self.episode_step_counter = 0
        if options is not None and "index" in options:
            self.index = options["index"]
        else:
            # set the scenario-start
            random_max_number = max((self.n_total_timesteps // self.episode_length) - 1, 0)
            # integers() is half-open, randint() was inclusive -- keep the last scenario reachable.
            scenario_index = int(self.np_random.integers(0, random_max_number + 1))
            self.index = scenario_index*self.episode_length
        self.episode_index = self.index // self.episode_length

        logger.debug("Options: %s", options)

        self.load_profile_timestep_into_net(self.index)


        return {}, {}



    def close(self) -> None:
        """Close function -- kept here for compatibility with gym."""
        return

    def load_action(self, *_: Any) -> None: # noqa: ANN401
        """
        Apply the action to the pandapower network.

        This base implementation resets the power flow status. Derived classes
        should override this method with their specific signature and call
        super().load_action() to ensure the status is reset.

        :param args: Positional arguments (defined by derived classes)
        """
        self.net.converged = None  # reset converged flag for new topology

    def initialize_info(self) -> dict:
        """Initialize info with the most important properties."""
        info: dict = {
            "current_step": self.current_step,
            "profile_index": self.index,
        }
        return info


    def observation_to_info(self, metric_key: str) -> dict:
        """Report intermediate obs values to info."""
        _ = metric_key
        return {}

    def create_observation(self) -> list[float] | dict:
        """
        Create the observation from the result of the powerflow calculation in net.res_line and net.res_bus.

        This function must be implemented in the derived class.
        The powerflow must have been run before, saving "self.net.converged" flag to True.

        :return: observation
        :rtype: TBD
        """
        msg = "The create_observation method must be implemented in a derived class."
        raise NotImplementedError(msg)

    @abstractmethod
    def calculate_reward(self) -> SupportsFloat:
        """
        Calculate the step reward.

        This function must be implemented in the derived class.

        :return: reward
        :rtype: TBD
        """
        msg = "The calculate_reward method must be implemented in a derived class."
        raise NotImplementedError(msg)
