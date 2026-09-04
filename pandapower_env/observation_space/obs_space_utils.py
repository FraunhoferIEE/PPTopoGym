from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import pandapower as pp


class ObsType(Enum):
    TABLE = auto()      # Direct lookup in net["table"]["column"]
    PROFILE = auto()    # Values from external time-series DataFrames
    AGGREGATE = auto()  # Pre-defined scalar calculations (e.g., total loss)
    CUSTOM = auto()  # Complex logic (e.g., adjacency matrices)
    TOPOLOGY = auto() # Structural info (e.g., bus ids of loads)

@dataclass(frozen=True, slots=True)
class ObservationConfig:
    """Metadata defining how to extract and normalize a specific observation."""

    table: str                  # pandapower table name or 'profile'/'aggregate'
    column: str                 # column name or logic key
    dtype: type                 # Target numpy dtype (e.g. np.float32)
    nan_value: float | int      # Fill value for missing data
    obs_type: ObsType = ObsType.TABLE
    low: float | None = None    # Min value for clipping
    high: float | None = None   # Max value for clipping
    spaces_shape: str |  tuple[str, int] | None = None #scalar, net-table
    # Optional override for specific logic
    handler: Callable[[pp.pandapowerNet], np.ndarray] | None = None


def build_observation_registry() -> dict[str, ObservationConfig]:
    """Build the complete observation configuration registry."""
    return {
        # ═══════════════════════════════════════════════════════════════════
        # BUS OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════════
        "bus_voltage_magnitude": ObservationConfig(
            table="res_bus",
            column="vm_pu",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=0.0,
            high=1.5,
            spaces_shape="bus",
        ),
        "bus_voltage_angle": ObservationConfig(
            table="res_bus",
            column="va_degree",
            dtype=np.float32,
            nan_value=360.0,
            obs_type=ObsType.TABLE,
            low=-360.0,
            high=360.0,
            spaces_shape="bus",
        ),
        # ═══════════════════════════════════════════════════════════════════
        # LINE OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════════
        "line_loadings": ObservationConfig(
            table="res_line",
            column="loading_percent",
            dtype=np.float32,
            nan_value=800.0,
            obs_type=ObsType.TABLE,
            low=0.0,
            high=800.0,
            spaces_shape="line",
        ),
        "line_power_flow_p_mw": ObservationConfig(
            table="res_line",
            column="p_from_mw",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=-1000.0,
            high=1000.0,
            spaces_shape="line",
        ),
        "line_power_flow_q_mvar": ObservationConfig(
            table="res_line",
            column="q_from_mvar",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=-1000.0,
            high=1000.0,
            spaces_shape="line",
        ),
        "line_status": ObservationConfig(
            table="line",
            column="in_service",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TABLE,
            low=0,
            high=1,
            spaces_shape="line",
        ),
        "line_thermal_limit": ObservationConfig(
            table="line",
            column="max_i_ka",
            dtype=np.float32,
            nan_value=1e-6,
            obs_type=ObsType.TABLE,
            low=0.0,
            high=100000.0,
            spaces_shape="line",
        ),
        # ═══════════════════════════════════════════════════════════════════
        # TRANSFORMER OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════════
        "transformer_loading_percent": ObservationConfig(
            table="res_trafo",
            column="loading_percent",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=0.0,
            high=800.0,
            spaces_shape="trafo",
        ),
        "transformer_power_flow_p_mw": ObservationConfig(
            table="res_trafo",
            column="p_hv_mw",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=-1000.0,
            high=1000.0,
            spaces_shape="trafo",
        ),
        "transformer_power_flow_q_mvar": ObservationConfig(
            table="res_trafo",
            column="q_hv_mvar",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=-1000.0,
            high=1000.0,
            spaces_shape="trafo",
        ),
        "transformer_tap_position": ObservationConfig(
            table="trafo",
            column="tap_pos",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TABLE,
            low=int(np.iinfo(np.int32).min),
            high=int(np.iinfo(np.int32).max),
            spaces_shape="trafo",
        ),
        "transformer_status": ObservationConfig(
            table="trafo",
            column="in_service",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TABLE,
            low=0,
            high=1,
            spaces_shape="trafo",
        ),
        # ═══════════════════════════════════════════════════════════════════
        # GENERATOR OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════════
        "gen_status": ObservationConfig(
            table="gen",
            column="in_service",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TABLE,
            low=0,
            high=1,
            spaces_shape="gen",
        ),
        "gen_power_p_mw_profile": ObservationConfig(
            table="profile_gen",
            column="p_mw",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.PROFILE,
            low=0.0,
            high=1000.0,
            spaces_shape="gen",
        ),
        "gen_power_p_mw_runpf": ObservationConfig(
            table="res_gen",
            column="p_mw",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=0.0,
            high=1000.0,
            spaces_shape="gen",
        ),
        "gen_vm_pu_profile": ObservationConfig(
            table="profile_gen",
            column="vm_pu",
            dtype=np.float32,
            nan_value=1.0,
            obs_type=ObsType.PROFILE,
            low=0.0,
            high=1.1,
            spaces_shape="gen",
        ),
        "gen_vm_pu_runpf": ObservationConfig(
            table="res_gen",
            column="vm_pu",
            dtype=np.float32,
            nan_value=1.0,
            obs_type=ObsType.TABLE,
            low=0.0,
            high=1.1,
            spaces_shape="gen",
        ),
        # ═══════════════════════════════════════════════════════════════════
        # STATIC GENERATOR (SGEN) OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════════
        "sgen_status": ObservationConfig(
            table="sgen",
            column="in_service",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TABLE,
            low=0,
            high=1,
            spaces_shape="sgen",
        ),
        "sgen_power_p_mw_profile": ObservationConfig(
            table="profile_sgen",
            column="p_mw",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.PROFILE,
            low=0.0,
            high=1000.0,
            spaces_shape="sgen",
        ),
        "sgen_power_p_mw_runpf": ObservationConfig(
            table="res_sgen",
            column="p_mw",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=0.0,
            high=1000.0,
            spaces_shape="sgen",
        ),
        "sgen_power_q_mvar_profile": ObservationConfig(
            table="profile_sgen",
            column="q_mvar",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.PROFILE,
            low=-1000.0,
            high=1000.0,
            spaces_shape="sgen",
        ),
        "sgen_power_q_mvar_runpf": ObservationConfig(
            table="res_sgen",
            column="q_mvar",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=-1000.0,
            high=1000.0,
            spaces_shape="sgen",
        ),
        # ═══════════════════════════════════════════════════════════════════
        # LOAD OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════════
        "load_status": ObservationConfig(
            table="load",
            column="in_service",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TABLE,
            low=0,
            high=1,
            spaces_shape="load",
        ),
        "load_power_p_mw_profile": ObservationConfig(
            table="profile_load",
            column="p_mw",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.PROFILE,
            low=-1000.0,
            high=1000.0,
            spaces_shape="load",
        ),
        "load_power_q_mvar_profile": ObservationConfig(
            table="profile_load",
            column="q_mvar",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.PROFILE,
            low=-1000.0,
            high=1000.0,
            spaces_shape="load",
        ),
        "load_power_p_mw_runpf": ObservationConfig(
            table="res_load",
            column="p_mw",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=-1000.0,
            high=1000.0,
            spaces_shape="load",
        ),
        "load_power_q_mvar_runpf": ObservationConfig(
            table="res_load",
            column="q_mvar",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.TABLE,
            low=-1000.0,
            high=1000.0,
            spaces_shape="load",
        ),
        # ═══════════════════════════════════════════════════════════════════
        # SWITCH OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════════
        "switch_positions": ObservationConfig(
            table="switch",
            column="closed",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TABLE,
            low=0,
            high=1,
            spaces_shape="switch",
        ),
        # ═══════════════════════════════════════════════════════════════════
        # AGGREGATE / SCALAR OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════════
        "total_power_demand_profile": ObservationConfig(
            table="",
            column="total_load_p_profile",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.AGGREGATE,
            low=-1e6,
            high=1e6,
            spaces_shape="scalar",
        ),
        "total_power_generation_profile": ObservationConfig(
            table="",
            column="total_gen_p_profile",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.AGGREGATE,
            low=-1e6,
            high=1e6,
            spaces_shape="scalar",
        ),
        "total_power_demand_runpf": ObservationConfig(
            table="",
            column="total_load_p_runpf",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.AGGREGATE,
            low=-1e6,
            high=1e6,
            spaces_shape="scalar",
        ),
        "total_power_generation_runpf": ObservationConfig(
            table="",
            column="total_gen_p_runpf",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.AGGREGATE,
            low=-1e6,
            high=1e6,
            spaces_shape="scalar",
        ),
        "system_losses": ObservationConfig(
            table="",
            column="system_losses",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.AGGREGATE,
            low=0.0,
            high=1e6,
            spaces_shape="scalar",
        ),
        # ═══════════════════════════════════════════════════════════════════
        # TOPOLOGY OBSERVATIONS
        # ═══════════════════════════════════════════════════════════════════
        "adjacency_matrix": ObservationConfig(
            table="",
            column="adjacency_matrix",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.CUSTOM,
            low=0,
            high=float(np.iinfo(np.int32).max),
            spaces_shape=("adjacency", 2),
        ),
        "node_slot_map": ObservationConfig(
            table="",
            column="node_slot_map",
            dtype=np.int32,
            nan_value=-1,
            obs_type=ObsType.CUSTOM,
            low=-1,
            high=float(np.iinfo(np.int32).max),
            spaces_shape="node_slots",
        ),
        "bus_lookup_table": ObservationConfig(
            table="",
            column="bus_lookup_table",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TOPOLOGY,
            low=0,
            high=float(np.iinfo(np.int32).max),
            spaces_shape="bus",
        ),
        "load_bus": ObservationConfig(
            table="load",
            column="bus",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TOPOLOGY,
            low=0,
            high=float(np.iinfo(np.int32).max),
            spaces_shape="load",
        ),
        "gen_bus": ObservationConfig(
            table="gen",
            column="bus",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TOPOLOGY,
            low=0,
            high=float(np.iinfo(np.int32).max),
            spaces_shape="gen",
        ),
        "sgen_bus": ObservationConfig(
            table="sgen",
            column="bus",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TOPOLOGY,
            low=0,
            high=float(np.iinfo(np.int32).max),
            spaces_shape="sgen",
        ),
        "line_from_bus": ObservationConfig(
            table="line",
            column="from_bus",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TOPOLOGY,
            low=0,
            high=float(np.iinfo(np.int32).max),
            spaces_shape="line",
        ),
        "line_to_bus": ObservationConfig(
            table="line",
            column="to_bus",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TOPOLOGY,
            low=0,
            high=float(np.iinfo(np.int32).max),
            spaces_shape="line",
        ),
        "trafo_hv_bus": ObservationConfig(
            table="trafo",
            column="hv_bus",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TOPOLOGY,
            low=0,
            high=float(np.iinfo(np.int32).max),
            spaces_shape="trafo",
        ),
        "trafo_lv_bus": ObservationConfig(
            table="trafo",
            column="lv_bus",
            dtype=np.int32,
            nan_value=0,
            obs_type=ObsType.TOPOLOGY,
            low=0,
            high=float(np.iinfo(np.int32).max),
            spaces_shape="trafo",
        ),
    }


def build_info_observation_registry() -> dict[str, ObservationConfig]:
    """Build the aggregates that are reported in ``info`` but are *not* part of the observation.

    These two keys are named by the default ``env_config["info_observations"]`` and are read by
    the evaluation metrics (``overload_energy_difference_abs_mvah`` and
    ``loading_improvement_optimization``), but they are deliberately kept out of
    :func:`build_observation_registry`: an environment defaults ``observation_keys`` to *every*
    registry key, so adding them there would grow the observation space and change the input
    dimension of any network already trained against it.

    ``PPTopoGym`` merges this registry into the one it uses to *compute* an explicitly requested
    observation key, while ``define_observation_space`` keeps using the main registry alone. The
    net effect is that ``info`` carries these aggregates again without the observation space
    moving. Both are computed by ``PPTopoGym._get_aggregate_value``, which has always
    implemented them -- only the config entries pointing at it were missing.

    :return: mapping of info-only observation name to its configuration.
    :rtype: dict[str, ObservationConfig]
    """
    return {
        "total_energy_overload": ObservationConfig(
            table="",
            column="total_energy_overload",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.AGGREGATE,
            low=0.0,
            high=1e6,
            spaces_shape="scalar",
        ),
        "max_loading_percent": ObservationConfig(
            table="",
            column="max_loading_percent",
            dtype=np.float32,
            nan_value=0.0,
            obs_type=ObsType.AGGREGATE,
            low=0.0,
            high=1e6,
            spaces_shape="scalar",
        ),
    }






