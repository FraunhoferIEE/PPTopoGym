from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
import simbench
import torch
from gymnasium import spaces
from pandapower.auxiliary import pandapowerNet
from pandapower.networks import case14, case89pegase

from pandapower_env.action_space.action_space import (
    add_actions_substation_line_switching,
)
from pandapower_env.data.example_configs import config_case30
from pandapower_env.environments.simulation_env import PPTopoGym
from pandapower_env.rlib_agents.gnn_agents import GINETorchRLModule
from pandapower_env.substation.create_double_busbar_substation import (
    can_convert_to_n_busbar_substation,
    create_3bb_with_pst_substation,
    create_all_double_busbar_substations,
    create_n_busbar_substation,
)
from pandapower_env.toolbox.utils_profiles import get_profile_names

# Note: The fixtures are already defined in the test file itself.

# This conftest.py can be used for additional shared configuration

# or fixtures that might be needed across multiple test modules.


@pytest.fixture(scope="session")
def test_project_root() -> Path:
    """
    Get the root directory of the test project.

    :return: Path to the project root
    :rtype: Path
    """
    return Path(__file__).parent.parent.resolve()


@pytest.fixture()
def temp_python_file(tmp_path: Path) -> Path:
    """
    Create a temporary Python file for testing.

    :param tmp_path: pytest tmp_path fixture
    :type tmp_path: Path
    :yield: Path to temporary Python file
    :rtype: Generator[Path, None, None]
    """
    file_path = tmp_path / "temp_module.py"
    file_path.write_text(
        "def example_function():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = a + b\n"
        "    return c\n",
        encoding="utf-8",
    )
    return file_path
    # Cleanup is automatic with tmp_path


@pytest.fixture()
def temp_git_repo(tmp_path: Path) -> Path:
    """
    Create a temporary directory structure mimicking a git repository.

    :param tmp_path: pytest tmp_path fixture
    :type tmp_path: Path
    :yield: Path to the fake git repo root
    :rtype: Generator[Path, None, None]
    """
    # Create .git directory
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    # Create source directory structure
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    return tmp_path

@pytest.fixture()
def overloaded_net() -> pandapowerNet:
    # Create a simple network: Slack -> Transformer -> Line -> Load
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=110.)
    b2 = pp.create_bus(net, vn_kv=20.)
    b3 = pp.create_bus(net, vn_kv=20.)

    pp.create_ext_grid(net, bus=b1)

    # Create a transformer with 20 MVA rating
    pp.create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="25 MVA 110/20 kV")

    # Create a line with a specific thermal limit (e.g., ~14 MVA)
    pp.create_line(net, from_bus=b2, to_bus=b3, length_km=1.0, std_type="NA2XS2Y 1x185 RM/25 12/20 kV")

    # Create a massive load (50 MW) to guarantee overload on both elements
    pp.create_load(net, bus=b3, p_mw=50.0, q_mvar=10.0)

    return net

@pytest.fixture()
def test_grid() -> pandapowerNet:
    """
    Pytest fixture for generating a test grid with 14 buses.

    :return: a simple 14-bus network
    :rtype: pandapowerNet
    """
    return case14()


@pytest.fixture()
def test_grid_double_bb_substations() -> pandapowerNet:
    """
    Test grid with 14 buses, with a double-busbar Dataframe already created.

    :return: a 14-bus network with a net.multi_bb_substation created.
    :rtype: pandapowerNet
    """
    net = case14()

    create_all_double_busbar_substations(net)

    return net


@pytest.fixture()
def test_grid_with_pst() -> pandapowerNet:
    """
    Test grid with 14 buses and a PST.

    :return: a 14-bus network with a PST.
    :rtype: pandapowerNet
    """
    net = case14()

    create_3bb_with_pst_substation(net, 3)

    return net


@pytest.fixture()
def test_grid_multi_bb_substations() -> pandapowerNet:
    """
    Test grid with 14 buses, with a double-busbar Dataframe already created.

    :return: a 14-bus network with a net.multi_bb_substation created.
    :rtype: pandapowerNet
    """
    net = case14()

    for ibus in net.bus.index:
        if not can_convert_to_n_busbar_substation(net, ibus):
            continue

        create_n_busbar_substation(net, ibus)

    return net


@pytest.fixture()
def test_grid_dbb_plus_simbench() -> pandapowerNet:
    """
    Test grid with 14 buses, with a double-busbar Dataframe already created, plus simbench-style profiles.

    :return: a 14-bus network with a n
    :rtype: pandapowerNet
    """
    net = case14()

    # Make sure loads, generators each have unique names
    net.gen["name"] = net.gen.index.to_series().apply(lambda x: f"Generator {x}")
    net.sgen["name"] = net.sgen.index.to_series().apply(
        lambda x: f"Static Generator {x}",
    )
    net.load["name"] = net.load.index.to_series().apply(lambda x: f"Load {x}")

    n_gen = len(net.gen)
    n_loads = len(net.load)
    profile = [3.0, 3.0, 3.0, 3.0, 2.0]
    profile_end = [3.0, 3.0, 3.0, 3.0, 3.0]  # last number differs
    columns_list = [profile] * 2 * (n_loads-1)
    columns_list += ([profile_end]*2)
    df_loads = pd.DataFrame(
        columns_list,
    ).T
    df_loads.columns = [
        col
        for i in range(1, n_loads + 1)
        for col in (f"load {i}_pload", f"load {i}_qload")
    ]
    df_powerplants = pd.DataFrame({
        f"profile{i + 1}": [5.0, 5.0, 5.0, 5.0, 5.0] for i in range(n_gen)
    })

    profiles = {"load": df_loads, "powerplants": df_powerplants}
    net.profiles = profiles

    for ibus in net.bus.index:
        if not can_convert_to_n_busbar_substation(net, ibus):
            continue

        create_n_busbar_substation(net, ibus)

    return net


@pytest.fixture()
def test_grid_with_sgens_plus_simbench() -> pandapowerNet:
    """
    Test grid with static generators plus simbench profiles.

    :return: test grid
    :rtype: pandapowerNet
    """
    net = case89pegase()
    net.profiles = simbench.get_all_simbench_profiles(0)

    # Make sure loads, generators each have unique names
    net.gen["name"] = net.gen.index.to_series().apply(lambda x: f"Generator {x}")
    net.sgen["name"] = net.sgen.index.to_series().apply(
        lambda x: f"Static Generator {x}",
    )
    net.load["name"] = net.load.index.to_series().apply(lambda x: f"Load {x}")

    # Load random profiles into net
    rng = np.random.default_rng(seed=12345)
    net.load["profile"] = rng.choice(
        get_profile_names(net.profiles["load"]),
        len(net.load),
    )
    net.gen["profile"] = rng.choice(
        get_profile_names(net.profiles["powerplants"]),
        len(net.gen),
    )
    net.sgen["profile"] = rng.choice(
        get_profile_names(net.profiles["renewables"]),
        len(net.sgen),
    )

    return net


@pytest.fixture()
def env_config(test_grid_dbb_plus_simbench) -> dict:
    """
    Fixture for a simple environment configuration.

    :return: a simple environment configuration
    :rtype: dict
    """
    net = test_grid_dbb_plus_simbench
    actions = add_actions_substation_line_switching(net)[:3]
    return {
        "net": net,
        "n_episodes": 1,
        "episode_length": 5,
        "action_space": actions,
        "nminus1": False,
    }


@pytest.fixture()
def simenv(test_grid_dbb_plus_simbench) -> PPTopoGym:
    """
    Fixture for a simple simulation environment.

    :param test_grid_dbb_plus_simbench: Grid with 14 buses, with a double-busbar Dataframe
        already created, plus simbench-style profiles.
    :type test_grid_dbb_plus_simbench: pandapowerNet
    :return: a simple simulation environment
    :rtype: PPTopoGym
    """
    net = test_grid_dbb_plus_simbench
    action_4 = defaultdict(list, {"action": 4, "disconnect_lines": [1]})

    dict_actions = [
        {"action": 0, "substations": [], "states": []},
        {"action": 1, "substations": [0], "states": ["0x110101"]},
        {
            "action": 2,
            "substations": [0, 1],
            "states": ["0x101101","0x1100"],
            "lines": [5],
            "disconnect_lines": [True],
        },
        {
            "action": 3,
            "substations": [],
            "states": [],
            "lines": [5],
            "disconnect_lines": [False],
        },
        {
            "action": 4,
            "substations": [],
            "states": [],
            "lines": [2],
            "disconnect_lines": [True],
        },
    ]

    dict_actions = [defaultdict(list, action) for action in dict_actions]
    dict_actions.append(action_4)

    env_config = {
        "n_episodes": 10,
        "episode_length": 5,
        "net": net,
        "action_space": dict_actions,
        "nminus1": False,
    }

    return PPTopoGym(env_config=env_config)



@pytest.fixture()
def simenv_oldobs(test_grid_dbb_plus_simbench) -> PPTopoGym:
    """
    Fixture for a simple simulation environment.

    :param test_grid_dbb_plus_simbench: Grid with 14 buses, with a double-busbar Dataframe
        already created, plus simbench-style profiles.
    :type test_grid_dbb_plus_simbench: pandapowerNet
    :return: a simple simulation environment
    :rtype: PPTopoGym
    """
    net = test_grid_dbb_plus_simbench
    action_4 = defaultdict(list, {"action": 4, "disconnect_lines": [1]})

    dict_actions = [
        {"action": 0, "substations": [], "states": []},
        {"action": 1, "substations": [0], "states": ["0x110101"]},
        {
            "action": 2,
            "substations": [0, 1],
            "states": ["0x101101","0x1100"],
            "lines": [5],
            "disconnect_lines": [True],
        },
        {
            "action": 3,
            "substations": [],
            "states": [],
            "lines": [5],
            "disconnect_lines": [False],
        },
        {
            "action": 4,
            "substations": [],
            "states": [],
            "lines": [2],
            "disconnect_lines": [True],
        },
    ]

    dict_actions = [defaultdict(list, action) for action in dict_actions]
    dict_actions.append(action_4)

    env_config = {
        "n_episodes": 10,
        "episode_length": 5,
        "net": net,
        "action_space": dict_actions,
        "nminus1": False,
        "fix_obs_space": False,
    }

    return PPTopoGym(env_config=env_config)


@pytest.fixture()
def simenv2(test_grid_dbb_plus_simbench) -> PPTopoGym:
    """
    Fixture for a simple simulation environment.

    :param test_grid_dbb_plus_simbench: Grid with 14 buses, with a double-busbar Dataframe
        already created, plus simbench-style profiles.
    :type test_grid_dbb_plus_simbench: pandapowerNet
    :return: a simple simulation environment
    :rtype: PPTopoGym
    """
    net = test_grid_dbb_plus_simbench
    action_4 = defaultdict(list, {"action": 4, "disconnect_lines": [1]})

    dict_actions = [
        {"action": 0, "substations": [], "states": []},
        {"action": 1, "substations": [0], "states": ["0x110101"]},
        {
            "action": 2,
            "substations": [0, 1],
            "states": ["0x101101", "0x1100"],
            "lines": [5],
            "disconnect_lines": [True],
        },
        {
            "action": 3,
            "substations": [],
            "states": [],
            "lines": [5],
            "disconnect_lines": [False],
        },
        {
            "action": 4,
            "substations": [],
            "states": [],
            "lines": [2],
            "disconnect_lines": [True],
        },
    ]

    dict_actions = [defaultdict(list, action) for action in dict_actions]
    dict_actions.append(action_4)

    env_config = {
        "n_episodes": 5,
        "episode_length": 5,
        "net": net,
        "action_space": dict_actions,
        "nminus1": False,
    }

    return PPTopoGym(env_config=env_config)

@pytest.fixture()
def dummy_result() -> dict[str, Any]:
    """Return a minimal RLlib-style train result."""
    return {
    "training_iteration": 1,
    "env_runners": {
        "episode_return_mean": 500.0,
        "episode_len_mean": 100.0,
        "custom_metrics": {
            "mean_loading_mean": 85.0,
            "crash_mean": 0.05,
            },
        },
    }

@pytest.fixture(scope="module")
def simple_gine_module() -> GINETorchRLModule:
    """Create A minimal, ready-to-use GINETorchRLModule instance."""
    num_buses = 4
    num_lines = 2
    num_trafos = 1

    obs_space = spaces.Dict(
        {
            # node features (shape: (num_buses,))
            "bus_voltage_magnitude": spaces.Box(-1.0, 1.0, (num_buses,), np.float32),
            "bus_voltage_angle": spaces.Box(-np.pi, np.pi, (num_buses,), np.float32),
            "bus_loads": spaces.Box(0.0, 10.0, (num_buses,), np.float32),
            "bus_generators": spaces.Box(0.0, 10.0, (num_buses,), np.float32),
            # line feature vectors (shape: (num_lines,))
            "line_loadings": spaces.Box(0.0, 100.0, (num_lines,), np.float32),
            "line_power_flow_p_mw": spaces.Box(-100.0, 100.0, (num_lines,), np.float32),
            "line_power_flow_q_mvar": spaces.Box(-100.0, 100.0, (num_lines,), np.float32),
            "line_status": spaces.Box(0, 1, (num_lines,), np.int32),
            "line_thermal_limit": spaces.Box(0.0, 150.0, (num_lines,), np.float32),
            # trafo feature vectors (shape: (num_trafos,))
            "transformer_loading_percent": spaces.Box(0.0, 100.0, (num_trafos,), np.float32),
            "transformer_power_flow_p_mw": spaces.Box(-100.0, 100.0, (num_trafos,), np.float32),
            "transformer_power_flow_q_mvar": spaces.Box(-100.0, 100.0, (num_trafos,), np.float32),
            "transformer_tap_position": spaces.Box(-5, 5, (num_trafos,), np.int32),
            "transformer_status": spaces.Box(0, 1, (num_trafos,), np.int32),
            # adjacency matrix (num_edges, 2)
            "adjacency_matrix": spaces.Box(0, num_buses - 1, (num_lines + num_trafos, 2), np.int32),
        },
    )

    action_space = spaces.Discrete(3)

    model_config: dict[str, Any] = {
        "gine_hidden_dims": [16, 16],
        "num_layers": 2,
    }

    return GINETorchRLModule(
        observation_space=obs_space,
        action_space=action_space,
        model_config_dict=model_config,
    )

def _fake_obs_tensor(space: spaces.Space) -> torch.Tensor:
    """Generate a tensor of random values matching the Box space (MyPy-safe)."""
    shape: tuple[int, ...] = space.shape or ()
    rng = np.random.default_rng(42)
    arr = rng.random(shape if shape else (1,))
    arr = arr.astype(space.dtype).reshape(shape)
    return torch.as_tensor(arr)

@pytest.fixture()
def sample_observation(simple_gine_module: GINETorchRLModule) -> dict[str, torch.Tensor]:
    """Create one-batch observation dict matching the module's obs space."""
    obs_dict: dict[str, torch.Tensor] = {}
    obs_space = cast(spaces.Dict, simple_gine_module.observation_space)
    for key, sp in obs_space.spaces.items():
        t = _fake_obs_tensor(sp)  # shape (...)
        # Add batch dim =1 for every tensor
        obs_dict[key] = t.unsqueeze(0)
    return obs_dict

@pytest.fixture()
def simenv30() -> PPTopoGym:
    cfg = config_case30()
    return PPTopoGym(cfg)
