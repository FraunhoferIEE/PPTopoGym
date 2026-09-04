from __future__ import annotations

import os
import sys

os.chdir("/PATH/TO/pandapower-env/")
PROJECT_ROOT = "/PATH/TO/pandapower-env/"
sys.path.insert(0, PROJECT_ROOT)

import logging
import os
import warnings


import ray
from gymnasium import spaces
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env
from ray.util.annotations import RayDeprecationWarning

from pandapower_env.environments.simulation_env import (
    PPTopoGym,
)
from pandapower_env.rlib_agents.callbacks import RayCallbacks
from pandapower_env.rlib_agents.gnn_agents import GINETorchRLModule

from pandapower_env.data.example_configs import config_case30



os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
warnings.filterwarnings("ignore", category=RayDeprecationWarning)

def env_creator(env_config):
    return PPTopoGym(env_config)
register_env("SimulationEnvV0", env_creator) # register env with env creator function

my_config = config_case30()
#my_config = {
#    "net": net,
#    "n_episodes": 366,
 #   "episode_length": 96,  # 1 day
#    "action_space": actions,
#    "obs_keys":  [
#           "bus_voltage_magnitude",
#           "bus_voltage_angle",
#           "line_loadings",
#            "transformer_loading_percent",
#            "adjacency_matrix",
#        ],
 #   }

# You need to create an instance of your environment to get observation_space & action_space
env = PPTopoGym(my_config)

# create config for the agent
model_cfg = {
    "gine_hidden_dims":     [512, 512, 512],   # each entry is one GINEConv
    "gine_mlp_layers":      3,
    "gine_mlp_activation":  "relu",            # relu | elu

    "pooling":              "max",            # mean | max | add

    "fc_hidden_dims":       [512, 512],
    "fc_activation":        "relu",
    "norm":                "graph",           # graph | layer | none
    "dropout":              0.02
}
action_space = spaces.Discrete(len(my_config["action_space"]))

MyModuleSpec = RLModuleSpec(
    module_class=GINETorchRLModule,
    observation_space=env.observation_space,
    action_space=action_space,
    model_config=model_cfg,
    inference_only=False,
    learner_only=False,
)

logging.getLogger("pandapower").setLevel(logging.ERROR)
# Initialize Ray
ray.init(ignore_reinit_error=True)

# create config for PPO
config = (
    PPOConfig()
      .api_stack(
          enable_rl_module_and_learner=True,
          enable_env_runner_and_connector_v2=True,
      )
      .environment("SimulationEnvV0", env_config=my_config)
      .rl_module(rl_module_spec=MyModuleSpec)
      .training(minibatch_size=256, num_epochs=4, gamma=0.99, lr=1e-3, entropy_coeff=0.02) # type: ignore[call-arg]
      .env_runners(num_cpus_per_env_runner=10, num_env_runners=1, batch_mode="complete_episodes", rollout_fragment_length="auto")
      .learners(num_learners=1, num_gpus_per_learner=1)
      .framework("torch")
)
config = config.callbacks(callbacks_class=RayCallbacks)

algo = config.build_algo()

checkpoint_root = os.path.abspath("./pandapower_env/rlib_agents/training_scripts/checkpoints_gnn")
os.makedirs(checkpoint_root, exist_ok=True)

for i in range(2500):
    result = algo.train()
    if (i) % 25 == 0:
        ckpt_path = algo.save(checkpoint_root)
        print(f"[Checkpoint] iter {i+1}: saved to {ckpt_path}")
