"""Train a small DQN against the vectorized env, to smoke-test the plumbing a learner needs.

Run with::

    python scripts/train_smoke.py                       # case14, SyncVectorEnv
    python scripts/train_smoke.py --grid case30 --steps 300 --async

Why this exists: the env's real consumer is a learning algorithm, and the failure that
matters most to one -- **observation shapes changing mid-run** -- is invisible to the unit
tests and to the golden record, both of which take short, single-env trajectories. A
substation split grows ``n_nodes`` (case30: 30 -> 35), and without
``static_obs_space`` the vector envs die with ``could not broadcast input array from shape
(31,) into shape (30,)``. That flag is therefore forced on here, and every step asserts the
per-key shapes have not moved.

Success is about plumbing, not about learning well: the run completes, shapes stay constant,
the loss stays finite, and the same seed reproduces the same reward trace. The reported
steps/s is the number that answers "how does this feel to a learner".

:raises SystemExit: if a shape moves, the loss goes non-finite, or a seed fails to reproduce.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

if TYPE_CHECKING:
    from pandapower_env.environments.multi_pp_env import SimpleVecEnvWrapper

# Kept 1-D and always present, so the network input width is fixed regardless of grid.
FEATURE_KEYS = ("line_loadings", "bus_voltage_magnitude")


def build_vec_env(grid: str, num_envs: int, *, use_async: bool) -> tuple[SimpleVecEnvWrapper, dict]:
    """Build a vectorized PPTopoGym with a static (paddable) observation space.

    :param grid: grid name understood by :func:`scripts.golden_record.build_config`.
    :param num_envs: how many environments to run in parallel.
    :param use_async: True for ``AsyncVectorEnv`` (separate processes).
    :return: the wrapped vector env and the config it was built from.
    """
    from pandapower_env.environments.multi_pp_env import (
        SimpleVecEnvWrapper,
        create_vectorized_env,
    )
    from pandapower_env.environments.simulation_env import PPTopoGym
    from golden_record import build_config

    config = build_config(grid)
    # Without this the observation space is a lie the moment a substation splits, and the
    # vector env raises on the broadcast (see CLAUDE.md).
    config["static_obs_space"] = True

    def factory() -> PPTopoGym:
        return PPTopoGym(copy.deepcopy(config))

    vec_env = create_vectorized_env(factory, num_envs, use_async=use_async, seed=0)
    return SimpleVecEnvWrapper(vec_env), config


def features(obs: dict[str, np.ndarray]) -> torch.Tensor:
    """Concatenate the fixed feature keys of a batched observation into a float tensor."""
    parts = [np.asarray(obs[key], dtype=np.float32).reshape(len(obs[key]), -1) for key in FEATURE_KEYS]
    return torch.from_numpy(np.concatenate(parts, axis=1))


def assert_shapes_stable(obs: dict[str, np.ndarray], reference: dict[str, tuple], step: int) -> None:
    """Fail loudly if any observation key changed shape since the first reset.

    :raises SystemExit: naming the key and both shapes, so the culprit is obvious.
    """
    for key, value in obs.items():
        if np.asarray(value).shape != reference[key]:
            sys.exit(
                f"SHAPE CHANGED at step {step}: {key} {reference[key]} -> "
                f"{np.asarray(value).shape}",
            )


def train(grid: str, steps: int, num_envs: int, *, use_async: bool, seed: int) -> list[float]:
    """Run the DQN loop and return the per-step mean reward trace.

    :raises SystemExit: on a shape change or a non-finite loss.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    env, _config = build_vec_env(grid, num_envs, use_async=use_async)
    n_actions = env.action_size
    obs, _ = env.reset(seed=seed)
    reference = {key: np.asarray(value).shape for key, value in obs.items()}

    net = nn.Sequential(
        nn.Linear(features(obs).shape[1], 64), nn.ReLU(), nn.Linear(64, n_actions),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    rewards: list[float] = []
    start = time.perf_counter()
    for step in range(steps):
        state = features(obs)
        epsilon = max(0.05, 1.0 - step / max(steps, 1))
        if rng.random() < epsilon:
            actions = rng.integers(0, n_actions, size=num_envs)
        else:
            with torch.no_grad():
                actions = net(state).argmax(dim=1).numpy()

        result = env.step(actions)
        assert_shapes_stable(result.observations, reference, step)

        next_state = features(result.observations)
        with torch.no_grad():
            target = torch.from_numpy(result.rewards.astype(np.float32)) + 0.99 * net(
                next_state,
            ).max(dim=1).values * torch.from_numpy(~result.terminateds).float()
        predicted = net(state).gather(1, torch.as_tensor(actions).long().unsqueeze(1)).squeeze(1)
        loss = nn.functional.mse_loss(predicted, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if not torch.isfinite(loss):
            sys.exit(f"NON-FINITE LOSS at step {step}: {loss.item()}")

        rewards.append(float(np.mean(result.rewards)))
        obs = result.observations
        if step % 25 == 0:
            print(f"  step {step:4d}  loss {loss.item():10.4f}  reward {rewards[-1]:8.3f}", flush=True)

    elapsed = time.perf_counter() - start
    print(
        f"  {steps} steps x {num_envs} envs in {elapsed:.1f}s "
        f"-> {steps * num_envs / elapsed:.1f} env-steps/s",
        flush=True,
    )
    env.close()
    return rewards


def main() -> None:
    """Parse arguments, train twice with the same seed, and check the traces agree."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="case14")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--async", dest="use_async", action="store_true")
    parser.add_argument(
        "--skip-repro", action="store_true", help="skip the second, reproducibility run",
    )
    args = parser.parse_args()

    print(f"training on {args.grid} ({'async' if args.use_async else 'sync'}) ...", flush=True)
    first = train(args.grid, args.steps, args.num_envs, use_async=args.use_async, seed=args.seed)

    if not args.skip_repro:
        print("re-running with the same seed ...", flush=True)
        second = train(
            args.grid, args.steps, args.num_envs, use_async=args.use_async, seed=args.seed,
        )
        if first != second:
            mismatch = next(i for i, (a, b) in enumerate(zip(first, second)) if a != b)
            sys.exit(f"NOT REPRODUCIBLE: reward traces diverge at step {mismatch}")
        print("reward trace reproduced exactly")

    print("train smoke OK")


if __name__ == "__main__":
    main()
