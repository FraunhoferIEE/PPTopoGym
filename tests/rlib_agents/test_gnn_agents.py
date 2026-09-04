"""Tests for pandapower_env.rlib_agents.gnn_agents.GINETorchRLModule."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch
from ray.rllib.core.columns import Columns
from torch_geometric.data import Batch

if TYPE_CHECKING:
    from pandapower_env.rlib_agents.gnn_agents import GINETorchRLModule


def test_transform_observation_returns_batch(
    simple_gine_module: GINETorchRLModule,
    sample_observation: dict[str, torch.Tensor],
) -> None:
    """transform_observation should return a PyG Batch on the same device."""
    batch = simple_gine_module.transform_observation(sample_observation)
    assert isinstance(batch, Batch)
    assert batch.x.size(1) == 4  # 4 node-feature channels # noqa: PLR2004
    assert batch.edge_index.shape[0] == 2 # noqa: PLR2004


@pytest.mark.parametrize("mode", ["train", "inference", "exploration"])
def test_forward_modes(
    mode: str,
    simple_gine_module: GINETorchRLModule,
    sample_observation: dict[str, torch.Tensor],
) -> None:
    """_forward_* methods must return action logits and value preds."""
    batch_input: dict[str, Any] = {Columns.OBS: sample_observation}

    if mode == "train":
        out = simple_gine_module._forward_train(batch_input)
        assert set(out.keys()) == {
            Columns.ACTION_DIST_INPUTS,
            Columns.EMBEDDINGS,
        }
    elif mode == "inference":
        out = simple_gine_module._forward_inference(batch_input)
        assert set(out.keys()) == {
            Columns.ACTION_DIST_INPUTS,
        }
    else:
        out = simple_gine_module._forward_exploration(batch_input)
        assert set(out.keys()) == {
            Columns.ACTION_DIST_INPUTS,

        }

    # Shape checks -----------------------------------------------------------
    assert out[Columns.ACTION_DIST_INPUTS].dim() == 2  # noqa: PLR2004



def test_compute_values(simple_gine_module: GINETorchRLModule, sample_observation: dict[str, torch.Tensor]) -> None:
    """compute_values should match vf head output size."""
    batch_input: dict[str, Any] = {Columns.OBS: sample_observation}
    vals = simple_gine_module.compute_values(batch_input)
    assert vals.shape == torch.Size([1])
