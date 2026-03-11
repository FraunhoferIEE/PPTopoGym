from typing import Any, cast

import torch
from gymnasium.spaces import Dict, Discrete
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, GraphNorm, LayerNorm, global_add_pool, global_max_pool, global_mean_pool


class GINETorchRLModule(TorchRLModule, ValueFunctionAPI):

    @override(TorchRLModule)
    def setup(self) -> None:
        raw_cfg = self.model_config or {}
        if isinstance(raw_cfg, dict):
            cfg: dict[str, Any] = raw_cfg
        else:
            cfg = getattr(raw_cfg, "__dict__", {})

        h_dims = cfg.get("gine_hidden_dims",[128]*cfg.get("num_layers", 2))

        self.pooling_type = cfg.get("pooling", "mean").lower()
        if self.pooling_type not in {"mean", "max", "add"}:
            msg = "pooling must be 'mean', 'max', or 'add'"
            raise ValueError(msg)


        act_map = {"relu": nn.ReLU, "elu": nn.ELU}

        self.act_cls_mlp = act_map.get(
            cfg.get("gine_mlp_activation", "relu").lower(), nn.ReLU)
        self.act_cls_fc  = act_map.get(
            cfg.get("fc_activation", "relu").lower(), nn.ReLU)


        in_ch   = 4 # change when use another feature key set
        edge_d  = 10
        depth_mlp = cfg.get("gine_mlp_layers",  2)

        self.norm_type = (cfg.get("norm", "graph") or "graph").lower()
        self.dropout = nn.Dropout(cfg.get("dropout", 0.0))
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for out_ch in h_dims:
            self.convs.append(
                GINEConv(
                    self._mlp(in_ch, out_ch, depth_mlp),  # <<<
                    edge_dim=edge_d,
                    train_eps=True,
                ),
            )
            self.norms.append(GraphNorm(out_ch) if self.norm_type == "graph"
                              else LayerNorm(out_ch) if self.norm_type == "layer"
                              else nn.Identity())
            in_ch = out_ch
        self._last_hidden = in_ch

        fc_hidden = cfg.get("fc_hidden_dims", [])
        num_actions = int(cast(Discrete, self.action_space).n)
        self.pi_head = self._fc_stack(fc_hidden, num_actions)
        vf_hidden = cfg.get("vf_hidden_dims", fc_hidden) if cfg.get("separate_value_fc", False) else fc_hidden
        self.v_head = self._fc_stack(vf_hidden, 1)


    def _mlp(self, in_dim: int, out_dim: int, depth: int) -> nn.Sequential:
        act_cls = self.act_cls_mlp
        layers  = []
        for i in range(depth):
            layers += [nn.Linear(in_dim if i == 0 else out_dim, out_dim),
                       act_cls()]
        layers.append(nn.Linear(out_dim, out_dim))
        return nn.Sequential(*layers)

    def _fc_stack(self, hidden_dims: list[int], out_dim: int) -> nn.Sequential:
        act_cls = self.act_cls_fc
        dims    = [self._last_hidden, *hidden_dims, out_dim]
        layers  = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), act_cls()]
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers)

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        if self.pooling_type == "max":
            return global_max_pool(x, batch)
        return global_add_pool(x, batch)  # 'add'

    def _encode(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        g = self.transform_observation(obs)
        x, e = g.x, g.edge_attr
        act  = self.act_cls_mlp()
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, g.edge_index, e)
            try:
                x = norm(x, g.batch)
            except TypeError:
                x = norm(x)  # Identity or nn.LayerNorm fallback
            x = self.dropout(act(x))
        return self._pool(x, g.batch)

    @override(TorchRLModule)
    def _forward(self, batch: dict[str, Any], **_: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Forward for inference/exploration: only policy logits."""
        z = self._encode(batch[Columns.OBS])
        return {Columns.ACTION_DIST_INPUTS: self.pi_head(z)}

    @override(TorchRLModule)
    def _forward_train(self, batch: dict[str, Any], **_: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Training forward: policy logits + embeddings (used by PPO for V)."""
        z = self._encode(batch[Columns.OBS])
        return {
            Columns.ACTION_DIST_INPUTS: self.pi_head(z),
            Columns.EMBEDDINGS:         z,             # <— let PPO call compute_values(z)
        }

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings: torch.Tensor | None = None) -> torch.Tensor:
        """PPO will pass EMBEDDINGS when available; otherwise we encode again."""
        if embeddings is None:
            embeddings = self._encode(batch[Columns.OBS])
        return self.v_head(embeddings).squeeze(-1)

    def transform_observation(self, obs: dict[str, torch.Tensor]) -> Data:
        """
        Convert the raw observation dict into a torch_geometric Data object.

        Expected keys:

        Node (bus) features:
          - "bus_voltage_magnitude": numpy array of shape (num_buses,)
          - "bus_voltage_angle": numpy array of shape (num_buses,)
          - "bus_loads": numpy array of shape (num_buses,)
          - "bus_generators": numpy array of shape (num_buses,)

        Main edge connectivity (lines and transformers):
          - "adjacency_matrix": numpy array of shape (n_lines + n_trafo, 2)
            * The first n_lines rows correspond to line edges.
            * The remaining rows correspond to transformer edges.

        Edge features:
          For line edges (first n_lines rows):
            - "line_loadings": numpy array of shape (n_lines,)
            - "line_power_flow_p_mw": numpy array of shape (n_lines,)
            - "line_power_flow_q_mvar": numpy array of shape (n_lines,)
            - "line_status": numpy array of shape (n_lines,)
            - "line_thermal_limit": numpy array of shape (n_lines,)

          For transformer edges (next n_trafo rows):
            - "transformer_loading_percent": numpy array of shape (n_trafo,)
            - "transformer_power_flow_p_mw": numpy array of shape (n_trafo,)
            - "transformer_power_flow_q_mvar": numpy array of shape (n_trafo,)
            - "transformer_tap_position": numpy array of shape (n_trafo,)
            - "transformer_status": numpy array of shape (n_trafo,)
        """
        device = next(self.v_head.parameters()).device

        node_keys = [
            "bus_voltage_magnitude",
            "bus_voltage_angle",
            "bus_loads",
            "bus_generators",
        ]
        line_keys = [
            "line_loadings",
            "line_power_flow_p_mw",
            "line_power_flow_q_mvar",
            "line_status",
            "line_thermal_limit",
        ]
        trafo_keys = [
            "transformer_loading_percent",
            "transformer_power_flow_p_mw",
            "transformer_power_flow_q_mvar",
            "transformer_tap_position",
            "transformer_status",
        ]

        # 1) Collapse any leading batch/time dims into one batch dimension
        obs_space = cast(Dict, self.observation_space)
        exp_ndims = {
            k: len(space.shape or ())
            for k, space in obs_space.spaces.items()
        }
        flat_obs = {}
        for k, tensor in obs.items():
            t = tensor.to(device)
            e_nd = exp_ndims[k]
            if t.dim() > e_nd:
                lead = t.shape[:-e_nd]
                b = 1
                for d in lead:
                    b *= d
                t = t.view(b, *t.shape[-e_nd:])
            flat_obs[k] = t

        batch_count = flat_obs[node_keys[0]].shape[0]
        num_line_feats = len(line_keys)
        num_trafo_feats = len(trafo_keys)

        data_list = []
        for i in range(batch_count):
            # Node features
            nf = [flat_obs[k][i].unsqueeze(-1) for k in node_keys]
            node_x = torch.cat(nf, dim=-1)

            # Line features
            lf = [flat_obs[k][i].unsqueeze(-1) for k in line_keys]
            line_feat = torch.cat(lf, dim=-1)

            # Trafo features
            tf = [flat_obs[k][i].unsqueeze(-1) for k in trafo_keys]
            trafo_feat = torch.cat(tf, dim=-1)

            # Pad to equal width = 5+5 = 10
            pad_l = torch.zeros(line_feat.size(0), num_trafo_feats, device=device)
            line_full = torch.cat([line_feat, pad_l], dim=1)

            pad_t = torch.zeros(trafo_feat.size(0), num_line_feats, device=device)
            trafo_full = torch.cat([pad_t, trafo_feat], dim=1)

            edge_attr = torch.cat([line_full, trafo_full], dim=0)

            # Build edge_index and cast to int64
            adj = flat_obs["adjacency_matrix"][i]  # shape: [n_edges, 2]
            n_lines = line_feat.size(0)
            e1 = adj[:n_lines].t().contiguous()
            e2 = adj[n_lines:].t().contiguous()
            edge_index = torch.cat([e1, e2], dim=1).long()  # ensure int64

            data_list.append(Data(x=node_x,
                                  edge_index=edge_index,
                                  edge_attr=edge_attr))

        return Batch.from_data_list(data_list).to(device)


