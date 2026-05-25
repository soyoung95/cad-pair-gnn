from __future__ import annotations




import argparse
import csv
import json
import os
import random
import time
import atexit
from datetime import datetime
from pathlib import Path
from typing import Any


from tqdm import tqdm


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool








# ── Constants ─────────────────────────────────────────────────────────────────




FACE_PNET_DIM = 32   # FaceMiniPointNet output dim per face node
EDGE_PNET_DIM = 16   # EdgeMiniPointNet output dim per directed edge
PNET_MAX_PTS  = 64   # max sample points per face / edge entity




# Attributes that are NOT batched automatically (variable-length or local indices)
_EXCLUDE_FROM_BATCH = (
    "face_points", "face_normals", "face_trimming_mask",
    "edge_entity_points", "edge_entity_tangents",
    "edge_to_entity_idx",   # local per-graph entity indices — would break if batched
)








# ── Utilities ─────────────────────────────────────────────────────────────────




def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)








def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)








def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)








def torch_load_graph(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")








def clean_graph(graph: Any, default_edge_dim: int = 16) -> Data:
    if not hasattr(graph, "x"):
        raise ValueError("Graph missing x.")
    if not hasattr(graph, "edge_index"):
        raise ValueError("Graph missing edge_index.")
    if not hasattr(graph, "edge_attr") or graph.edge_attr is None:
        n_edges = int(graph.edge_index.shape[1])
        graph.edge_attr = torch.zeros((n_edges, default_edge_dim), dtype=torch.float32)




    kwargs: dict[str, Any] = {
        "x":          graph.x.float(),
        "edge_index": graph.edge_index.long(),
        "edge_attr":  graph.edge_attr.float(),
    }
    for attr in _EXCLUDE_FROM_BATCH:
        if hasattr(graph, attr) and getattr(graph, attr) is not None:
            kwargs[attr] = getattr(graph, attr)
    return Data(**kwargs)








def iter_pair_records(data: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []




    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            keys = {str(k).lower() for k in obj.keys()}
            if "body_one" in keys and "body_two" in keys:
                records.append(obj)
                return
            for value in obj.values():
                visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)




    visit(data)
    return records








def get_value(record: dict[str, Any], key: str, default: Any = "") -> Any:
    if key in record:
        return record[key]
    lower_key = key.lower()
    for k, value in record.items():
        if str(k).lower() == lower_key:
            return value
    return default








# ── Dataset ───────────────────────────────────────────────────────────────────




class PairGraphDataset(Dataset):
    def __init__(self, pair_file: Path, graph_dir: Path, limit: int | None) -> None:
        data    = load_json(pair_file)
        records = iter_pair_records(data)
        if limit is not None:
            records = records[:limit]
        if not records:
            raise ValueError(f"No records found: {pair_file}")
        self.records   = records
        self.graph_dir = graph_dir




    def __len__(self) -> int:
        return len(self.records)




    def label_counts(self) -> dict[str, int]:
        positive = sum(1 for r in self.records if int(get_value(r, "label", 0)) == 1)
        negative = len(self.records) - positive
        return {"positive": positive, "negative": negative, "total": len(self.records)}




    def __getitem__(self, index: int) -> dict[str, Any]:
        record   = self.records[index]
        body_one = str(get_value(record, "body_one"))
        body_two = str(get_value(record, "body_two"))
        label    = int(get_value(record, "label", 0))
        for path, name in [
            (self.graph_dir / f"{body_one}.pt", "A"),
            (self.graph_dir / f"{body_two}.pt", "B"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"Missing graph {name}: {path}")
        graph_a = clean_graph(torch_load_graph(self.graph_dir / f"{body_one}.pt"))
        graph_b = clean_graph(torch_load_graph(self.graph_dir / f"{body_two}.pt"))
        return {
            "graph_a": graph_a,
            "graph_b": graph_b,
            "label":   torch.tensor(label, dtype=torch.float32),
        }








def _attr_or_empty(graph: Data, name: str) -> Any:
    """Return graph attribute without using Tensor truth-value checks."""
    value = getattr(graph, name, None)
    return [] if value is None else value




def _collect_point_data(graphs: list[Data]) -> dict[str, list | None]:
    """Extract per-graph point cloud data before batching.


    Important: do not write `tensor or []` here. PyTorch tensors with more than
    one value cannot be converted to bool, and that can crash collation.
    """
    has_pts = any(
        hasattr(g, "face_points") and getattr(g, "face_points") is not None
        for g in graphs
    )
    if not has_pts:
        return {
            "face_pts": None, "face_nrm": None, "face_msk": None,
            "edge_pts": None, "edge_tan": None, "edge_idx": None,
        }
    return {
        "face_pts": [_attr_or_empty(g, "face_points") for g in graphs],
        "face_nrm": [_attr_or_empty(g, "face_normals") for g in graphs],
        "face_msk": [_attr_or_empty(g, "face_trimming_mask") for g in graphs],
        "edge_pts": [_attr_or_empty(g, "edge_entity_points") for g in graphs],
        "edge_tan": [_attr_or_empty(g, "edge_entity_tangents") for g in graphs],
        "edge_idx": [getattr(g, "edge_to_entity_idx", None) for g in graphs],
    }








def pair_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    graphs_a = [item["graph_a"] for item in batch]
    graphs_b = [item["graph_b"] for item in batch]




    pt_a = _collect_point_data(graphs_a)
    pt_b = _collect_point_data(graphs_b)




    has_pts = pt_a["face_pts"] is not None
    excl    = list(_EXCLUDE_FROM_BATCH) if has_pts else []




    batch_a = Batch.from_data_list(graphs_a, exclude_keys=excl)
    batch_b = Batch.from_data_list(graphs_b, exclude_keys=excl)




    return {
        "graph_a": batch_a,
        "graph_b": batch_b,
        "label":   torch.stack([item["label"] for item in batch]),
        # Face point clouds (per graph in batch)
        "face_pts_a": pt_a["face_pts"], "face_nrm_a": pt_a["face_nrm"],
        "face_msk_a": pt_a["face_msk"],
        "face_pts_b": pt_b["face_pts"], "face_nrm_b": pt_b["face_nrm"],
        "face_msk_b": pt_b["face_msk"],
        # Edge point clouds (per graph in batch)
        "edge_pts_a": pt_a["edge_pts"], "edge_tan_a": pt_a["edge_tan"],
        "edge_idx_a": pt_a["edge_idx"],
        "edge_pts_b": pt_b["edge_pts"], "edge_tan_b": pt_b["edge_tan"],
        "edge_idx_b": pt_b["edge_idx"],
    }








# ── Model ─────────────────────────────────────────────────────────────────────




class MiniPointNet(nn.Module):
    """
    Generic PointNet-style encoder for paired point clouds.




    Takes two parallel [P_i, 3] tensors per node (e.g. points + normals,
    or edge_points + tangents), concatenates them to [P_i, 6], applies a
    shared MLP, then max-pools over P to get a fixed [out_dim] embedding.




    An optional float mask (1.0 = valid, 0.0 = trimmed/invalid) can be
    supplied to exclude trimmed sample points before max-pooling.
    This is used for face_trimming_mask.
    """
    IN_DIM = 6  # 3 + 3




    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.out_dim = out_dim
        mid = max(out_dim, 32)
        self.mlp = nn.Sequential(
            nn.Linear(self.IN_DIM, mid),
            nn.ReLU(),
            nn.Linear(mid, out_dim),
            nn.ReLU(),
        )




    def encode_graph(
        self,
        list_a:    list[torch.Tensor],          # [P_i, 3] per node
        list_b:    list[torch.Tensor],          # [P_i, 3] per node
        mask_list: list[torch.Tensor] | None,   # [P_i] float mask, or None
        device:    torch.device,
    ) -> torch.Tensor:
        """Returns [N_nodes, out_dim]."""
        N = len(list_a)
        if N == 0:
            return torch.zeros((0, self.out_dim), device=device)




        max_p = min(
            max((t.size(0) for t in list_a), default=0),
            PNET_MAX_PTS,
        )
        if max_p == 0:
            return torch.zeros((N, self.out_dim), device=device)




        buf_a = torch.zeros(N, max_p, 3, device=device)
        buf_b = torch.zeros(N, max_p, 3, device=device)
        valid = torch.zeros(N, max_p, dtype=torch.bool, device=device)




        for i, (ta, tb) in enumerate(zip(list_a, list_b)):
            p = min(ta.size(0), max_p)
            if p == 0:
                continue
            buf_a[i, :p] = ta[:p].to(device)
            if tb.size(0) >= p:
                buf_b[i, :p] = tb[:p].to(device)




            if mask_list is not None and i < len(mask_list):
                msk = mask_list[i]
                if msk.numel() >= p:
                    valid[i, :p] = msk[:p].to(device) > 0.5
                else:
                    valid[i, :p] = True
            else:
                valid[i, :p] = True




        # [N, max_p, 6] -> MLP -> masked max-pool -> [N, out_dim]
        feats = self.mlp(torch.cat([buf_a, buf_b], dim=-1))  # [N, max_p, out_dim]
        feats = feats.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        result = feats.max(dim=1).values                      # [N, out_dim]
        # Replace -inf (all-masked faces) with zeros
        result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result








class GATv2NodeEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")
        out_channels = hidden_dim // heads
        self.convs   = nn.ModuleList()
        self.norms   = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for i in range(num_layers):
            in_dim = node_dim if i == 0 else hidden_dim
            self.convs.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_channels,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))




    def forward(
        self,
        graph:         Batch,
        x_in:          torch.Tensor | None = None,
        edge_attr_in:  torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x         = x_in         if x_in         is not None else graph.x
        edge_attr = edge_attr_in if edge_attr_in  is not None else graph.edge_attr
        edge_index = graph.edge_index
        batch      = graph.batch
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
        z = global_mean_pool(x, batch)
        return x, batch, z








class GeometricCompatibility(nn.Module):
    """
    Hand-crafted pairwise features from raw node features.




    has_normals=True  (v2_geometry, node_dim=72):
      dims  0-6  : surface_type one-hot
      dim   7    : log_area
      dims 15-17 : face normal xyz  (verified by range analysis)
      -> 10 output features




    has_normals=False (v5_slim, node_dim=10):
      dims 0-6 : surface_type one-hot
      dim  7   : reversed flag
      dim  8   : log_area  (normals are in face_normals → handled by MiniPointNet)
      -> 6 output features
    """




    def __init__(self, has_normals: bool = True) -> None:
        super().__init__()
        self.has_normals = has_normals




    @property
    def out_dim(self) -> int:
        return 10 if self.has_normals else 6




    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        return self._with_normals(x_a, x_b) if self.has_normals else self._area_type(x_a, x_b)




    def _with_normals(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        type_sim  = torch.mm(x_a[:, 0:7], x_b[:, 0:7].T)
        area_diff = (x_a[:, 7].unsqueeze(1) - x_b[:, 7].unsqueeze(0)).abs()
        n_a  = F.normalize(x_a[:, 15:18], dim=-1)
        n_b  = F.normalize(x_b[:, 15:18], dim=-1)
        dots = torch.mm(n_a, n_b.T)
        f10  = torch.tensor(
            x_a.shape[0] / (x_a.shape[0] + x_b.shape[0] + 1e-8),
            dtype=torch.float32, device=x_a.device,
        )
        return torch.stack([
            dots.min(), dots.mean(),
            (dots < -0.5).float().mean(), (dots < -0.8).float().mean(),
            area_diff.min(), area_diff.mean(),
            type_sim.max(), type_sim.mean(),
            ((dots < -0.7) & (area_diff < 1.0)).float().mean(),
            f10,
        ])




    def _area_type(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        type_sim  = torch.mm(x_a[:, 0:7], x_b[:, 0:7].T)
        area_diff = (x_a[:, 8].unsqueeze(1) - x_b[:, 8].unsqueeze(0)).abs()
        f6 = torch.tensor(
            x_a.shape[0] / (x_a.shape[0] + x_b.shape[0] + 1e-8),
            dtype=torch.float32, device=x_a.device,
        )
        return torch.stack([
            area_diff.min(), area_diff.mean(),
            type_sim.max(), type_sim.mean(),
            (area_diff < 0.5).float().mean(),
            f6,
        ])








class PairMatchCompatibility(nn.Module):
    """
    Add direct face/edge candidate matching features between two CAD parts.


    This does not replace the GNN. It gives the final classifier extra numbers
    that describe whether some face/edge candidates from Part A and Part B
    look geometrically compatible.
    """


    def __init__(self) -> None:
        super().__init__()


    @property
    def out_dim(self) -> int:
        return 16


    def forward(
        self,
        x_a: torch.Tensor,
        x_b: torch.Tensor,
        edge_a: torch.Tensor,
        edge_b: torch.Tensor,
    ) -> torch.Tensor:
        device = x_a.device
        face_feat = self._face_match(x_a, x_b, device)
        edge_feat = self._edge_match(edge_a, edge_b, device)


        face_ratio = torch.tensor(
            x_a.shape[0] / max(x_a.shape[0] + x_b.shape[0], 1),
            dtype=torch.float32,
            device=device,
        )
        edge_ratio = torch.tensor(
            edge_a.shape[0] / max(edge_a.shape[0] + edge_b.shape[0], 1),
            dtype=torch.float32,
            device=device,
        )


        return torch.cat([face_feat, edge_feat, face_ratio.view(1), edge_ratio.view(1)], dim=0)


    def _topk_mean(self, values: torch.Tensor, k: int = 3) -> torch.Tensor:
        flat = values.flatten()
        if flat.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=values.device)
        kk = min(k, flat.numel())
        return torch.topk(flat, kk).values.mean()


    def _safe_max(self, values: torch.Tensor) -> torch.Tensor:
        if values.numel() == 0:
            return torch.zeros((), dtype=torch.float32, device=values.device)
        return values.max()


    def _face_center(self, x: torch.Tensor) -> torch.Tensor | None:
        dim = x.shape[1]
        if dim >= 83:
            return x[:, 74:77]
        if dim >= 34:
            return x[:, 12:15]
        if dim >= 21:
            return x[:, 12:15]
        return None


    def _face_normal(self, x: torch.Tensor) -> torch.Tensor | None:
        if x.shape[1] >= 18:
            n = x[:, 15:18]
            if n.numel() == 0:
                return None
            return F.normalize(n, dim=-1, eps=1e-8)
        return None


    def _face_area(self, x: torch.Tensor) -> torch.Tensor | None:
        if x.shape[1] > 7:
            return x[:, 7]
        return None


    def _edge_center(self, edge_attr: torch.Tensor) -> torch.Tensor | None:
        dim = edge_attr.shape[1]
        if dim >= 62:
            return edge_attr[:, 59:62]
        if dim >= 37:
            return edge_attr[:, 19:22]
        if dim >= 22:
            return edge_attr[:, 19:22]
        return None


    def _edge_tangent(self, edge_attr: torch.Tensor) -> torch.Tensor | None:
        dim = edge_attr.shape[1]
        if dim >= 62:
            return F.normalize(edge_attr[:, 55:58], dim=-1, eps=1e-8)
        if dim >= 18:
            return F.normalize(edge_attr[:, 15:18], dim=-1, eps=1e-8)
        return None


    def _face_match(self, x_a: torch.Tensor, x_b: torch.Tensor, device: torch.device) -> torch.Tensor:
        if x_a.shape[0] == 0 or x_b.shape[0] == 0:
            return torch.zeros(8, dtype=torch.float32, device=device)


        center_a = self._face_center(x_a)
        center_b = self._face_center(x_b)
        normal_a = self._face_normal(x_a)
        normal_b = self._face_normal(x_b)
        area_a = self._face_area(x_a)
        area_b = self._face_area(x_b)


        if center_a is not None and center_b is not None:
            dist = torch.cdist(center_a.float(), center_b.float(), p=2)
            center_close = 1.0 / (1.0 + dist)
            center_close_max = self._safe_max(center_close)
            center_close_top3 = self._topk_mean(center_close, k=3)
        else:
            center_close = None
            center_close_max = torch.zeros((), dtype=torch.float32, device=device)
            center_close_top3 = torch.zeros((), dtype=torch.float32, device=device)


        if normal_a is not None and normal_b is not None:
            dots = torch.mm(normal_a.float(), normal_b.float().T)
            opposite = torch.clamp(-dots, min=0.0, max=1.0)
            abs_sim = dots.abs()
            opposite_max = self._safe_max(opposite)
            abs_max = self._safe_max(abs_sim)
            opposite_top3 = self._topk_mean(opposite, k=3)
        else:
            opposite = None
            opposite_max = torch.zeros((), dtype=torch.float32, device=device)
            abs_max = torch.zeros((), dtype=torch.float32, device=device)
            opposite_top3 = torch.zeros((), dtype=torch.float32, device=device)


        if area_a is not None and area_b is not None:
            area_diff = (area_a.float().unsqueeze(1) - area_b.float().unsqueeze(0)).abs()
            area_sim = torch.exp(-area_diff)
            area_sim_max = self._safe_max(area_sim)
        else:
            area_sim = None
            area_sim_max = torch.zeros((), dtype=torch.float32, device=device)


        type_sim = torch.mm(x_a[:, 0:7].float(), x_b[:, 0:7].float().T)
        type_sim_max = self._safe_max(type_sim)


        components = []
        if center_close is not None:
            components.append(center_close)
        if opposite is not None:
            components.append(opposite)
        if area_sim is not None:
            components.append(area_sim)
        components.append(type_sim.clamp(min=0.0, max=1.0))


        combined = torch.zeros_like(type_sim)
        for comp in components:
            combined = combined + comp
        combined = combined / max(len(components), 1)
        combined_max = self._safe_max(combined)


        return torch.stack([
            center_close_max,
            center_close_top3,
            opposite_max,
            abs_max,
            opposite_top3,
            area_sim_max,
            type_sim_max,
            combined_max,
        ])


    def _edge_match(
        self,
        edge_a: torch.Tensor,
        edge_b: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        if edge_a.shape[0] == 0 or edge_b.shape[0] == 0:
            return torch.zeros(6, dtype=torch.float32, device=device)


        center_a = self._edge_center(edge_a)
        center_b = self._edge_center(edge_b)
        tan_a = self._edge_tangent(edge_a)
        tan_b = self._edge_tangent(edge_b)


        if center_a is not None and center_b is not None:
            dist = torch.cdist(center_a.float(), center_b.float(), p=2)
            center_close = 1.0 / (1.0 + dist)
            center_close_max = self._safe_max(center_close)
            center_close_top3 = self._topk_mean(center_close, k=3)
        else:
            center_close = None
            center_close_max = torch.zeros((), dtype=torch.float32, device=device)
            center_close_top3 = torch.zeros((), dtype=torch.float32, device=device)


        if tan_a is not None and tan_b is not None:
            dots = torch.mm(tan_a.float(), tan_b.float().T)
            abs_sim = dots.abs()
            opposite = torch.clamp(-dots, min=0.0, max=1.0)
            abs_max = self._safe_max(abs_sim)
            abs_top3 = self._topk_mean(abs_sim, k=3)
            opposite_max = self._safe_max(opposite)
        else:
            abs_sim = None
            abs_max = torch.zeros((), dtype=torch.float32, device=device)
            abs_top3 = torch.zeros((), dtype=torch.float32, device=device)
            opposite_max = torch.zeros((), dtype=torch.float32, device=device)


        components = []
        if center_close is not None:
            components.append(center_close)
        if abs_sim is not None:
            components.append(abs_sim)


        if components:
            combined = torch.zeros_like(components[0])
            for comp in components:
                combined = combined + comp
            combined = combined / len(components)
            combined_max = self._safe_max(combined)
        else:
            combined_max = torch.zeros((), dtype=torch.float32, device=device)


        return torch.stack([
            center_close_max,
            center_close_top3,
            abs_max,
            abs_top3,
            opposite_max,
            combined_max,
        ])



class CrossAttentionPairModel(nn.Module):
    def __init__(
        self,
        node_dim:      int,
        edge_dim:      int,
        hidden_dim:    int,
        num_layers:    int,
        gat_heads:     int,
        cross_heads:   int,
        dropout:       float,
        use_point_net: bool = False,
        use_pair_match_features: bool = False,
    ) -> None:
        super().__init__()
        self.use_point_net = use_point_net
        self.use_pair_match_features = use_pair_match_features
        self.profile_forward = False
        self.profile_interval = 20
        self._forward_calls = 0




        # Augmented input dimensions when MiniPointNet is active
        gat_node_in = node_dim + FACE_PNET_DIM if use_point_net else node_dim
        gat_edge_in = edge_dim + EDGE_PNET_DIM if use_point_net else edge_dim




        self.encoder = GATv2NodeEncoder(
            node_dim=gat_node_in, edge_dim=gat_edge_in,
            hidden_dim=hidden_dim, num_layers=num_layers,
            heads=gat_heads, dropout=dropout,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=cross_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_b = nn.LayerNorm(hidden_dim)




        if use_point_net:
            # Face encoder: face_points + face_normals (with trimming_mask)
            self.face_pnet = MiniPointNet(out_dim=FACE_PNET_DIM)
            # Edge encoder: edge_entity_points + edge_entity_tangents
            self.edge_pnet = MiniPointNet(out_dim=EDGE_PNET_DIM)
        else:
            self.face_pnet = None
            self.edge_pnet = None




        self.geo_compat = GeometricCompatibility(has_normals=not use_point_net)
        self.pair_match = PairMatchCompatibility() if use_pair_match_features else None
        pair_match_dim = self.pair_match.out_dim if self.pair_match is not None else 0
        mlp_in = hidden_dim * 6 + self.geo_compat.out_dim + pair_match_dim
        self.classifier = nn.Sequential(
            nn.Linear(mlp_in, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )




    # ── Point cloud helpers ────────────────────────────────────────────────────




    def _face_emb(
        self,
        pts_lists: list, nrm_lists: list, msk_lists: list,
        device: torch.device,
    ) -> torch.Tensor:
        """[total_face_nodes, FACE_PNET_DIM] for all graphs in batch."""
        return torch.cat(
            [self.face_pnet.encode_graph(pts, nrm, msk, device)
             for pts, nrm, msk in zip(pts_lists, nrm_lists, msk_lists)],
            dim=0,
        )




    def _edge_emb(
        self,
        batched_graph: Batch,
        edge_pts_lists: list,
        edge_tan_lists: list,
        edge_idx_list:  list,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute EdgeMiniPointNet embeddings and map to directed edges.
        Returns [total_directed_edges, EDGE_PNET_DIM].
        """
        all_edge_embs = []
        for pts_list, tan_list, eidx in zip(edge_pts_lists, edge_tan_lists, edge_idx_list):
            entity_emb = self.edge_pnet.encode_graph(pts_list, tan_list, None, device)
            n_edges = eidx.numel() if eidx is not None else 0
            if n_edges == 0 or entity_emb.size(0) == 0:
                all_edge_embs.append(torch.zeros(n_edges, EDGE_PNET_DIM, device=device))
            else:
                idx = eidx.to(device).clamp(0, entity_emb.size(0) - 1)
                all_edge_embs.append(entity_emb[idx])
        return torch.cat(all_edge_embs, dim=0)  # [total_edges, EDGE_PNET_DIM]




    # ── Architecture helpers ───────────────────────────────────────────────────




    def _cross_pool(
        self,
        node_a: torch.Tensor, batch_a: torch.Tensor,
        node_b: torch.Tensor, batch_b: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out_a, out_b = [], []
        for i in range(batch_size):
            a_i = node_a[batch_a == i].unsqueeze(0)
            b_i = node_b[batch_b == i].unsqueeze(0)
            if a_i.size(1) == 0 or b_i.size(1) == 0:
                raise ValueError("Empty graph in batch.")
            ca, _ = self.cross_attention(a_i, b_i, b_i, need_weights=False)
            cb, _ = self.cross_attention(b_i, a_i, a_i, need_weights=False)
            out_a.append(self.norm_a(ca + a_i).mean(dim=1).squeeze(0))
            out_b.append(self.norm_b(cb + b_i).mean(dim=1).squeeze(0))
        return torch.stack(out_a), torch.stack(out_b)




    # ── Forward ───────────────────────────────────────────────────────────────




    def forward(
        self,
        graph_a: Batch,
        graph_b: Batch,
        # Face point clouds (per graph in batch, from v5_slim)
        face_pts_a: list | None = None,
        face_nrm_a: list | None = None,
        face_msk_a: list | None = None,
        face_pts_b: list | None = None,
        face_nrm_b: list | None = None,
        face_msk_b: list | None = None,
        # Edge point clouds (per graph in batch, from v5_slim)
        edge_pts_a: list | None = None,
        edge_tan_a: list | None = None,
        edge_idx_a: list | None = None,
        edge_pts_b: list | None = None,
        edge_tan_b: list | None = None,
        edge_idx_b: list | None = None,
    ) -> torch.Tensor:
        self._forward_calls += 1
        do_profile = (
            self.profile_forward
            and (
                self._forward_calls == 1
                or self._forward_calls % max(self.profile_interval, 1) == 0
            )
        )


        def sync_if_cuda() -> None:
            if torch.cuda.is_available() and graph_a.x.is_cuda:
                torch.cuda.synchronize()


        def now() -> float:
            sync_if_cuda()
            return time.perf_counter()


        raw_x_a = graph_a.x
        raw_x_b = graph_b.x
        dev = raw_x_a.device


        if do_profile:
            print(f"[FORWARD {self._forward_calls}] start", flush=True)


        if self.use_point_net and face_pts_a is not None:
            # Augmented node features: x || face_pnet_embedding
            if do_profile:
                t = now()
                print(f"[FORWARD {self._forward_calls}] face emb A start", flush=True)
            face_emb_a = self._face_emb(face_pts_a, face_nrm_a, face_msk_a, dev)
            if do_profile:
                print(f"[FORWARD {self._forward_calls}] face emb A done: {now() - t:.2f}s", flush=True)


            if do_profile:
                t = now()
                print(f"[FORWARD {self._forward_calls}] face emb B start", flush=True)
            face_emb_b = self._face_emb(face_pts_b, face_nrm_b, face_msk_b, dev)
            if do_profile:
                print(f"[FORWARD {self._forward_calls}] face emb B done: {now() - t:.2f}s", flush=True)


            aug_x_a = torch.cat([raw_x_a, face_emb_a], dim=-1)
            aug_x_b = torch.cat([raw_x_b, face_emb_b], dim=-1)


            # Augmented edge features: edge_attr || edge_pnet_embedding
            if do_profile:
                t = now()
                print(f"[FORWARD {self._forward_calls}] edge emb A start", flush=True)
            edge_emb_a = self._edge_emb(graph_a, edge_pts_a, edge_tan_a, edge_idx_a, dev)
            if do_profile:
                print(f"[FORWARD {self._forward_calls}] edge emb A done: {now() - t:.2f}s", flush=True)


            if do_profile:
                t = now()
                print(f"[FORWARD {self._forward_calls}] edge emb B start", flush=True)
            edge_emb_b = self._edge_emb(graph_b, edge_pts_b, edge_tan_b, edge_idx_b, dev)
            if do_profile:
                print(f"[FORWARD {self._forward_calls}] edge emb B done: {now() - t:.2f}s", flush=True)


            aug_ea_a = torch.cat([graph_a.edge_attr, edge_emb_a], dim=-1)
            aug_ea_b = torch.cat([graph_b.edge_attr, edge_emb_b], dim=-1)
        else:
            aug_x_a, aug_x_b = raw_x_a, raw_x_b
            aug_ea_a, aug_ea_b = None, None


        # GATv2 encoding
        if do_profile:
            t = now()
            print(f"[FORWARD {self._forward_calls}] GAT encoder A/B start", flush=True)


        node_a, batch_a, z_a = self.encoder(graph_a, x_in=aug_x_a, edge_attr_in=aug_ea_a)
        node_b, batch_b, z_b = self.encoder(graph_b, x_in=aug_x_b, edge_attr_in=aug_ea_b)


        if do_profile:
            print(f"[FORWARD {self._forward_calls}] GAT encoder A/B done: {now() - t:.2f}s", flush=True)


        B = z_a.size(0)


        # Cross-attention
        if do_profile:
            t = now()
            print(f"[FORWARD {self._forward_calls}] cross attention start", flush=True)


        cross_a, cross_b = self._cross_pool(node_a, batch_a, node_b, batch_b, B)


        if do_profile:
            print(f"[FORWARD {self._forward_calls}] cross attention done: {now() - t:.2f}s", flush=True)


        # Geometric compatibility (uses raw x, pre-augmentation)
        if do_profile:
            t = now()
            print(f"[FORWARD {self._forward_calls}] geo compatibility start", flush=True)


        geo = torch.stack([
            self.geo_compat(raw_x_a[batch_a == i], raw_x_b[batch_b == i])
            for i in range(B)
        ])


        if do_profile:
            print(f"[FORWARD {self._forward_calls}] geo compatibility done: {now() - t:.2f}s", flush=True)


        feat_parts = [z_a, z_b, (z_a - z_b).abs(), z_a * z_b, cross_a, cross_b, geo]


        if self.pair_match is not None:
            if do_profile:
                t = now()
                print(f"[FORWARD {self._forward_calls}] pair match features start", flush=True)


            edge_batch_a = graph_a.batch[graph_a.edge_index[0]]
            edge_batch_b = graph_b.batch[graph_b.edge_index[0]]
            pair_match = torch.stack([
                self.pair_match(
                    raw_x_a[batch_a == i],
                    raw_x_b[batch_b == i],
                    graph_a.edge_attr[edge_batch_a == i],
                    graph_b.edge_attr[edge_batch_b == i],
                )
                for i in range(B)
            ])
            feat_parts.append(pair_match)


            if do_profile:
                print(f"[FORWARD {self._forward_calls}] pair match features done: {now() - t:.2f}s", flush=True)


        feat = torch.cat(feat_parts, dim=-1)


        if do_profile:
            t = now()
            print(f"[FORWARD {self._forward_calls}] classifier start", flush=True)


        logits = self.classifier(feat).squeeze(-1)


        if do_profile:
            print(f"[FORWARD {self._forward_calls}] classifier done: {now() - t:.2f}s", flush=True)
            print(f"[FORWARD {self._forward_calls}] done", flush=True)


        return logits








# ── Metrics ───────────────────────────────────────────────────────────────────




def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(int)
    n_pos  = int((labels == 1).sum())
    n_neg  = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.zeros(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))








def compute_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float = 0.5,
) -> dict[str, float]:
    labels = labels.astype(int)
    preds  = (scores >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    return {
        "accuracy":   float((tp + tn) / max(len(labels), 1)),
        "precision":  float(prec),
        "recall":     float(rec),
        "f1":         float(2.0 * prec * rec / max(prec + rec, 1e-12)),
        "auc":        compute_auc(scores, labels),
        "mean_score": float(scores.mean()),
        "tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn),
    }








@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, threshold: float,
) -> dict[str, float]:
    model.eval()
    all_scores, all_labels = [], []
    for batch in loader:
        logits = model(
            batch["graph_a"].to(device), batch["graph_b"].to(device),
            batch.get("face_pts_a"), batch.get("face_nrm_a"), batch.get("face_msk_a"),
            batch.get("face_pts_b"), batch.get("face_nrm_b"), batch.get("face_msk_b"),
            batch.get("edge_pts_a"), batch.get("edge_tan_a"), batch.get("edge_idx_a"),
            batch.get("edge_pts_b"), batch.get("edge_tan_b"), batch.get("edge_idx_b"),
        )
        all_scores.append(torch.sigmoid(logits).cpu())
        all_labels.append(batch["label"].cpu())
    return compute_metrics(
        torch.cat(all_scores).numpy(),
        torch.cat(all_labels).numpy().astype(int),
        threshold=threshold,
    )








def infer_dims(dataset: PairGraphDataset) -> tuple[int, int]:
    item = dataset[0]
    return int(item["graph_a"].x.shape[1]), int(item["graph_a"].edge_attr.shape[1])








def print_label_summary(name: str, dataset: PairGraphDataset) -> dict[str, int]:
    counts = dataset.label_counts()
    total  = max(counts["total"], 1)
    print(f"{name}: positive={counts['positive']}  negative={counts['negative']}"
          f"  ratio={counts['positive']/total:.3f}")
    return counts








def save_checkpoint(
    path: Path, model: nn.Module, optimizer: Any, epoch: int,
    metrics: dict[str, float], args: argparse.Namespace,
    node_dim: int, edge_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":              metrics,
        "args":                 vars(args),
        "node_dim":             node_dim,
        "edge_dim":             edge_dim,
    }, path)








# ── Main ──────────────────────────────────────────────────────────────────────




def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_index_dir", type=str,   default="outputs/pair_index")
    parser.add_argument("--graph_dir",      type=str,   default="processed/body_graphs_v2_geometry")
    parser.add_argument("--output_dir",     type=str,   default="outputs/gat_v2")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--batch_size",     type=int,   default=64)
    parser.add_argument("--hidden_dim",     type=int,   default=128)
    parser.add_argument("--num_layers",     type=int,   default=2)
    parser.add_argument("--gat_heads",      type=int,   default=8)
    parser.add_argument("--cross_heads",    type=int,   default=8)
    parser.add_argument("--dropout",        type=float, default=0.3)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--threshold",      type=float, default=0.5)
    parser.add_argument("--monitor",        type=str,   default="auc",
                        choices=["auc", "f1", "accuracy"])
    parser.add_argument("--patience",       type=int,   default=20)
    parser.add_argument("--num_workers",    type=int,   default=min(4, os.cpu_count() or 1))
    parser.add_argument("--log_interval",    type=int,   default=20)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--limit_train",    type=int,   default=None)
    parser.add_argument("--limit_val",      type=int,   default=None)
    parser.add_argument("--cpu",            action="store_true")
    parser.add_argument("--use_point_net",  action="store_true",
                        help="Enable MiniPointNet for face & edge point clouds. "
                             "Requires v5_slim dataset (face_points, face_normals, "
                             "face_trimming_mask, edge_entity_points, edge_entity_tangents).")
    parser.add_argument("--use_pair_match_features", action="store_true",
                        help="Add direct face/edge candidate-pair matching features to the classifier.")
    parser.add_argument("--profile_forward", action="store_true",
                        help="Print detailed timing inside model.forward().")
    parser.add_argument("--profile_interval", type=int, default=20,
                        help="Print forward timing every N forward calls when profiling is enabled.")
    parser.add_argument("--no_progress", action="store_true",
                        help="Disable tqdm progress bar.")
    parser.add_argument("--debug_max_train_steps", type=int, default=0,
                        help="Stop each epoch after N train steps. 0 means full epoch.")
    args = parser.parse_args()




    t0         = time.perf_counter()
    started_at = datetime.now()




    def _on_exit():
        elapsed = time.perf_counter() - t0
        print(f"TOTAL_TRAINING_TIME_SEC: {elapsed:.2f}")
        print(f"TOTAL_TRAINING_TIME_MIN: {elapsed / 60.0:.2f}")
        try:
            d = Path(args.output_dir) / f"seed_{args.seed}"
            d.mkdir(parents=True, exist_ok=True)
            save_json(d / "training_time.json", {
                "started_at":  started_at.isoformat(timespec="seconds"),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_sec": round(elapsed, 3),
                "elapsed_min": round(elapsed / 60.0, 3),
            })
        except Exception:
            pass




    atexit.register(_on_exit)
    set_seed(args.seed)




    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)




    train_ds = PairGraphDataset(
        Path(args.pair_index_dir) / "train_pairs.json",
        Path(args.graph_dir), args.limit_train,
    )
    val_ds = PairGraphDataset(
        Path(args.pair_index_dir) / "validation_pairs.json",
        Path(args.graph_dir), args.limit_val,
    )
    node_dim, edge_dim = infer_dims(train_ds)




    loader_kw = dict(
        batch_size=args.batch_size,
        collate_fn=pair_collate,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kw)




    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")




    model = CrossAttentionPairModel(
        node_dim=node_dim, edge_dim=edge_dim,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        gat_heads=args.gat_heads, cross_heads=args.cross_heads,
        dropout=args.dropout, use_point_net=args.use_point_net,
        use_pair_match_features=args.use_pair_match_features,
    ).to(device)
    model.profile_forward = args.profile_forward
    model.profile_interval = args.profile_interval




    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-2,
    )




    save_json(output_dir / "config.json",
              {"args": vars(args), "node_dim": node_dim, "edge_dim": edge_dim})




    print("=" * 70)
    pnet_info = (f"+ FaceMiniPointNet(dim={FACE_PNET_DIM}, mask=trimming_mask)"
                 f" + EdgeMiniPointNet(dim={EDGE_PNET_DIM})"
                 if args.use_point_net else "(no point net)")
    print(f"Siamese GATv2 + Cross-Attention  {pnet_info}")
    print("=" * 70)
    print(f"  device      : {device}")
    print(f"  node_dim    : {node_dim}  edge_dim: {edge_dim}")
    if args.use_point_net:
        print(f"  GAT input   : node={node_dim}+{FACE_PNET_DIM}={node_dim+FACE_PNET_DIM}"
              f"  edge={edge_dim}+{EDGE_PNET_DIM}={edge_dim+EDGE_PNET_DIM}")
    print(f"  hidden_dim  : {args.hidden_dim}  gat_heads={args.gat_heads}  cross_heads={args.cross_heads}")
    print(f"  batch_size  : {args.batch_size}  lr: {args.lr}  dropout: {args.dropout}")
    print(f"  epochs      : {args.epochs}  patience: {args.patience}")
    print(f"  num_workers : {args.num_workers}  profile_forward: {args.profile_forward}")
    print(f"  pair_match  : {args.use_pair_match_features}")
    if args.debug_max_train_steps > 0:
        print(f"  debug steps : {args.debug_max_train_steps} train steps per epoch")
    print()




    train_counts = print_label_summary("train", train_ds)
    val_counts   = print_label_summary("val",   val_ds)
    save_json(output_dir / "label_summary.json",
              {"train": train_counts, "validation": val_counts})




    _pos = train_counts["positive"]
    _neg = train_counts["negative"]
    _pw  = _neg / max(_pos, 1)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([_pw], dtype=torch.float32).to(device)
    )
    print(f"  pos_weight  : {_pw:.2f}  (neg={_neg} / pos={_pos})")
    print()




    log_path   = output_dir / "training_log.csv"
    best_path  = output_dir / "best_model.pt"
    last_path  = output_dir / "last_model.pt"
    fieldnames = [
        "epoch", "train_loss", "lr",
        "val_accuracy", "val_precision", "val_recall", "val_f1", "val_auc",
        "val_mean_score", "val_tp", "val_tn", "val_fp", "val_fn", "is_best",
    ]




    best_score     = -float("inf")
    best_epoch     = -1
    best_metrics: dict[str, float] = {}
    no_improve_cnt = 0




    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()




        for epoch in range(1, args.epochs + 1):
            epoch_t0 = time.perf_counter()
            model.train()
            total_loss, total_n = 0.0, 0
            progress = tqdm(
                train_loader,
                total=len(train_loader),
                desc=f"Epoch {epoch:03d}/{args.epochs}",
                dynamic_ncols=True,
                mininterval=1.0,
                disable=args.no_progress,
            )


            for step, batch in enumerate(progress, 1):
                batch_t0 = time.perf_counter()


                move_t0 = time.perf_counter()
                ga = batch["graph_a"].to(device)
                gb = batch["graph_b"].to(device)
                lb = batch["label"].to(device)
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()
                move_time = time.perf_counter() - move_t0


                optimizer.zero_grad(set_to_none=True)


                forward_t0 = time.perf_counter()
                logits = model(
                    ga, gb,
                    batch.get("face_pts_a"), batch.get("face_nrm_a"), batch.get("face_msk_a"),
                    batch.get("face_pts_b"), batch.get("face_nrm_b"), batch.get("face_msk_b"),
                    batch.get("edge_pts_a"), batch.get("edge_tan_a"), batch.get("edge_idx_a"),
                    batch.get("edge_pts_b"), batch.get("edge_tan_b"), batch.get("edge_idx_b"),
                )
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()
                forward_time = time.perf_counter() - forward_t0


                loss_t0 = time.perf_counter()
                loss = criterion(logits, lb)
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()
                loss_time = time.perf_counter() - loss_t0


                backward_t0 = time.perf_counter()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if torch.cuda.is_available() and device.type == "cuda":
                    torch.cuda.synchronize()
                backward_time = time.perf_counter() - backward_t0


                total_loss += float(loss.item()) * len(lb)
                total_n += len(lb)


                batch_elapsed = time.perf_counter() - batch_t0
                avg_loss = total_loss / max(total_n, 1)


                progress.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{avg_loss:.4f}",
                    "move": f"{move_time:.1f}s",
                    "fwd": f"{forward_time:.1f}s",
                    "bwd": f"{backward_time:.1f}s",
                    "batch": f"{batch_elapsed:.1f}s",
                })


                if step == 1 or (args.log_interval > 0 and step % args.log_interval == 0):
                    print(
                        f"Epoch {epoch:03d} step {step:04d}/{len(train_loader)} | "
                        f"loss={loss.item():.4f} | "
                        f"avg_loss={avg_loss:.4f} | "
                        f"move={move_time:.2f}s | "
                        f"forward={forward_time:.2f}s | "
                        f"loss_time={loss_time:.2f}s | "
                        f"backward={backward_time:.2f}s | "
                        f"batch_time={batch_elapsed:.2f}s | "
                        f"seen={total_n}",
                        flush=True,
                    )


                if args.debug_max_train_steps > 0 and step >= args.debug_max_train_steps:
                    print(
                        f"[DEBUG] Stopping epoch {epoch:03d} after "
                        f"{args.debug_max_train_steps} train steps.",
                        flush=True,
                    )
                    break




            scheduler.step()
            train_loss  = total_loss / max(total_n, 1)
            current_lr  = scheduler.get_last_lr()[0]
            val_m       = evaluate(model, val_loader, device, args.threshold)
            monitor_val = float(val_m[args.monitor])
            is_best     = monitor_val > best_score




            if is_best:
                best_score, best_epoch, best_metrics = monitor_val, epoch, dict(val_m)
                no_improve_cnt = 0
                save_checkpoint(best_path, model, optimizer, epoch, val_m, args, node_dim, edge_dim)
            else:
                no_improve_cnt += 1




            save_checkpoint(last_path, model, optimizer, epoch, val_m, args, node_dim, edge_dim)




            writer.writerow({
                "epoch":          epoch,
                "train_loss":     train_loss,
                "lr":             current_lr,
                "val_accuracy":   val_m["accuracy"],
                "val_precision":  val_m["precision"],
                "val_recall":     val_m["recall"],
                "val_f1":         val_m["f1"],
                "val_auc":        val_m["auc"],
                "val_mean_score": val_m["mean_score"],
                "val_tp":  int(val_m["tp"]), "val_tn": int(val_m["tn"]),
                "val_fp":  int(val_m["fp"]), "val_fn": int(val_m["fn"]),
                "is_best": int(is_best),
            })
            f.flush()




            print(
                f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                f"auc={val_m['auc']:.4f} | acc={val_m['accuracy']:.4f} | "
                f"lr={current_lr:.2e} | best={best_epoch:03d} | "
                f"epoch_time={time.perf_counter() - epoch_t0:.2f}s"
            )




            if args.patience > 0 and no_improve_cnt >= args.patience:
                print(f"[EarlyStopping] No improvement for {args.patience} epochs.")
                break




    save_json(output_dir / "best_metrics.json", {
        "best_epoch":   best_epoch,
        "best_monitor": args.monitor,
        "best_score":   best_score,
        "best_metrics": best_metrics,
        "best_model":   str(best_path),
        "last_model":   str(last_path),
        "log":          str(log_path),
    })




    print()
    print("=" * 70)
    print("TRAINING DONE")
    print("=" * 70)
    print(f"Best epoch  : {best_epoch}")
    print(f"Best {args.monitor:<8}: {best_score:.4f}")
    print(f"Best model  : {best_path}")








if __name__ == "__main__":
    main()






