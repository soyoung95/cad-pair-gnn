
from __future__ import annotations

import argparse
import json
import math
from itertools import islice
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool


ID_COLUMNS = ["assembly_key", "pair_key", "body_one", "body_two"]
LABEL_COLUMN = "label"


class GATEncoder(nn.Module):
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

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for layer_index in range(num_layers):
            in_channels = node_dim if layer_index == 0 else hidden_dim

            conv = GATv2Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=True,
                edge_dim=edge_dim,
                dropout=dropout,
            )

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(self, graph: Batch) -> torch.Tensor:
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        batch = graph.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)

        z = global_mean_pool(x, batch)
        return z


class SiameseGATPairClassifier(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        mlp_hidden_dim: int,
    ) -> None:
        super().__init__()

        self.encoder = GATEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

        pair_dim = hidden_dim * 4

        self.classifier = nn.Sequential(
            nn.Linear(pair_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def make_pair_feature(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                z_a,
                z_b,
                torch.abs(z_a - z_b),
                z_a * z_b,
            ],
            dim=-1,
        )

    def forward(self, graph_a: Batch, graph_b: Batch) -> torch.Tensor:
        z_a = self.encoder(graph_a)
        z_b = self.encoder(graph_b)
        pair_feature = self.make_pair_feature(z_a, z_b)
        return self.classifier(pair_feature).squeeze(-1)

    def score_pair_feature(self, pair_feature: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(pair_feature).squeeze(-1)
        return torch.sigmoid(logits)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def torch_load_graph(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")

    if "config" not in checkpoint:
        raise ValueError("Checkpoint does not contain config.")

    return checkpoint


def infer_dims_from_state_dict(state_dict: dict[str, Any]) -> tuple[int, int]:
    node_dim = int(state_dict["encoder.convs.0.lin_l.weight"].shape[1])
    edge_dim = int(state_dict["encoder.convs.0.lin_edge.weight"].shape[1])
    return node_dim, edge_dim


def build_model_from_checkpoint(checkpoint: dict[str, Any]) -> SiameseGATPairClassifier:
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    node_dim, edge_dim = infer_dims_from_state_dict(state_dict)

    model = SiameseGATPairClassifier(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        heads=int(config["heads"]),
        dropout=float(config["dropout"]),
        mlp_hidden_dim=int(config["mlp_hidden_dim"]),
    )

    model.load_state_dict(state_dict)
    return model


def iter_batches(items: list[str], batch_size: int) -> list[list[str]]:
    batches = []
    for start in range(0, len(items), batch_size):
        batches.append(items[start:start + batch_size])
    return batches


def collect_body_ids(pair_records: list[dict[str, Any]]) -> list[str]:
    body_ids = set()

    for record in pair_records:
        body_ids.add(str(record["body_one"]))
        body_ids.add(str(record["body_two"]))

    return sorted(body_ids)


def load_pair_records(pair_file: Path, limit: int | None) -> list[dict[str, Any]]:
    records = load_json(pair_file)

    if not isinstance(records, list):
        raise ValueError(f"Pair file must contain a list: {pair_file}")

    if limit is not None:
        records = list(islice(records, limit))

    return records


@torch.no_grad()
def compute_body_embeddings(
    model: SiameseGATPairClassifier,
    body_ids: list[str],
    graph_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    model.eval()

    embeddings: dict[str, torch.Tensor] = {}

    total_batches = math.ceil(len(body_ids) / batch_size)

    for batch_index, batch_body_ids in enumerate(iter_batches(body_ids, batch_size), start=1):
        graphs = []

        for body_id in batch_body_ids:
            graph_path = graph_dir / f"{body_id}.pt"

            if not graph_path.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_path}")

            graph = torch_load_graph(graph_path)
            graphs.append(graph)

        batch_graph = Batch.from_data_list(graphs).to(device)
        z = model.encoder(batch_graph).detach().cpu()

        for body_id, embedding in zip(batch_body_ids, z):
            embeddings[body_id] = embedding

        print(
            f"[BODY] batch {batch_index}/{total_batches} | "
            f"encoded={len(embeddings)}/{len(body_ids)}"
        )

    return embeddings


@torch.no_grad()
def build_pair_embedding_dataframe(
    model: SiameseGATPairClassifier,
    pair_records: list[dict[str, Any]],
    embeddings: dict[str, torch.Tensor],
    device: torch.device,
    pair_batch_size: int,
) -> pd.DataFrame:
    model.eval()

    rows: list[dict[str, Any]] = []
    total_batches = math.ceil(len(pair_records) / pair_batch_size)

    for batch_index in range(total_batches):
        start = batch_index * pair_batch_size
        end = min(start + pair_batch_size, len(pair_records))
        batch_records = pair_records[start:end]

        z_a = torch.stack(
            [embeddings[str(record["body_one"])] for record in batch_records],
            dim=0,
        ).to(device)

        z_b = torch.stack(
            [embeddings[str(record["body_two"])] for record in batch_records],
            dim=0,
        ).to(device)

        pair_feature = model.make_pair_feature(z_a, z_b)
        scores = model.score_pair_feature(pair_feature)

        pair_feature_np = pair_feature.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        for row_index, record in enumerate(batch_records):
            row: dict[str, Any] = {
                "assembly_key": str(record.get("assembly_key", "")),
                "pair_key": str(record.get("pair_key", "")),
                "body_one": str(record["body_one"]),
                "body_two": str(record["body_two"]),
                "label": int(record.get("label", 0)),
                "gnn_score": float(scores_np[row_index]),
            }

            for feature_index in range(pair_feature_np.shape[1]):
                row[f"gnn_pair_{feature_index:03d}"] = float(pair_feature_np[row_index, feature_index])

            rows.append(row)

        print(
            f"[PAIR] batch {batch_index + 1}/{total_batches} | "
            f"rows={len(rows)}/{len(pair_records)}"
        )

    return pd.DataFrame(rows)


def process_split(
    split_name: str,
    pair_file: Path,
    model: SiameseGATPairClassifier,
    graph_dir: Path,
    output_root: Path,
    device: torch.device,
    graph_batch_size: int,
    pair_batch_size: int,
    limit: int | None,
) -> dict[str, Any]:
    print("=" * 80)
    print(f"PROCESS SPLIT: {split_name}")
    print("=" * 80)
    print(f"Pair file : {pair_file}")

    pair_records = load_pair_records(pair_file, limit=limit)
    body_ids = collect_body_ids(pair_records)

    print(f"Pairs     : {len(pair_records)}")
    print(f"Body IDs  : {len(body_ids)}")

    embeddings = compute_body_embeddings(
        model=model,
        body_ids=body_ids,
        graph_dir=graph_dir,
        device=device,
        batch_size=graph_batch_size,
    )

    split_df = build_pair_embedding_dataframe(
        model=model,
        pair_records=pair_records,
        embeddings=embeddings,
        device=device,
        pair_batch_size=pair_batch_size,
    )

    output_csv = output_root / f"{split_name}_gnn_pair_embeddings.csv"
    split_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    label_counts = split_df["label"].value_counts().sort_index().to_dict()

    print(f"Saved CSV : {output_csv}")
    print(f"Shape     : {split_df.shape}")
    print(f"Labels    : {label_counts}")

    return {
        "split": split_name,
        "pair_file": str(pair_file),
        "output_csv": str(output_csv),
        "rows": int(len(split_df)),
        "columns": int(len(split_df.columns)),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "unique_body_ids": int(len(body_ids)),
    }


def parse_split_names(split_text: str) -> list[str]:
    return [item.strip() for item in split_text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pair_dir", type=str, default="outputs/pair_index")
    parser.add_argument("--graph_dir", type=str, default="processed/body_graphs")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--splits", type=str, default="train,validation,test")
    parser.add_argument("--graph_batch_size", type=int, default=256)
    parser.add_argument("--pair_batch_size", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    pair_dir = Path(args.pair_dir)
    graph_dir = Path(args.graph_dir)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()

    config = checkpoint["config"]
    split_names = parse_split_names(args.splits)

    pair_files = {
        "train": pair_dir / "train_pairs.json",
        "validation": pair_dir / "validation_pairs.json",
        "test": pair_dir / "test_pairs.json",
    }

    print("=" * 80)
    print("EXPORT GNN PAIR EMBEDDINGS")
    print("=" * 80)
    print(f"Checkpoint       : {checkpoint_path}")
    print(f"Pair dir         : {pair_dir}")
    print(f"Graph dir        : {graph_dir}")
    print(f"Output root      : {output_root}")
    print(f"Device           : {device}")
    print(f"Checkpoint config: {config}")
    print("")

    split_reports = []

    for split_name in split_names:
        if split_name not in pair_files:
            raise ValueError(f"Unknown split name: {split_name}")

        report = process_split(
            split_name=split_name,
            pair_file=pair_files[split_name],
            model=model,
            graph_dir=graph_dir,
            output_root=output_root,
            device=device,
            graph_batch_size=args.graph_batch_size,
            pair_batch_size=args.pair_batch_size,
            limit=args.limit,
        )

        split_reports.append(report)

    manifest = {
        "checkpoint": str(checkpoint_path),
        "pair_dir": str(pair_dir),
        "graph_dir": str(graph_dir),
        "output_root": str(output_root),
        "checkpoint_config": config,
        "splits": split_reports,
        "feature_prefix": "gnn_pair_",
        "feature_count": 256,
    }

    save_json(output_root / "manifest.json", manifest)

    readme_lines = [
        "# GNN Pair Embedding Folder",
        "",
        "## Purpose",
        "",
        "This folder contains frozen pair-level embeddings from a trained Siamese GATv2 model.",
        "",
        "## Pair representation",
        "",
        "`concat(z_a, z_b, abs(z_a - z_b), z_a * z_b)`",
        "",
        "## Files",
        "",
    ]

    for report in split_reports:
        readme_lines.append(f"- `{Path(report['output_csv']).name}`")

    readme_lines.extend(
        [
            "- `manifest.json`",
            "",
            "## Notes",
            "",
            "- These features are extracted from a fixed checkpoint.",
            "- They can be combined with axis-pair features in a fusion experiment.",
        ]
    )

    (output_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print("")
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Output root : {output_root}")
    print(f"Manifest    : {output_root / 'manifest.json'}")
    print(f"README      : {output_root / 'README.md'}")


if __name__ == "__main__":
    main()
