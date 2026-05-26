
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool


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
    edge_count = int(graph.edge_index.shape[1])

    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        edge_dim = int(graph.edge_attr.shape[1])
    else:
        edge_dim = default_edge_dim

    zero_edge_attr = torch.zeros((edge_count, edge_dim), dtype=torch.float32)

    return Data(
        x=graph.x.float(),
        edge_index=graph.edge_index.long(),
        edge_attr=zero_edge_attr,
    )


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


class PairGraphDataset(Dataset):
    def __init__(self, pair_file: Path, graph_dir: Path, limit: int | None) -> None:
        data = load_json(pair_file)
        records = iter_pair_records(data)

        if limit is not None:
            records = records[:limit]

        if not records:
            raise ValueError(f"No records found: {pair_file}")

        self.records = records
        self.graph_dir = graph_dir

    def __len__(self) -> int:
        return len(self.records)

    def label_counts(self) -> dict[str, int]:
        positive = 0
        negative = 0

        for record in self.records:
            label = int(get_value(record, "label", 0))
            if label == 1:
                positive += 1
            else:
                negative += 1

        return {
            "positive": positive,
            "negative": negative,
            "total": positive + negative,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]

        body_one = str(get_value(record, "body_one"))
        body_two = str(get_value(record, "body_two"))
        label = int(get_value(record, "label", 0))

        graph_a_path = self.graph_dir / f"{body_one}.pt"
        graph_b_path = self.graph_dir / f"{body_two}.pt"

        if not graph_a_path.exists():
            raise FileNotFoundError(f"Missing graph A: {graph_a_path}")

        if not graph_b_path.exists():
            raise FileNotFoundError(f"Missing graph B: {graph_b_path}")

        graph_a = clean_graph(torch_load_graph(graph_a_path))
        graph_b = clean_graph(torch_load_graph(graph_b_path))

        return {
            "graph_a": graph_a,
            "graph_b": graph_b,
            "label": torch.tensor(label, dtype=torch.float32),
        }


def pair_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "graph_a": Batch.from_data_list([item["graph_a"] for item in batch]),
        "graph_b": Batch.from_data_list([item["graph_b"] for item in batch]),
        "label": torch.stack([item["label"] for item in batch]),
    }


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

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for layer_index in range(num_layers):
            in_dim = node_dim if layer_index == 0 else hidden_dim

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

    def forward(self, graph: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return x, batch, z


class CrossAttentionPairModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int,
        gat_heads: int,
        cross_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.encoder = GATv2NodeEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=gat_heads,
            dropout=dropout,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=cross_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_b = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 6, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def make_cross_features(
        self,
        node_a: torch.Tensor,
        batch_a: torch.Tensor,
        node_b: torch.Tensor,
        batch_b: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cross_a_list = []
        cross_b_list = []

        for i in range(batch_size):
            a_i = node_a[batch_a == i].unsqueeze(0)
            b_i = node_b[batch_b == i].unsqueeze(0)

            if a_i.size(1) == 0 or b_i.size(1) == 0:
                raise ValueError("Empty graph found in batch.")

            attended_a, _ = self.cross_attention(
                query=a_i,
                key=b_i,
                value=b_i,
                need_weights=False,
            )

            attended_b, _ = self.cross_attention(
                query=b_i,
                key=a_i,
                value=a_i,
                need_weights=False,
            )

            attended_a = self.norm_a(attended_a + a_i)
            attended_b = self.norm_b(attended_b + b_i)

            cross_a_list.append(attended_a.mean(dim=1).squeeze(0))
            cross_b_list.append(attended_b.mean(dim=1).squeeze(0))

        return torch.stack(cross_a_list), torch.stack(cross_b_list)

    def forward(self, graph_a: Batch, graph_b: Batch) -> torch.Tensor:
        node_a, batch_a, z_a = self.encoder(graph_a)
        node_b, batch_b, z_b = self.encoder(graph_b)

        batch_size = z_a.size(0)

        cross_a, cross_b = self.make_cross_features(
            node_a=node_a,
            batch_a=batch_a,
            node_b=node_b,
            batch_b=batch_b,
            batch_size=batch_size,
        )

        pair_feature = torch.cat(
            [
                z_a,
                z_b,
                torch.abs(z_a - z_b),
                z_a * z_b,
                cross_a,
                cross_b,
            ],
            dim=-1,
        )

        return self.classifier(pair_feature).squeeze(-1)


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(int)
    pos_mask = labels == 1
    neg_mask = labels == 0

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]

    ranks = np.zeros(len(scores), dtype=np.float64)

    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1

        average_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = average_rank
        i = j

    sum_pos_ranks = float(ranks[pos_mask].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_binary_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = labels.astype(int)
    preds = (scores >= threshold).astype(int)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    total = max(len(labels), 1)
    accuracy = float((tp + tn) / total)

    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    auc = compute_auc(scores, labels)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "mean_score": float(scores.mean()),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()

    all_scores = []
    all_labels = []

    for batch in loader:
        graph_a = batch["graph_a"].to(device)
        graph_b = batch["graph_b"].to(device)
        labels = batch["label"].to(device)

        logits = model(graph_a, graph_b)
        scores = torch.sigmoid(logits)

        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    scores_np = torch.cat(all_scores).numpy()
    labels_np = torch.cat(all_labels).numpy().astype(int)

    return compute_binary_metrics(scores_np, labels_np, threshold=threshold)


def infer_dims(dataset: PairGraphDataset) -> tuple[int, int]:
    item = dataset[0]
    graph = item["graph_a"]

    node_dim = int(graph.x.shape[1])
    edge_dim = int(graph.edge_attr.shape[1])

    return node_dim, edge_dim


def print_label_summary(name: str, dataset: PairGraphDataset) -> dict[str, int]:
    counts = dataset.label_counts()
    total = max(counts["total"], 1)
    pos_ratio = counts["positive"] / total
    neg_ratio = counts["negative"] / total
    majority_baseline = max(counts["positive"], counts["negative"]) / total

    print(f"{name} labels")
    print(f"  positive          : {counts['positive']}")
    print(f"  negative          : {counts['negative']}")
    print(f"  positive ratio    : {pos_ratio:.4f}")
    print(f"  negative ratio    : {neg_ratio:.4f}")
    print(f"  majority baseline : {majority_baseline:.4f}")

    return counts


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    args: argparse.Namespace,
    node_dim: int,
    edge_dim: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--pair_index_dir", type=str, default="outputs/pair_index")
    parser.add_argument("--graph_dir", type=str, default="processed/body_graphs")
    parser.add_argument("--output_dir", type=str, default="outputs/gat_cross_attention_full")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--cross_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--monitor", type=str, default="auc", choices=["auc", "f1", "accuracy"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    pair_index_dir = Path(args.pair_index_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir) / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PairGraphDataset(
        pair_file=pair_index_dir / "train_pairs.json",
        graph_dir=graph_dir,
        limit=args.limit_train,
    )

    val_dataset = PairGraphDataset(
        pair_file=pair_index_dir / "validation_pairs.json",
        graph_dir=graph_dir,
        limit=args.limit_val,
    )

    node_dim, edge_dim = infer_dims(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=pair_collate,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=pair_collate,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    model = CrossAttentionPairModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gat_heads=args.gat_heads,
        cross_heads=args.cross_heads,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    save_json(
        output_dir / "config.json",
        {
            "args": vars(args),
            "node_dim": node_dim,
            "edge_dim": edge_dim,
        },
    )

    log_path = output_dir / "training_log.csv"
    best_path = output_dir / "best_model.pt"
    last_path = output_dir / "last_model.pt"

    print("=" * 80)
    print("CROSS-ATTENTION FULL TRAINING")
    print("=" * 80)
    print(f"Device        : {device}")
    print(f"Output dir    : {output_dir}")
    print(f"Train samples : {len(train_dataset)}")
    print(f"Val samples   : {len(val_dataset)}")
    print(f"Node dim      : {node_dim}")
    print(f"Edge dim      : {edge_dim}")
    print(f"Hidden dim    : {args.hidden_dim}")
    print(f"GAT heads     : {args.gat_heads}")
    print(f"Cross heads   : {args.cross_heads}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Epochs        : {args.epochs}")
    print(f"Monitor       : {args.monitor}")
    print("")

    train_counts = print_label_summary("Train", train_dataset)
    val_counts = print_label_summary("Validation", val_dataset)
    print("")

    save_json(
        output_dir / "label_summary.json",
        {
            "train": train_counts,
            "validation": val_counts,
        },
    )

    fieldnames = [
        "epoch",
        "train_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auc",
        "val_mean_score",
        "val_tp",
        "val_tn",
        "val_fp",
        "val_fn",
        "is_best",
    ]

    best_score = -float("inf")
    best_epoch = -1
    best_metrics: dict[str, float] = {}

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            total_count = 0

            for step, batch in enumerate(train_loader, start=1):
                graph_a = batch["graph_a"].to(device)
                graph_b = batch["graph_b"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                logits = model(graph_a, graph_b)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * len(labels)
                total_count += len(labels)

                if step == 1:
                    print(
                        f"Epoch {epoch:03d} first batch OK | "
                        f"logits shape={tuple(logits.shape)} | "
                        f"loss={loss.item():.4f}"
                    )

            train_loss = total_loss / max(total_count, 1)
            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                threshold=args.threshold,
            )

            monitor_score = float(val_metrics[args.monitor])
            is_best = monitor_score > best_score

            if is_best:
                best_score = monitor_score
                best_epoch = epoch
                best_metrics = dict(val_metrics)

                save_checkpoint(
                    path=best_path,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    args=args,
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                )

            save_checkpoint(
                path=last_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                args=args,
                node_dim=node_dim,
                edge_dim=edge_dim,
            )

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_auc": val_metrics["auc"],
                "val_mean_score": val_metrics["mean_score"],
                "val_tp": int(val_metrics["tp"]),
                "val_tn": int(val_metrics["tn"]),
                "val_fp": int(val_metrics["fp"]),
                "val_fn": int(val_metrics["fn"]),
                "is_best": int(is_best),
            }

            writer.writerow(row)
            f.flush()

            print(
                f"Epoch {epoch:03d} done | "
                f"train_loss={train_loss:.4f} | "
                f"val_acc={val_metrics['accuracy']:.4f} | "
                f"val_prec={val_metrics['precision']:.4f} | "
                f"val_rec={val_metrics['recall']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | "
                f"val_mean_score={val_metrics['mean_score']:.4f} | "
                f"best_epoch={best_epoch:03d}"
            )

    save_json(
        output_dir / "best_metrics.json",
        {
            "best_epoch": best_epoch,
            "best_monitor": args.monitor,
            "best_score": best_score,
            "best_metrics": best_metrics,
            "best_model_path": str(best_path),
            "last_model_path": str(last_path),
            "log_path": str(log_path),
        },
    )

    print("")
    print("=" * 80)
    print("FULL TRAINING DONE")
    print("=" * 80)
    print(f"Best epoch      : {best_epoch}")
    print(f"Best {args.monitor:<8}: {best_score:.4f}")
    print(f"Best model path : {best_path}")
    print(f"Last model path : {last_path}")
    print(f"CSV log path    : {log_path}")


if __name__ == "__main__":
    main()
