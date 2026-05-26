
import argparse
import json
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)
from tqdm import tqdm


FACE_FEATURE_DIM = 12
EDGE_FEATURE_DIM = 16


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def torch_load_graph(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


class PairGraphDataset(Dataset):
    def __init__(self, pair_json_path, graph_dir, limit=None):
        self.pair_json_path = Path(pair_json_path)
        self.graph_dir = Path(graph_dir)
        self.samples = load_json(self.pair_json_path)

        if limit is not None:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def get_graph_path(self, body_id):
        return self.graph_dir / f"{body_id}.pt"

    def __getitem__(self, index):
        sample = self.samples[index]

        body_one = sample["body_one"]
        body_two = sample["body_two"]
        label = float(sample["label"])

        graph_one = torch_load_graph(self.get_graph_path(body_one))
        graph_two = torch_load_graph(self.get_graph_path(body_two))

        graph_one.body_id = body_one
        graph_two.body_id = body_two

        y = torch.tensor([label], dtype=torch.float32)

        return graph_one, graph_two, y


class GATEncoder(nn.Module):
    def __init__(
        self,
        in_channels=FACE_FEATURE_DIM,
        edge_dim=EDGE_FEATURE_DIM,
        hidden_dim=64,
        num_layers=2,
        heads=4,
        dropout=0.2,
    ):
        super().__init__()

        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads.")

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_dim = in_channels
        out_channels_per_head = hidden_dim // heads

        for _ in range(num_layers):
            conv = GATv2Conv(
                in_channels=current_dim,
                out_channels=out_channels_per_head,
                heads=heads,
                concat=True,
                edge_dim=edge_dim,
                dropout=dropout,
                add_self_loops=True,
                fill_value=0.0,
            )

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
            current_dim = hidden_dim

        self.activation = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph):
        x = graph.x
        edge_index = graph.edge_index
        edge_attr = graph.edge_attr
        batch = graph.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

        z = global_mean_pool(x, batch)

        return z


class SiameseGATPairClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        num_layers=2,
        heads=4,
        dropout=0.2,
        mlp_hidden_dim=128,
    ):
        super().__init__()

        self.encoder = GATEncoder(
            in_channels=FACE_FEATURE_DIM,
            edge_dim=EDGE_FEATURE_DIM,
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
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, 1),
        )

    def forward(self, graph_a, graph_b):
        z_a = self.encoder(graph_a)
        z_b = self.encoder(graph_b)

        pair_feature = torch.cat(
            [
                z_a,
                z_b,
                torch.abs(z_a - z_b),
                z_a * z_b,
            ],
            dim=1,
        )

        return self.classifier(pair_feature)


def find_best_threshold(y_true, y_prob):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    best_threshold = 0.5
    best_f1 = 0.0

    for i, threshold in enumerate(thresholds):
        p = precision[i]
        r = recall[i]

        if p + r == 0:
            f1 = 0.0
        else:
            f1 = 2 * p * r / (p + r)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)

    return best_threshold, float(best_f1)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if len(np.unique(y_true)) == 2:
        result["auroc"] = float(roc_auc_score(y_true, y_prob))
        result["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        result["auroc"] = None
        result["auprc"] = None

    best_threshold, best_f1 = find_best_threshold(y_true, y_prob)
    result["best_threshold"] = best_threshold
    result["best_threshold_f1"] = best_f1

    y_pred_best = (y_prob >= best_threshold).astype(int)
    tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_true, y_pred_best, labels=[0, 1]).ravel()

    result["best_threshold_accuracy"] = float(accuracy_score(y_true, y_pred_best))
    result["best_threshold_precision"] = float(precision_score(y_true, y_pred_best, zero_division=0))
    result["best_threshold_recall"] = float(recall_score(y_true, y_pred_best, zero_division=0))
    result["best_threshold_tn"] = int(tn_b)
    result["best_threshold_fp"] = int(fp_b)
    result["best_threshold_fn"] = int(fn_b)
    result["best_threshold_tp"] = int(tp_b)

    return result


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probs = []

    progress = tqdm(loader, desc=f"Train epoch {epoch}", leave=False)

    for graph_a, graph_b, y in progress:
        graph_a = graph_a.to(device)
        graph_b = graph_b.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(graph_a, graph_b)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = torch.sigmoid(logits).detach().cpu().view(-1).numpy()
        labels = y.detach().cpu().view(-1).numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

        progress.set_postfix(loss=loss.item())

    metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
    metrics["loss"] = float(total_loss / max(total_samples, 1))

    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device, split_name):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probs = []

    progress = tqdm(loader, desc=f"Eval {split_name}", leave=False)

    for graph_a, graph_b, y in progress:
        graph_a = graph_a.to(device)
        graph_b = graph_b.to(device)
        y = y.to(device)

        logits = model(graph_a, graph_b)
        loss = criterion(logits, y)

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        probs = torch.sigmoid(logits).detach().cpu().view(-1).numpy()
        labels = y.detach().cpu().view(-1).numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    metrics = compute_metrics(all_labels, all_probs, threshold=0.5)
    metrics["loss"] = float(total_loss / max(total_samples, 1))

    return metrics


def make_dataloader(pair_path, graph_dir, batch_size, shuffle, num_workers, limit=None):
    dataset = PairGraphDataset(
        pair_json_path=pair_path,
        graph_dir=graph_dir,
        limit=limit,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, loader


def save_markdown_report(history, args, output_path, best_summary):
    lines = []

    lines.append("# GAT Pair Training Report V2")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Experiment name: `{args.exp_name}`")
    lines.append(f"- Batch size: {args.batch_size}")
    lines.append(f"- Epochs: {args.epochs}")
    lines.append(f"- Learning rate: {args.lr}")
    lines.append(f"- Hidden dim: {args.hidden_dim}")
    lines.append(f"- Num layers: {args.num_layers}")
    lines.append(f"- Heads: {args.heads}")
    lines.append(f"- Dropout: {args.dropout}")
    lines.append(f"- Device: `{args.device}`")
    lines.append(f"- Best checkpoint metric: `validation AUROC`")
    lines.append("")

    lines.append("## Best Checkpoint Summary")
    lines.append("")
    lines.append(f"- Best epoch: {best_summary['best_epoch']}")
    lines.append(f"- Best validation AUROC: {best_summary['best_val_auroc']:.6f}")
    lines.append(f"- Best model path: `{best_summary['best_model_path']}`")
    lines.append("")

    if best_summary.get("best_test_metrics") is not None:
        m = best_summary["best_test_metrics"]
        lines.append("### Test Metrics at Best Validation AUROC")
        lines.append("")
        lines.append(f"- Loss: {m['loss']:.6f}")
        lines.append(f"- Accuracy: {m['accuracy']:.6f}")
        lines.append(f"- Precision: {m['precision']:.6f}")
        lines.append(f"- Recall: {m['recall']:.6f}")
        lines.append(f"- F1: {m['f1']:.6f}")
        lines.append(f"- AUROC: {m['auroc']:.6f}")
        lines.append(f"- AUPRC: {m['auprc']:.6f}")
        lines.append(f"- Confusion matrix at threshold 0.5: TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}")
        lines.append(f"- Best threshold: {m['best_threshold']:.6f}")
        lines.append(f"- Best-threshold F1: {m['best_threshold_f1']:.6f}")
        lines.append(f"- Confusion matrix at best threshold: TN={m['best_threshold_tn']}, FP={m['best_threshold_fp']}, FN={m['best_threshold_fn']}, TP={m['best_threshold_tp']}")
        lines.append("")

    lines.append("## Epoch Metrics")
    lines.append("")
    lines.append("| Epoch | Split | Loss | Acc | Precision | Recall | F1 | AUROC | AUPRC | Best Thr | Best Thr F1 |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for item in history:
        epoch = item["epoch"]

        for split in ["train", "validation", "test"]:
            m = item[split]

            auroc = "None" if m["auroc"] is None else f"{m['auroc']:.4f}"
            auprc = "None" if m["auprc"] is None else f"{m['auprc']:.4f}"

            lines.append(
                f"| {epoch} | {split} | "
                f"{m['loss']:.4f} | "
                f"{m['accuracy']:.4f} | "
                f"{m['precision']:.4f} | "
                f"{m['recall']:.4f} | "
                f"{m['f1']:.4f} | "
                f"{auroc} | "
                f"{auprc} | "
                f"{m['best_threshold']:.4f} | "
                f"{m['best_threshold_f1']:.4f} |"
            )

    lines.append("")
    lines.append("## Confusion Matrix at Threshold 0.5")
    lines.append("")
    lines.append("| Epoch | Split | TN | FP | FN | TP |")
    lines.append("|---:|---|---:|---:|---:|---:|")

    for item in history:
        epoch = item["epoch"]

        for split in ["train", "validation", "test"]:
            m = item[split]
            lines.append(
                f"| {epoch} | {split} | {m['tn']} | {m['fp']} | {m['fn']} | {m['tp']} |"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pair_index_dir", type=str, default=r"C:\so_zero\cad_pair_gnn\outputs\pair_index")
    parser.add_argument("--graph_dir", type=str, default=r"C:\so_zero\cad_pair_gnn\processed\body_graphs")
    parser.add_argument("--output_dir", type=str, default=r"C:\so_zero\cad_pair_gnn\outputs\runs")

    parser.add_argument("--exp_name", type=str, default="gat_pair_v2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mlp_hidden_dim", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--limit_test", type=int, default=None)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.exp_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    pair_index_dir = Path(args.pair_index_dir)
    graph_dir = Path(args.graph_dir)

    train_pair_path = pair_index_dir / "train_pairs.json"
    val_pair_path = pair_index_dir / "validation_pairs.json"
    test_pair_path = pair_index_dir / "test_pairs.json"

    print("=" * 80)
    print("[STEP] Preparing datasets")
    print("=" * 80)

    train_dataset, train_loader = make_dataloader(
        pair_path=train_pair_path,
        graph_dir=graph_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        limit=args.limit_train,
    )

    val_dataset, val_loader = make_dataloader(
        pair_path=val_pair_path,
        graph_dir=graph_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        limit=args.limit_val,
    )

    test_dataset, test_loader = make_dataloader(
        pair_path=test_pair_path,
        graph_dir=graph_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        limit=args.limit_test,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Device: {device}")

    print("\n" + "=" * 80)
    print("[STEP] Building model")
    print("=" * 80)

    model = SiameseGATPairClassifier(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        mlp_hidden_dim=args.mlp_hidden_dim,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f"Trainable parameters: {num_params}")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = vars(args).copy()
    config["run_dir"] = str(run_dir)
    config["train_samples"] = len(train_dataset)
    config["validation_samples"] = len(val_dataset)
    config["test_samples"] = len(test_dataset)
    config["trainable_parameters"] = num_params
    config["best_checkpoint_metric"] = "validation_auroc"

    save_json(run_dir / "config.json", config)

    history = []

    best_val_auroc = -1.0
    best_epoch = None
    best_test_metrics = None
    best_model_path = run_dir / "best_model_by_val_auroc.pt"

    print("\n" + "=" * 80)
    print("[STEP] Training")
    print("=" * 80)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            split_name="validation",
        )

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            split_name="test",
        )

        epoch_result = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": val_metrics,
            "test": test_metrics,
        }

        history.append(epoch_result)

        val_auroc = val_metrics["auroc"]

        if val_auroc is not None and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            best_test_metrics = test_metrics

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_auroc": best_val_auroc,
                "config": config,
                "validation_metrics": val_metrics,
                "test_metrics": test_metrics,
            }

            torch.save(checkpoint, best_model_path)

        print("\n" + "-" * 80)
        print(f"Epoch {epoch}")
        print(
            f"Train      loss={train_metrics['loss']:.4f}, "
            f"acc={train_metrics['accuracy']:.4f}, "
            f"f1={train_metrics['f1']:.4f}, "
            f"auroc={train_metrics['auroc']:.4f}, "
            f"auprc={train_metrics['auprc']:.4f}"
        )
        print(
            f"Validation loss={val_metrics['loss']:.4f}, "
            f"acc={val_metrics['accuracy']:.4f}, "
            f"f1={val_metrics['f1']:.4f}, "
            f"auroc={val_metrics['auroc']:.4f}, "
            f"auprc={val_metrics['auprc']:.4f}"
        )
        print(
            f"Test       loss={test_metrics['loss']:.4f}, "
            f"acc={test_metrics['accuracy']:.4f}, "
            f"f1={test_metrics['f1']:.4f}, "
            f"auroc={test_metrics['auroc']:.4f}, "
            f"auprc={test_metrics['auprc']:.4f}"
        )
        print(f"Best validation AUROC so far: {best_val_auroc:.4f} at epoch {best_epoch}")

    best_summary = {
        "best_epoch": best_epoch,
        "best_val_auroc": best_val_auroc,
        "best_model_path": str(best_model_path),
        "best_test_metrics": best_test_metrics,
    }

    save_json(run_dir / "history.json", history)
    save_json(run_dir / "best_summary.json", best_summary)
    save_markdown_report(
        history=history,
        args=args,
        output_path=run_dir / "training_report_v2.md",
        best_summary=best_summary,
    )

    print("\n" + "=" * 80)
    print("[DONE] Training completed")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation AUROC: {best_val_auroc:.4f}")
    print(f"Training report: {run_dir / 'training_report_v2.md'}")
    print(f"Best model: {best_model_path}")


if __name__ == "__main__":
    main()
