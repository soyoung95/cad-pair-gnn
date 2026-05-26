


from pathlib import Path
import argparse
import csv
import importlib.util
import json
import sys
from typing import Any


import numpy as np
import torch
from torch.utils.data import DataLoader




try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass




def import_training_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("train_gat_cross_attention_pair_full", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import training script: {script_path}")


    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)


    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, but got: {type(checkpoint)}")


    return checkpoint




def get_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    candidate_keys = [
        "model_state_dict",
        "state_dict",
        "model",
    ]


    for key in candidate_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value


    tensor_like_count = sum(torch.is_tensor(value) for value in checkpoint.values())
    if tensor_like_count > 0:
        return checkpoint


    raise KeyError(f"Could not find model state dict. Checkpoint keys: {list(checkpoint.keys())}")




def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))




def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(int)
    scores = scores.astype(float)


    pos = scores[labels == 1]
    neg = scores[labels == 0]


    if len(pos) == 0 or len(neg) == 0:
        return float("nan")


    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)


    pos_ranks = ranks[labels == 1]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)




def compute_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(int)
    preds = (scores >= threshold).astype(int)


    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())


    total = len(labels)
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    auc = compute_auc(scores=scores, labels=labels)


    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "mean_score": float(scores.mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": int(total),
        "positive": int((labels == 1).sum()),
        "negative": int((labels == 0).sum()),
    }




def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}


    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value


    return moved




def get_batch_item(batch: dict[str, Any], candidates: list[str]):
    for key in candidates:
        if key in batch:
            return batch[key]
    raise KeyError(f"None of the candidate keys exist: {candidates}. Available keys: {list(batch.keys())}")




@torch.no_grad()
def evaluate_test(model, loader, device: torch.device):
    model.eval()


    all_scores = []
    all_labels = []
    all_rows = []


    for batch_idx, batch in enumerate(loader):
        batch = move_batch_to_device(batch, device)


        graph_a = get_batch_item(batch, ["graph_a", "a", "body_one_graph", "graph_one"])
        graph_b = get_batch_item(batch, ["graph_b", "b", "body_two_graph", "graph_two"])
        labels = get_batch_item(batch, ["label", "labels", "y"])


        logits = model(graph_a, graph_b)


        if logits.ndim > 1:
            logits = logits.view(-1)


        labels = labels.view(-1).float()


        scores = torch.sigmoid(logits).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy().astype(int)


        all_scores.append(scores)
        all_labels.append(labels_np)


        batch_size = len(labels_np)
        for i in range(batch_size):
            row = {
                "batch_idx": batch_idx,
                "row_idx": i,
                "label": int(labels_np[i]),
                "score": float(scores[i]),
                "pred_05": int(scores[i] >= 0.5),
            }


            for key in ["assembly_key", "pair_key", "body_one", "body_two"]:
                value = batch.get(key)
                if value is not None:
                    if isinstance(value, list):
                        row[key] = value[i]
                    elif isinstance(value, tuple):
                        row[key] = value[i]
                    else:
                        row[key] = str(value)


            all_rows.append(row)


    scores_np = np.concatenate(all_scores, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)


    return scores_np, labels_np, all_rows




def save_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)


    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_script", default="models/train_gat_cross_attention_pair_full.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pair_index_dir", default="outputs/pair_index")
    parser.add_argument("--graph_dir", default="processed/body_graphs")
    parser.add_argument("--output_dir", required=True)


    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)


    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gat_heads", type=int, default=8)
    parser.add_argument("--cross_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)


    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()


    training_script = Path(args.training_script)
    checkpoint_path = Path(args.checkpoint)
    pair_index_dir = Path(args.pair_index_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)


    print("=" * 100)
    print("CROSS-ATTENTION TEST SET EVALUATION")
    print("=" * 100)
    print(f"Training script : {training_script}")
    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Pair index dir  : {pair_index_dir}")
    print(f"Graph dir       : {graph_dir}")
    print(f"Output dir      : {output_dir}")
    print(f"Threshold       : {args.threshold}")


    if not training_script.exists():
        raise FileNotFoundError(f"Training script not found: {training_script}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not (pair_index_dir / "test_pairs.json").exists():
        raise FileNotFoundError(f"Test pair file not found: {pair_index_dir / 'test_pairs.json'}")
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph dir not found: {graph_dir}")


    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device          : {device}")


    train_module = import_training_module(training_script)


    test_dataset = train_module.PairGraphDataset(
        pair_file=pair_index_dir / "test_pairs.json",
        graph_dir=graph_dir,
        limit=None,
    )


    node_dim, edge_dim = train_module.infer_dims(test_dataset)


    print(f"Test samples    : {len(test_dataset)}")
    print(f"Node dim        : {node_dim}")
    print(f"Edge dim        : {edge_dim}")


    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=train_module.pair_collate,
    )


    model = train_module.CrossAttentionPairModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        gat_heads=args.gat_heads,
        cross_heads=args.cross_heads,
        dropout=args.dropout,
    ).to(device)


    checkpoint = load_checkpoint(checkpoint_path, device=device)
    state_dict = get_state_dict(checkpoint)
    model.load_state_dict(state_dict)


    scores, labels, pred_rows = evaluate_test(
        model=model,
        loader=test_loader,
        device=device,
    )


    metrics = compute_metrics(
        scores=scores,
        labels=labels,
        threshold=args.threshold,
    )


    output_dir.mkdir(parents=True, exist_ok=True)


    metrics_path = output_dir / "test_metrics.json"
    predictions_path = output_dir / "test_predictions.csv"


    save_json(metrics_path, metrics)
    save_predictions(predictions_path, pred_rows)


    print()
    print("=" * 100)
    print("TEST RESULTS")
    print("=" * 100)
    for key, value in metrics.items():
        print(f"{key}: {value}")


    print()
    print(f"Saved metrics     : {metrics_path}")
    print(f"Saved predictions : {predictions_path}")




if __name__ == "__main__":
    main()


