
from pathlib import Path
import argparse
import json
import sys

import pandas as pd


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def flatten_best_metrics(best_metrics: dict) -> dict:
    row = {}

    row["best_epoch"] = best_metrics.get("best_epoch")
    row["best_monitor"] = best_metrics.get("best_monitor")
    row["best_score"] = best_metrics.get("best_score")
    row["best_model_path"] = best_metrics.get("best_model_path")
    row["last_model_path"] = best_metrics.get("last_model_path")
    row["log_path"] = best_metrics.get("log_path")

    metrics = best_metrics.get("best_metrics", {})
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            row[f"best_{key}"] = value

    return row


def flatten_config(config: dict) -> dict:
    row = {}

    args = config.get("args", {})
    if isinstance(args, dict):
        for key, value in args.items():
            row[key] = value

    if "node_dim" in config:
        row["node_dim"] = config["node_dim"]
    if "edge_dim" in config:
        row["edge_dim"] = config["edge_dim"]

    return row


def summarize_training_log(path: Path) -> dict:
    row = {
        "training_log_exists": path.exists(),
    }

    if not path.exists():
        return row

    df = pd.read_csv(path, encoding="utf-8-sig")
    row["training_log_rows"] = len(df)

    if len(df) == 0:
        return row

    metric_columns = [
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auc",
        "val_mean_score",
    ]

    for col in metric_columns:
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        best_idx = values.idxmax()
        best_row = df.loc[best_idx]

        row[f"log_best_{col}"] = float(values.max())

        if "epoch" in df.columns:
            row[f"log_best_{col}_epoch"] = int(best_row["epoch"])

        if "val_accuracy" in df.columns:
            row[f"log_best_{col}_val_accuracy"] = float(best_row["val_accuracy"])
        if "val_f1" in df.columns:
            row[f"log_best_{col}_val_f1"] = float(best_row["val_f1"])
        if "val_auc" in df.columns:
            row[f"log_best_{col}_val_auc"] = float(best_row["val_auc"])

    last_row = df.iloc[-1]
    if "epoch" in df.columns:
        row["last_epoch"] = int(last_row["epoch"])
    if "train_loss" in df.columns:
        row["last_train_loss"] = float(last_row["train_loss"])
    if "val_accuracy" in df.columns:
        row["last_val_accuracy"] = float(last_row["val_accuracy"])
    if "val_f1" in df.columns:
        row["last_val_f1"] = float(last_row["val_f1"])
    if "val_auc" in df.columns:
        row["last_val_auc"] = float(last_row["val_auc"])

    return row


def summarize_one_run(run_dir: Path) -> dict:
    seed_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])

    if seed_dirs:
        seed_dir = seed_dirs[0]
    else:
        seed_dir = run_dir

    row = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "seed_dir": str(seed_dir),
    }

    config_path = seed_dir / "config.json"
    best_metrics_path = seed_dir / "best_metrics.json"
    training_log_path = seed_dir / "training_log.csv"
    best_model_path = seed_dir / "best_model.pt"
    last_model_path = seed_dir / "last_model.pt"

    row["config_exists"] = config_path.exists()
    row["best_metrics_exists"] = best_metrics_path.exists()
    row["training_log_exists"] = training_log_path.exists()
    row["best_model_exists"] = best_model_path.exists()
    row["last_model_exists"] = last_model_path.exists()

    if config_path.exists():
        row.update(flatten_config(load_json(config_path)))

    if best_metrics_path.exists():
        row.update(flatten_best_metrics(load_json(best_metrics_path)))

    row.update(summarize_training_log(training_log_path))

    return row


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    priority_cols = [
        "best_accuracy",
        "log_best_val_accuracy",
        "best_f1",
        "log_best_val_f1",
        "best_auc",
        "log_best_val_auc",
    ]

    sort_cols = [col for col in priority_cols if col in df.columns]

    if not sort_cols:
        return df

    for col in sort_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(sort_cols, ascending=[False] * len(sort_cols))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Cross-Attention tuning root folder.")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root)

    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    if not run_dirs:
        raise FileNotFoundError(f"No run folders found under: {root}")

    rows = [summarize_one_run(run_dir) for run_dir in run_dirs]

    df = pd.DataFrame(rows)
    df = sort_results(df)

    if args.output_csv is None:
        output_csv = root / "cross_attention_tuning_summary.csv"
    else:
        output_csv = Path(args.output_csv)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print("=" * 100)
    print("Cross-Attention tuning summary")
    print("=" * 100)
    print(f"Root     : {root}")
    print(f"Runs     : {len(df)}")
    print(f"Saved CSV: {output_csv}")
    print()

    display_cols = [
        "run_name",
        "batch_size",
        "hidden_dim",
        "num_layers",
        "gat_heads",
        "cross_heads",
        "dropout",
        "lr",
        "weight_decay",
        "best_epoch",
        "best_monitor",
        "best_score",
        "best_accuracy",
        "best_precision",
        "best_recall",
        "best_f1",
        "best_auc",
        "log_best_val_accuracy",
        "log_best_val_accuracy_epoch",
        "log_best_val_f1",
        "log_best_val_auc",
        "last_val_accuracy",
        "last_val_f1",
        "last_val_auc",
    ]

    display_cols = [col for col in display_cols if col in df.columns]

    print(df[display_cols].head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()
