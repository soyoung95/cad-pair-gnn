
from pathlib import Path
import argparse
import itertools
import json
import os
import random
import subprocess
import sys
from datetime import datetime

import pandas as pd


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


TRAIN_SCRIPT = Path("models/train_gat_cross_attention_pair_full.py")


SEARCH_SPACE = {
    "lr": [0.1, 0.01, 0.001, 0.0001],
    "batch_size": [32, 64, 128],
    "dropout": [0.0, 0.1, 0.5],
}


FIXED_CONFIG = {
    "epochs": 40,
    "hidden_dim": 128,
    "num_layers": 2,
    "gat_heads": 8,
    "cross_heads": 8,
    "weight_decay": 0.0001,
    "threshold": 0.5,
    "monitor": "accuracy",
}


def build_all_configs() -> list[dict]:
    configs = []

    keys = list(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[key] for key in keys]

    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        config.update(FIXED_CONFIG)
        configs.append(config)

    return configs


def sample_configs(num_configs: int, sampling_seed: int) -> list[dict]:
    all_configs = build_all_configs()
    rng = random.Random(sampling_seed)
    rng.shuffle(all_configs)

    if num_configs <= 0 or num_configs >= len(all_configs):
        return all_configs

    return all_configs[:num_configs]


def format_float_for_name(value: float) -> str:
    return str(value).replace(".", "p")


def make_run_name(config: dict, seed: int, index: int) -> str:
    lr_str = format_float_for_name(config["lr"])
    dropout_str = format_float_for_name(config["dropout"])

    return (
        f"prof_rand_cfg{index:03d}"
        f"_seed{seed}"
        f"_h{config['hidden_dim']}"
        f"_b{config['batch_size']}"
        f"_l{config['num_layers']}"
        f"_g{config['gat_heads']}"
        f"_c{config['cross_heads']}"
        f"_d{dropout_str}"
        f"_lr{lr_str}"
        f"_wd{format_float_for_name(config['weight_decay'])}"
        f"_ep{config['epochs']}"
    )


def build_command(
    output_dir: Path,
    config: dict,
    seed: int,
    limit_train: int | None,
    limit_val: int | None,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--pair_index_dir",
        "outputs/pair_index",
        "--graph_dir",
        "processed/body_graphs",
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(config["epochs"]),
        "--batch_size",
        str(config["batch_size"]),
        "--hidden_dim",
        str(config["hidden_dim"]),
        "--num_layers",
        str(config["num_layers"]),
        "--gat_heads",
        str(config["gat_heads"]),
        "--cross_heads",
        str(config["cross_heads"]),
        "--dropout",
        str(config["dropout"]),
        "--lr",
        str(config["lr"]),
        "--weight_decay",
        str(config["weight_decay"]),
        "--threshold",
        str(config["threshold"]),
        "--monitor",
        str(config["monitor"]),
        "--seed",
        str(seed),
    ]

    if limit_train is not None:
        command.extend(["--limit_train", str(limit_train)])

    if limit_val is not None:
        command.extend(["--limit_val", str(limit_val)])

    return command


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


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

    metric_cols = [
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_auc",
        "val_mean_score",
    ]

    for col in metric_cols:
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


def summarize_run(output_dir: Path, config: dict, seed: int, return_code: int) -> dict:
    seed_dir = output_dir / f"seed_{seed}"

    best_metrics_path = seed_dir / "best_metrics.json"
    training_log_path = seed_dir / "training_log.csv"
    best_model_path = seed_dir / "best_model.pt"
    last_model_path = seed_dir / "last_model.pt"

    row = {
        "run_name": output_dir.name,
        "output_dir": str(output_dir),
        "seed": seed,
        "return_code": return_code,
        "best_metrics_exists": best_metrics_path.exists(),
        "training_log_exists": training_log_path.exists(),
        "best_model_exists": best_model_path.exists(),
        "last_model_exists": last_model_path.exists(),
    }

    row.update(config)

    if best_metrics_path.exists():
        best_metrics = load_json(best_metrics_path)
        row["best_epoch"] = best_metrics.get("best_epoch")
        row["best_monitor"] = best_metrics.get("best_monitor")
        row["best_score"] = best_metrics.get("best_score")

        metrics = best_metrics.get("best_metrics", {})
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                row[f"best_{key}"] = value

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
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling_seed", type=int, default=2026)
    parser.add_argument("--num_configs", type=int, default=12)
    parser.add_argument("--limit_train", type=int, default=None)
    parser.add_argument("--limit_val", type=int, default=None)
    parser.add_argument("--gpu_id", type=str, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    configs = sample_configs(
        num_configs=args.num_configs,
        sampling_seed=args.sampling_seed,
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "seed": args.seed,
        "sampling_seed": args.sampling_seed,
        "num_configs_requested": args.num_configs,
        "num_configs_actual": len(configs),
        "search_space": SEARCH_SPACE,
        "fixed_config": FIXED_CONFIG,
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "gpu_id": args.gpu_id,
        "selection_rule": "Select by validation accuracy. Test set is not used for hyperparameter tuning.",
    }

    with (output_root / "professor_random_tuning_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    rows = []

    env = os.environ.copy()
    if args.gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    for index, config in enumerate(configs, start=1):
        run_name = make_run_name(config=config, seed=args.seed, index=index)
        run_output_dir = output_root / run_name
        seed_dir = run_output_dir / f"seed_{args.seed}"
        best_metrics_path = seed_dir / "best_metrics.json"

        if args.skip_existing and best_metrics_path.exists():
            print(f"[SKIP] Existing run: {run_output_dir}")
            return_code = 0
        else:
            command = build_command(
                output_dir=run_output_dir,
                config=config,
                seed=args.seed,
                limit_train=args.limit_train,
                limit_val=args.limit_val,
            )

            print()
            print("=" * 100)
            print(f"[RUN] {run_name}")
            print("=" * 100)
            print(" ".join(command))

            completed = subprocess.run(command, env=env)
            return_code = completed.returncode

        row = summarize_run(
            output_dir=run_output_dir,
            config=config,
            seed=args.seed,
            return_code=return_code,
        )
        rows.append(row)

        partial_df = pd.DataFrame(rows)
        partial_df = sort_results(partial_df)
        partial_df.to_csv(
            output_root / "professor_random_tuning_partial.csv",
            index=False,
            encoding="utf-8-sig",
        )

        if return_code != 0:
            print(f"[WARN] Run failed: {run_name}")

    result_df = pd.DataFrame(rows)
    result_df = sort_results(result_df)

    result_csv = output_root / "professor_random_tuning_results.csv"
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    print()
    print("=" * 100)
    print("Professor random hyperparameter tuning complete")
    print("=" * 100)
    print(f"Saved: {result_csv}")
    print()

    display_cols = [
        "run_name",
        "batch_size",
        "lr",
        "dropout",
        "hidden_dim",
        "gat_heads",
        "cross_heads",
        "best_epoch",
        "best_monitor",
        "best_score",
        "best_accuracy",
        "best_precision",
        "best_recall",
        "best_f1",
        "best_auc",
        "log_best_val_accuracy",
        "log_best_val_f1",
        "log_best_val_auc",
        "last_val_accuracy",
        "last_val_f1",
        "last_val_auc",
        "return_code",
    ]

    display_cols = [col for col in display_cols if col in result_df.columns]

    if display_cols:
        print(result_df[display_cols].head(30).to_string(index=False))
    else:
        print(result_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
