
from pathlib import Path
import argparse
import json
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


def build_search_space() -> dict:
    return {
        "epochs": [40],
        "batch_size": [32, 64],
        "hidden_dim": [64, 128],
        "num_layers": [2, 3],
        "gat_heads": [4, 8],
        "cross_heads": [2, 4, 8],
        "dropout": [0.1, 0.2, 0.3],
        "lr": [0.001, 0.0005, 0.0003],
        "weight_decay": [0.0001, 0.001],
    }


def sample_configs(search_space: dict, num_configs: int, sampling_seed: int) -> list[dict]:
    rng = random.Random(sampling_seed)
    seen = set()
    configs = []

    max_possible = 1
    for values in search_space.values():
        max_possible *= len(values)

    target_count = min(num_configs, max_possible)

    while len(configs) < target_count:
        config = {
            key: rng.choice(values)
            for key, values in search_space.items()
        }

        # Multi-head attention usually requires hidden_dim to be divisible by the number of heads.
        if config["hidden_dim"] % config["gat_heads"] != 0:
            continue
        if config["hidden_dim"] % config["cross_heads"] != 0:
            continue

        signature = tuple(sorted(config.items()))
        if signature in seen:
            continue

        seen.add(signature)
        configs.append(config)

    return configs


def make_run_name(config: dict, seed: int, index: int) -> str:
    dropout_str = str(config["dropout"]).replace(".", "p")
    lr_str = str(config["lr"]).replace(".", "p")
    wd_str = str(config["weight_decay"]).replace(".", "p")

    return (
        f"cross_rand_cfg{index:03d}"
        f"_seed{seed}"
        f"_h{config['hidden_dim']}"
        f"_b{config['batch_size']}"
        f"_l{config['num_layers']}"
        f"_g{config['gat_heads']}"
        f"_c{config['cross_heads']}"
        f"_d{dropout_str}"
        f"_lr{lr_str}"
        f"_wd{wd_str}"
        f"_ep{config['epochs']}"
    )


def build_command(
    config: dict,
    output_dir: Path,
    seed: int,
    limit_train: int | None,
    limit_val: int | None,
    cpu: bool,
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
        "0.5",
        "--monitor",
        "accuracy",
        "--seed",
        str(seed),
    ]

    if limit_train is not None:
        command.extend(["--limit_train", str(limit_train)])

    if limit_val is not None:
        command.extend(["--limit_val", str(limit_val)])

    if cpu:
        command.append("--cpu")

    return command


def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return None


def flatten_dict(data: dict, prefix: str = "") -> dict:
    flat = {}

    for key, value in data.items():
        new_key = f"{prefix}_{key}" if prefix else str(key)

        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key))
        elif isinstance(value, (int, float, str, bool)) or value is None:
            flat[new_key] = value
        else:
            flat[new_key] = str(value)

    return flat


def collect_run_outputs(output_dir: Path) -> dict:
    row = {}

    json_files = sorted(output_dir.rglob("*.json"))
    csv_files = sorted(output_dir.rglob("*.csv"))
    pt_files = sorted(output_dir.rglob("*.pt"))

    row["json_file_count"] = len(json_files)
    row["csv_file_count"] = len(csv_files)
    row["pt_file_count"] = len(pt_files)

    # Try common summary names first.
    priority_names = [
        "result.json",
        "best_summary.json",
        "summary.json",
        "metrics.json",
        "config.json",
    ]

    for name in priority_names:
        matches = [p for p in json_files if p.name == name]
        for path in matches:
            data = read_json(path)
            if isinstance(data, dict):
                row[f"found_{name}"] = str(path)
                row.update(flatten_dict(data, prefix=path.stem))
                break

    # Read history if available.
    history_files = [p for p in csv_files if p.name.lower() == "history.csv"]
    if history_files:
        history_path = history_files[0]
        try:
            df = pd.read_csv(history_path, encoding="utf-8-sig")
            row["history_path"] = str(history_path)
            row["history_rows"] = len(df)

            metric_candidates = [
                "val_accuracy",
                "val_acc",
                "accuracy",
                "val_f1",
                "val_auc",
                "val_auroc",
                "val_auprc",
            ]

            for col in metric_candidates:
                if col in df.columns:
                    values = pd.to_numeric(df[col], errors="coerce")
                    row[f"history_best_{col}"] = float(values.max())
                    best_idx = values.idxmax()
                    if "epoch" in df.columns:
                        row[f"history_best_{col}_epoch"] = int(df.loc[best_idx, "epoch"])
        except Exception as exc:
            row["history_read_error"] = str(exc)

    return row


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    priority_cols = [
        "history_best_val_accuracy",
        "history_best_val_acc",
        "best_summary_best_val_accuracy",
        "best_summary_best_val_acc",
        "result_best_val_accuracy",
        "result_best_val_acc",
        "history_best_val_f1",
        "history_best_val_auroc",
        "history_best_val_auc",
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
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Training script not found: {TRAIN_SCRIPT}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    search_space = build_search_space()
    configs = sample_configs(
        search_space=search_space,
        num_configs=args.num_configs,
        sampling_seed=args.sampling_seed,
    )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_root": str(output_root),
        "seed": args.seed,
        "sampling_seed": args.sampling_seed,
        "num_configs": args.num_configs,
        "actual_config_count": len(configs),
        "limit_train": args.limit_train,
        "limit_val": args.limit_val,
        "search_space": search_space,
        "selection_rule": "Use validation accuracy for tuning. Test set should only be used for final confirmation.",
    }

    with (output_root / "cross_attention_random_tuning_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    rows = []

    for index, config in enumerate(configs, start=1):
        run_name = make_run_name(config, args.seed, index)
        run_output_dir = output_root / run_name

        # A run is considered existing if it has at least one checkpoint or summary JSON.
        existing_files = list(run_output_dir.rglob("*.pt")) + list(run_output_dir.rglob("*.json"))

        if args.skip_existing and existing_files:
            print(f"[SKIP] Existing run folder: {run_output_dir}")
            return_code = 0
        else:
            command = build_command(
                config=config,
                output_dir=run_output_dir,
                seed=args.seed,
                limit_train=args.limit_train,
                limit_val=args.limit_val,
                cpu=args.cpu,
            )

            print()
            print("=" * 100)
            print(f"[RUN] {run_name}")
            print("=" * 100)
            print(" ".join(command))

            completed = subprocess.run(command)
            return_code = completed.returncode

        row = {
            "run_name": run_name,
            "output_dir": str(run_output_dir),
            "seed": args.seed,
            "return_code": return_code,
        }
        row.update(config)
        row.update(collect_run_outputs(run_output_dir))
        rows.append(row)

        partial_df = pd.DataFrame(rows)
        partial_df = sort_results(partial_df)
        partial_df.to_csv(output_root / "cross_attention_random_tuning_partial.csv", index=False, encoding="utf-8-sig")

        if return_code != 0:
            print(f"[WARN] Run failed: {run_name}")

    result_df = pd.DataFrame(rows)
    result_df = sort_results(result_df)

    result_csv = output_root / "cross_attention_random_tuning_results.csv"
    result_df.to_csv(result_csv, index=False, encoding="utf-8-sig")

    print()
    print("=" * 100)
    print("Cross-Attention random tuning complete")
    print("=" * 100)
    print(f"Saved: {result_csv}")
    print()

    display_cols = [
        "run_name",
        "hidden_dim",
        "batch_size",
        "num_layers",
        "gat_heads",
        "cross_heads",
        "dropout",
        "lr",
        "weight_decay",
        "history_best_val_accuracy",
        "history_best_val_accuracy_epoch",
        "history_best_val_f1",
        "history_best_val_auroc",
        "return_code",
    ]

    display_cols = [col for col in display_cols if col in result_df.columns]
    if display_cols:
        print(result_df[display_cols].head(30).to_string(index=False))
    else:
        print(result_df.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
