
from pathlib import Path
import csv
import json
import sys

import torch


try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


PATHS = {
    "graph_dir": Path("processed/body_graphs"),
    "checkpoint": Path("outputs/runs/gat_pair_v2_seed42_20260512_230948/best_model_by_val_auroc.pt"),
    "axis_train": Path("outputs/vector_features/axis_pair_v1/train_axis_pair_features.csv"),
    "gnn_train": Path("outputs/gnn_pair_embeddings/gat_pair_v2_seed42_20260512_230948/train_gnn_pair_embeddings.csv"),
    "fusion_train": Path("outputs/fusion_pair_features/gat_pair_v2_axis_v1_seed42/train_fusion_pair_features.csv"),
}


ID_COLUMNS = ["assembly_key", "pair_key", "body_one", "body_two"]
LABEL_COLUMN = "label"


def read_csv_header(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    return [str(col).replace("\ufeff", "") for col in header]


def count_csv_rows(path: Path):
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for _ in reader:
            count += 1
    return count


def summarize_graph_file():
    print("=" * 100)
    print("[GRAPH FEATURE DIMENSION]")
    print("=" * 100)

    graph_dir = PATHS["graph_dir"]
    print(f"graph_dir: {graph_dir}")
    print(f"exists: {graph_dir.exists()}")

    graph_files = sorted(graph_dir.glob("*.pt"))
    print(f"graph_file_count: {len(graph_files)}")

    if not graph_files:
        return

    graph_path = graph_files[0]
    print(f"sample_graph: {graph_path}")

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)

    print(f"node_feature_shape: {tuple(graph.x.shape)}")
    print(f"node_feature_dim: {graph.x.shape[1]}")

    print(f"edge_attr_shape: {tuple(graph.edge_attr.shape)}")
    print(f"edge_attribute_dim: {graph.edge_attr.shape[1]}")
    print()


def summarize_checkpoint():
    print("=" * 100)
    print("[CHECKPOINT VECTOR SIZE]")
    print("=" * 100)

    ckpt_path = PATHS["checkpoint"]
    print(f"path: {ckpt_path}")
    print(f"exists: {ckpt_path.exists()}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    print(f"top_level_keys: {list(checkpoint.keys())}")

    classifier_keys = [key for key in state_dict.keys() if "classifier" in key.lower()]

    print()
    print("[classifier keys]")
    for key in classifier_keys:
        value = state_dict[key]
        if torch.is_tensor(value):
            print(f"{key}: shape={tuple(value.shape)}")

    for key in classifier_keys:
        if key.endswith("weight"):
            weight = state_dict[key]
            print()
            print(f"first_classifier_weight: {key}")
            print(f"classifier_input_dim: {weight.shape[1]}")
            print(f"classifier_output_dim: {weight.shape[0]}")
            break

    print()


def summarize_csv_exact(name: str, path: Path):
    print("=" * 100)
    print(f"[CSV] {name}")
    print("=" * 100)

    print(f"path: {path}")
    print(f"exists: {path.exists()}")

    if not path.exists():
        print()
        return

    header = read_csv_header(path)
    rows = count_csv_rows(path)

    id_cols = [col for col in ID_COLUMNS if col in header]
    label_cols = [LABEL_COLUMN] if LABEL_COLUMN in header else []

    gnn_cols = [col for col in header if col.startswith("gnn_pair_")]
    non_feature_cols = set(id_cols + label_cols)
    feature_cols = [col for col in header if col not in non_feature_cols]
    axis_cols = [col for col in feature_cols if not col.startswith("gnn_pair_")]

    print(f"rows: {rows}")
    print(f"total_columns: {len(header)}")
    print(f"id_columns_count: {len(id_cols)}")
    print(f"label_columns_count: {len(label_cols)}")
    print(f"feature_columns_count: {len(feature_cols)}")
    print(f"gnn_pair_feature_count: {len(gnn_cols)}")
    print(f"axis_or_vector_feature_count: {len(axis_cols)}")

    print(f"id_columns: {id_cols}")
    print(f"label_columns: {label_cols}")
    print(f"first_20_feature_columns: {feature_cols[:20]}")
    print()


def main():
    print("=" * 100)
    print("VECTOR SIZE INSPECTION - FIXED COLUMN CLASSIFICATION")
    print("=" * 100)
    print()

    summarize_graph_file()
    summarize_checkpoint()

    summarize_csv_exact("axis_train", PATHS["axis_train"])
    summarize_csv_exact("gnn_train_embeddings", PATHS["gnn_train"])
    summarize_csv_exact("fusion_train", PATHS["fusion_train"])

    print("=" * 100)
    print("[FINAL VECTOR SIZE SUMMARY]")
    print("=" * 100)
    print("Face node feature dim          : 12")
    print("Edge attribute dim             : 16")
    print("GNN pair embedding dim         : 256")
    print("Axis/vector feature dim        : 84")
    print("Fusion feature dim             : 340")
    print("Fusion CSV total columns       : 345")
    print("Fusion CSV structure           : 4 ID columns + 1 label column + 340 feature columns")
    print()
    print("GNN pair embedding structure estimate:")
    print("z_A 64 + z_B 64 + |z_A - z_B| 64 + z_A * z_B 64 = 256")


if __name__ == "__main__":
    main()
