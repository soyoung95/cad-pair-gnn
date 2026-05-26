


from pathlib import Path
import argparse
import csv
import json
import math
import sys
from typing import Any


import numpy as np
import torch




try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass




ID_KEYS = ["assembly_key", "pair_key", "body_one", "body_two"]
LABEL_KEYS = ["label", "y", "target"]
BODY_ONE_KEYS = ["body_one", "body_a", "part_one", "part_a", "source_body", "body1"]
BODY_TWO_KEYS = ["body_two", "body_b", "part_two", "part_b", "target_body", "body2"]


NEGATIVE_TYPE_KEYS = [
    "negative_type",
    "neg_type",
    "negative_source",
    "source",
    "pair_type",
    "candidate_type",
    "sampling_type",
    "is_hard_negative",
    "hard_negative",
]




def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)




def iter_pair_records(data: Any) -> list[dict[str, Any]]:
    records = []


    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            if any(key in obj for key in LABEL_KEYS) and any(key in obj for key in BODY_ONE_KEYS):
                records.append(obj)
            else:
                for value in obj.values():
                    visit(value)
        elif isinstance(obj, list):
            for item in obj:
                visit(item)


    visit(data)
    return records




def get_first(record: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return default




def normalize_body_id(value: Any) -> str:
    return str(value).strip()




def parse_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)


    if isinstance(value, (int, float)):
        return int(value)


    text = str(value).strip().lower()
    if text in ["1", "true", "positive", "pos", "yes"]:
        return 1
    if text in ["0", "false", "negative", "neg", "no"]:
        return 0


    raise ValueError(f"Cannot parse label: {value}")




def torch_load_graph(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")




def tensor_to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0, 0), dtype=np.float32)


    if torch.is_tensor(value):
        return value.detach().cpu().float().numpy()


    return np.asarray(value, dtype=np.float32)




def summarize_graph(graph: Any) -> dict[str, Any]:
    x = tensor_to_numpy(getattr(graph, "x", None))
    edge_attr = tensor_to_numpy(getattr(graph, "edge_attr", None))
    edge_index = getattr(graph, "edge_index", None)


    if x.ndim != 2:
        x = np.zeros((0, 0), dtype=np.float32)


    if edge_attr.ndim != 2:
        edge_attr = np.zeros((0, 0), dtype=np.float32)


    node_count = int(x.shape[0])
    node_dim = int(x.shape[1]) if x.ndim == 2 and x.shape[1] > 0 else 0


    if edge_index is not None and torch.is_tensor(edge_index) and edge_index.ndim == 2:
        edge_count = int(edge_index.shape[1])
    else:
        edge_count = int(edge_attr.shape[0])


    edge_dim = int(edge_attr.shape[1]) if edge_attr.ndim == 2 and edge_attr.shape[1] > 0 else 0


    x_mean = x.mean(axis=0) if node_count > 0 else np.zeros((node_dim,), dtype=np.float32)
    x_std = x.std(axis=0) if node_count > 0 else np.zeros((node_dim,), dtype=np.float32)


    e_mean = edge_attr.mean(axis=0) if edge_attr.shape[0] > 0 else np.zeros((edge_dim,), dtype=np.float32)
    e_std = edge_attr.std(axis=0) if edge_attr.shape[0] > 0 else np.zeros((edge_dim,), dtype=np.float32)


    vector = np.concatenate(
        [
            np.asarray([node_count, edge_count], dtype=np.float32),
            x_mean.astype(np.float32),
            x_std.astype(np.float32),
            e_mean.astype(np.float32),
            e_std.astype(np.float32),
        ],
        axis=0,
    )


    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "vector": vector,
    }




def load_body_vectors(records_by_split: dict[str, list[dict[str, Any]]], graph_dir: Path) -> dict[str, dict[str, Any]]:
    body_ids = set()


    for records in records_by_split.values():
        for record in records:
            body_one = normalize_body_id(get_first(record, BODY_ONE_KEYS))
            body_two = normalize_body_id(get_first(record, BODY_TWO_KEYS))
            body_ids.add(body_one)
            body_ids.add(body_two)


    body_data = {}
    missing = []


    for body_id in sorted(body_ids):
        graph_path = graph_dir / f"{body_id}.pt"


        if not graph_path.exists():
            missing.append(str(graph_path))
            continue


        graph = torch_load_graph(graph_path)
        summary = summarize_graph(graph)
        body_data[body_id] = summary


    if missing:
        print()
        print("[WARNING] Missing graph files")
        print(f"missing_count: {len(missing)}")
        for path in missing[:20]:
            print(f"  {path}")


    return body_data




def build_normalization(body_data: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    vectors = [item["vector"] for item in body_data.values()]


    if not vectors:
        raise ValueError("No body vectors found.")


    max_dim = max(len(v) for v in vectors)
    padded = []


    for vector in vectors:
        if len(vector) < max_dim:
            vector = np.pad(vector, (0, max_dim - len(vector)), mode="constant")
        padded.append(vector)


    matrix = np.stack(padded, axis=0).astype(np.float32)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std < 1e-8] = 1.0


    return mean, std




def normalize_vector(vector: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if len(vector) < len(mean):
        vector = np.pad(vector, (0, len(mean) - len(vector)), mode="constant")
    elif len(vector) > len(mean):
        vector = vector[: len(mean)]


    return (vector - mean) / std




def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))


    if denom < 1e-8:
        return 0.0


    return float(np.dot(a, b) / denom)




def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))




def classify_negative(distance: float, low_threshold: float, high_threshold: float) -> str:
    if math.isnan(low_threshold) or math.isnan(high_threshold):
        return "unknown"


    if distance <= low_threshold:
        return "hard_like"
    if distance >= high_threshold:
        return "easy_like"
    return "medium_like"




def detect_negative_type(record: dict[str, Any]) -> str:
    for key in NEGATIVE_TYPE_KEYS:
        if key in record:
            return str(record[key])
    return ""




def analyze_records(
    split: str,
    records: list[dict[str, Any]],
    body_data: dict[str, dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> list[dict[str, Any]]:
    rows = []


    for index, record in enumerate(records):
        label = parse_label(get_first(record, LABEL_KEYS))
        body_one = normalize_body_id(get_first(record, BODY_ONE_KEYS))
        body_two = normalize_body_id(get_first(record, BODY_TWO_KEYS))


        if body_one not in body_data or body_two not in body_data:
            continue


        a = normalize_vector(body_data[body_one]["vector"], mean, std)
        b = normalize_vector(body_data[body_two]["vector"], mean, std)


        distance = float(np.linalg.norm(a - b))
        cosine = cosine_similarity(a, b)


        negative_type = detect_negative_type(record)


        if label == 0:
            difficulty = classify_negative(distance, low_threshold, high_threshold)
        else:
            difficulty = "positive"


        row = {
            "split": split,
            "index": index,
            "assembly_key": str(record.get("assembly_key", "")),
            "pair_key": str(record.get("pair_key", "")),
            "body_one": body_one,
            "body_two": body_two,
            "label": label,
            "negative_type_from_metadata": negative_type,
            "difficulty_proxy": difficulty,
            "graph_distance": distance,
            "graph_cosine_similarity": cosine,
            "node_count_one": body_data[body_one]["node_count"],
            "node_count_two": body_data[body_two]["node_count"],
            "node_count_abs_diff": abs(body_data[body_one]["node_count"] - body_data[body_two]["node_count"]),
            "edge_count_one": body_data[body_one]["edge_count"],
            "edge_count_two": body_data[body_two]["edge_count"],
            "edge_count_abs_diff": abs(body_data[body_one]["edge_count"] - body_data[body_two]["edge_count"]),
        }


        rows.append(row)


    return rows




def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


    if not rows:
        path.write_text("", encoding="utf-8")
        return


    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)


    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)




def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []


    splits = sorted(set(row["split"] for row in rows))


    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]


        for label in [0, 1]:
            label_rows = [row for row in split_rows if int(row["label"]) == label]
            distances = [float(row["graph_distance"]) for row in label_rows]
            cosines = [float(row["graph_cosine_similarity"]) for row in label_rows]


            summary.append(
                {
                    "split": split,
                    "group": f"label_{label}",
                    "count": len(label_rows),
                    "distance_mean": float(np.mean(distances)) if distances else "",
                    "distance_p25": percentile(distances, 25) if distances else "",
                    "distance_p50": percentile(distances, 50) if distances else "",
                    "distance_p75": percentile(distances, 75) if distances else "",
                    "cosine_mean": float(np.mean(cosines)) if cosines else "",
                }
            )


        negative_rows = [row for row in split_rows if int(row["label"]) == 0]
        difficulty_values = sorted(set(row["difficulty_proxy"] for row in negative_rows))


        for difficulty in difficulty_values:
            group_rows = [row for row in negative_rows if row["difficulty_proxy"] == difficulty]
            distances = [float(row["graph_distance"]) for row in group_rows]


            summary.append(
                {
                    "split": split,
                    "group": f"negative_{difficulty}",
                    "count": len(group_rows),
                    "distance_mean": float(np.mean(distances)) if distances else "",
                    "distance_p25": percentile(distances, 25) if distances else "",
                    "distance_p50": percentile(distances, 50) if distances else "",
                    "distance_p75": percentile(distances, 75) if distances else "",
                    "cosine_mean": "",
                }
            )


    return summary




def summarize_metadata_negative_types(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = {}


    for row in rows:
        if int(row["label"]) != 0:
            continue


        key = (
            row["split"],
            row["negative_type_from_metadata"] if row["negative_type_from_metadata"] else "(missing)",
        )
        counts[key] = counts.get(key, 0) + 1


    result = []


    for (split, negative_type), count in sorted(counts.items()):
        result.append(
            {
                "split": split,
                "negative_type_from_metadata": negative_type,
                "count": count,
            }
        )


    return result




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_index_dir", default="outputs/pair_index")
    parser.add_argument("--graph_dir", default="processed/body_graphs")
    parser.add_argument("--output_dir", default="outputs/diagnostics/negative_pair_difficulty")
    args = parser.parse_args()


    pair_index_dir = Path(args.pair_index_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)


    split_files = {
        "train": pair_index_dir / "train_pairs.json",
        "validation": pair_index_dir / "validation_pairs.json",
        "test": pair_index_dir / "test_pairs.json",
    }


    print("=" * 100)
    print("NEGATIVE PAIR DIFFICULTY ANALYSIS")
    print("=" * 100)
    print(f"pair_index_dir: {pair_index_dir}")
    print(f"graph_dir: {graph_dir}")
    print(f"output_dir: {output_dir}")


    records_by_split = {}


    for split, path in split_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing pair file: {path}")


        data = load_json(path)
        records = iter_pair_records(data)
        records_by_split[split] = records


        label_counts = {0: 0, 1: 0}
        for record in records:
            label = parse_label(get_first(record, LABEL_KEYS))
            label_counts[label] += 1


        print()
        print(f"[{split}]")
        print(f"path: {path}")
        print(f"records: {len(records)}")
        print(f"label_0: {label_counts[0]}")
        print(f"label_1: {label_counts[1]}")


    print()
    print("[LOAD BODY GRAPH FEATURES]")
    body_data = load_body_vectors(records_by_split, graph_dir)
    print(f"loaded_body_graphs: {len(body_data)}")


    mean, std = build_normalization(body_data)


    train_temp_rows = analyze_records(
        split="train",
        records=records_by_split["train"],
        body_data=body_data,
        mean=mean,
        std=std,
        low_threshold=float("nan"),
        high_threshold=float("nan"),
    )


    train_negative_distances = [
        float(row["graph_distance"])
        for row in train_temp_rows
        if int(row["label"]) == 0
    ]


    low_threshold = percentile(train_negative_distances, 33.333)
    high_threshold = percentile(train_negative_distances, 66.666)


    print()
    print("[NEGATIVE DIFFICULTY THRESHOLDS]")
    print("Criterion: graph distance among train negative pairs")
    print(f"hard_like <= {low_threshold}")
    print(f"easy_like >= {high_threshold}")


    all_rows = []


    for split, records in records_by_split.items():
        rows = analyze_records(
            split=split,
            records=records,
            body_data=body_data,
            mean=mean,
            std=std,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        all_rows.extend(rows)


    pair_csv = output_dir / "pair_difficulty_rows.csv"
    summary_csv = output_dir / "negative_difficulty_summary.csv"
    metadata_csv = output_dir / "negative_metadata_type_counts.csv"


    write_csv(pair_csv, all_rows)
    write_csv(summary_csv, summarize_rows(all_rows))
    write_csv(metadata_csv, summarize_metadata_negative_types(all_rows))


    print()
    print("[SAVED]")
    print(f"pair rows: {pair_csv}")
    print(f"summary  : {summary_csv}")
    print(f"metadata : {metadata_csv}")


    print()
    print("[SUMMARY PREVIEW]")
    for row in summarize_rows(all_rows):
        print(row)


    print()
    print("[METADATA NEGATIVE TYPE PREVIEW]")
    metadata_rows = summarize_metadata_negative_types(all_rows)
    for row in metadata_rows[:30]:
        print(row)


    print()
    print("Done.")




if __name__ == "__main__":
    main()


