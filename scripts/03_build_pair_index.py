
import argparse
import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from itertools import combinations


def safe_load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), None
    except UnicodeDecodeError:
        try:
            with open(path, "r", encoding="cp949") as f:
                return json.load(f), None
        except Exception as e:
            return None, str(e)
    except Exception as e:
        return None, str(e)


def is_joint_set_file(data):
    if not isinstance(data, dict):
        return False
    return "body_one" in data and "body_two" in data and "joints" in data


def is_body_graph_file(data):
    if not isinstance(data, dict):
        return False
    required = {"nodes", "links", "graph", "properties"}
    return required.issubset(set(data.keys()))


def get_body_id_from_path(path: Path):
    return path.stem


def get_assembly_key(body_id: str, mode: str = "first2"):
    parts = body_id.split("_")

    if mode == "first2":
        if len(parts) >= 2:
            return "_".join(parts[:2])
        return body_id

    if mode == "first3":
        if len(parts) >= 3:
            return "_".join(parts[:3])
        return body_id

    if mode == "drop_last":
        if len(parts) >= 2:
            return "_".join(parts[:-1])
        return body_id

    raise ValueError(f"Unknown assembly key mode: {mode}")


def make_pair_key(body_one: str, body_two: str):
    return tuple(sorted([body_one, body_two]))


def make_pair_key_string(pair_key):
    return f"{pair_key[0]}|{pair_key[1]}"


def scan_dataset(dataset_dir: Path):
    json_files = sorted(dataset_dir.rglob("*.json"))

    body_graph_files = {}
    joint_set_files = []
    load_errors = []

    for path in json_files:
        data, error = safe_load_json(path)

        if error is not None:
            load_errors.append((str(path), error))
            continue

        if is_body_graph_file(data):
            body_id = get_body_id_from_path(path)
            body_graph_files[body_id] = str(path)

        elif is_joint_set_file(data):
            joint_set_files.append(path)

    return body_graph_files, joint_set_files, load_errors


def build_positive_pairs(joint_set_files, body_graph_files, group_mode):
    positive_map = {}
    duplicate_count = 0
    missing_body_count = 0
    cross_group_count = 0
    empty_joint_count = 0

    for path in joint_set_files:
        data, error = safe_load_json(path)
        if error is not None:
            continue

        body_one = data.get("body_one")
        body_two = data.get("body_two")
        joints = data.get("joints", [])

        if body_one is None or body_two is None:
            continue

        if len(joints) == 0:
            empty_joint_count += 1
            continue

        if body_one not in body_graph_files or body_two not in body_graph_files:
            missing_body_count += 1
            continue

        group_one = get_assembly_key(body_one, mode=group_mode)
        group_two = get_assembly_key(body_two, mode=group_mode)

        if group_one != group_two:
            cross_group_count += 1
            continue

        pair_key = make_pair_key(body_one, body_two)

        if pair_key in positive_map:
            duplicate_count += 1
            positive_map[pair_key]["joint_set_files"].append(str(path))
            positive_map[pair_key]["num_joint_sets"] += 1
            positive_map[pair_key]["total_joints"] += len(joints)
        else:
            positive_map[pair_key] = {
                "body_one": pair_key[0],
                "body_two": pair_key[1],
                "pair_key": make_pair_key_string(pair_key),
                "assembly_key": group_one,
                "label": 1,
                "source": "positive_joint",
                "joint_set_files": [str(path)],
                "num_joint_sets": 1,
                "total_joints": len(joints),
            }

    stats = {
        "raw_joint_set_files": len(joint_set_files),
        "unique_positive_pairs_used": len(positive_map),
        "duplicate_positive_records": duplicate_count,
        "missing_body_records": missing_body_count,
        "cross_group_positive_records_excluded": cross_group_count,
        "empty_joint_records": empty_joint_count,
    }

    return positive_map, stats


def build_body_groups(body_graph_files, group_mode):
    groups = defaultdict(list)

    for body_id in sorted(body_graph_files.keys()):
        group_key = get_assembly_key(body_id, mode=group_mode)
        groups[group_key].append(body_id)

    return groups


def split_assembly_keys(positive_pairs, seed, train_ratio, val_ratio):
    rng = random.Random(seed)

    assembly_keys = sorted(set(item["assembly_key"] for item in positive_pairs))
    rng.shuffle(assembly_keys)

    n_total = len(assembly_keys)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_keys = set(assembly_keys[:n_train])
    val_keys = set(assembly_keys[n_train:n_train + n_val])
    test_keys = set(assembly_keys[n_train + n_val:])

    return {
        "train": train_keys,
        "validation": val_keys,
        "test": test_keys,
    }


def assign_split(sample, split_keys):
    assembly_key = sample["assembly_key"]

    for split_name, keys in split_keys.items():
        if assembly_key in keys:
            return split_name

    raise RuntimeError(f"Assembly key was not assigned to any split: {assembly_key}")


def build_negative_candidates(groups, positive_pair_set):
    candidates_by_group = defaultdict(list)

    for group_key, members in groups.items():
        if len(members) < 2:
            continue

        for body_one, body_two in combinations(members, 2):
            pair_key = make_pair_key(body_one, body_two)

            if pair_key in positive_pair_set:
                continue

            candidates_by_group[group_key].append(
                {
                    "body_one": pair_key[0],
                    "body_two": pair_key[1],
                    "pair_key": make_pair_key_string(pair_key),
                    "assembly_key": group_key,
                    "label": 0,
                    "source": "negative_same_assembly_random",
                }
            )

    return candidates_by_group


def sample_negatives_for_split(
    split_name,
    split_keys,
    positive_count,
    candidates_by_group,
    rng,
):
    candidate_pool = []

    for group_key in split_keys:
        candidate_pool.extend(candidates_by_group.get(group_key, []))

    rng.shuffle(candidate_pool)

    if len(candidate_pool) < positive_count:
        raise RuntimeError(
            f"Not enough negative candidates for {split_name}: "
            f"needed={positive_count}, available={len(candidate_pool)}"
        )

    selected = candidate_pool[:positive_count]
    return selected, len(candidate_pool)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default=r"C:\so_zero\j1.0.0_all\joint",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\outputs\pair_index",
    )

    parser.add_argument(
        "--group_mode",
        type=str,
        default="first2",
        choices=["first2", "first3", "drop_last"],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    output_dir = Path(args.output_dir)
    rng = random.Random(args.seed)

    print("=" * 80)
    print("[STEP] Scanning dataset")
    print("=" * 80)

    body_graph_files, joint_set_files, load_errors = scan_dataset(dataset_dir)

    print(f"Body graph files : {len(body_graph_files)}")
    print(f"Joint set files  : {len(joint_set_files)}")
    print(f"Load errors      : {len(load_errors)}")

    print("\n" + "=" * 80)
    print("[STEP] Building positive pairs")
    print("=" * 80)

    positive_map, positive_stats = build_positive_pairs(
        joint_set_files=joint_set_files,
        body_graph_files=body_graph_files,
        group_mode=args.group_mode,
    )

    positive_pairs = list(positive_map.values())
    positive_pair_set = set(positive_map.keys())

    for key, value in positive_stats.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 80)
    print("[STEP] Building body groups")
    print("=" * 80)

    groups = build_body_groups(body_graph_files, group_mode=args.group_mode)

    group_size_counter = Counter(len(v) for v in groups.values())

    print(f"Number of assembly groups: {len(groups)}")
    print("Group size distribution:")
    for size, count in group_size_counter.most_common(15):
        print(f"  size={size}: {count}")

    print("\n" + "=" * 80)
    print("[STEP] Creating assembly-level split")
    print("=" * 80)

    split_keys = split_assembly_keys(
        positive_pairs=positive_pairs,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    for split_name, keys in split_keys.items():
        print(f"{split_name} assembly groups: {len(keys)}")

    positives_by_split = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for sample in positive_pairs:
        split_name = assign_split(sample, split_keys)
        positives_by_split[split_name].append(sample)

    for split_name, samples in positives_by_split.items():
        print(f"{split_name} positive pairs: {len(samples)}")

    print("\n" + "=" * 80)
    print("[STEP] Building negative candidates")
    print("=" * 80)

    candidates_by_group = build_negative_candidates(
        groups=groups,
        positive_pair_set=positive_pair_set,
    )

    total_negative_candidates = sum(len(v) for v in candidates_by_group.values())

    print(f"Total same-assembly negative candidates: {total_negative_candidates}")

    print("\n" + "=" * 80)
    print("[STEP] Sampling negatives with 1:1 ratio")
    print("=" * 80)

    negatives_by_split = {}
    available_negative_by_split = {}

    for split_name in ["train", "validation", "test"]:
        positive_count = len(positives_by_split[split_name])

        selected_negatives, available_count = sample_negatives_for_split(
            split_name=split_name,
            split_keys=split_keys[split_name],
            positive_count=positive_count,
            candidates_by_group=candidates_by_group,
            rng=rng,
        )

        negatives_by_split[split_name] = selected_negatives
        available_negative_by_split[split_name] = available_count

        print(
            f"{split_name}: positives={positive_count}, "
            f"available_negatives={available_count}, "
            f"selected_negatives={len(selected_negatives)}"
        )

    print("\n" + "=" * 80)
    print("[STEP] Saving pair index files")
    print("=" * 80)

    final_splits = {}

    for split_name in ["train", "validation", "test"]:
        samples = positives_by_split[split_name] + negatives_by_split[split_name]
        rng.shuffle(samples)

        final_splits[split_name] = samples

        out_path = output_dir / f"{split_name}_pairs.json"
        save_json(out_path, samples)

        label_counter = Counter(sample["label"] for sample in samples)
        print(f"{split_name}: total={len(samples)}, labels={dict(label_counter)}")
        print(f"  saved to: {out_path}")

    split_key_output = {
        split_name: sorted(list(keys))
        for split_name, keys in split_keys.items()
    }

    save_json(output_dir / "assembly_split_keys.json", split_key_output)

    body_graph_map_output = {
        body_id: path
        for body_id, path in sorted(body_graph_files.items())
    }

    save_json(output_dir / "body_graph_files.json", body_graph_map_output)

    summary = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "group_mode": args.group_mode,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "body_graph_files": len(body_graph_files),
        "joint_set_files": len(joint_set_files),
        "positive_stats": positive_stats,
        "positive_pairs_used": len(positive_pairs),
        "total_negative_candidates": total_negative_candidates,
        "available_negative_by_split": available_negative_by_split,
        "final_split_counts": {
            split_name: {
                "total": len(samples),
                "positive": sum(1 for sample in samples if sample["label"] == 1),
                "negative": sum(1 for sample in samples if sample["label"] == 0),
            }
            for split_name, samples in final_splits.items()
        },
    }

    save_json(output_dir / "pair_index_summary.json", summary)

    print("\n" + "=" * 80)
    print("[DONE] Pair index generation completed")
    print("=" * 80)
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
