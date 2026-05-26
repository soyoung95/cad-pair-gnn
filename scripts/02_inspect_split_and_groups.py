
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
from pprint import pprint


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


def is_split_file(data):
    if not isinstance(data, dict):
        return False
    split_keys = {"train", "validation", "test"}
    return split_keys.issubset(set(data.keys()))


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


def inspect_dataset(dataset_dir: Path, report_path: Path):
    json_files = sorted(dataset_dir.rglob("*.json"))

    body_graph_files = {}
    joint_set_files = []
    split_files = []

    load_errors = []

    print("=" * 80)
    print("[STEP] Scanning dataset")
    print("=" * 80)

    for path in json_files:
        data, error = safe_load_json(path)

        if error is not None:
            load_errors.append((str(path), error))
            continue

        if is_body_graph_file(data):
            body_id = get_body_id_from_path(path)
            body_graph_files[body_id] = path

        elif is_joint_set_file(data):
            joint_set_files.append(path)

        elif is_split_file(data):
            split_files.append(path)

    print(f"Total JSON files      : {len(json_files)}")
    print(f"Body graph files      : {len(body_graph_files)}")
    print(f"Joint set files       : {len(joint_set_files)}")
    print(f"Split files           : {len(split_files)}")
    print(f"Load errors           : {len(load_errors)}")

    print("\n[INFO] Split file candidates:")
    for path in split_files:
        print(f" - {path}")

    print("\n" + "=" * 80)
    print("[STEP] Inspecting split files")
    print("=" * 80)

    split_summaries = []

    for path in split_files:
        data, _ = safe_load_json(path)

        print(f"\n--- SPLIT FILE: {path} ---")
        summary = {}

        for key, value in data.items():
            if isinstance(value, list):
                summary[key] = {
                    "type": "list",
                    "length": len(value),
                    "first_items": value[:5],
                }
                print(f"{key}: list length = {len(value)}, first items = {value[:5]}")
            else:
                summary[key] = {
                    "type": type(value).__name__,
                    "value": value,
                }
                print(f"{key}: {type(value).__name__}")

        split_summaries.append(
            {
                "file": str(path),
                "summary": summary,
            }
        )

    print("\n" + "=" * 80)
    print("[STEP] Loading positive pairs from joint_set files")
    print("=" * 80)

    positive_pairs = []
    joint_count_counter = Counter()
    missing_body_graph_count = 0
    missing_body_examples = []

    for path in joint_set_files:
        data, _ = safe_load_json(path)

        body_one = data.get("body_one")
        body_two = data.get("body_two")
        joints = data.get("joints", [])

        if body_one is None or body_two is None:
            continue

        pair = tuple(sorted([body_one, body_two]))

        positive_pairs.append(
            {
                "joint_set_file": str(path),
                "body_one": body_one,
                "body_two": body_two,
                "pair_key": pair,
                "num_joints": len(joints),
            }
        )

        joint_count_counter[len(joints)] += 1

        if body_one not in body_graph_files or body_two not in body_graph_files:
            missing_body_graph_count += 1
            if len(missing_body_examples) < 10:
                missing_body_examples.append(
                    {
                        "joint_set_file": str(path),
                        "body_one": body_one,
                        "body_one_exists": body_one in body_graph_files,
                        "body_two": body_two,
                        "body_two_exists": body_two in body_graph_files,
                    }
                )

    unique_positive_pair_keys = set(item["pair_key"] for item in positive_pairs)

    print(f"Positive pair records          : {len(positive_pairs)}")
    print(f"Unique positive pairs          : {len(unique_positive_pair_keys)}")
    print(f"Missing body graph pair records: {missing_body_graph_count}")

    print("\n[INFO] Number of joints per joint_set:")
    for count, freq in joint_count_counter.most_common(20):
        print(f"num_joints={count}: {freq}")

    if missing_body_examples:
        print("\n[WARN] Missing body graph examples:")
        pprint(missing_body_examples, width=120)

    print("\n" + "=" * 80)
    print("[STEP] Inspecting assembly group candidates")
    print("=" * 80)

    group_modes = ["first2", "first3", "drop_last"]
    group_summaries = {}

    body_ids = sorted(body_graph_files.keys())

    for mode in group_modes:
        groups = defaultdict(list)

        for body_id in body_ids:
            group_key = get_assembly_key(body_id, mode=mode)
            groups[group_key].append(body_id)

        group_sizes = [len(v) for v in groups.values()]
        group_size_counter = Counter(group_sizes)

        possible_pair_count = 0
        for size in group_sizes:
            if size >= 2:
                possible_pair_count += size * (size - 1) // 2

        positive_inside_group = 0
        positive_cross_group = 0

        for item in positive_pairs:
            b1 = item["body_one"]
            b2 = item["body_two"]

            g1 = get_assembly_key(b1, mode=mode)
            g2 = get_assembly_key(b2, mode=mode)

            if g1 == g2:
                positive_inside_group += 1
            else:
                positive_cross_group += 1

        largest_groups = sorted(groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]

        print(f"\n--- Group mode: {mode} ---")
        print(f"Number of groups              : {len(groups)}")
        print(f"Possible same-group body pairs: {possible_pair_count}")
        print(f"Positive pairs inside group   : {positive_inside_group}")
        print(f"Positive pairs crossing group : {positive_cross_group}")
        print("Group size distribution:")
        for size, freq in group_size_counter.most_common(10):
            print(f"  size={size}: {freq}")

        print("Largest group examples:")
        for group_key, members in largest_groups[:5]:
            print(f"  {group_key}: size={len(members)}, first members={members[:5]}")

        group_summaries[mode] = {
            "num_groups": len(groups),
            "possible_same_group_pairs": possible_pair_count,
            "positive_inside_group": positive_inside_group,
            "positive_cross_group": positive_cross_group,
            "group_size_distribution": dict(group_size_counter),
            "largest_groups": [
                {
                    "group_key": group_key,
                    "size": len(members),
                    "first_members": members[:20],
                }
                for group_key, members in largest_groups
            ],
        }

    print("\n" + "=" * 80)
    print("[STEP] Estimating same-assembly negative candidates")
    print("=" * 80)

    selected_mode = "first2"
    groups = defaultdict(list)

    for body_id in body_ids:
        group_key = get_assembly_key(body_id, mode=selected_mode)
        groups[group_key].append(body_id)

    positive_pair_set = set(unique_positive_pair_keys)

    negative_candidate_count = 0
    group_negative_summary = []

    for group_key, members in groups.items():
        if len(members) < 2:
            continue

        total_pairs = len(members) * (len(members) - 1) // 2
        positive_pairs_in_group = 0

        for b1, b2 in combinations(members, 2):
            pair = tuple(sorted([b1, b2]))
            if pair in positive_pair_set:
                positive_pairs_in_group += 1

        negative_pairs_in_group = total_pairs - positive_pairs_in_group
        negative_candidate_count += negative_pairs_in_group

        if negative_pairs_in_group > 0:
            group_negative_summary.append(
                {
                    "group_key": group_key,
                    "num_bodies": len(members),
                    "total_pairs": total_pairs,
                    "positive_pairs": positive_pairs_in_group,
                    "negative_candidates": negative_pairs_in_group,
                    "first_members": members[:10],
                }
            )

    group_negative_summary = sorted(
        group_negative_summary,
        key=lambda x: x["negative_candidates"],
        reverse=True,
    )

    print(f"Selected group mode               : {selected_mode}")
    print(f"Total positive pairs              : {len(positive_pairs)}")
    print(f"Unique positive pairs             : {len(unique_positive_pair_keys)}")
    print(f"Same-group negative candidates    : {negative_candidate_count}")
    print(f"Target negative count for 1:1     : {len(positive_pairs)}")

    if negative_candidate_count >= len(positive_pairs):
        print("[OK] There are enough same-group negative candidates for a 1:1 dataset.")
    else:
        print("[WARN] Not enough same-group negative candidates for a 1:1 dataset.")

    print("\n[INFO] Groups with many negative candidates:")
    pprint(group_negative_summary[:10], width=140)

    report = {
        "dataset_dir": str(dataset_dir),
        "total_json_files": len(json_files),
        "body_graph_files": len(body_graph_files),
        "joint_set_files": len(joint_set_files),
        "split_files": [str(path) for path in split_files],
        "split_summaries": split_summaries,
        "positive_pair_records": len(positive_pairs),
        "unique_positive_pairs": len(unique_positive_pair_keys),
        "missing_body_graph_count": missing_body_graph_count,
        "missing_body_examples": missing_body_examples,
        "joint_count_distribution": dict(joint_count_counter),
        "group_summaries": group_summaries,
        "selected_group_mode": selected_mode,
        "same_group_negative_candidates": negative_candidate_count,
        "target_negative_count_1_to_1": len(positive_pairs),
        "top_negative_candidate_groups": group_negative_summary[:50],
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("[DONE] Report saved")
    print("=" * 80)
    print(f"Report path: {report_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default=r"C:\so_zero\j1.0.0_all\joint",
    )

    parser.add_argument(
        "--report",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\outputs\split_and_group_report.json",
    )

    args = parser.parse_args()

    inspect_dataset(
        dataset_dir=Path(args.dataset),
        report_path=Path(args.report),
    )


if __name__ == "__main__":
    main()
