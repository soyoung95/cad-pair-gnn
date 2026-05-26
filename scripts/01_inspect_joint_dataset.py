
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from pprint import pprint


def check_environment():
    print("=" * 80)
    print("[CHECK] Python and CUDA environment")
    print("=" * 80)

    try:
        import torch

        print(f"PyTorch version : {torch.__version__}")
        print(f"CUDA available  : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version    : {torch.version.cuda}")
            print(f"GPU count       : {torch.cuda.device_count()}")
            print(f"GPU name        : {torch.cuda.get_device_name(0)}")
        else:
            print("[INFO] CUDA is not available. This inspection script does not require GPU.")

    except Exception as e:
        print("[WARN] Failed to import PyTorch.")
        print(f"[WARN] Error type: {type(e).__name__}")
        print(f"[WARN] Error message: {e}")
        print("[INFO] This inspection script will continue without PyTorch.")


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


def summarize_value(value, max_items=3):
    if isinstance(value, dict):
        return {
            "type": "dict",
            "num_keys": len(value),
            "keys": list(value.keys())[:20],
        }

    if isinstance(value, list):
        sample = value[:max_items]
        return {
            "type": "list",
            "length": len(value),
            "sample_types": [type(x).__name__ for x in sample],
        }

    return {
        "type": type(value).__name__,
        "value": value,
    }


def summarize_json_structure(data, max_depth=2, depth=0):
    if depth >= max_depth:
        return summarize_value(data)

    if isinstance(data, dict):
        result = {}
        for key, value in list(data.items())[:50]:
            result[key] = summarize_json_structure(value, max_depth, depth + 1)
        return result

    if isinstance(data, list):
        if len(data) == 0:
            return {"type": "list", "length": 0}

        return {
            "type": "list",
            "length": len(data),
            "first_item": summarize_json_structure(data[0], max_depth, depth + 1),
        }

    return summarize_value(data)


def find_keys_recursive(obj, target_words, max_depth=4, depth=0, prefix=""):
    found = []

    if depth > max_depth:
        return found

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            key_lower = key_str.lower()
            full_key = f"{prefix}.{key_str}" if prefix else key_str

            if any(word in key_lower for word in target_words):
                found.append(full_key)

            found.extend(
                find_keys_recursive(
                    value,
                    target_words,
                    max_depth=max_depth,
                    depth=depth + 1,
                    prefix=full_key,
                )
            )

    elif isinstance(obj, list):
        for index, item in enumerate(obj[:3]):
            full_key = f"{prefix}[{index}]"
            found.extend(
                find_keys_recursive(
                    item,
                    target_words,
                    max_depth=max_depth,
                    depth=depth + 1,
                    prefix=full_key,
                )
            )

    return found


def extract_possible_group_ids(data):
    result = {}

    candidate_exact_keys = {
        "assembly_id",
        "assembly",
        "assembly_name",
        "design_id",
        "design",
        "dataset_id",
        "root_id",
        "root",
        "file_id",
        "folder",
    }

    def search(obj, prefix="", depth=0):
        if depth > 4:
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                key_str = str(key)
                key_lower = key_str.lower()
                full_key = f"{prefix}.{key_str}" if prefix else key_str

                matched = (
                    key_lower in candidate_exact_keys
                    or "assembly" in key_lower
                    or "design" in key_lower
                )

                if matched:
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        result[full_key] = value
                    else:
                        result[full_key] = summarize_value(value)

                search(value, full_key, depth + 1)

        elif isinstance(obj, list):
            for index, item in enumerate(obj[:3]):
                search(item, f"{prefix}[{index}]", depth + 1)

    search(data)
    return result


def inspect_dataset(dataset_dir: Path, max_files: int, report_path: Path):
    print("=" * 80)
    print("[CHECK] Dataset path")
    print("=" * 80)
    print(f"Dataset directory: {dataset_dir}")

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_dir}")

    json_files = sorted(dataset_dir.rglob("*.json"))

    print(f"Total JSON files: {len(json_files)}")

    if len(json_files) == 0:
        raise RuntimeError("No JSON files were found. Check the dataset path.")

    print("\n[INFO] First 10 JSON files:")
    for path in json_files[:10]:
        print(f" - {path}")

    scan_files = json_files[:max_files]
    print(f"\n[INFO] Number of JSON files to scan: {len(scan_files)} / {len(json_files)}")

    filename_counter = Counter()
    top_key_counter = Counter()

    has_body_one_two = 0
    has_joint_key = 0
    load_errors = []

    likely_joint_files = []
    likely_body_files = []
    likely_graph_files = []

    key_examples = defaultdict(list)
    group_id_examples = []

    for path in scan_files:
        name_lower = path.name.lower()

        if "joint" in name_lower:
            filename_counter["filename_contains_joint"] += 1
        if "body" in name_lower:
            filename_counter["filename_contains_body"] += 1
        if "graph" in name_lower:
            filename_counter["filename_contains_graph"] += 1

        data, error = safe_load_json(path)
        if error is not None:
            load_errors.append((str(path), error))
            continue

        if not isinstance(data, dict):
            continue

        keys = list(data.keys())
        lower_keys = {str(key).lower() for key in keys}

        for key in keys:
            top_key_counter[key] += 1

        if "body_one" in lower_keys and "body_two" in lower_keys:
            has_body_one_two += 1
            likely_joint_files.append(path)

        joint_keys = find_keys_recursive(data, ["joint"], max_depth=3)
        body_keys = find_keys_recursive(data, ["body"], max_depth=3)
        graph_keys = find_keys_recursive(data, ["graph"], max_depth=3)
        group_keys = find_keys_recursive(data, ["assembly", "design"], max_depth=3)

        if len(joint_keys) > 0:
            has_joint_key += 1
            likely_joint_files.append(path)

        if len(body_keys) > 0:
            likely_body_files.append(path)

        if len(graph_keys) > 0:
            likely_graph_files.append(path)

        for key in joint_keys[:10]:
            if len(key_examples[key]) < 3:
                key_examples[key].append(str(path))

        for key in body_keys[:10]:
            if len(key_examples[key]) < 3:
                key_examples[key].append(str(path))

        for key in graph_keys[:10]:
            if len(key_examples[key]) < 3:
                key_examples[key].append(str(path))

        for key in group_keys[:10]:
            if len(key_examples[key]) < 3:
                key_examples[key].append(str(path))

        group_candidates = extract_possible_group_ids(data)
        if group_candidates and len(group_id_examples) < 5:
            group_id_examples.append(
                {
                    "file": str(path),
                    "group_candidates": group_candidates,
                }
            )

    unique_joint_files = list(dict.fromkeys(likely_joint_files))
    unique_body_files = list(dict.fromkeys(likely_body_files))
    unique_graph_files = list(dict.fromkeys(likely_graph_files))

    print("\n" + "=" * 80)
    print("[RESULT] Filename pattern summary")
    print("=" * 80)
    pprint(filename_counter)

    print("\n" + "=" * 80)
    print("[RESULT] Top-level key frequency")
    print("=" * 80)
    for key, count in top_key_counter.most_common(50):
        print(f"{key}: {count}")

    print("\n" + "=" * 80)
    print("[RESULT] Core structure summary")
    print("=" * 80)
    print(f"Files with top-level body_one/body_two : {has_body_one_two}")
    print(f"Files with joint-related keys          : {has_joint_key}")
    print(f"Likely joint files                     : {len(unique_joint_files)}")
    print(f"Likely body files                      : {len(unique_body_files)}")
    print(f"Likely graph files                     : {len(unique_graph_files)}")

    print("\n[INFO] Example likely joint files:")
    for path in unique_joint_files[:10]:
        print(f" - {path}")

    print("\n" + "=" * 80)
    print("[RESULT] Key path examples")
    print("=" * 80)
    for key, paths in list(key_examples.items())[:80]:
        print(key)
        for path in paths:
            print(f"  - {path}")

    print("\n" + "=" * 80)
    print("[RESULT] Assembly or design group candidate examples")
    print("=" * 80)
    if len(group_id_examples) == 0:
        print("[WARN] No assembly or design group candidate was found.")
        print("[WARN] Same-assembly negative sampling may require another metadata file.")
    else:
        pprint(group_id_examples, width=120)

    print("\n" + "=" * 80)
    print("[RESULT] JSON load errors")
    print("=" * 80)
    if len(load_errors) == 0:
        print("No load errors.")
    else:
        print(f"Number of load errors: {len(load_errors)}")
        for path, error in load_errors[:10]:
            print(f" - {path}: {error}")

    print("\n" + "=" * 80)
    print("[SAMPLE] Representative JSON structure")
    print("=" * 80)

    sample_candidates = unique_joint_files[:3]
    if len(sample_candidates) == 0:
        sample_candidates = json_files[:3]

    sample_summaries = []

    for path in sample_candidates:
        data, error = safe_load_json(path)
        if error is not None:
            continue

        print(f"\n--- SAMPLE FILE: {path} ---")
        summary = summarize_json_structure(data, max_depth=2)
        pprint(summary, width=120)

        sample_summaries.append(
            {
                "file": str(path),
                "summary": summary,
            }
        )

    report = {
        "dataset_dir": str(dataset_dir),
        "total_json_files": len(json_files),
        "scanned_json_files": len(scan_files),
        "filename_counter": dict(filename_counter),
        "top_key_counter": dict(top_key_counter),
        "has_body_one_two": has_body_one_two,
        "has_joint_key": has_joint_key,
        "likely_joint_files_count": len(unique_joint_files),
        "likely_body_files_count": len(unique_body_files),
        "likely_graph_files_count": len(unique_graph_files),
        "likely_joint_files_examples": [str(path) for path in unique_joint_files[:30]],
        "group_id_examples": group_id_examples,
        "load_errors": load_errors[:50],
        "sample_summaries": sample_summaries,
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
        help="Path to the Assembly-Joint dataset directory.",
    )

    parser.add_argument(
        "--max_files",
        type=int,
        default=5000,
        help="Maximum number of JSON files to scan.",
    )

    parser.add_argument(
        "--report",
        type=str,
        default=r"C:\so_zero\cad_pair_gnn\outputs\dataset_inspection_report.json",
        help="Path to the output report JSON file.",
    )

    args = parser.parse_args()

    check_environment()

    inspect_dataset(
        dataset_dir=Path(args.dataset),
        max_files=args.max_files,
        report_path=Path(args.report),
    )


if __name__ == "__main__":
    main()
