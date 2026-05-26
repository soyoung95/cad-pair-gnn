


from pathlib import Path
import argparse
import json
import random
import sys
from typing import Any




try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass




LABEL_KEYS = ["label", "y", "target"]
BODY_ONE_KEYS = ["body_one", "body_a", "part_one", "part_a", "source_body", "body1"]
BODY_TWO_KEYS = ["body_two", "body_b", "part_two", "part_b", "target_body", "body2"]




def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)




def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)




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




def get_assembly_key(record: dict[str, Any]) -> str:
    return str(record.get("assembly_key", "")).strip()




def get_body_one(record: dict[str, Any]) -> str:
    return str(get_first(record, BODY_ONE_KEYS)).strip()




def get_body_two(record: dict[str, Any]) -> str:
    return str(get_first(record, BODY_TWO_KEYS)).strip()




def collect_body_pool(records: list[dict[str, Any]]) -> list[dict[str, str]]:
    pool = []
    seen = set()


    for record in records:
        assembly_key = get_assembly_key(record)


        for body_id in [get_body_one(record), get_body_two(record)]:
            key = (assembly_key, body_id)
            if body_id and key not in seen:
                seen.add(key)
                pool.append(
                    {
                        "assembly_key": assembly_key,
                        "body_id": body_id,
                    }
                )


    return pool




def make_easy_negative(
    positive_record: dict[str, Any],
    body_pool: list[dict[str, str]],
    rng: random.Random,
    split: str,
    index: int,
) -> dict[str, Any]:
    anchor_assembly = get_assembly_key(positive_record)
    body_one = get_body_one(positive_record)


    candidates = [
        item for item in body_pool
        if item["assembly_key"] != anchor_assembly and item["body_id"] != body_one
    ]


    if not candidates:
        raise ValueError(f"No cross-assembly candidate found for split={split}, index={index}")


    sampled = rng.choice(candidates)


    return {
        "assembly_key": anchor_assembly,
        "pair_key": f"easy_negative_{split}_{index:06d}",
        "body_one": body_one,
        "body_two": sampled["body_id"],
        "label": 0,
        "negative_type": "negative_cross_assembly_random_easy",
        "source_positive_pair_key": str(positive_record.get("pair_key", "")),
        "source_body_two_original": get_body_two(positive_record),
        "sampled_negative_assembly_key": sampled["assembly_key"],
    }




def normalize_positive(record: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    return {
        "assembly_key": get_assembly_key(record),
        "pair_key": str(record.get("pair_key", f"positive_{split}_{index:06d}")),
        "body_one": get_body_one(record),
        "body_two": get_body_two(record),
        "label": 1,
        "pair_type": "positive_original",
    }




def process_split(
    split: str,
    input_path: Path,
    output_path: Path,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)


    data = load_json(input_path)
    records = iter_pair_records(data)


    positives = [record for record in records if parse_label(get_first(record, LABEL_KEYS)) == 1]
    negatives = [record for record in records if parse_label(get_first(record, LABEL_KEYS)) == 0]


    body_pool = collect_body_pool(records)


    output_records = []


    for idx, positive in enumerate(positives):
        output_records.append(normalize_positive(positive, split=split, index=idx))


    for idx, positive in enumerate(positives):
        output_records.append(
            make_easy_negative(
                positive_record=positive,
                body_pool=body_pool,
                rng=rng,
                split=split,
                index=idx,
            )
        )


    rng.shuffle(output_records)


    save_json(output_path, output_records)


    summary = {
        "split": split,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_records": len(records),
        "input_positive": len(positives),
        "input_negative": len(negatives),
        "output_records": len(output_records),
        "output_positive": len(positives),
        "output_negative_easy": len(positives),
        "body_pool_count": len(body_pool),
        "seed": seed,
    }


    return summary




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_pair_index_dir", default="outputs/pair_index")
    parser.add_argument("--output_pair_index_dir", default="outputs/pair_index_easy_negative")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    input_dir = Path(args.input_pair_index_dir)
    output_dir = Path(args.output_pair_index_dir)


    split_files = {
        "train": "train_pairs.json",
        "validation": "validation_pairs.json",
        "test": "test_pairs.json",
    }


    print("=" * 100)
    print("BUILD EASY NEGATIVE PAIR INDEX")
    print("=" * 100)
    print(f"input_pair_index_dir : {input_dir}")
    print(f"output_pair_index_dir: {output_dir}")
    print(f"seed                 : {args.seed}")


    summaries = []


    for split, filename in split_files.items():
        input_path = input_dir / filename
        output_path = output_dir / filename


        if not input_path.exists():
            raise FileNotFoundError(f"Missing input pair file: {input_path}")


        summary = process_split(
            split=split,
            input_path=input_path,
            output_path=output_path,
            seed=args.seed,
        )


        summaries.append(summary)


        print()
        print(f"[{split}]")
        for key, value in summary.items():
            print(f"{key}: {value}")


    manifest = {
        "description": "Pair index with cross-assembly random easy negatives. This is a diagnostic dataset.",
        "input_pair_index_dir": str(input_dir),
        "output_pair_index_dir": str(output_dir),
        "seed": args.seed,
        "splits": summaries,
    }


    save_json(output_dir / "manifest_easy_negative.json", manifest)


    print()
    print("=" * 100)
    print("DONE")
    print("=" * 100)
    print(f"manifest: {output_dir / 'manifest_easy_negative.json'}")




if __name__ == "__main__":
    main()


