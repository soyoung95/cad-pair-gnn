




from pathlib import Path
import argparse
import csv
import json
import sys


import pandas as pd




try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass




def compute_binary_metrics(labels, scores, threshold):
    labels = labels.astype(int)
    preds = (scores >= threshold).astype(int)


    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())


    total = int(len(labels))
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


    return {
        "count": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_positive": int(preds.sum()),
        "pred_negative": int((preds == 0).sum()),
        "actual_positive": int((labels == 1).sum()),
        "actual_negative": int((labels == 0).sum()),
    }




def summarize_group(df, group_name, threshold):
    labels = df["label"].astype(int)
    scores = df["score"].astype(float)


    metrics = compute_binary_metrics(labels=labels, scores=scores, threshold=threshold)
    metrics["group"] = group_name
    metrics["threshold"] = threshold


    if len(df) > 0:
        metrics["score_mean"] = float(df["score"].mean())
        metrics["score_p25"] = float(df["score"].quantile(0.25))
        metrics["score_p50"] = float(df["score"].quantile(0.50))
        metrics["score_p75"] = float(df["score"].quantile(0.75))
    else:
        metrics["score_mean"] = ""
        metrics["score_p25"] = ""
        metrics["score_p50"] = ""
        metrics["score_p75"] = ""


    return metrics




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions_csv",
        default=(
            "outputs/professor_hparam_random_tuning/full_all36_seed42/"
            "prof_rand_cfg028_seed42_h128_b32_l2_g8_c8_d0p1_lr0p001_wd0p0001_ep40/"
            "seed_42/test_eval/test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--difficulty_csv",
        default="outputs/diagnostics/negative_pair_difficulty/pair_difficulty_rows.csv",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/diagnostics/test_errors_by_negative_difficulty",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()


    predictions_csv = Path(args.predictions_csv)
    difficulty_csv = Path(args.difficulty_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    if not predictions_csv.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {predictions_csv}")


    if not difficulty_csv.exists():
        raise FileNotFoundError(f"Missing difficulty CSV: {difficulty_csv}")


    pred_df = pd.read_csv(predictions_csv, encoding="utf-8-sig")
    diff_df = pd.read_csv(difficulty_csv, encoding="utf-8-sig")


    test_diff = diff_df[diff_df["split"] == "test"].copy().reset_index(drop=True)
    pred_df = pred_df.copy().reset_index(drop=True)


    print("=" * 100)
    print("TEST ERRORS BY NEGATIVE DIFFICULTY")
    print("=" * 100)
    print(f"predictions_csv: {predictions_csv}")
    print(f"difficulty_csv : {difficulty_csv}")
    print(f"test_diff rows : {len(test_diff)}")
    print(f"pred rows      : {len(pred_df)}")


    if len(test_diff) != len(pred_df):
        raise ValueError(
            f"Row count mismatch. test difficulty rows={len(test_diff)}, predictions rows={len(pred_df)}"
        )


    if "label" not in pred_df.columns:
        raise KeyError(f"Prediction CSV must contain label column. Columns: {list(pred_df.columns)}")


    if "score" not in pred_df.columns:
        raise KeyError(f"Prediction CSV must contain score column. Columns: {list(pred_df.columns)}")


    label_match = (test_diff["label"].astype(int).values == pred_df["label"].astype(int).values).mean()
    print(f"label_order_match_ratio: {label_match}")


    if label_match < 0.99:
        print("[WARNING] Label order may not match. Check pair IDs before trusting the merge.")


    merged = test_diff.copy()
    merged["score"] = pred_df["score"].astype(float)
    merged["pred_05"] = (merged["score"] >= args.threshold).astype(int)
    merged["case_type"] = "UNKNOWN"


    merged.loc[(merged["label"] == 1) & (merged["pred_05"] == 1), "case_type"] = "TP"
    merged.loc[(merged["label"] == 1) & (merged["pred_05"] == 0), "case_type"] = "FN"
    merged.loc[(merged["label"] == 0) & (merged["pred_05"] == 0), "case_type"] = "TN"
    merged.loc[(merged["label"] == 0) & (merged["pred_05"] == 1), "case_type"] = "FP"


    summary_rows = []


    summary_rows.append(summarize_group(merged, "all_test", args.threshold))
    summary_rows.append(summarize_group(merged[merged["label"] == 1], "positive_all", args.threshold))
    summary_rows.append(summarize_group(merged[merged["label"] == 0], "negative_all", args.threshold))


    for difficulty in ["easy_like", "medium_like", "hard_like"]:
        subset = merged[(merged["label"] == 0) & (merged["difficulty_proxy"] == difficulty)]
        summary_rows.append(summarize_group(subset, f"negative_{difficulty}", args.threshold))


    summary_df = pd.DataFrame(summary_rows)


    case_counts = (
        merged.groupby(["difficulty_proxy", "label", "case_type"])
        .size()
        .reset_index(name="count")
        .sort_values(["difficulty_proxy", "label", "case_type"])
    )


    merged_path = output_dir / "test_predictions_with_difficulty.csv"
    summary_path = output_dir / "test_metrics_by_difficulty.csv"
    case_counts_path = output_dir / "case_counts_by_difficulty.csv"


    merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    case_counts.to_csv(case_counts_path, index=False, encoding="utf-8-sig")


    print()
    print("[SUMMARY]")
    print(summary_df.to_string(index=False))


    print()
    print("[CASE COUNTS]")
    print(case_counts.to_string(index=False))


    print()
    print("[SAVED]")
    print(f"merged     : {merged_path}")
    print(f"summary    : {summary_path}")
    print(f"case counts: {case_counts_path}")




if __name__ == "__main__":
    main()


