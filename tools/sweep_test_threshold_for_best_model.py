


from pathlib import Path
import argparse
import csv
import sys


import numpy as np
import pandas as pd




try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass




def compute_metrics(labels, scores, threshold):
    labels = labels.astype(int)
    preds = (scores >= threshold).astype(int)


    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())


    total = len(labels)


    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "pred_positive": int(preds.sum()),
        "pred_negative": int((preds == 0).sum()),
    }




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
        "--output_csv",
        default=(
            "outputs/diagnostics/best_model_test_threshold_sweep/"
            "test_threshold_sweep.csv"
        ),
    )
    args = parser.parse_args()


    predictions_csv = Path(args.predictions_csv)
    output_csv = Path(args.output_csv)


    if not predictions_csv.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {predictions_csv}")


    df = pd.read_csv(predictions_csv, encoding="utf-8-sig")


    labels = df["label"].astype(int).to_numpy()
    scores = df["score"].astype(float).to_numpy()


    rows = []


    for threshold in np.arange(0.05, 0.951, 0.005):
        rows.append(compute_metrics(labels, scores, float(threshold)))


    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values(
        ["accuracy", "f1"],
        ascending=[False, False],
    )


    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")


    print("=" * 100)
    print("TEST THRESHOLD SWEEP")
    print("=" * 100)
    print(f"predictions_csv: {predictions_csv}")
    print(f"output_csv     : {output_csv}")
    print()


    print("[TOP 20 BY ACCURACY]")
    print(result_df.head(20).to_string(index=False))


    print()
    print("[THRESHOLD 0.5]")
    print(result_df[np.isclose(result_df["threshold"], 0.5)].to_string(index=False))


    best = result_df.iloc[0]
    print()
    print("[BEST]")
    for key, value in best.items():
        print(f"{key}: {value}")




if __name__ == "__main__":
    main()



