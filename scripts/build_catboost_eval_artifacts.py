from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score, roc_curve


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data/processed/evaluation/churn/catboost"
PRED_PATH = EVAL_DIR / "eval_predictions.parquet"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    pred_df = pd.read_parquet(PRED_PATH)
    y_true = pred_df["y_true"].astype(int).to_numpy()
    y_score = pred_df["y_score"].astype(float).to_numpy()
    threshold = float(pred_df["threshold"].iloc[0])
    y_pred = (y_score >= threshold).astype(int)

    write_json(
        EVAL_DIR / "confusion_matrix.json",
        {
            "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
            "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
            "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
            "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        },
    )

    thresholds = np.round(np.arange(0.1, 0.91, 0.01), 2)
    write_json(
        EVAL_DIR / "threshold_curve.json",
        {
            "threshold": thresholds.tolist(),
            "precision": [float(precision_score(y_true, y_score >= t, zero_division=0)) for t in thresholds],
            "recall": [float(recall_score(y_true, y_score >= t, zero_division=0)) for t in thresholds],
            "f1": [float(f1_score(y_true, y_score >= t, zero_division=0)) for t in thresholds],
        },
    )

    roc_fpr, roc_tpr, roc_threshold = roc_curve(y_true, y_score)
    write_json(
        EVAL_DIR / "roc_curve.json",
        {
            "fpr": roc_fpr.tolist(),
            "tpr": roc_tpr.tolist(),
            "threshold": roc_threshold.tolist(),
        },
    )

    pr_precision, pr_recall, pr_threshold = precision_recall_curve(y_true, y_score)
    write_json(
        EVAL_DIR / "pr_curve.json",
        {
            "precision": pr_precision.tolist(),
            "recall": pr_recall.tolist(),
            "threshold": pr_threshold.tolist(),
        },
    )

    bin_edges = np.linspace(0.0, 1.0, 11)
    prob_pred = []
    prob_true = []
    counts = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_score >= left) & (y_score < right)
        if right == 1.0:
            mask = (y_score >= left) & (y_score <= right)
        if mask.any():
            prob_pred.append(float(y_score[mask].mean()))
            prob_true.append(float(y_true[mask].mean()))
            counts.append(int(mask.sum()))
    write_json(
        EVAL_DIR / "calibration_curve.json",
        {"prob_pred": prob_pred, "prob_true": prob_true, "count": counts},
    )

    bins = np.round(bin_edges, 2)
    non_churn_count = []
    churn_count = []
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_score >= left) & (y_score < right)
        if right == 1.0:
            mask = (y_score >= left) & (y_score <= right)
        non_churn_count.append(int(((y_true == 0) & mask).sum()))
        churn_count.append(int(((y_true == 1) & mask).sum()))
    write_json(
        EVAL_DIR / "score_distribution.json",
        {
            "bins": bins.tolist(),
            "non_churn_count": non_churn_count,
            "churn_count": churn_count,
        },
    )

    order = np.argsort(-y_score)
    sorted_true = y_true[order]
    total_positive = max(int(y_true.sum()), 1)
    top_percent = [1, 5, 10, 20, 30]
    capture_rate = []
    lift = []
    base_rate = float(y_true.mean())
    for pct in top_percent:
        n = max(int(len(y_true) * pct / 100), 1)
        positives = int(sorted_true[:n].sum())
        capture = positives / total_positive
        top_rate = positives / n
        capture_rate.append(float(capture))
        lift.append(float(top_rate / base_rate))
    write_json(
        EVAL_DIR / "lift_curve.json",
        {"top_percent": top_percent, "capture_rate": capture_rate, "lift": lift},
    )


if __name__ == "__main__":
    main()
