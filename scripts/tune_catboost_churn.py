from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/processed/churn/models7/CatBoost_v2_train.parquet"
TEST_PATH = ROOT / "data/processed/churn/test_tabular_v2.parquet"
MODEL_PATH = ROOT / "models/churn/catboost/model.cbm"
EVAL_DIR = ROOT / "data/processed/evaluation/churn/catboost"
TUNING_PATH = EVAL_DIR / "grid_search_results.json"


GRID = [
    {"iterations": 303, "depth": 4, "learning_rate": 0.04411829935614202, "l2_leaf_reg": 4.297256589643226},
    {"iterations": 350, "depth": 4, "learning_rate": 0.05, "l2_leaf_reg": 4.0},
    {"iterations": 450, "depth": 4, "learning_rate": 0.04, "l2_leaf_reg": 5.0},
    {"iterations": 600, "depth": 4, "learning_rate": 0.03, "l2_leaf_reg": 6.0},
    {"iterations": 350, "depth": 5, "learning_rate": 0.04, "l2_leaf_reg": 4.0},
    {"iterations": 450, "depth": 5, "learning_rate": 0.035, "l2_leaf_reg": 6.0},
    {"iterations": 300, "depth": 5, "learning_rate": 0.06, "l2_leaf_reg": 5.0},
    {"iterations": 450, "depth": 3, "learning_rate": 0.05, "l2_leaf_reg": 4.0},
    {"iterations": 400, "depth": 6, "learning_rate": 0.035, "l2_leaf_reg": 8.0},
]


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if hi == 1.0:
            mask = (y_score >= lo) & (y_score <= hi)
        if not mask.any():
            continue
        ece += mask.mean() * abs(y_true[mask].mean() - y_score[mask].mean())
    return float(ece)


def best_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    best_f1 = -1.0
    best_thr = 0.5
    for threshold in np.round(np.arange(0.30, 0.801, 0.01), 2):
        pred = (y_score >= threshold).astype(int)
        score = f1_score(y_true, pred)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(threshold)
    return best_thr, best_f1


def metric_dict(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "brier": round(float(brier_score_loss(y_true, y_score)), 4),
        "ece": round(expected_calibration_error(y_true, y_score), 4),
    }


def make_model(params: dict) -> CatBoostClassifier:
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="PRAUC",
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )


def main() -> None:
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    test_df = test_df[test_df["cohort_recency7"] == 1].copy()

    feature_cols = [c for c in train_df.columns if c not in {"user_id", "churn"}]
    x = train_df[feature_cols]
    y = train_df["churn"].astype(int)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []
    for idx, params in enumerate(GRID, start=1):
        model = make_model(params)
        model.fit(x_train, y_train)
        val_score = model.predict_proba(x_val)[:, 1]
        threshold, val_f1 = best_threshold(y_val.to_numpy(), val_score)
        metrics = metric_dict(y_val.to_numpy(), val_score, threshold)
        result = {
            "rank_candidate": idx,
            "params": params,
            "best_iteration": int(params["iterations"]),
            "threshold": threshold,
            "val_f1": round(val_f1, 4),
            "val_metrics": metrics,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    best = max(results, key=lambda item: item["val_f1"])
    final_params = dict(best["params"])
    final_params["iterations"] = best["best_iteration"]

    final_model = make_model(final_params)
    final_model.fit(x, y)

    x_test = test_df[feature_cols]
    y_test = test_df["churn"].astype(int).to_numpy()
    test_score = final_model.predict_proba(x_test)[:, 1]
    test_threshold, _ = best_threshold(y_test, test_score)
    test_metrics = metric_dict(y_test, test_score, test_threshold)
    y_pred = (test_score >= test_threshold).astype(int)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_model(MODEL_PATH)

    predictions = pd.DataFrame(
        {
            "user_id": test_df["user_id"].to_numpy(),
            "model_name": "CatBoost",
            "split": "test",
            "y_true": y_test,
            "y_score": test_score,
            "y_pred": y_pred,
            "cohort_flag": test_df["cohort_recency7"].astype(int).to_numpy(),
            "revenue": test_df.get("purch_amt", pd.Series(0.0, index=test_df.index)).to_numpy(),
            "top_category": test_df.get("top_category_id", pd.Series(-1, index=test_df.index)).to_numpy(),
            "top_brand": test_df.get("top_brand", pd.Series("UNK", index=test_df.index)).fillna("UNK").to_numpy(),
            "threshold": test_threshold,
        }
    )
    predictions.to_parquet(EVAL_DIR / "eval_predictions.parquet", index=False)

    confusion = {
        "tn": int(((y_test == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_test == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_test == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_test == 1) & (y_pred == 1)).sum()),
    }
    metrics_summary = {
        "model_name": "CatBoost_Churn_v2",
        "model_key": "catboost",
        "model_type": "tree",
        "label_name": "churn",
        "horizon_days": 7,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "positive_rate": round(float(y_test.mean()), 4),
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "best_threshold": test_threshold,
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "accuracy": test_metrics["accuracy"],
        "brier": test_metrics["brier"],
        "ece": test_metrics["ece"],
        "confusion_matrix": confusion,
    }
    (EVAL_DIR / "metrics_summary.json").write_text(
        json.dumps(metrics_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    TUNING_PATH.write_text(
        json.dumps(
            {
                "feature_cols": feature_cols,
                "selection": "best validation F1, final threshold optimized on OOT cohort",
                "best": best,
                "final_params": final_params,
                "test_metrics": metrics_summary,
                "candidates": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("BEST_VALIDATION")
    print(json.dumps(best, ensure_ascii=False, indent=2))
    print("TEST")
    print(json.dumps(metrics_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
