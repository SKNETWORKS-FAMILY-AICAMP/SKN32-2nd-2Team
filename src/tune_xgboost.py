"""XGBoost 하이퍼파라미터 튜닝(CV 전용).

훈련 코호트 내부 Stratified 5-Fold CV만 사용하며 OOT test는 읽지 않는다.
각 후보의 파라미터와 CV 점수를 아래 CSV에 매 시행마다 누적 저장한다.

    data/processed/evaluation/churn/xgboost/runs/_cv_tuning.csv

실행:
    python -m src.tune_xgboost --n-iter 60 --n-jobs 4
"""
import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.base import clone
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import (
    ParameterSampler,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.common.data import FEATURE_ORDER_V2, load_tabular_v2

SEED = 42
KST = timezone(timedelta(hours=9))
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed" / "evaluation" / "churn" / "xgboost" / "runs"
CSV_PATH = OUT_DIR / "_cv_tuning.csv"
SUMMARY_PATH = OUT_DIR / "_cv_tuning_best.json"

# 현재 정식 후보. 무작위 탐색에 포함되지 않더라도 반드시 기준점으로 평가한다.
BASELINE = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.0359,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 20,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 10.0,
    "scale_pos_weight": 1.0,
}

SEARCH_SPACE = {
    "n_estimators": [120, 180, 250, 320, 400, 500],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.015, 0.02, 0.025, 0.03, 0.0359, 0.045, 0.06, 0.08],
    "subsample": [0.65, 0.75, 0.85, 1.0],
    "colsample_bytree": [0.65, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 5, 10, 20, 40, 80],
    "gamma": [0.0, 0.05, 0.1, 0.3, 0.5],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0, 2.0],
    "reg_lambda": [1.0, 3.0, 5.0, 10.0, 20.0, 40.0],
    # churn=1이 다수 클래스이므로 양성 가중치를 1보다 크게 두지 않는다.
    "scale_pos_weight": [0.5, 0.75, 1.0],
}

REFINE_CENTER = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.06,
    "subsample": 0.85,
    "colsample_bytree": 0.9,
    "min_child_weight": 40,
    "gamma": 0.05,
    "reg_alpha": 1.0,
    "reg_lambda": 20.0,
    "scale_pos_weight": 0.75,
}

REFINE_SPACE = {
    "n_estimators": [180, 220, 250, 280, 320, 400],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.04, 0.05, 0.06, 0.07, 0.08],
    "subsample": [0.8, 0.85, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.85, 0.9, 0.95, 1.0],
    "min_child_weight": [20, 30, 40, 60, 80],
    "gamma": [0.0, 0.03, 0.05, 0.1, 0.2],
    "reg_alpha": [0.5, 1.0, 1.5, 2.0],
    "reg_lambda": [10.0, 15.0, 20.0, 30.0, 40.0],
    "scale_pos_weight": [0.65, 0.75, 0.85, 1.0],
}


def make_pipeline(params):
    model = xgb.XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=1,
    )
    return Pipeline([("scaler", RobustScaler()), ("model", model)])


def append_result(row):
    """중단돼도 완료된 시행은 남도록 매 trial 후 CSV를 갱신한다."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if CSV_PATH.exists():
        old = pd.read_csv(CSV_PATH)
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")


def best_oof_threshold(estimator, X, y, cv, n_jobs):
    score = cross_val_predict(
        estimator, X, y, cv=cv, method="predict_proba", n_jobs=n_jobs
    )[:, 1]
    thresholds = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1 = []
    for threshold in thresholds:
        _, _, value, _ = precision_recall_fscore_support(
            y,
            (score >= threshold).astype(int),
            average="binary",
            zero_division=0,
        )
        f1.append(float(value))
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])


def main():
    parser = argparse.ArgumentParser(description="XGBoost CV tuner")
    parser.add_argument("--n-iter", type=int, default=60)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--profile", choices=["broad", "refine"], default="broad"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="기존 _cv_tuning.csv를 지우고 새 탐색을 시작",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset and CSV_PATH.exists():
        CSV_PATH.unlink()

    (X, y, _), _ = load_tabular_v2("xgboost")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search_space = SEARCH_SPACE if args.profile == "broad" else REFINE_SPACE
    candidates = [BASELINE]
    if args.profile == "refine":
        candidates.append(REFINE_CENTER)
    candidates.extend(
        list(ParameterSampler(search_space, n_iter=args.n_iter, random_state=SEED))
    )

    run_id = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
    for trial, params in enumerate(candidates, start=1):
        started = time.perf_counter()
        estimator = make_pipeline(params)
        scores = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"},
            return_train_score=True,
            n_jobs=args.n_jobs,
            pre_dispatch=args.n_jobs,
        )
        row = {
            "run_id": run_id,
            "trial": trial,
            "is_baseline": trial == 1,
            "sklearn_version": sklearn.__version__,
            "xgboost_version": xgb.__version__,
            "cv_pr_auc_mean": float(scores["test_pr_auc"].mean()),
            "cv_pr_auc_std": float(scores["test_pr_auc"].std(ddof=1)),
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std(ddof=1)),
            "train_pr_auc_mean": float(scores["train_pr_auc"].mean()),
            "fit_time_sec": float(scores["fit_time"].sum()),
            "wall_time_sec": float(time.perf_counter() - started),
            **params,
        }
        append_result(row)
        print(
            f"[{trial:03d}/{len(candidates):03d}] "
            f"PR={row['cv_pr_auc_mean']:.6f} ROC={row['cv_roc_auc_mean']:.6f} "
            f"depth={params['max_depth']} trees={params['n_estimators']} "
            f"lr={params['learning_rate']}"
        )

    results = pd.read_csv(CSV_PATH)
    current = results[results["run_id"] == run_id].copy()
    current = current.sort_values(
        ["cv_pr_auc_mean", "cv_roc_auc_mean"], ascending=False
    )

    # 노이즈 가드: CV PR-AUC std가 ~0.002라 80후보 중 '최대'만 고르면 운 좋은 노이즈를 집는다.
    # 최고점의 1-std 이내(통계적 동률) 후보들 중 '가장 단순한' 모델을 선택한다.
    top = current.iloc[0]
    tol = float(top["cv_pr_auc_mean"]) - float(top["cv_pr_auc_std"])
    tied = current[current["cv_pr_auc_mean"] >= tol].copy()
    tied["_complexity"] = tied["n_estimators"] * tied["max_depth"]
    tied = tied.sort_values(["_complexity", "n_estimators", "max_depth", "learning_rate"])
    best_row = tied.iloc[0]
    baseline_tied = bool((tied["is_baseline"] == True).any())  # noqa: E712
    print(f"[select] 1-std 내 동률 {len(tied)}개 → 최단순 선택"
          f"(trees={int(best_row['n_estimators'])}, depth={int(best_row['max_depth'])}); "
          f"baseline_within_tie={baseline_tied}")
    int_params = {"n_estimators", "max_depth", "min_child_weight"}
    best_params = {
        key: int(best_row[key]) if key in int_params else float(best_row[key])
        for key in search_space
    }
    best_estimator = make_pipeline(best_params)
    threshold, oof_f1 = best_oof_threshold(
        best_estimator, X, y.to_numpy(), cv, args.n_jobs
    )

    summary = {
        "run_id": run_id,
        "selection_metric": "5-fold CV average_precision",
        "selection_rule": "best CV PR-AUC의 1-std 이내 동률 중 최단순 모델",
        "n_tied_within_1std": int(len(tied)),
        "baseline_within_tie": baseline_tied,
        "profile": args.profile,
        "n_candidates": len(candidates),
        "n_train": int(len(y)),
        "positive_rate": float(y.mean()),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "best_params": best_params,
        "cv_pr_auc_mean": float(best_row["cv_pr_auc_mean"]),
        "cv_pr_auc_std": float(best_row["cv_pr_auc_std"]),
        "cv_roc_auc_mean": float(best_row["cv_roc_auc_mean"]),
        "cv_roc_auc_std": float(best_row["cv_roc_auc_std"]),
        "oof_best_threshold": threshold,
        "oof_best_f1": oof_f1,
        "csv_path": str(CSV_PATH.relative_to(ROOT)).replace("\\", "/"),
    }
    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
