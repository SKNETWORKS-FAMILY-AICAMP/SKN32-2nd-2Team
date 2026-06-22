"""Session Bounce 공통 유틸: 데이터 로드, 평가 산출물 저장 (계획서 §3.4/§3.6).

churn 평가 모듈(src/common/evaluation.py)은 label=churn/horizon=7 고정이라
session_bounce(churn30, 30분 horizon)는 본 모듈로 분리한다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[3]
DATA_NPZ = ROOT / "data" / "processed" / "session_bounce" / "gru" / "dataset.npz"
EVAL_DIR = ROOT / "data" / "processed" / "evaluation" / "session_bounce" / "gru"
MODEL_DIR = ROOT / "models" / "session_bounce" / "gru"


def load_dataset():
    z = np.load(DATA_NPZ)
    return {
        "X_num": z["X_num"].astype("float32"),
        "X_cat": z["X_cat"].astype("int64"),
        "y": z["y"].astype("int64"),
        "user_id": z["user_id"].astype("int64"),
        "split": z["split"].astype("int64"),
    }


def split_masks(split):
    return split == 0, split == 1, split == 2


def pick_threshold(y_true, y_score, step=0.01):
    """validation에서 F1 최대 임계값 선택 (test 사용 금지, 계획서 §3.4)."""
    ts = np.round(np.arange(0.05, 0.96, step), 4)
    best_t, best_f1 = 0.5, -1.0
    for t in ts:
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, (y_score >= t).astype(int), average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t


def first_event_bounce_recall(y_true, y_pred, x_cat):
    """첫 이벤트(시퀀스 유효길이 1: 카테고리 패딩이 9개) bounce recall (계획서 §3.4)."""
    is_first = (x_cat[:, :-1] == 0).all(axis=1)  # 마지막만 비패딩
    m = is_first & (y_true == 1)
    return float(y_pred[m].mean()) if m.sum() else None


def evaluate_and_save(eval_dir, *, model_name, y_true, y_score, user_id,
                      threshold, n_train, x_cat=None, training_history=None,
                      latency_ms=None, extra=None):
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    roc = float(roc_auc_score(y_true, y_score))
    pr = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())

    summary = {
        "model_name": model_name,
        "label_name": "churn30",
        "horizon_minutes": 30,
        "n_train": int(n_train),
        "n_test": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": roc,
        "pr_auc": pr,
        "brier": brier,
        "threshold": float(threshold),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if x_cat is not None:
        summary["first_event_bounce_recall"] = first_event_bounce_recall(y_true, y_pred, x_cat)
    if latency_ms is not None:
        summary["cpu_latency_ms_per_1k"] = float(latency_ms)
    if extra:
        summary.update(extra)

    (eval_dir / "metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (eval_dir / "training_history.json").write_text(
        json.dumps(training_history or {"epoch": [], "train_loss": [], "val_loss": []},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame({
        "user_id": np.asarray(user_id).astype("int64"),
        "y_true": y_true.astype("int32"),
        "y_score": y_score.astype("float64"),
        "y_pred": y_pred.astype("int32"),
        "split": "test",
        "threshold": float(threshold),
        "model_name": model_name,
    }).to_parquet(eval_dir / "eval_predictions.parquet", index=False)
    return summary
