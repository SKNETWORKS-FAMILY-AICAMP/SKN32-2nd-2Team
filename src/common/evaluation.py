"""y_true / y_score 로부터 19-3 §6 평가 산출물을 생성한다.

생성 파일(평가 디렉터리에 저장):
  metrics_summary.json, threshold_curve.json, calibration_curve.json,
  lift_curve.json, score_distribution.json, training_history.json,
  shap_summary.json, business_value.json, eval_predictions.parquet
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# business_value 기본 가정 (19-3 §6.10)
DEFAULT_ASSUMPTIONS = {"coupon_cost": 3000, "save_rate": 0.08, "avg_revenue": 42000}


def _dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _threshold_curve(y_true, y_score, thresholds):
    prec, rec, f1 = [], [], []
    for t in thresholds:
        yp = (y_score >= t).astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            y_true, yp, average="binary", zero_division=0
        )
        prec.append(float(p))
        rec.append(float(r))
        f1.append(float(f))
    return prec, rec, f1


def _calibration(y_true, y_score, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    prob_pred, prob_true, count = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score >= lo) & (y_score < hi) if i < n_bins - 1 else (y_score >= lo) & (y_score <= hi)
        c = int(mask.sum())
        count.append(c)
        prob_pred.append(float(y_score[mask].mean()) if c else float((lo + hi) / 2))
        prob_true.append(float(y_true[mask].mean()) if c else 0.0)
    return prob_pred, prob_true, count


def _lift_curve(y_true, y_score, top_percents):
    order = np.argsort(-y_score)
    yt = np.asarray(y_true)[order]
    n = len(yt)
    total_pos = int(yt.sum())
    cap, lift = [], []
    for pct in top_percents:
        k = max(1, int(n * pct / 100))
        captured = float(yt[:k].sum() / total_pos) if total_pos else 0.0
        cap.append(captured)
        lift.append(float(captured / (k / n)) if k else 0.0)
    return cap, lift


def _score_distribution(y_true, y_score, n_bins=10):
    edges = np.round(np.linspace(0.0, 1.0, n_bins + 1), 2)
    yt = np.asarray(y_true)
    non_churn, churn = [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_score >= lo) & (y_score < hi) if i < n_bins - 1 else (y_score >= lo) & (y_score <= hi)
        non_churn.append(int(((yt == 0) & mask).sum()))
        churn.append(int(((yt == 1) & mask).sum()))
    return [float(b) for b in edges[:-1]], non_churn, churn


def _business_value(y_true, y_score, assumptions):
    a = assumptions or DEFAULT_ASSUMPTIONS
    order = np.argsort(-y_score)
    yt = np.asarray(y_true)[order]
    n = len(yt)
    top_percent = [5, 10, 20]
    target_users, value_at_risk, expected_recovery = [], [], []
    for pct in top_percent:
        k = max(1, int(n * pct / 100))
        churners = int(yt[:k].sum())
        var = churners * a["avg_revenue"]
        recovery = var * a["save_rate"] - k * a["coupon_cost"]
        target_users.append(k)
        value_at_risk.append(int(var))
        expected_recovery.append(int(recovery))
    return {
        "assumptions": a,
        "top_percent": top_percent,
        "target_users": target_users,
        "value_at_risk": value_at_risk,
        "expected_recovery": expected_recovery,
    }


def evaluate_and_save(
    eval_dir,
    *,
    model_name,
    model_key,
    model_type,
    user_id,
    y_true,
    y_score,
    n_train,
    training_history=None,
    shap_summary=None,
    business_assumptions=None,
    split="test",
    fixed_threshold=None,
    threshold_grid_step=0.05,
):
    """평가 산출물 9종을 저장하고 metrics 요약 dict를 반환한다.

    산출물 ↔ 대시보드 차트 매핑:
      metrics_summary(#4,#10) · threshold_curve(#9) · calibration_curve(#12) ·
      lift_curve(#11) · score_distribution(#8) · shap_summary(#13) ·
      business_value(#14,#15) · training_history(#5) · eval_predictions(#6,#7 재계산 원천)

    fixed_threshold: 운영 임계값을 외부(예: train OOF)에서 정해 넘기면 그 값으로 고정한다.
      None이면 평가셋 F1 최대점을 자동 선택(기존 동작). 평가셋으로 임계값을 고르는 건 누수이므로
      운영에선 train OOF에서 구한 값을 넘기는 것을 권장(19-3 §11.2).
    threshold_grid_step: threshold_curve와 자동선택에 쓰는 그리드 간격(기본 0.05).
    """
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    # 임계값 그리드(곡선 + 자동선택용). PR-AUC/ROC는 임계값 무관(모델 품질),
    # precision/recall/f1/confusion만 이 임계값 기준값.
    thresholds = np.round(np.arange(0.05, 0.96, threshold_grid_step), 2)
    prec, rec, f1 = _threshold_curve(y_true, y_score, thresholds)
    if fixed_threshold is not None:
        # 운영 임계값 고정(누수 없음). 그리드에 없을 수 있어 직접 P/R/F1 계산.
        best_t = float(fixed_threshold)
        bp, br, bf, _ = precision_recall_fscore_support(
            y_true, (y_score >= best_t).astype(int), average="binary", zero_division=0)
        best_prec, best_rec, best_f1 = float(bp), float(br), float(bf)
    else:
        best_i = int(np.argmax(f1))
        best_t = float(thresholds[best_i])
        best_prec, best_rec, best_f1 = prec[best_i], rec[best_i], f1[best_i]
    y_pred = (y_score >= best_t).astype(int)

    # PR-AUC = average_precision, ROC-AUC = roc_auc_score (둘 다 임계값 무관 = 순위 품질).
    roc = float(roc_auc_score(y_true, y_score))
    pr = float(average_precision_score(y_true, y_score))
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel())

    _dump(
        eval_dir / "metrics_summary.json",
        {
            "model_name": model_name,
            "model_key": model_key,
            "model_type": model_type,
            "label_name": "churn",
            "horizon_days": 7,
            "n_train": int(n_train),
            "n_test": int(len(y_true)),
            "positive_rate": float(y_true.mean()),
            "roc_auc": roc,
            "pr_auc": pr,
            "best_threshold": best_t,
            "precision": best_prec,
            "recall": best_rec,
            "f1": best_f1,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        },
    )

    _dump(
        eval_dir / "threshold_curve.json",
        {"threshold": [float(t) for t in thresholds], "precision": prec, "recall": rec, "f1": f1},
    )

    prob_pred, prob_true, cal_count = _calibration(y_true, y_score)
    _dump(
        eval_dir / "calibration_curve.json",
        {"prob_pred": prob_pred, "prob_true": prob_true, "count": cal_count},
    )

    top_percents = [1, 5, 10, 20, 30]
    cap, lift = _lift_curve(y_true, y_score, top_percents)
    _dump(eval_dir / "lift_curve.json", {"top_percent": top_percents, "capture_rate": cap, "lift": lift})

    bins, non_churn, churn = _score_distribution(y_true, y_score)
    _dump(
        eval_dir / "score_distribution.json",
        {"bins": bins, "non_churn_count": non_churn, "churn_count": churn},
    )

    _dump(
        eval_dir / "training_history.json",
        training_history or {"epoch": [], "train_loss": [], "val_loss": []},
    )

    if shap_summary is not None:
        _dump(eval_dir / "shap_summary.json", shap_summary)

    _dump(eval_dir / "business_value.json", _business_value(y_true, y_score, business_assumptions))

    pd.DataFrame(
        {
            "user_id": np.asarray(user_id).astype("int64"),
            "y_true": y_true.astype("int32"),
            "y_score": y_score.astype("float64"),
            "y_pred": y_pred.astype("int32"),
            "split": split,
            "threshold": best_t,
            "model_name": model_name,
        }
    ).to_parquet(eval_dir / "eval_predictions.parquet", index=False)

    return {
        "roc_auc": roc,
        "pr_auc": pr,
        "best_threshold": best_t,
        "best_f1": best_f1,
        "precision": best_prec,
        "recall": best_rec,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
