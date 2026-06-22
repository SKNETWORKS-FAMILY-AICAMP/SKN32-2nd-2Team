# -*- coding: utf-8 -*-
"""infrastructure/files — 모델팀 산출물(19-3 §6) artifact-first 리더 + 19-4 §9 chart 변환.
우선순위: per-model evaluation/churn/{key}/*.json → 집계 curves.json → 빈 상태.
백엔드는 재학습/재계산하지 않고, 학습 종료 시점 산출물을 chart-ready JSON으로 변환만 한다."""
import json
from app.config import EVAL_DIR

CHURN_DIR = EVAL_DIR / "churn"
FALLBACK_CHURN_DIR = EVAL_DIR / "_fallback" / "churn"   # primary 파일 없을 때 대체(이전 산출물)

# 모델명 → model_key (19-3)
KEY = {"catboost": "catboost", "lightgbm": "lightgbm", "xgboost": "xgboost",
       "randomforest": "randomforest", "decisiontree": "decisiontree",
       "logreg": "logreg", "transformer": "transformer"}
NAME = {"CatBoost": "catboost", "LightGBM": "lightgbm", "XGBoost": "xgboost",
        "RandomForest": "randomforest", "DecisionTree": "decisiontree",
        "LogReg": "logreg", "Transformer": "transformer"}


def _json(p):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def resolve_key(model):
    """model = 이름('CatBoost')·model_key('catboost')·등록명('CatBoost_Churn_v2') 모두 허용."""
    if model in NAME:
        return NAME[model]
    s = str(model).lower().replace("_churn_v2", "").replace("_v2", "")
    return KEY.get(s, s)


def _artifact(mkey, fname):
    """primary(모델팀 최신 산출물) 우선, 없으면 _fallback(이전 산출물) 사용."""
    p = CHURN_DIR / mkey / fname
    if not p.exists():
        p = FALLBACK_CHURN_DIR / mkey / fname
    return _json(p)


def _wrap(chart_name, chart_type, x, y, data, model=None, source="artifact", **extra):
    out = {"chart_name": chart_name, "chart_type": chart_type, "x": x, "y": y,
           "data": data, "meta": {"schema_version": "chart.v1", "source": source}}
    if model:
        out["model"] = model
    out.update(extra)
    return out


# ---------- per-model charts (19-4 §8 5~15) ----------
def model_chart(model, chart_name):
    """kebab chart_name → 19-4 §9 chart JSON. 없으면 빈 data."""
    mkey = resolve_key(model)
    cn = chart_name.replace("_", "-")

    if cn in ("roc-auc", "roc"):
        a = _artifact(mkey, "roc_curve.json")
        rows = [{"fpr": f, "tpr": t} for f, t in zip(a["fpr"], a["tpr"])] if a else []
        return _wrap("roc_auc", "line", "fpr", "tpr", rows, model)
    if cn in ("pr-auc", "pr"):
        a = _artifact(mkey, "pr_curve.json")
        rows = [{"recall": r, "precision": p} for r, p in zip(a["recall"], a["precision"])] if a else []
        return _wrap("pr_auc", "line", "recall", "precision", rows, model)
    if cn == "threshold":
        a = _artifact(mkey, "threshold_curve.json")
        rows = []
        if isinstance(a, dict):
            thr = a.get("threshold") or a.get("thresholds") or []       # 신 스키마=thresholds(복수)
            prec, rec = a.get("precision") or [], a.get("recall") or []
            f1s = a.get("f1")
            n = min(len(thr), len(prec), len(rec))
            for i in range(n):
                p_, r_ = prec[i], rec[i]
                f_ = f1s[i] if (f1s and i < len(f1s)) else (2 * p_ * r_ / (p_ + r_) if (p_ + r_) else 0.0)
                rows.append({"threshold": thr[i], "precision": p_, "recall": r_, "f1": round(f_, 4)})
        return _wrap("threshold", "line", "threshold", ["precision", "recall", "f1"], rows, model)
    if cn == "calibration":
        a = _artifact(mkey, "calibration_curve.json")
        rows = []
        if isinstance(a, dict):
            pp = a.get("prob_pred") or a.get("mean_predicted_value") or []   # 신 스키마=mean_predicted_value
            pt = a.get("prob_true") or a.get("fraction_of_positives") or []
            cnts = a.get("count", [None] * len(pp))
            for i in range(min(len(pp), len(pt))):
                rows.append({"prob_pred": pp[i], "prob_true": pt[i], "count": cnts[i] if i < len(cnts) else None})
        return _wrap("calibration", "line", "prob_pred", "prob_true", rows, model)
    if cn == "confusion-matrix":
        a = _artifact(mkey, "confusion_matrix.json")
        rows = ([{"actual": 0, "predicted": 0, "count": a["tn"]}, {"actual": 0, "predicted": 1, "count": a["fp"]},
                 {"actual": 1, "predicted": 0, "count": a["fn"]}, {"actual": 1, "predicted": 1, "count": a["tp"]}]
                if a else [])
        return _wrap("confusion_matrix", "matrix", "predicted", "actual", rows, model)
    if cn == "lift":
        a = _artifact(mkey, "lift_curve.json")
        rows = []
        if isinstance(a, list):           # 신 스키마: [{percentage_of_population, cumulative_response_rate, lift}]
            for it in a:
                rows.append({"top_percent": it.get("percentage_of_population", it.get("top_percent")),
                             "capture_rate": it.get("cumulative_response_rate", it.get("capture_rate")),
                             "lift": it.get("lift")})
        elif isinstance(a, dict):          # 구 스키마: 배열
            for tp, cr, lf in zip(a.get("top_percent", []), a.get("capture_rate", []), a.get("lift", [])):
                rows.append({"top_percent": tp, "capture_rate": cr, "lift": lf})
        return _wrap("lift", "line", "top_percent", ["capture_rate", "lift"], rows, model)
    if cn == "score-distribution":
        a = _artifact(mkey, "score_distribution.json") or {}
        bins = a.get("bins", [])
        if a.get("churn_count") is not None and a.get("non_churn_count") is not None:   # 스키마A: 라벨 분리
            rows = [{"bin": b, "churn_count": c, "non_churn_count": n}
                    for b, c, n in zip(bins, a["churn_count"], a["non_churn_count"])]
            return _wrap("score_distribution", "histogram", "bin", ["churn_count", "non_churn_count"], rows, model)
        counts = a.get("counts") or a.get("count") or []                                # 스키마B: 단일 히스토그램
        rows = [{"bin": b, "count": c} for b, c in zip(bins, counts)]
        return _wrap("score_distribution", "histogram", "bin", ["count"], rows, model)
    if cn in ("shap-summary", "shap", "feature-importance"):
        a = _artifact(mkey, "shap_summary.json") or {}
        feats = a.get("feature") or a.get("feature_names") or []          # 스키마A/B 모두 지원
        vals = a.get("mean_abs_shap") or a.get("mean_abs_shap_values") or []
        pairs = sorted(zip(feats, vals), key=lambda t: -(t[1] or 0))[:15]  # 중요도 내림차순 top15
        rows = [{"feature": f, "mean_abs_shap": round(float(s), 4), "rank": i + 1}
                for i, (f, s) in enumerate(pairs)]
        return _wrap("shap_summary", "bar", "feature", "mean_abs_shap", rows, model)
    if cn == "value-at-risk":
        a = _artifact(mkey, "value_at_risk.json")
        return _wrap("value_at_risk", "treemap", "segment", "value_at_risk", a or [], model)
    if cn == "revenue-recovery":
        a = _artifact(mkey, "business_value.json")
        if isinstance(a, dict) and "confusion_matrix" in a:
            # 신 스키마: CM + 가정 비용/가치 → 항목별 가치 기여 막대
            cm, ass = a.get("confusion_matrix", {}), a.get("assumptions", {})

            def _won(s):
                try:
                    return float(str(s).split()[0].replace(",", ""))
                except Exception:
                    return 0.0
            tp = cm.get("True Positives", 0)
            fp = cm.get("False Positives", 0)
            fn = cm.get("False Negatives", 0)
            rows = [
                {"item": "TP 회복가치", "value_KRW": tp * _won(ass.get("True Positive Value", "5000"))},
                {"item": "FP 비용", "value_KRW": -fp * _won(ass.get("False Positive Cost", "1000"))},
                {"item": "FN 손실", "value_KRW": -fn * _won(ass.get("False Negative Cost", "10000"))},
                {"item": "추정 순가치", "value_KRW": a.get("estimated_total_value_KRW", 0)},
            ]
            return _wrap("revenue_recovery", "bar", "item", ["value_KRW"], rows, model, summary=ass)
        # 구 스키마: top_percent 곡선
        rows = ([{"top_percent": tp, "value_at_risk": v, "expected_recovery": e}
                 for tp, v, e in zip(a["top_percent"], a["value_at_risk"], a["expected_recovery"])]
                if isinstance(a, dict) and "top_percent" in a else [])
        return _wrap("revenue_recovery", "bar", "top_percent", ["value_at_risk", "expected_recovery"], rows, model,
                     summary=(a.get("assumptions") if isinstance(a, dict) else None))
    if cn in ("train-val-loss", "training-history"):
        a = _artifact(mkey, "training_history.json") or {}
        rows = [{"epoch": e, "train_loss": tl, "val_loss": vl}
                for e, tl, vl in zip(a.get("epoch", []), a.get("train_loss", []), a.get("val_loss", []))]
        return _wrap("train_val_loss", "line", "epoch", ["train_loss", "val_loss"], rows, model)
    return None


def model_metrics(model):
    m = _artifact(resolve_key(model), "metrics_summary.json")
    if isinstance(m, list):              # 신 스키마: [{...}] → 첫 항목 dict
        return m[0] if m else None
    return m


def all_metrics():
    """전 모델 metrics를 per-model 산출물(primary→_fallback)에서 정규화해 모음.
    요약·드롭다운·베이스라인이 이 단일 소스를 쓴다(모델팀 값 우선, 없으면 폴백)."""
    out = {}
    for name, mkey in NAME.items():
        m = _artifact(mkey, "metrics_summary.json")
        if isinstance(m, list):                 # 신 스키마: [{model, roc_auc, ...}] → 첫 항목
            m = m[0] if m else None
        if not isinstance(m, dict):
            continue
        auc = m.get("roc_auc", m.get("auc", m.get("val_auc")))
        if auc is None:
            continue
        out[name] = {
            "auc": auc, "pr_auc": m.get("pr_auc"), "f1": m.get("f1"),
            "threshold": m.get("best_threshold", m.get("threshold")),
            "brier": m.get("brier"), "ece": m.get("ece"),
            "n": m.get("n_test", m.get("n")),
            "val_only": m.get("pr_auc") is None,        # 시퀀스(Transformer) 등 test 미산출
        }
    return out


def has_artifacts(model):
    return (CHURN_DIR / resolve_key(model)).exists()
