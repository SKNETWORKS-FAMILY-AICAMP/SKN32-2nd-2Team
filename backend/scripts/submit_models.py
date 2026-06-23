# -*- coding: utf-8 -*-
"""scripts/submit_models — 7모델 일괄 제출(26-9 P1 #3, 19-2 §13 step5).
metrics_summary.json + 디스크 artifact 경로 → POST /models/submit 로 registry/evaluation 적재.
실행(서버 기동 후): python scripts/submit_models.py [--base http://127.0.0.1:8090] [--key anchor-dev-key]
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]              # backend/
GAJIMA = ROOT.parent
EVAL = GAJIMA / "data" / "processed" / "evaluation"

# 모델명 → (type, artifact_path) 매핑(디스크 실재 경로 기준)
MODEL_MAP = {
    "DecisionTree": ("tree", "models/preprocessors/prep_DecisionTree_v2.joblib"),
    "RandomForest": ("tree", "models/preprocessors/prep_RandomForest_v2.joblib"),
    "LogReg":       ("linear", "models/preprocessors/prep_LogReg_v2.joblib"),
    "XGBoost":      ("tree", "models/preprocessors/prep_XGBoost_v2.joblib"),
    "LightGBM":     ("tree", "models/preprocessors/prep_LightGBM_v2.joblib"),
    "CatBoost":     ("tree", "models/preprocessors/prep_CatBoost_v2.joblib"),
    "Transformer":  ("sequence", "models/sequence/Transformer_train_seq.npz"),
}

FEATURE_ORDER = ["recency_days", "tenure_days", "ndays", "n_events", "n_view",
                 "n_cart", "n_remove_from_cart", "n_purchase", "avg_price", "purch_amt"]


def post(base, key, payload):
    req = urllib.request.Request(
        f"{base}/models/submit", method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "x-api-key": key})
    with urllib.request.urlopen(req, timeout=15) as r:
        res = json.loads(r.read().decode("utf-8"))
        return res.get("data", res) if isinstance(res, dict) and "ok" in res else res   # 봉투 해제


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8090")
    ap.add_argument("--key", default="anchor-dev-key")
    args = ap.parse_args()

    agg = json.loads((EVAL / "metrics_summary.json").read_text(encoding="utf-8"))

    def per_model(name):
        """per-model 산출물(primary→_fallback) 우선, 없으면 집계본. 모델팀 값과 일치."""
        mkey = name.lower()
        for p in (EVAL / "churn" / mkey / "metrics_summary.json",
                  EVAL / "_fallback" / "churn" / mkey / "metrics_summary.json"):
            if p.exists():
                 m = json.loads(p.read_text(encoding="utf-8"))
                 if isinstance(m, list) and len(m) > 0:
                     m = m[0]
                 return {"roc_auc": m.get("roc_auc", m.get("auc", m.get("val_auc"))),
                         "pr_auc": m.get("pr_auc"),
                         "best_threshold": m.get("best_threshold", m.get("threshold", 0.5)),
                         "best_f1": m.get("f1"), "key": mkey}
        v = agg.get(name, {})
        return {"roc_auc": v.get("auc", v.get("val_auc")), "pr_auc": v.get("pr_auc"),
                "best_threshold": v.get("threshold"), "best_f1": v.get("f1"), "key": mkey}

    pm = {name: per_model(name) for name in MODEL_MAP}
    best_name = max(pm, key=lambda k: pm[k]["roc_auc"] or 0)

    results = []
    for name, (mtype, artifact) in MODEL_MAP.items():
        v = pm[name]; mkey = v["key"]
        # per-model 학습입력본 경로(models7). Transformer 는 시퀀스 npz.
        dataset_path = (f"data/processed/churn/models7/{name}_v2_train.parquet"
                        if mtype != "sequence"
                        else "data/processed/churn/models7/Transformer_train_seq.npz")
        payload = {
            "model_name": f"{name}_Churn_v2",
            "model_type": mtype,
            "feature_schema_version": "v2",
            "label_name": "churn",
            "horizon_days": 7,
            "preprocessing_config": {
                "scale": "log+scaler" if mtype == "linear" else "none",
                "feature_order": FEATURE_ORDER,
            },
            "dataset_path": dataset_path,
            "artifact_path": artifact,
            "metrics": {
                "roc_auc": v["roc_auc"],
                "pr_auc": v["pr_auc"],
                "best_threshold": v["best_threshold"],
                "best_f1": v["best_f1"],
            },
            "evaluation": {
                "eval_predictions_path": f"data/processed/evaluation/churn/{mkey}/eval_predictions.parquet",
                "shap_summary_path": f"data/processed/evaluation/churn/{mkey}/shap_summary.json",
            },
            "is_active": (name == best_name),       # 최고 AUC 모델만 active
        }
        try:
            out = post(args.base, args.key, payload)
            print(f"  [OK] {name:14s} -> model_id={out.get('model_id')} eval_id={out.get('eval_id')} "
                  f"mode={out.get('mode')}{'  [ACTIVE]' if payload['is_active'] else ''}")
            results.append(out)
        except Exception as e:
            print(f"  [X]  {name:14s} FAIL: {e}")
    print(f"\n제출 완료 {len(results)}/{len(MODEL_MAP)}  (active={best_name}_Churn_v2)")
    return 0 if len(results) == len(MODEL_MAP) else 1


if __name__ == "__main__":
    sys.exit(main())
