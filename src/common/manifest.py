"""model_run_manifest.json 생성 (19-3 §6.1).

백엔드 `POST /models/submit` 등록 기준 파일. 모든 path는 레포 루트 기준 상대경로.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))


def write_manifest(
    eval_dir,
    *,
    model_name,
    model_key,
    model_type,
    input_train,
    input_test,
    artifact_path,
    metrics,
    preprocessing_config,
    is_active_candidate=True,
):
    eval_dir = Path(eval_dir)
    # evaluation.* 경로는 레포 루트 기준 상대경로로 박아둔다(백엔드가 그대로 읽음).
    base = f"data/processed/evaluation/churn/{model_key}"
    manifest = {
        "model_name": model_name,
        "model_key": model_key,
        "model_type": model_type,
        "label_name": "churn",      # 운영 라벨 고정(19-3 §1)
        "horizon_days": 7,          # 향후 7일 이탈
        "feature_schema_version": "v2",  # 현재 데이터가 22피처 v2이므로 고정. v1로 돌아가면 수정
        "input_train_path": input_train,
        "input_test_path": input_test,
        "dataset_path": input_train,
        "artifact_path": artifact_path,
        "preprocessing_config": preprocessing_config,
        "metrics": {
            "roc_auc": round(metrics["roc_auc"], 4),
            "pr_auc": round(metrics["pr_auc"], 4),
            "best_threshold": metrics["best_threshold"],
            "best_f1": round(metrics["best_f1"], 4),
        },
        "evaluation": {
            "eval_predictions_path": f"{base}/eval_predictions.parquet",
            "metrics_summary_path": f"{base}/metrics_summary.json",
            "threshold_curve_path": f"{base}/threshold_curve.json",
            "calibration_curve_path": f"{base}/calibration_curve.json",
            "lift_curve_path": f"{base}/lift_curve.json",
            "score_distribution_path": f"{base}/score_distribution.json",
            "shap_summary_path": f"{base}/shap_summary.json",
            "business_value_path": f"{base}/business_value.json",
        },
        "is_active_candidate": is_active_candidate,
        "created_at": datetime.now(KST).isoformat(),
    }
    (eval_dir / "model_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest
