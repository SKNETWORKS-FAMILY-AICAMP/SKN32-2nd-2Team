# -*- coding: utf-8 -*-
"""application — 대시보드 요약/차트 usecase(19-2 §9.3 / 19-7-1 §5.2).
요약은 운영 KPI(active 모델·예측수·고위험수·평균확률·최신시각·회복매출). 모델 비교표는
별도 차트(/dashboard/charts/baseline-comparison)에서 제공한다."""
from app.infrastructure.files import artifact_store as art
from app.infrastructure.files import eval_artifacts as ea
from app.infrastructure.mysql.session import prediction_repository, model_repository


def _active_model_name():
    """active 모델명(짧은 이름). 미등록 시 per-model 산출물 중 best로 폴백."""
    try:
        rows = model_repository.active()
        if rows:
            return str(rows[0].get("model_name", "")).replace("_Churn_v2", "").replace("_v2", "") or None
    except Exception:
        pass
    m = ea.all_metrics()
    return max(m, key=lambda k: m[k]["auc"]) if m else None


def _expected_revenue_recovery(model) -> float:
    """모델별 회복 예상 매출 — eval_predictions 실데이터로 **일관 계산**(정탐 이탈자 revenue × save_rate).
    (구) business_value.json은 모델마다 avg_revenue 가정(51 vs 42000)·단위($/₩)·0/버그가 섞여
    비교 불가했음 → 단일 공식으로 대체. 단위는 데이터 revenue(소액 달러급) 기준."""
    if not model:
        return 0.0
    from app.infrastructure.files import dataset_reader as ds
    return ds.revenue_recovery(model)


def get_dashboard_summary(model: str = None) -> dict:
    """운영 요약 KPI (19-7-1 §5.2 계약 필드).
    model 지정 시 그 모델 기준(eval_predictions per-model 통계 + 해당 모델 회복매출).
    미지정 시 active 모델 + 런타임 prediction_log 집계."""
    active = model or _active_model_name()
    s = None
    if model:                                    # 선택 모델의 eval 기반 per-model 통계
        from app.infrastructure.files import dataset_reader as ds
        s = ds.model_summary_stats(model)
    if s is None:                                # 폴백: 런타임 prediction_log 집계
        s = prediction_repository.summary_stats()
    return {"active_model": active,
            "total_predictions": s["total"],
            "high_risk_count": s["high_risk"],
            "avg_churn_probability": s["avg"],
            "latest_prediction_at": s.get("latest_at"),
            "expected_revenue_recovery": _expected_revenue_recovery(active),
            "label": "churn", "horizon_days": 7}


def get_model_charts(model: str, name: str) -> dict:
    """PR/ROC/threshold/calibration/shap/feature_importance chart JSON(19-2 §12)."""
    data = art.chart(model, name)
    if data is None:
        return {"_status": 404, "error": f"chart 없음: {model}/{name}"}
    return {"model": model, "chart": name, "data": data}
