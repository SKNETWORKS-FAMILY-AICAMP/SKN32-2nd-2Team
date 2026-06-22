# -*- coding: utf-8 -*-
"""application — 차트/대시보드 조회 usecase(19-4 §8). artifact-first(모델팀 산출물 우선)."""
from app.infrastructure.files import eval_artifacts as art
from app.infrastructure.files import dataset_reader as ds


def get_model_chart(model, chart_name) -> dict:
    """5~15번 모델 종속 차트. 산출물 없으면 빈 data(19-4 빈상태 규칙)."""
    out = art.model_chart(model, chart_name)
    if out is None:
        return {"_status": 404, "error": f"미지원 chart: {chart_name}"}
    return out


def get_dashboard_chart(chart_name) -> dict:
    """1~4번 모델 비종속 차트."""
    cn = chart_name.replace("_", "-")
    if cn == "baseline-comparison":
        return {"chart_name": "baseline_comparison", "chart_type": "bar", "x": "model_name",
                "y": ["roc_auc", "pr_auc", "f1"], "data": ds.baseline_comparison(),
                "meta": {"schema_version": "chart.v1", "source": "metrics_summary"}}
    if cn == "data-distribution":
        return {"chart_name": "data_distribution", "chart_type": "histogram", "x": "bin", "y": "count",
                "data": ds.data_distribution(), "meta": {"schema_version": "chart.v1", "source": "canonical"}}
    if cn == "cohort-retention":
        # tenure_days 기반 실데이터 잔존(survival) 곡선. week별 retention_rate 라인.
        rows = ds.cohort_retention()
        return {"chart_name": "cohort_retention", "chart_type": "line",
                "x": "week_index", "y": "retention_rate", "data": rows,
                "meta": {"schema_version": "chart.v1", "source": "canonical:tenure"}}
    if cn == "system-architecture":
        return {"chart_name": "system_architecture", "chart_type": "image", "x": None, "y": None,
                "data": {"asset_path": "assets/architecture.svg"},
                "meta": {"schema_version": "chart.v1", "source": "static"}}
    return {"_status": 404, "error": f"미지원 dashboard chart: {chart_name}"}


def get_user_dashboard(user_id) -> dict:
    """유저 개인 대시보드. 없으면 200 빈 객체(프론트가 '데이터 없음' 빈상태로 표시)."""
    return ds.user_dashboard(user_id) or {}


def get_recommendations(user_id) -> dict:
    """추천(19-7-1 §5.4 계약): top_categories + recommendations(상품) + source."""
    rec = ds.recommendations(user_id)
    cats = [{"category_id": c.get("category_id"), "category_name": c.get("name"),
             "score": c.get("score"), "reason": "category_similarity"}
            for c in (rec.get("categories") or [])]
    return {"user_id": user_id, "source": "fallback:category_similarity",
            "top_categories": cats, "recommendations": rec.get("items") or []}


def get_model_names() -> dict:
    return {"models": ds.model_names()}


def get_sample_users(model="CatBoost", n=60) -> dict:
    return {"model": model, "users": ds.sample_users(model, n)}


def get_session_bounce() -> dict:
    return ds.session_bounce()
