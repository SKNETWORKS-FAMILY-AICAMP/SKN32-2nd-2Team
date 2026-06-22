# -*- coding: utf-8 -*-
"""interfaces/http — 대시보드 요약/차트(모델 비종속)/유저/추천/리텐션/앙상블(19-4 §5). 봉투 적용."""
from fastapi import APIRouter, Depends
from app.interfaces.http.deps import require_api_key, unwrap
from app.interfaces.http.responses import ok
from app.schemas.model_submit_schema import EnsembleIn
from app.schemas.ops_schema import RetentionActionIn
from app.application.dashboard_usecase import get_dashboard_summary
from app.application import charts_usecase as cu
from app.application import predict_usecase as puc
from app.infrastructure.mysql import ops_repository as ops

router = APIRouter(tags=["dashboard"], dependencies=[Depends(require_api_key)])


@router.get("/dashboard/summary")
async def dashboard_summary():
    return ok(get_dashboard_summary())


@router.get("/dashboard/charts/{chart_name}")
async def dashboard_chart(chart_name: str):
    """모델 비종속: system-architecture·data-distribution·cohort-retention·baseline-comparison."""
    return ok(unwrap(cu.get_dashboard_chart(chart_name)))


@router.get("/dashboard/user/{user_id}")
async def dashboard_user(user_id: str):
    return ok(unwrap(cu.get_user_dashboard(user_id)))


@router.get("/dashboard/models")
async def dashboard_models():
    """드롭다운용 모델 목록(짧은 이름)."""
    return ok(cu.get_model_names())


@router.get("/churn-policy")
async def churn_policy_get():
    """현재 Churn Rate 산정 정책(시뮬·대시보드 공통 기준)."""
    from app.application import sim_usecase as sim
    return ok(sim.get_churn_policy())


@router.post("/churn-policy")
async def churn_policy_set(body: dict):
    """대시보드가 정책 설정 → 서버가 적용 → 시뮬은 받은 값만 표시.
    body: {mode: max|ensemble|bounce_scaled|select, select_key, bounce_floor, bounce_ceiling, weights}."""
    from app.application import sim_usecase as sim
    return ok(sim.set_churn_policy(**(body or {})))


@router.get("/samples/users")
async def sample_users(model: str = "CatBoost", n: int = 60):
    return ok(cu.get_sample_users(model, n))


@router.get("/session-bounce")
async def session_bounce():
    """실시간 세션 바운스 메타 + 샘플 세션."""
    return ok(cu.get_session_bounce())


@router.get("/ensemble/aux-summary")
async def ensemble_aux_summary():
    """보조 태스크(bounce·category) 앙상블 요약 — 모델별 + 합산 성능."""
    from app.infrastructure.files import dataset_reader as ds
    return ok(ds.aux_ensemble_summary())


@router.get("/recommendations/{user_id}")
async def recommendations(user_id: str):
    return ok(cu.get_recommendations(user_id))


@router.get("/coupons/summary")
async def coupons_summary():
    """쿠폰 타게팅 요약(2+장바구니 대상자·등급별 인원). artifact-first."""
    from app.infrastructure.files import coupons as cp
    return ok(cp.summary())


@router.get("/coupons/targets")
async def coupons_targets(grade: str = None, limit: int = 100):
    """쿠폰 대상자 목록(이탈확률 내림차순). grade로 등급필터."""
    from app.infrastructure.files import coupons as cp
    return ok({"targets": cp.targets(grade, limit)})


@router.post("/retention-actions")
async def retention_action(body: RetentionActionIn):
    return ok(ops.add_retention_action(body.user_id, body.action_type, body.message, body.prediction_id))


@router.post("/ensemble/run")
async def ensemble_run(body: EnsembleIn):
    return ok(puc.run_ensemble([m.model_dump() for m in body.members]))


@router.post("/internal/cleanup")
async def internal_cleanup():
    """보존정책 수동 실행(ops). 평소엔 스케줄러가 주기 실행."""
    from app.infrastructure.mysql import retention
    from app.application import sim_usecase
    return ok({"db_retention": retention.cleanup_db(), "sessions_swept": sim_usecase.sweep_sessions()})
