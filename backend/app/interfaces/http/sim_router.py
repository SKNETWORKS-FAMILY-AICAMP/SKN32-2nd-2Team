# -*- coding: utf-8 -*-
"""interfaces/http — 시뮬 사이트 실시간 루프(26-9 P2). 카탈로그·이벤트 수집·세션 실시간 점수·리텐션.
봉투 적용. require_api_key(시뮬 사이트는 dev 키 동봉)."""
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from typing import Optional
from app.interfaces.http.deps import require_api_key, unwrap
from app.interfaces.http.responses import ok
from app.infrastructure.files import catalog_store as cat
from app.application import sim_usecase as sim

router = APIRouter(prefix="/sim", tags=["simulation"], dependencies=[Depends(require_api_key)])


class SimEventIn(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    event_type: str = Field(..., pattern="^(view|cart|remove|purchase)$")
    product_id: Optional[str] = None
    category_id: Optional[str] = None
    brand: Optional[str] = None
    price: Optional[float] = None
    profile: str = "returning"


@router.get("/products")
async def products(limit: int = Query(24, ge=1, le=100), category_id: Optional[str] = None):
    return ok(cat.products(limit=limit, category_id=category_id))


@router.get("/categories")
async def categories(limit: int = Query(12, ge=1, le=60)):
    return ok(cat.categories(limit=limit))


@router.post("/event")
async def event(body: SimEventIn):
    """행동 1건 기록 → 즉시 세션 재점수(이탈확률·위험등급·추천 반환)."""
    return ok(unwrap(sim.record_event(
        body.session_id, body.user_id or body.session_id, body.event_type,
        body.product_id, body.category_id, body.brand, body.price, body.profile)))


@router.get("/score")
async def score(session_id: str = Query(...), model: Optional[str] = None):
    return ok(unwrap(sim.score_session(session_id, model)))


@router.get("/user-score")
async def user_score(user_id: str = Query(...)):
    """유저의 가장 최근 시뮬 세션 실시간 이탈 점수(활동 없으면 200 빈 객체)."""
    return ok(sim.latest_score_by_user(user_id) or {})


class ActiveUserIn(BaseModel):
    user_id: Optional[str] = None
    refresh_interval_sec: Optional[int] = Field(default=None, ge=1, le=60)


@router.post("/active-user")
async def set_active_user(body: ActiveUserIn):
    """대시보드에서 현재 진단 대상 유저 설정 → 시뮬 사이트가 읽어 표시/동작."""
    return ok(sim.set_active_user(body.user_id, body.refresh_interval_sec))


@router.get("/active-user")
async def get_active_user():
    return ok(sim.get_active_user())


@router.post("/reset")
async def reset(session_id: str = Query(...)):
    return ok(sim.reset(session_id))
