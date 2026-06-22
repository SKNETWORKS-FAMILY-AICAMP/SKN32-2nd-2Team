# -*- coding: utf-8 -*-
"""interfaces/http — 시뮬 사이트(ecom-churn-simulation) 외부 계약 어댑터.
앱의 fastApiClient.ts가 기대하는 raw JSON(봉투 X, api-key X)으로 응답.
실추론은 sim_usecase(prep 번들)·추천은 catalog_store 재사용. CORS는 main에서 허용."""
import os
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Header, HTTPException, Depends
from app.schemas.sim_external_schema import ChurnPredictIn, RecommendationIn, EventIn
from app.application import sim_usecase as sim
from app.infrastructure.files import catalog_store as cat
from app.infrastructure.mysql.session import sim_event_repository

# 해시키 상호인증(옵션). SIM_SYNC_KEY 환경변수가 설정됐을 때만 X-Sync-Key 일치 요구(하위호환).
SIM_SYNC_KEY = os.getenv("SIM_SYNC_KEY", "")


def verify_sync(x_sync_key: str = Header(default=None)):
    if SIM_SYNC_KEY and x_sync_key != SIM_SYNC_KEY:
        raise HTTPException(status_code=403, detail="sync key mismatch")


router = APIRouter(tags=["sim-external"], dependencies=[Depends(verify_sync)])
KST = timezone(timedelta(hours=9))


def _now():
    return datetime.now(KST).isoformat()


@router.post("/api/churn/predict")
async def churn_predict(body: ChurnPredictIn):
    # 실시간 세션 이탈값 = 3단 폴백(B-모델 models/ → B-데이터 data/ → A-하자드).
    # 모델팀 세션모델 배포 전엔 A(하자드, 학습 0·설명가능)가 동작.
    events = [e.model_dump() for e in body.events]
    scored = sim.realtime_session_score(body.session_id, body.user_id, events)
    p = scored.get("churn_probability")
    prob_pct = round(float(p) * 100, 1) if isinstance(p, (int, float)) else 0.0
    action = sim.action_from_events(p, events)        # 이탈방지 액션(3 시나리오)
    return {"session_id": body.session_id, "churn_probability": prob_pct,
            "risk_level": scored.get("risk_level", "low"),
            "source": scored.get("source"), "recommended_action": action,
            "timestamp": _now()}


@router.post("/api/recommendations")
async def recommendations(body: RecommendationIn):
    recs = []
    sims = cat.similar_categories(body.category_id, k=3) if body.category_id else []
    for s in sims:
        for pr in cat.products(limit=1, category_id=s["category_id"]):
            recs.append({
                "product_id": str(pr.get("product_id")),
                "name": pr.get("category_name") or str(pr.get("product_id")),
                "category_id": str(pr.get("category_id")),
                "brand": pr.get("brand"),
                "price": pr.get("price"),
                "score": s.get("cosine"),
                "reason": f"유사 카테고리(코사인 {s.get('cosine')})",
            })
    return {"session_id": body.session_id, "user_id": body.user_id,
            "current_product_id": body.current_product_id,
            "recommendations": recs, "timestamp": _now()}


@router.post("/api/events")
async def ingest_event(body: EventIn):
    sim_event_repository.log({
        "user_id": str(body.user_id), "session_id": body.session_id,
        "event_type": body.event_type, "product_id": body.product_id,
        "category_id": body.category_id, "brand": body.brand, "price": body.price,
        "churn_prob": None, "risk_level": None})
    return {"status": "ok", "event_id": body.event_id, "timestamp": _now()}


@router.get("/api/analytics/session/{session_id}")
async def analytics(session_id: str):
    a = sim.session_analytics(session_id)
    return {"session_id": session_id, "average_session_duration": 0, **a}


@router.get("/api/catalog/products")
async def catalog_products(limit: int = 60, category: str = None, brand: str = None, q: str = None):
    """시뮬 상품목록(REES46 seed 카탈로그, 인기순+필터). 정적 productData 대체용."""
    items = cat.search_products(min(limit, 200), category, brand, q)
    return {"total": cat.total_products(), "count": len(items),
            "products": [{"product_id": str(p["product_id"]), "category_id": str(p["category_id"]),
                          "category_name": p["category_name"], "brand": p["brand"],
                          "price": p["price"], "n_events": p["n_events"], "name": p["name"]} for p in items]}


@router.get("/api/catalog/facets")
async def catalog_facets():
    """드롭다운용 카테고리/브랜드 목록."""
    return {"categories": [{"category_id": str(c["category_id"]), "name": c["display_name"]} for c in cat.categories(20)],
            "brands": [b["brand"] for b in cat.brands(30)]}
