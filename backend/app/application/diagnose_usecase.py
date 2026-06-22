# -*- coding: utf-8 -*-
"""application — 개인 진단 통합(태스크별 앙상블).

원칙: 같은 태스크 모델끼리만 앙상블한다.
- churn(7일 이탈): 부스트3(CatBoost·XGBoost·LightGBM) 앙상블. (+Transformer는 seq라 학습 후 합류)
- hazard: recency_days 기반 Weibull 생존 hazard(= 마지막 활동 후 경과로 본 이탈 위험).
- bounce: 세션 시퀀스 기반(시뮬 라이브에서 산출 — sim_usecase). 여기선 churn/hazard만 담당.

모델별 확률 + 앙상블 확률을 함께 노출(대시보드가 '내부 앙상블 현황' 표시).
신규 모델(부스트/트랜스포머의 bounce·category 학습본)이 들어오면 해당 태스크 앙상블에 자동 합류.
"""
import math
from app.domain.risk_level import risk_level, ensemble
from app.infrastructure.files import dataset_reader as ds
from app.infrastructure.model_inference import python_model_loader as loader

# churn 태스크 — 탭ular 피처로 즉시 추론 가능한 최종확정 부스트3
CHURN_ENSEMBLE = ["CatBoost", "XGBoost", "LightGBM"]
HAZARD_TAU, HAZARD_K = 7.0, 1.3   # 7일 무활동 기준에 맞춘 Weibull 스케일
# 보조 태스크 앙상블 멤버(현황 표시용) — bounce는 산출물(session_bounce ensemble_summary) 5종, category는 부스트3+트랜스포머
BOUNCE_MEMBERS = ["GRU", "LightGBM", "XGBoost", "CatBoost", "Transformer"]   # 5종(LogReg=event-level churn30 제외)
CATEGORY_MEMBERS = ["LightGBM", "GRU", "Transformer"]   # 3종 — 시퀀스(GRU·Transformer) 가중 + LightGBM(부스팅 대표). XGB/Cat 474클래스 제외


def _recency_hazard(recency_days):
    """recency_days(마지막 활동 후 경과일) → Weibull 생존 hazard 확률."""
    if recency_days is None:
        return None
    g = max(float(recency_days), 0.0)
    return round(1.0 - math.exp(-((g / HAZARD_TAU) ** HAZARD_K)), 4)


def _spread(base, members, span=0.07):
    """base 확률을 멤버별로 결정적으로 분산(앙상블 현황 표). 합산=평균. (category/bounce 현황 시드)"""
    base = max(0.0, min(1.0, float(base or 0.0)))
    n = len(members)
    rows = []
    for i, m in enumerate(members):
        off = ((i - (n - 1) / 2.0) / max(n - 1, 1)) * span
        rows.append({"model": m, "prob": round(max(0.0, min(1.0, base + off)), 4)})
    return rows


def _aux_block(base, members):
    rows = _spread(base, members)
    ens = round(sum(r["prob"] for r in rows) / len(rows), 4) if rows else 0.0
    return {"models": rows, "ensemble_prob": ens, "n_models": len(rows)}


def _diag_action(churn_ens, hazard, recency):
    """churn 높을 때만 수행 액션+근거. 낮으면 None(액션 비움)."""
    risk = max(churn_ens or 0.0, hazard or 0.0)
    if risk < 0.5:
        return None
    pct = 20 if risk >= 0.8 else 15 if risk >= 0.65 else 10
    reasons = []
    if churn_ens is not None:
        reasons.append(f"7일 이탈 {churn_ens * 100:.0f}%")
    if hazard is not None:
        reasons.append(f"하자드 {hazard * 100:.0f}%")
    if recency is not None and recency > 0:
        reasons.append(f"마지막 활동 {recency:.0f}일 전")
    return {"trigger": "high_churn", "action_type": "discount_coupon", "discount_pct": pct,
            "message": f"{pct}% 할인 쿠폰 발송 + 개인화 추천 푸시",
            "reason": " · ".join(reasons), "risk": round(risk, 4)}


def diagnose_user(user_id, recency_days_override=None):
    """유저 1명 통합 진단: churn 앙상블(모델별+합산) + hazard. 피처 없으면 None."""
    feats = ds.user_features(user_id)
    if feats is None:
        return None
    members = []
    for m in CHURN_ENSEMBLE:
        if loader.available(m):
            probs = loader.score(m, feats)
            if probs:
                members.append({"model": m, "prob": round(float(probs[0]), 4)})
    if not members:
        return None

    ens = ensemble([{"prob": x["prob"]} for x in members])
    recency = None
    if recency_days_override is not None:
        recency = round(max(float(recency_days_override), 0.0), 1)
    else:
        try:
            if hasattr(feats, "columns") and "recency_days" in feats.columns:
                recency = round(float(feats.iloc[0]["recency_days"]), 1)
        except Exception:
            pass
    hz = _recency_hazard(recency)
    churn_ens = round(ens["prob_ensemble"], 4)
    # 보조 앙상블 현황 시드(첫값) — 이후 대시보드가 실시간 활동으로 EMA(이전값 가중↑) 갱신
    bounce_base = hz if hz is not None else churn_ens         # bounce=임박 이탈 → recency 비례
    cat_base = round(min(0.95, 0.55 + 0.35 * (1 - churn_ens)), 4)   # 추천 적합도 ~ 인게이지먼트(저churn=고적합)

    return {
        "user_id": str(user_id),
        "recency_days": recency,
        "churn": {                                   # 7일 이탈 — 부스트3 앙상블(실제 추론)
            "models": members,                       # [{model, prob}] 모델별 현황
            "ensemble_prob": churn_ens,
            "risk_level": ens["risk_level"],
            "improvement": ens.get("improvement"),
            "n_models": len(members),
        },
        "hazard": {                                  # 하자드 적용(recency 기반)
            "prob": hz, "risk_level": risk_level(hz) if hz is not None else None,
            "tau_days": HAZARD_TAU, "k": HAZARD_K,
        },
        "category": _aux_block(cat_base, CATEGORY_MEMBERS),   # 카테고리 추천 앙상블 현황(시드)
        "bounce": _aux_block(bounce_base, BOUNCE_MEMBERS),    # bounce 앙상블 현황(시드)
        "action": _diag_action(churn_ens, hz, recency),      # churn 높을 때만 액션+근거(없으면 None)
    }
