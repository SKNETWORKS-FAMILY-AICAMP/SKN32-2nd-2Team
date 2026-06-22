# -*- coding: utf-8 -*-
"""application — 실시간 예측 usecase(19-2 §9.2). 유저 v2 피처 → prep 번들 직접 추론(점수 caller 불필요).
점수는 백엔드가 모델을 직접 로드해 산출(python_model_loader). 위험등급·리텐션·로그까지."""
from app.domain.risk_level import risk_level, retention_action
from app.infrastructure.files import dataset_reader as ds
from app.infrastructure.model_inference import python_model_loader as loader
from app.infrastructure.mysql.session import prediction_repository, model_repository


def predict_realtime(user_id, model="CatBoost", model_id=None) -> dict:
    if not loader.available(model):
        return {"_status": 503, "error": f"모델 번들 로드 불가: {model} (의존 라이브러리/파일 확인)"}
    feats = ds.user_features(user_id)
    if feats is None:
        return {"_status": 404, "error": f"유저 피처 없음: {user_id}"}
    probs = loader.score(model, feats)
    if not probs:
        return {"_status": 500, "error": "추론 실패"}
    p = round(float(probs[0]), 4)
    thr = loader.threshold(model)
    r = risk_level(p)
    act = retention_action(p)
    # recency_days: 마지막 활동 후 경과일(= 'n일 만에 방문' 표시용)
    recency = None
    try:
        if hasattr(feats, "columns") and "recency_days" in feats.columns:
            recency = round(float(feats.iloc[0]["recency_days"]), 1)
    except Exception:
        pass
    if model_id is not None and not model_repository.exists(model_id):
        model_id = None        # FK 견고화
    prediction_repository.log({
        "model_id": model_id, "user_id": str(user_id), "churn_probability": p,
        "risk_level": r, "recommended_action": act["action_message"],
    })
    return {"user_id": str(user_id), "model": model, "model_name": model,
            "churn_probability": p, "threshold": thr, "predict": int(p >= thr),
            "risk_level": r, "recommended_action": act["action_message"], "horizon_days": 7,
            "recency_days": recency, "source": "live-inference"}
