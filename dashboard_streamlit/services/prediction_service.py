from services.api_client import request_json


def get_top_risk() -> dict:
    return request_json("GET", "/predictions/top-risk")


def get_latest_prediction(user_id: str) -> dict:
    return request_json("GET", "/predictions/latest", params={"user_id": user_id})


def run_realtime_prediction(user_id: str, model: str = "CatBoost") -> dict:
    """유저 v2 피처로 active 모델 라이브 추론(POST /predict/realtime). 저장 예측 없어도 즉시 산출."""
    return request_json("POST", "/predict/realtime", json={"user_id": user_id, "model": model})


def get_sim_user_score(user_id: str) -> dict:
    """유저의 시뮬 사이트 실시간 세션 이탈 점수(GET /sim/user-score). 활동 없으면 빈 객체."""
    return request_json("GET", "/sim/user-score", params={"user_id": user_id})


def get_diagnose(user_id: str, recency_days: float | None = None) -> dict:
    """개인 진단 통합(GET /predict/diagnose): churn 부스트3 앙상블(모델별+합산) + hazard."""
    params = {"user_id": user_id}
    if recency_days is not None:
        params["recency_days"] = recency_days
    return request_json("GET", "/predict/diagnose", params=params)


def set_active_user(user_id: str, refresh_interval_sec: int | None = None) -> dict:
    """현재 진단 대상 유저를 서버에 설정(POST /sim/active-user) → 시뮬 사이트가 읽어 표시."""
    body = {"user_id": user_id}
    if refresh_interval_sec is not None:
        body["refresh_interval_sec"] = int(refresh_interval_sec)
    return request_json("POST", "/sim/active-user", json=body)

def get_session_bounce(session_id: str) -> dict:
    return request_json("GET", "/session-bounce/latest", params={"session_id": session_id})
