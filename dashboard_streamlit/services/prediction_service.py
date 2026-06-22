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

def get_session_bounce(session_id: str) -> dict:
    return request_json("GET", "/session-bounce/latest", params={"session_id": session_id})