from services.api_client import request_json


def get_dashboard_summary(model: str = None) -> dict:
    """운영 요약. model 지정 시 그 모델 기준 KPI(Active 전환)."""
    params = {"model": model} if model else None
    return request_json("GET", "/dashboard/summary", params=params)


def get_user_dashboard(user_id: str) -> dict:
    return request_json("GET", f"/dashboard/user/{user_id}")

def get_model_names() -> dict:
    return request_json("GET", "/dashboard/models")


def get_sample_users(model: str = "CatBoost", n: int = 5) -> dict:
    """실제 유저 ID 샘플(입력 예시용). 임의 ID는 피처가 없어 예측 불가."""
    return request_json("GET", "/samples/users", params={"model": model, "n": n})

def get_aux_ensemble() -> dict:
    """보조 태스크(bounce·category) 앙상블 요약(모델별+합산 성능). GET /ensemble/aux-summary."""
    return request_json("GET", "/ensemble/aux-summary")

def get_churn_policy() -> dict:
    """현재 Churn Rate 산정 정책(GET /churn-policy)."""
    return request_json("GET", "/churn-policy")


def set_churn_policy(payload: dict) -> dict:
    """Churn Rate 정책 설정(POST /churn-policy) → 서버 적용 → 시뮬이 받아 표시."""
    return request_json("POST", "/churn-policy", json=payload)


get_summary = get_dashboard_summary