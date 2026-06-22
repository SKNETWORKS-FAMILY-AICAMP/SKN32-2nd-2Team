from services.api_client import request_json


def get_dashboard_summary() -> dict:
    return request_json("GET", "/dashboard/summary")


def get_user_dashboard(user_id: str) -> dict:
    return request_json("GET", f"/dashboard/user/{user_id}")

def get_model_names() -> dict:
    return request_json("GET", "/dashboard/models")


def get_sample_users(model: str = "CatBoost", n: int = 5) -> dict:
    """실제 유저 ID 샘플(입력 예시용). 임의 ID는 피처가 없어 예측 불가."""
    return request_json("GET", "/samples/users", params={"model": model, "n": n})

get_summary = get_dashboard_summary