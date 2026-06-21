from services.api_client import request_json


def get_recommendations(user_id: str) -> dict:
    return request_json("GET", f"/recommendations/{user_id}")


def create_retention_action(payload: dict) -> dict:
    return request_json("POST", "/retention-actions", json=payload)
