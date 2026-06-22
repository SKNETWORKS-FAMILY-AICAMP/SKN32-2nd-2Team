from services.api_client import request_json


def get_recommendations(user_id: str) -> dict:
    return request_json("GET", f"/recommendations/{user_id}")


def create_retention_action(user_id: str, prediction_id: int, action_type: str, message: str) -> dict:
    payload = {
        "user_id": user_id,
        "prediction_id": prediction_id,
        "action_type": action_type,
        "message": message
    }
    return request_json("POST", "/retention-actions", json=payload)