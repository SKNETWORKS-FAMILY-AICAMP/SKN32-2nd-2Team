from services.api_client import request_json


def get_top_risk() -> dict:
    return request_json("GET", "/predictions/top-risk")


def get_latest_prediction(user_id: str) -> dict:
    return request_json("GET", "/predictions/latest", params={"user_id": user_id})

def get_session_bounce(session_id: str) -> dict:
    return request_json("GET", "/session-bounce/latest", params={"session_id": session_id})