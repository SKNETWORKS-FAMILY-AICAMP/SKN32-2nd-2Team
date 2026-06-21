from services.api_client import request_json


def get_dashboard_summary() -> dict:
    return request_json("GET", "/dashboard/summary")


def get_user_dashboard(user_id: str) -> dict:
    return request_json("GET", f"/dashboard/user/{user_id}")
