from services.api_client import request_json


MODEL_CHARTS = [
    "train-val-loss",
    "pr-auc",
    "roc-auc",
    "score-distribution",
    "threshold",
    "confusion-matrix",
    "lift",
    "calibration",
    "shap-summary",
    "value-at-risk",
    "revenue-recovery",
]

DASHBOARD_CHARTS = [
    "system-architecture",
    "data-distribution",
    "cohort-retention",
    "baseline-comparison",
]


# 대시보드 드롭다운에 채울 활성 모델 목록 조회 함수를 추가했습니다.
def get_active_models() -> dict:
    return request_json("GET", "/models/active")


def get_dashboard_chart(chart_name: str) -> dict:
    return request_json("GET", f"/dashboard/charts/{chart_name}")


def get_system_chart(chart_name: str) -> dict:
    return request_json("GET", f"/dashboard/charts/{chart_name}")


# 모델 ID가 "CatBoost_Churn_v2" 같은 '문자열'로 들어오기 때문에,
# 기존 int 타입 힌트를 제거하여 에러가 나지 않도록 유연하게 수정했습니다.
def get_model_chart(model_id, chart_name: str) -> dict:
    return request_json("GET", f"/models/{model_id}/charts/{chart_name}")