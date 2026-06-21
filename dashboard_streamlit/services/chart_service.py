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


def get_dashboard_chart(chart_name: str) -> dict:
    return request_json("GET", f"/dashboard/charts/{chart_name}")


def get_model_chart(model_id: int, chart_name: str) -> dict:
    return request_json("GET", f"/models/{model_id}/charts/{chart_name}")
