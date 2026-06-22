"""실험 버전 관리 — run_tag별 산출물 경로 분리 + 모델별 리더보드 CSV 누적.

- run_tag 있으면: models/churn/{key}/runs/{tag}/, evaluation/churn/{key}/runs/{tag}/
- run_tag 없으면: 정식 경로(제출용 베스트 슬롯)
- 매 실행마다 evaluation/churn/{key}/runs/_leaderboard.csv 에 1행 append
"""
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.common.data import ROOT

KST = timezone(timedelta(hours=9))
MODELS_ROOT = ROOT / "models" / "churn"
EVAL_ROOT = ROOT / "data" / "processed" / "evaluation" / "churn"


def resolve_dirs(model_key, run_tag):
    """run_tag 유무에 따라 (artifact_dir, eval_dir)를 만들어 반환.

    run_tag=None  -> 정식(canonical) 경로 = 백엔드 제출용 베스트 슬롯.
    run_tag="..." -> runs/{tag}/ 실험 경로(덮어쓰기 방지, 튜닝 비교용).
    """
    if run_tag:
        artifact_dir = MODELS_ROOT / model_key / "runs" / run_tag
        eval_dir = EVAL_ROOT / model_key / "runs" / run_tag
    else:
        artifact_dir = MODELS_ROOT / model_key
        eval_dir = EVAL_ROOT / model_key
    artifact_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir, eval_dir


def artifact_rel_path(model_key, run_tag, filename):
    """manifest에 기록할 레포 기준 상대경로 문자열."""
    sub = f"/runs/{run_tag}" if run_tag else ""
    return f"models/churn/{model_key}{sub}/{filename}"


def log_run(model_key, run_tag, params, metrics):
    """리더보드 CSV에 이번 실험 1행 추가 (지표 + 하이퍼파라미터)."""
    runs_dir = EVAL_ROOT / model_key / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / "_leaderboard.csv"

    row = {
        "created_at": datetime.now(KST).strftime("%Y-%m-%d %H:%M"),
        "run_tag": run_tag or "baseline",
        "pr_auc": round(metrics["pr_auc"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "f1": round(metrics["best_f1"], 4),
        "best_threshold": metrics["best_threshold"],
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
    }
    for k, v in params.items():
        row[k] = v

    new = pd.DataFrame([row])
    if path.exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False, encoding="utf-8-sig")
    return path
