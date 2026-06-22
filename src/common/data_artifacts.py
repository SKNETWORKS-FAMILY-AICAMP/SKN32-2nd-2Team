"""데이터 기반 산출물 생성 — data_distribution.json, cohort_summary.json.

모델과 무관하게 입력 데이터(라벨/피처/코호트 분포)에서만 계산되는 두 산출물을 만든다.
다른 평가 산출물(evaluation.py)이 모델 예측 기반인 것과 달리, 여기 두 파일은 데이터 자체를
요약하므로 재학습 없이도 생성·갱신할 수 있다. 양식은 19-3 계약 §12.1.1 / §12.1.2를 따른다.

  data_distribution.json: label_distribution + split_distribution + feature_distribution (#2 차트)
  cohort_summary.json   : cohort_retention 코호트×주차 잔존율 (#3 차트)

주의: cohort_retention(코호트×주차)은 시계열 원천(canonical_5m 이벤트 로그)이 있어야 정확히
계산된다. 현재 tabular_v2에는 날짜/주차 컬럼이 없어 recency_days 기반 잔존곡선으로 근사하며,
meta.source / meta.note 에 근사임을 명시한다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.data import (
    FEATURE_ORDER_V2,
    TARGET,
    V2_DIR,
)

COHORT_COL = "cohort_recency7"
HORIZON_DAYS = 7
# data_distribution feature_distribution 에 노출할 대표 피처(계약 §12.1.1: 3~5개 권장)
DIST_FEATURES = ["recency_days", "tenure_days", "n_events", "n_purchase", "purch_amt"]
DIST_BINS = 8


def _dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _cohort(df: pd.DataFrame) -> pd.DataFrame:
    return df[df[COHORT_COL] == 1]


def build_data_distribution() -> dict:
    """19-3 §12.1.1 양식. label/split/feature 분포. (평가 코호트 기준)"""
    tr = _cohort(pd.read_parquet(V2_DIR / "train_tabular_v2.parquet"))
    te = _cohort(pd.read_parquet(V2_DIR / "test_tabular_v2.parquet"))

    y = te[TARGET].astype(int)
    n = int(len(te))
    label_distribution = [
        {"label": int(lab), "count": int((y == lab).sum()),
         "ratio": round(float((y == lab).mean()), 4)}
        for lab in (0, 1)
    ]
    split_distribution = [
        {"split": "train", "count": int(len(tr))},
        {"split": "test", "count": int(len(te))},
    ]

    feature_distribution = []
    for name in DIST_FEATURES:
        s = te[name].astype("float64")
        lo, hi = float(s.min()), float(s.max())
        edges = np.linspace(lo, hi, DIST_BINS + 1)
        counts, _ = np.histogram(s, bins=edges)
        for i, c in enumerate(counts):
            feature_distribution.append({
                "feature": name,
                "bin": f"{edges[i]:.2f}-{edges[i + 1]:.2f}",
                "count": int(c),
            })

    return {
        "chart_name": "data_distribution",
        "chart_type": "bar",
        "source": "test_tabular_v2.parquet",
        "label_distribution": label_distribution,
        "split_distribution": split_distribution,
        "feature_distribution": feature_distribution,
        "meta": {
            "schema_version": "artifact.data_distribution.v1",
            "label_name": TARGET,
            "horizon_days": HORIZON_DAYS,
            "cohort_filter": f"{COHORT_COL}==1",
            "n_users": n,
        },
    }


def build_cohort_summary() -> dict:
    """19-3 §12.1.2 양식(cohort_retention). 시계열 원천이 없어 recency_days로 잔존율 근사.

    week_index w 의 retention_rate = (recency_days > w*7/HORIZON 환산) 비율이 아니라,
    관측창(7일) 내 'recency_days >= 경계'인 유저 비율로 단조감소 잔존곡선을 만든다.
    정확한 월코호트×주차 heatmap은 canonical_5m 이벤트 로그 확보 후 재생성해야 한다.
    """
    te = _cohort(pd.read_parquet(V2_DIR / "test_tabular_v2.parquet"))
    rec = te["recency_days"].astype("float64").to_numpy()
    n = int(len(te))
    # 관측창 7일을 일 단위 주차 경계로 근사(0..7일). week_index는 일 경과로 해석.
    cohorts = []
    for w in range(0, HORIZON_DAYS + 1):
        retained = int((rec >= w).sum())  # w일 시점까지 마지막활동이 남아있던 유저 근사
        cohorts.append({
            "cohort": "all",
            "week_index": w,
            "users": n,
            "retained_users": retained,
            "retention_rate": round(retained / n, 4) if n else 0.0,
        })
    return {
        "chart_name": "cohort_retention",
        "chart_type": "heatmap",
        "source": "test_tabular_v2.parquet (approx; canonical_5m 미보유)",
        "cohorts": cohorts,
        "meta": {
            "schema_version": "artifact.cohort_summary.v1",
            "cohort_unit": "all",
            "period_unit": "day",
            "label_name": TARGET,
            "horizon_days": HORIZON_DAYS,
            "note": "recency_days 기반 잔존 근사. 정확한 월코호트×주차는 canonical_5m 이벤트 로그 필요.",
        },
    }


def write_data_artifacts(eval_dir) -> None:
    """eval_dir에 cohort_summary.json, data_distribution.json 두 파일을 쓴다."""
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    _dump(eval_dir / "cohort_summary.json", build_cohort_summary())
    _dump(eval_dir / "data_distribution.json", build_data_distribution())


if __name__ == "__main__":
    # 정식 경로 두 모델 디렉터리에 생성. (데이터 기반이라 두 모델 내용은 동일)
    from src.common.registry import EVAL_ROOT

    for key in ("xgboost", "decisiontree"):
        d = EVAL_ROOT / key
        write_data_artifacts(d)
        print(f"[{key}] wrote cohort_summary.json, data_distribution.json -> {d}")
