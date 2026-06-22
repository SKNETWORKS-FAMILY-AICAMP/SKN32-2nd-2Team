"""공통 데이터 로딩.

입력 정본은 19-3 §3 스키마를 따른다.
- 트리 모델(DecisionTree/XGBoost): v2 22피처 tabular. **`data/processed/churn/`(V2_DIR) 고정** —
  `load_tabular_v2`가 직접 읽으며 DATA_DIR/CHURN_DATA_DIR 환경변수의 영향을 받지 않는다.
- Transformer: recency<=7 코호트 raw 시퀀스. `processed_5m/`(DATA_DIR)에서 읽으며,
  Colab 등에선 CHURN_DATA_DIR 환경변수로 경로를 바꿀 수 있다(시퀀스 경로에만 적용).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# src/common/data.py -> parents[2] = 레포 루트
ROOT = Path(__file__).resolve().parents[2]
# 로컬 기본값은 레포의 processed_5m/. Colab 등에서는 CHURN_DATA_DIR 환경변수로
# Google Drive 경로를 지정하면 코드 수정 없이 데이터 위치만 바꿀 수 있다.
DATA_DIR = Path(os.environ.get("CHURN_DATA_DIR", ROOT / "processed_5m"))

# 19-3 §3.1 — feature 순서 고정. user_id는 feature 금지, churn이 target.
FEATURE_ORDER = [
    "recency_days",
    "tenure_days",
    "ndays",
    "n_events",
    "n_view",
    "n_cart",
    "n_remove_from_cart",
    "n_purchase",
    "avg_price",
    "purch_amt",
]
TARGET = "churn"
ID_COL = "user_id"

# 시퀀스 채널(19-3 §3.3, 17-5-1: view/cart/purchase 계열)
SEQ_FEATURES = ["view", "cart", "purchase"]

# === v2 (22피처) — 전처리팀 v4 산출물 (data/processed/churn) ===
# 기존 10피처 + category/brand/session/price 12피처. 범주형 ID는 제외된 순수 숫자형.
V2_DIR = ROOT / "data" / "processed" / "churn"
# DEPRECATED: 사전 스케일된 models7 사본 경로(스케일 불일치 버그의 원인). 신규 코드 사용 금지.
# 원시 train_tabular_v2/test_tabular_v2(V2_DIR)로 통일됨. (benchmark.py 하위호환 위해 정의만 유지)
V2_MODELS7 = V2_DIR / "models7"
FEATURE_ORDER_V2 = FEATURE_ORDER + [
    "min_price",
    "max_price",
    "std_price",
    "purchase_avg_price",
    "remove_ratio",
    "cart_purchase_ratio",
    "n_categories",
    "cat_entropy",
    "n_brands",
    "brand_loyalty",
    "n_sessions",
    "events_per_session",
]
# model_key -> models7 파일 접두어 (CamelCase)
V2_FILE = {
    "decisiontree": "DecisionTree",
    "xgboost": "XGBoost",
    "randomforest": "RandomForest",
    "logreg": "LogReg",
    "catboost": "CatBoost",
    "lightgbm": "LightGBM",
}


def make_scaler(name):
    """전처리 config의 scaler 이름 -> sklearn 변환기 (none은 None).

    리뷰 노트: 트리(XGB/DT/RF)는 스케일 불변이라 robust/none 결과가 같다. 그래도 변환기를
    fit해 preprocessor.joblib로 저장하는 이유는, 백엔드 실시간 서빙이 학습과 동일한 변환을
    적용해야 하기 때문(선형 모델 LogReg에는 실제로 영향을 줌).
    """
    return {
        "none": None,
        "robust": RobustScaler(),
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
    }[name]


def load_tabular(path):
    """tabular parquet -> (X[float64, 컬럼순서 고정], y[int32], user_id[int64])."""
    df = pd.read_parquet(path)
    X = df[FEATURE_ORDER].astype("float64")
    y = df[TARGET].astype("int32")
    uid = df[ID_COL].astype("int64")
    return X, y, uid


def load_sequence(path):
    """시퀀스 npz -> (X[float32, (N, seq_len, n_features)], y[int32], user_id[int64])."""
    z = np.load(path)
    X = z["X"].astype("float32")
    y = z["churn"].astype("int32")
    uid = z["user_id"].astype("int64")
    return X, y, uid


def load_tabular_v2(model_key=None):
    """v2 22피처 **원시값** 로드 -> (X_tr, y_tr, uid_tr), (X_te, y_te, uid_te).

    train/test 모두 train_tabular_v2/test_tabular_v2 에서 cohort_recency7==1 로 코호트 필터.
    스케일링은 하지 않고, 각 trainer가 자기 scaler로 train에 fit → train·test 동일 적용한다.

    ★ 버그수정: 이전엔 models7/{Model}_v2_train.parquet 을 로드했는데, 이 파일은 모델별로
      이미 스케일링된 사본이다(예: XGBoost_v2_train = RobustScaler(원시)). 그 상태에서 trainer가
      scaler를 다시 fit하고 원시 test에 적용하면 train(스케일)·test(원시) 스케일이 어긋나
      트리 분기 임계값이 안 맞아 성능이 깎였다. → 원시 train_tabular_v2 로 통일해 해결.
    (model_key 인자는 호출 호환성 위해 남겨두며 사용하지 않는다.)
    """
    tr = pd.read_parquet(V2_DIR / "train_tabular_v2.parquet")
    te = pd.read_parquet(V2_DIR / "test_tabular_v2.parquet")
    tr = tr[tr["cohort_recency7"] == 1]  # 코호트 필터 — recency<=7 모집단
    te = te[te["cohort_recency7"] == 1]  # test도 동일 모집단 맞춤(빠뜨리면 점수 부풀려짐)

    def split(df):
        # 22 feature는 float64, target(churn)=int32, user_id=int64.
        # user_id는 식별자라 feature에 넣지 않는다(19-3).
        return (
            df[FEATURE_ORDER_V2].astype("float64"),
            df[TARGET].astype("int32"),
            df[ID_COL].astype("int64"),
        )

    return split(tr), split(te)
