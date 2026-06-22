"""XGBoost 이탈(churn) 예측 학습기 — v2 22피처.

[데이터]
  - 입력: data/processed/churn/{train_tabular_v2, test_tabular_v2}.parquet 의 cohort_recency7==1
    (원시값. load_tabular_v2가 코호트 필터까지 수행)
  - 평가는 Feb OOT 코호트 109,736명 기준.

[전처리] RobustScaler를 **원시 train에 fit → train·test 동일 적용** → preprocessor.joblib 저장.
  (구버전 버그: 사전 스케일된 models7/XGBoost_v2_train을 로드해 scaler를 재fit하고 원시 test에
   적용 → train/test 스케일 불일치 → 트리 분기 임계값 어긋나 성능 하락. DecisionTree_XGBoost_
   교차검증_검토보고서.md §3에서 지적, 원시본 통일 + CV 재튜닝으로 수정)

[하이퍼파라미터] 올바른 전처리 위에서 CV(StratifiedKFold5, PR-AUC) 재튜닝(RandomizedSearch).
  CV PR-AUC 0.9372 / OOT PR-AUC 0.9361 / OOT ROC-AUC 0.7908
  (버그버전 0.9337/0.7841 대비 ROC +0.0067, 전처리팀 리포트 0.7904와 정합)
"""
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.common.data import FEATURE_ORDER_V2, load_tabular_v2, make_scaler
from src.common.evaluation import evaluate_and_save
from src.common.manifest import write_manifest
from src.common.registry import artifact_rel_path, log_run, resolve_dirs

MODEL_KEY = "xgboost"
MODEL_NAME = "XGBoost_Churn_v2"
MODEL_TYPE = "tree"
SEED = 42


def _oof_threshold(X_raw, y, common, scaler_name):
    """train 내부 5-fold OOF 예측으로 F1 최대 운영 임계값(0.01 그리드)을 구한다.

    평가셋(OOT)을 쓰지 않으므로 누수가 없다. 스케일러는 fold마다 fit(Pipeline)해 누수 차단.
    """
    model = xgb.XGBClassifier(**common)
    sc = make_scaler(scaler_name)
    estimator = Pipeline([("scaler", sc), ("model", model)]) if sc is not None else model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = cross_val_predict(estimator, X_raw, y, cv=cv, method="predict_proba", n_jobs=4)[:, 1]
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1 = [precision_recall_fscore_support(
            y, (oof >= t).astype(int), average="binary", zero_division=0)[2] for t in grid]
    return float(grid[int(np.argmax(f1))])

# 정식 config = 올바른 전처리 위 CV(PR-AUC) 재튜닝 best.
DEFAULTS = {
    "scaler": "robust",            # 원시 train에 fit → train/test 동일 적용
    "n_estimators": 250,
    "max_depth": 3,                # CV best: 얕은 트리
    "learning_rate": 0.0359,
    "subsample": 0.8,
    "colsample_bytree": 1.0,
    "min_child_weight": 20,        # 규제 ↑
    "gamma": 0,                    # 분할 최소 손실감소(규제). CV 튜닝 탐색 대상
    "reg_alpha": 0,
    "reg_lambda": 10,              # L2 ↑
    "scale_pos_weight": 1,         # 올바른 전처리에선 1이 최적(과거 4는 버그 상태 산물)
    "early_stopping_rounds": 0,    # 0=비활성 → 전체 train으로 학습(최종 refit)
}


def train(run_tag=None, **overrides):
    """학습 + 산출물 생성. run_tag 없으면 정식(canonical) 경로에 저장."""
    hp = dict(DEFAULTS)
    hp.update(overrides)

    # 원시 v2 22피처(코호트). load_tabular_v2가 train/test 모두 cohort_recency7==1 필터.
    (X_full, y_full, _), (X_te, y_te, uid_te) = load_tabular_v2(MODEL_KEY)
    X_full_raw = X_full  # 스케일 전 원시(OOF 운영임계값 계산용)

    # RobustScaler를 원시 train에 fit → train·test 동일 적용(스케일 일치 보장).
    scaler = make_scaler(hp["scaler"])
    if scaler is not None:
        X_full = scaler.fit_transform(X_full)
        X_te = scaler.transform(X_te)

    es = hp["early_stopping_rounds"]
    common = dict(
        n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
        learning_rate=hp["learning_rate"], subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"], min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"], reg_alpha=hp["reg_alpha"], reg_lambda=hp["reg_lambda"],
        scale_pos_weight=hp["scale_pos_weight"], objective="binary:logistic",
        eval_metric="logloss", tree_method="hist", random_state=SEED,
    )
    if es and es > 0:
        # 튜닝 모드: 조기종료용 val 분리(85% 학습)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_full, y_full, test_size=0.15, stratify=y_full, random_state=SEED)
        clf = xgb.XGBClassifier(early_stopping_rounds=es, **common)
        clf.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        res = clf.evals_result()
        tl = [float(x) for x in res["validation_0"]["logloss"]]
        vl = [float(x) for x in res["validation_1"]["logloss"]]
        n_train = len(y_tr)
    else:
        # 최종 모드: 전체 train으로 학습(보고서 §5). val은 train loss 기록용으로만.
        clf = xgb.XGBClassifier(**common)
        clf.fit(X_full, y_full, eval_set=[(X_full, y_full)], verbose=False)
        res = clf.evals_result()
        tl = [float(x) for x in res["validation_0"]["logloss"]]
        vl = []
        n_train = len(y_full)

    y_score = clf.predict_proba(X_te)[:, 1]  # churn=1 확률 (19-3 §6.2)

    artifact_dir, eval_dir = resolve_dirs(MODEL_KEY, run_tag)
    clf.save_model(str(artifact_dir / "model.json"))
    if scaler is not None:
        joblib.dump(scaler, artifact_dir / "preprocessor.joblib")  # 서빙용 동일 변환기

    # 최종 refit 모드엔 검증셋이 없어 val_loss=[](epoch/train_loss만 채워짐). 의도된 계약 —
    # 대시보드 #5는 train 곡선만 그리고 val 라인은 생략. (튜닝 모드에서만 val_loss 채워짐)
    training_history = {"epoch": list(range(1, len(tl) + 1)), "train_loss": tl, "val_loss": vl}

    # shap_summary(#13): XGBoost 네이티브 TreeSHAP(mean|SHAP|). gain proxy 대신 실제 기여도.
    # (gain과 SHAP은 순위가 크게 다름 — gain은 분기 손실감소, SHAP은 예측 기여. 원인설명엔 SHAP.)
    dtest = xgb.DMatrix(X_te, feature_names=list(FEATURE_ORDER_V2))
    contribs = clf.get_booster().predict(dtest, pred_contribs=True)  # (N, F+1), 마지막=bias
    mean_abs = np.abs(contribs[:, :-1]).mean(axis=0)
    order = np.argsort(-mean_abs)
    shap_summary = {
        "feature": [FEATURE_ORDER_V2[i] for i in order],
        "mean_abs_shap": [float(mean_abs[i]) for i in order],
        "rank": list(range(1, len(FEATURE_ORDER_V2) + 1)),
        "note": "XGBoost native TreeSHAP (mean|SHAP|) on test set",
    }

    # 운영 임계값: train OOF에서 F1 최대(누수 없음). 평가셋으로 고르지 않는다(19-3 §11.2).
    oof_threshold = _oof_threshold(X_full_raw, y_full, common, hp["scaler"])

    metrics = evaluate_and_save(
        eval_dir,
        model_name=MODEL_NAME, model_key=MODEL_KEY, model_type=MODEL_TYPE,
        user_id=uid_te, y_true=y_te, y_score=y_score, n_train=n_train,
        training_history=training_history, shap_summary=shap_summary,
        fixed_threshold=oof_threshold, threshold_grid_step=0.01,
    )

    write_manifest(
        eval_dir,
        model_name=MODEL_NAME, model_key=MODEL_KEY, model_type=MODEL_TYPE,
        input_train="data/processed/churn/train_tabular_v2.parquet",
        input_test="data/processed/churn/test_tabular_v2.parquet",
        artifact_path=artifact_rel_path(MODEL_KEY, run_tag, "model.json"),
        metrics=metrics,
        preprocessing_config={
            "input_format": "parquet",
            "cohort_filter": "cohort_recency7==1",
            "scale": hp["scaler"],
            "feature_order": FEATURE_ORDER_V2,
            "id_column": "user_id",
            "target_column": "churn",
            "objective": "binary:logistic",
            **{k: hp[k] for k in (
                "n_estimators", "max_depth", "learning_rate", "subsample",
                "colsample_bytree", "min_child_weight", "gamma", "reg_alpha",
                "reg_lambda", "scale_pos_weight")},
        },
    )
    log_run(MODEL_KEY, run_tag, hp, metrics)
    print(f"[{MODEL_KEY}] run={run_tag or 'baseline'} n_train={n_train} "
          f"ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} F1={metrics['best_f1']:.4f}")
    return metrics


if __name__ == "__main__":
    train()
