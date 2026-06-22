# -*- coding: utf-8 -*-
"""category 다중분류 부스터 보강 — sklearn 래퍼 한계/메모리 이슈 우회.

- XGBoost: train split이 전체 클래스를 안 담아 sklearn 래퍼가 거부 → 저수준 xgb.train(DMatrix, num_class)로 학습.
- CatBoost: 471클래스 MultiClass가 RAM 초과(bad allocation) → train 행 서브샘플 + 경량 하이퍼로 재시도.
자원: 스레드 3개. 산출 형식은 evaluate_aux_ensembles._booster_proba가 읽는 joblib 규약 준수.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "3"
import json
import time
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
from scripts.train_aux_ensembles import build_tabular, multiclass_metrics
MODELS = ROOT / "models"
EVAL = ROOT / "data" / "processed" / "evaluation"
N = 3


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save_metrics(model, metrics):
    d = EVAL / "next_category" / model
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  {model}: {metrics}")


def main():
    z = np.load(ROOT / "data/processed/next_category/category_gru_v1.npz", allow_pickle=True)
    Xc_tr, Xn_tr, y_tr = z["X_cat_train"], z["X_num_train"], z["y_train"].astype(int)
    Xc_va, Xn_va, y_va = z["X_cat_val"], z["X_num_val"], z["y_val"].astype(int)
    Xc_te, Xn_te, y_te = z["X_cat_test"], z["X_num_test"], z["y_test"].astype(int)
    Xtr = build_tabular(Xn_tr, Xc_tr); Xva = build_tabular(Xn_va, Xc_va); Xte = build_tabular(Xn_te, Xc_te)
    le = LabelEncoder().fit(np.concatenate([y_tr, y_va, y_te]))
    classes = le.classes_
    nc = len(classes)
    ytr_e, yva_e = le.transform(y_tr), le.transform(y_va)

    # ── XGBoost 저수준 ──
    try:
        import xgboost as xgb
        t0 = time.time()
        dtr = xgb.DMatrix(Xtr, label=ytr_e)
        dva = xgb.DMatrix(Xva, label=yva_e)
        params = {"objective": "multi:softprob", "num_class": nc, "max_depth": 8, "eta": 0.1,
                  "tree_method": "hist", "nthread": N, "subsample": 0.8, "colsample_bytree": 0.6,
                  "eval_metric": "mlogloss"}
        bst = xgb.train(params, dtr, num_boost_round=120, evals=[(dva, "val")],
                        early_stopping_rounds=15, verbose_eval=False)
        proba = bst.predict(xgb.DMatrix(Xte))
        m = multiclass_metrics(y_te, proba, classes)
        m["train_sec"] = round(time.time() - t0, 1)
        d = MODELS / "next_category" / "xgboost"; d.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": bst, "kind": "multiclass", "framework": "xgboost",
                     "framework_api": "xgb_booster", "classes": classes.tolist()}, d / "model.joblib")
        save_metrics("xgboost", m)
    except Exception:
        import traceback; log("XGBoost 실패\n" + traceback.format_exc())

    # ── CatBoost 경량(서브샘플 + 얕게) ──
    try:
        from catboost import CatBoostClassifier
        t0 = time.time()
        rng = np.random.RandomState(42)
        idx = rng.choice(len(Xtr), size=min(120000, len(Xtr)), replace=False)   # 메모리 절감
        m_cb = CatBoostClassifier(iterations=80, learning_rate=0.2, depth=6, thread_count=N,
                                  loss_function="MultiClass", max_ctr_complexity=1, verbose=False)
        m_cb.fit(Xtr[idx], ytr_e[idx], eval_set=(Xva, yva_e), early_stopping_rounds=15)
        proba = m_cb.predict_proba(Xte)
        m = multiclass_metrics(y_te, proba, classes)
        m["train_sec"] = round(time.time() - t0, 1)
        m["note"] = "train 12만 서브샘플(메모리 절감)"
        d = MODELS / "next_category" / "catboost"; d.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": m_cb, "kind": "multiclass", "framework": "catboost",
                     "classes": classes.tolist()}, d / "model.joblib")
        save_metrics("catboost", m)
    except Exception:
        import traceback; log("CatBoost 실패\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
    log("category 부스터 보강 완료")
