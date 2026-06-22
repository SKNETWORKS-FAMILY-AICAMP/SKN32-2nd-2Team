"""Session Bounce Logistic Regression 베이스라인 (계획서 §3.1/§3.4).

GRU와 **동일한 dataset.npz / 동일한 사용자 단위 split**으로 LogReg(balanced)를 학습해
공정 비교 기준선을 만든다. 시퀀스를 tabular 피처로 평탄화:
  현재(마지막) 이벤트 6채널 + 윈도우 누적 카운트/통계.
산출: data/processed/evaluation/session_bounce/logreg/{metrics_summary,eval_predictions,...}
"""
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.session_bounce.common import (
    ROOT, evaluate_and_save, load_dataset, pick_threshold, split_masks)

SEED = 42
EVAL_DIR = ROOT / "data" / "processed" / "evaluation" / "session_bounce" / "logreg"


def featurize(X_num, X_cat):
    """(N,L,6)+(N,L) -> (N, F) tabular. 마지막 이벤트 + 윈도우 집계."""
    last = X_num[:, -1, :]                              # 현재 이벤트 6채널
    valid = (X_cat != 0).astype(np.float32)            # 비패딩 마스크
    wlen = valid.sum(axis=1, keepdims=True)            # 윈도우 유효 길이
    cum = (X_num[:, :, :4] * valid[:, :, None]).sum(axis=1)  # view/cart/remove/purchase 누적
    gap_mean = (X_num[:, :, 4] * valid).sum(axis=1, keepdims=True) / np.clip(wlen, 1, None)
    price_mean = (X_num[:, :, 5] * valid).sum(axis=1, keepdims=True) / np.clip(wlen, 1, None)
    is_first = (wlen <= 1).astype(np.float32)
    return np.concatenate([last, cum, wlen, gap_mean, price_mean, is_first], axis=1).astype(np.float32)


def main():
    d = load_dataset()
    tr, va, te = split_masks(d["split"])
    X = featurize(d["X_num"], d["X_cat"])
    y = d["y"]

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    clf.fit(X[tr], y[tr])
    p_val = clf.predict_proba(X[va])[:, 1]
    thr = pick_threshold(y[va], p_val)
    p_te = clf.predict_proba(X[te])[:, 1]

    summary = evaluate_and_save(
        EVAL_DIR,
        model_name="SessionBounce_LogReg_baseline",
        y_true=y[te], y_score=p_te, user_id=d["user_id"][te],
        threshold=thr, n_train=int(tr.sum()), x_cat=d["X_cat"][te],
    )
    print(f"[logreg] ROC-AUC={summary['roc_auc']:.4f} PR-AUC={summary['pr_auc']:.4f} "
          f"Brier={summary['brier']:.4f} F1={summary['f1']:.4f} thr={thr}")


if __name__ == "__main__":
    main()
