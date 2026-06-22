"""모델별 결과표 벤치마크 (팀 공유용 — accuracy/precision/recall/f1/roc_auc/pr_auc/train_time_sec).

데이터: v2 22피처(트리) / 받은 시퀀스(Transformer, test 미제공 → 내부 80/20 분할).
지표는 threshold=0.5 기준(predict 관례), roc_auc·pr_auc는 threshold-free.

실행: .venv/Scripts/python.exe -m src.benchmark
"""
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

# 우리 담당 모델별 임계값 (v2 bayes / 리포트 기준). None = F1 최적값 자동 계산.
THRESHOLDS = {"DecisionTree": 0.42, "XGBoost": 0.54, "Transformer": None}


def _best_f1_threshold(y_true, y_score):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = []
    for t in grid:
        _, _, f, _ = precision_recall_fscore_support(
            y_true, (y_score >= t).astype(int), average="binary", zero_division=0
        )
        f1s.append(f)
    return float(grid[int(np.argmax(f1s))])


def metrics_row(model_name, y_true, y_score, train_time_sec):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thr = THRESHOLDS.get(model_name)
    if thr is None:
        thr = _best_f1_threshold(y_true, y_score)
    y_pred = (y_score >= thr).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "threshold": thr,
        "train_time_sec": train_time_sec,
    }


def run_tree(model_key, model_name, trainer):
    t0 = time.perf_counter()
    trainer.train()
    dt = time.perf_counter() - t0
    ep = pd.read_parquet(f"data/processed/evaluation/churn/{model_key}/eval_predictions.parquet")
    return metrics_row(model_name, ep["y_true"].to_numpy(), ep["y_score"].to_numpy(), dt)


def run_transformer():
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    from src.common.data import V2_MODELS7, load_sequence
    from src.models.churn.transformer_trainer import ChurnTransformer, _predict_scores

    X, y, _ = load_sequence(V2_MODELS7 / "Transformer_train_seq.npz")  # (N,17,3) 받은 데이터
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    seq_len, n_features = X.shape[1], X.shape[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype("float32"))),
        batch_size=128, shuffle=True,
    )
    model = ChurnTransformer(n_features=n_features, seq_len=seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    t0 = time.perf_counter()
    for _ in range(30):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            opt.step()
    dt = time.perf_counter() - t0
    y_score = _predict_scores(model, Xte, device)
    return metrics_row("Transformer", yte, y_score, dt)


def main():
    from src.models.churn import decisiontree_trainer, xgboost_trainer

    rows = [
        run_tree("decisiontree", "DecisionTree", decisiontree_trainer),
        run_tree("xgboost", "XGBoost", xgboost_trainer),
        run_transformer(),
    ]
    df = pd.DataFrame(rows)[
        ["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "threshold", "train_time_sec"]
    ]
    out = "data/processed/evaluation/churn/benchmark_summary.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
