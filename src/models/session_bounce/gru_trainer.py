"""Session Bounce GRU 학습기 (계획서 §3.5/§3.6).

- 입력: data/processed/session_bounce/gru/dataset.npz (사용자 단위 split 포함)
- early stopping: validation PR-AUC (계획서 §3.5)
- threshold: validation F1 최대점 (test 사용 금지, §3.4)
- 기본 적용: LR 스케줄러(ReduceLROnPlateau) + grad clip(1.0)
- 손실: bce / bce_pos(pos_weight) / focal
- 산출: models/session_bounce/gru/{model.pt, model_config.json}
        data/processed/evaluation/session_bounce/gru/{metrics_summary, eval_predictions,
        training_history, model_run_manifest}.json

fit()은 탐색(tune.py)과 최종 학습이 공유한다.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.session_bounce.common import (
    EVAL_DIR, MODEL_DIR, evaluate_and_save, load_dataset,
    pick_threshold, split_masks)
from src.models.session_bounce.gru_model import SessionBounceGRU

SEED = 42
DEFAULTS = {
    "epochs": 30, "batch_size": 512, "lr": 1e-3, "weight_decay": 1e-4,
    "patience": 4, "cat_emb": 32, "num_proj": 16, "hidden": 64,
    "layers": 1, "dropout": 0.2,
    # 탐색/기본 동작 관련
    "loss": "bce",          # bce | bce_pos | focal
    "focal_gamma": 2.0,
    "scheduler": "plateau",  # plateau | cosine | none
    "grad_clip": 1.0,
}


def _set_threads():
    try:
        torch.set_num_threads(min(16, torch.get_num_threads() or 8))
    except Exception:
        pass


def build_model(n_cat, hp, device):
    return SessionBounceGRU(
        n_categories=n_cat, n_num=6, cat_emb=hp["cat_emb"], num_proj=hp["num_proj"],
        hidden=hp["hidden"], layers=hp["layers"], dropout=hp["dropout"],
    ).to(device)


def make_criterion(hp, y_train, device):
    if hp["loss"] == "bce_pos":
        pos = float(y_train.mean())
        w = (1.0 - pos) / max(pos, 1e-6)
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w, device=device))
    if hp["loss"] == "focal":
        g = float(hp["focal_gamma"])

        def focal(logit, target):
            p = torch.sigmoid(logit)
            ce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
            pt = torch.where(target == 1, p, 1 - p)
            return ((1 - pt) ** g * ce).mean()

        return focal
    return nn.BCEWithLogitsLoss()


def _loader(d, mask, bs, shuffle):
    ds = TensorDataset(
        torch.from_numpy(d["X_num"][mask]),
        torch.from_numpy(d["X_cat"][mask]),
        torch.from_numpy(d["y"][mask].astype("float32")),
    )
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)


def _scores(model, X_num, X_cat, device, bs=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_num), bs):
            xn = torch.from_numpy(X_num[i:i + bs]).to(device)
            xc = torch.from_numpy(X_cat[i:i + bs]).to(device)
            out.append(torch.sigmoid(model(xn, xc)).cpu().numpy())
    return np.concatenate(out)


def fit(d, tr, va, n_cat, hp, *, max_epochs=None, verbose=True):
    """train/val로 학습. (model, history, best_pr, device) 반환. test 미사용."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    _set_threads()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = max_epochs or hp["epochs"]

    train_loader = _loader(d, tr, hp["batch_size"], True)
    val_loader = _loader(d, va, hp["batch_size"], False)
    model = build_model(n_cat, hp, device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    criterion = make_criterion(hp, d["y"][tr], device)

    sched = None
    if hp["scheduler"] == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    elif hp["scheduler"] == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    y_val = d["y"][va]
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_pr_auc": []}
    best_pr, best_state, bad = -1.0, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for xn, xc, yb in train_loader:
            xn, xc, yb = xn.to(device), xc.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xn, xc), yb)
            loss.backward()
            if hp["grad_clip"]:
                nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
            opt.step()
            tl += loss.item() * len(xn)
        tl /= int(tr.sum())

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xn, xc, yb in val_loader:
                xn, xc, yb = xn.to(device), xc.to(device), yb.to(device)
                vl += criterion(model(xn, xc), yb).item() * len(xn)
        vl /= int(va.sum())
        p_val = _scores(model, d["X_num"][va], d["X_cat"][va], device)
        pr = float(average_precision_score(y_val, p_val))
        if sched is not None:
            sched.step(pr) if hp["scheduler"] == "plateau" else sched.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(round(tl, 6))
        history["val_loss"].append(round(vl, 6))
        history["val_pr_auc"].append(round(pr, 6))
        if verbose:
            print(f"[gru] epoch {epoch:2d} train_loss={tl:.4f} val_loss={vl:.4f} val_PR-AUC={pr:.4f}")

        if pr > best_pr:
            best_pr, bad = pr, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= hp["patience"]:
                if verbose:
                    print(f"[gru] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_pr, device


def train(**overrides):
    hp = dict(DEFAULTS)
    hp.update(overrides)
    print(f"[gru] FINAL hp={hp}")
    d = load_dataset()
    tr, va, te = split_masks(d["split"])
    schema = json.loads((MODEL_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    n_cat = schema["n_categories_incl_pad"]

    model, history, best_pr, device = fit(d, tr, va, n_cat, hp)

    p_val = _scores(model, d["X_num"][va], d["X_cat"][va], device)
    thr = pick_threshold(d["y"][va], p_val)

    t0 = time.perf_counter()
    p_te = _scores(model, d["X_num"][te], d["X_cat"][te], device)
    latency_ms = (time.perf_counter() - t0) * 1000.0 / max(1, len(p_te)) * 1000.0  # ms/1k

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_DIR / "model.pt")
    (MODEL_DIR / "model_config.json").write_text(json.dumps({
        "arch": "SessionBounceGRU", "n_categories_incl_pad": n_cat, "n_num": 6,
        **hp, "seed": SEED, "early_stopping_metric": "val_pr_auc",
        "best_val_pr_auc": round(best_pr, 6),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = evaluate_and_save(
        EVAL_DIR, model_name="SessionBounce_GRU_v1",
        y_true=d["y"][te], y_score=p_te, user_id=d["user_id"][te],
        threshold=thr, n_train=int(tr.sum()), x_cat=d["X_cat"][te],
        training_history=history, latency_ms=latency_ms,
    )

    manifest = {
        "model_name": "SessionBounce_GRU_v1", "model_key": "session_bounce_gru",
        "model_type": "sequence", "label_name": "churn30", "horizon_minutes": 30,
        "input_dataset": "data/processed/session_bounce/gru/dataset.npz",
        "artifact_path": "models/session_bounce/gru/model.pt",
        "feature_schema_path": "models/session_bounce/gru/feature_schema.json",
        "category_index_map_path": "models/session_bounce/gru/category_index_map.json",
        "preprocessing_config": {
            "input_format": "npz", "seq_len": schema["seq_len"], "n_num": 6,
            "num_features": schema["num_features"], "cat_padding_index": 0,
            "split": "user-group 70/15/15", "seed": SEED,
        },
        "metrics": {k: summary[k] for k in
                    ["roc_auc", "pr_auc", "brier", "threshold", "precision", "recall", "f1"]},
        "evaluation": {
            "metrics_summary_path": "data/processed/evaluation/session_bounce/gru/metrics_summary.json",
            "eval_predictions_path": "data/processed/evaluation/session_bounce/gru/eval_predictions.parquet",
            "training_history_path": "data/processed/evaluation/session_bounce/gru/training_history.json",
        },
        "hyperparameters": hp,
    }
    (EVAL_DIR / "model_run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[gru] DONE ROC-AUC={summary['roc_auc']:.4f} PR-AUC={summary['pr_auc']:.4f} "
          f"Brier={summary['brier']:.4f} F1={summary['f1']:.4f} thr={thr} "
          f"latency={latency_ms:.2f}ms/1k")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = vars(ap.parse_args())
    train(**args)
