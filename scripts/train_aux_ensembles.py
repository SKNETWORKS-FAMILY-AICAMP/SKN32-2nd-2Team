# -*- coding: utf-8 -*-
"""보조 태스크 앙상블용 모델 학습 — bounce·category에 부스트3(+Transformer) 추가.

원칙: 같은 태스크 모델끼리 앙상블.
- bounce(이진): LightGBM·XGBoost·CatBoost·Transformer 학습 → 기존 GRU·LogReg와 합쳐 6종.
- category(다중분류 471): LightGBM·XGBoost·CatBoost·Transformer 학습 → 4종.

입력은 시퀀스 npz(X_cat (N,10), X_num (N,10,6)). 부스팅은 시퀀스를 직접 못 먹으므로
flatten+집계로 탭ular 피처(81개)를 만들어 학습한다. Transformer는 시퀀스 그대로 사용.

자원: 스레드 3개로 제한(8코어 중 5코어 여유). 모델별 try/except로 부분 실패 격리.
산출: models/{task}/{model}/...  +  data/processed/evaluation/{task}/{model}/metrics_summary.json
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"   # BLAS/OMP 스레드 제한(자원 여유 확보)

import json
import time
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
EVAL = ROOT / "data" / "processed" / "evaluation"
N_THREADS = 3


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_tabular(X_num, X_cat):
    """시퀀스 → 탭ular 피처. flatten(60) + 집계(mean/max/last 18) + cat통계(3) = 81."""
    N = X_num.shape[0]
    flat = X_num.reshape(N, -1)                       # 60
    mean = X_num.mean(axis=1)                         # 6
    mx = X_num.max(axis=1)                            # 6
    last = X_num[:, -1, :]                            # 6
    valid = (X_cat != 0)
    seqlen = valid.sum(axis=1, keepdims=True).astype(np.float32)        # 1
    nuniq = np.array([[len(np.unique(r[r != 0]))] for r in X_cat], dtype=np.float32)  # 1
    lastcat = X_cat[:, -1:].astype(np.float32)        # 1
    return np.hstack([flat, mean, mx, last, seqlen, nuniq, lastcat]).astype(np.float32)


def save_metrics(task, model, metrics):
    d = EVAL / task / model
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  saved metrics → {task}/{model}: {metrics}")


# ── 메트릭 ────────────────────────────────────────────────────────────────
def binary_metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    pred = (p >= 0.5).astype(int)
    return {"auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "f1": round(float(f1_score(y, pred)), 4),
            "n_test": int(len(y))}


def multiclass_metrics(y, proba, classes):
    """top-1 / top-5 정확도. proba: (N, n_class), classes: 열→원래 라벨."""
    order = np.argsort(-proba, axis=1)[:, :5]
    top1 = classes[order[:, 0]]
    acc1 = float((top1 == y).mean())
    top5 = classes[order]
    acc5 = float(np.any(top5 == y[:, None], axis=1).mean())
    return {"top1_acc": round(acc1, 4), "top5_acc": round(acc5, 4),
            "n_class": int(len(classes)), "n_test": int(len(y))}


# ── 부스터 학습 ────────────────────────────────────────────────────────────
def train_boosters_binary(task, Xtr, ytr, Xva, yva, Xte, yte):
    import joblib
    for name, fit in [("lightgbm", _lgb_bin), ("xgboost", _xgb_bin), ("catboost", _cat_bin)]:
        try:
            t0 = time.time()
            model, proba = fit(Xtr, ytr, Xva, yva, Xte)
            m = binary_metrics(yte, proba)
            m["train_sec"] = round(time.time() - t0, 1)
            d = MODELS / task / name
            d.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "kind": "binary", "framework": name}, d / "model.joblib")
            save_metrics(task, name, m)
        except Exception:
            log(f"  !! {task}/{name} 실패\n{traceback.format_exc()}")


def train_boosters_multiclass(task, Xtr, ytr, Xva, yva, Xte, yte):
    import joblib
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder().fit(np.concatenate([ytr, yva, yte]))
    classes = le.classes_
    ytr_e, yva_e, yte_e = le.transform(ytr), le.transform(yva), le.transform(yte)
    n_class = len(classes)
    for name, fit in [("lightgbm", _lgb_mc), ("xgboost", _xgb_mc), ("catboost", _cat_mc)]:
        try:
            t0 = time.time()
            model, proba = fit(Xtr, ytr_e, Xva, yva_e, Xte, n_class)
            m = multiclass_metrics(yte, proba, classes)
            m["train_sec"] = round(time.time() - t0, 1)
            d = MODELS / task / name
            d.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": model, "kind": "multiclass", "framework": name,
                         "classes": classes.tolist()}, d / "model.joblib")
            save_metrics(task, name, m)
        except Exception:
            log(f"  !! {task}/{name} 실패\n{traceback.format_exc()}")


def _lgb_bin(Xtr, ytr, Xva, yva, Xte):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63,
                           n_jobs=N_THREADS, subsample=0.8, colsample_bytree=0.8, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m, m.predict_proba(Xte)[:, 1]


def _xgb_bin(Xtr, ytr, Xva, yva, Xte):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, tree_method="hist",
                          n_jobs=N_THREADS, subsample=0.8, colsample_bytree=0.8,
                          eval_metric="auc", early_stopping_rounds=30)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, m.predict_proba(Xte)[:, 1]


def _cat_bin(Xtr, ytr, Xva, yva, Xte):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6, thread_count=N_THREADS,
                           loss_function="Logloss", eval_metric="AUC", verbose=False)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=40)
    return m, m.predict_proba(Xte)[:, 1]


def _lgb_mc(Xtr, ytr, Xva, yva, Xte, n_class):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=80, learning_rate=0.1, num_leaves=63, objective="multiclass",
                           num_class=n_class, n_jobs=N_THREADS, subsample=0.8,
                           colsample_bytree=0.6, verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(15, verbose=False)])
    return m, m.predict_proba(Xte)


def _xgb_mc(Xtr, ytr, Xva, yva, Xte, n_class):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=80, learning_rate=0.1, max_depth=8, tree_method="hist",
                          objective="multi:softprob", num_class=n_class, n_jobs=N_THREADS,
                          subsample=0.8, colsample_bytree=0.6, early_stopping_rounds=15)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m, m.predict_proba(Xte)


def _cat_mc(Xtr, ytr, Xva, yva, Xte, n_class):
    from catboost import CatBoostClassifier
    m = CatBoostClassifier(iterations=120, learning_rate=0.2, depth=8, thread_count=N_THREADS,
                           loss_function="MultiClass", verbose=False)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=20)
    return m, m.predict_proba(Xte)


# ── Transformer(시퀀스) 학습 ───────────────────────────────────────────────
def train_transformer(task, kind, Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va,
                      Xc_te, Xn_te, y_te, n_cat, n_out, epochs=4):
    try:
        import torch
        import torch.nn as nn
        torch.set_num_threads(N_THREADS)
        t0 = time.time()
        dev = "cpu"

        class SeqTransformer(nn.Module):
            def __init__(self, n_cat, n_num, n_out, d=64, layers=2, heads=4):
                super().__init__()
                self.emb = nn.Embedding(n_cat, d, padding_idx=0)
                self.num = nn.Linear(n_num, d)
                self.pos = nn.Parameter(torch.zeros(1, Xc_tr.shape[1], d))
                enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1, batch_first=True)
                self.tr = nn.TransformerEncoder(enc, layers)
                self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))

            def forward(self, xc, xn):
                h = self.emb(xc) + self.num(xn) + self.pos
                mask = (xc == 0)
                h = self.tr(h, src_key_padding_mask=mask)
                pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
                return self.head(pooled)

        out_dim = 1 if kind == "binary" else n_out
        model = SeqTransformer(n_cat, Xn_tr.shape[2], out_dim).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        lossf = nn.BCEWithLogitsLoss() if kind == "binary" else nn.CrossEntropyLoss()
        Xc_tr_t = torch.tensor(Xc_tr, dtype=torch.long)
        Xn_tr_t = torch.tensor(Xn_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32 if kind == "binary" else torch.long)
        bs = 1024
        n = len(Xc_tr_t)
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(n)
            tot = 0.0
            for i in range(0, n, bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                out = model(Xc_tr_t[idx], Xn_tr_t[idx])
                if kind == "binary":
                    loss = lossf(out.squeeze(-1), y_tr_t[idx])
                else:
                    loss = lossf(out, y_tr_t[idx])
                loss.backward()
                opt.step()
                tot += float(loss) * len(idx)
            log(f"  [{task}/transformer] epoch {ep+1}/{epochs} loss={tot/n:.4f}")

        model.eval()
        with torch.no_grad():
            outs = []
            Xc_te_t = torch.tensor(Xc_te, dtype=torch.long)
            Xn_te_t = torch.tensor(Xn_te, dtype=torch.float32)
            for i in range(0, len(Xc_te_t), 4096):
                o = model(Xc_te_t[i:i + 4096], Xn_te_t[i:i + 4096])
                outs.append(o)
            out = torch.cat(outs)
            if kind == "binary":
                proba = torch.sigmoid(out.squeeze(-1)).numpy()
                m = binary_metrics(y_te, proba)
            else:
                proba = torch.softmax(out, dim=1).numpy()
                m = multiclass_metrics(y_te, proba, np.arange(out_dim))
        m["train_sec"] = round(time.time() - t0, 1)
        d = MODELS / task / "transformer"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), d / "model.pt")
        (d / "config.json").write_text(json.dumps(
            {"kind": kind, "n_cat": int(n_cat), "n_num": int(Xn_tr.shape[2]),
             "seq_len": int(Xc_tr.shape[1]), "n_out": int(out_dim), "d": 64,
             "layers": 2, "heads": 4}, ensure_ascii=False, indent=2), encoding="utf-8")
        save_metrics(task, "transformer", m)
    except Exception:
        log(f"  !! {task}/transformer 실패\n{traceback.format_exc()}")


# ── 태스크 실행 ────────────────────────────────────────────────────────────
def run_bounce():
    log("=== BOUNCE 학습 시작 ===")
    z = np.load(ROOT / "data/processed/session_bounce/gru/dataset.npz", allow_pickle=True)
    sp = z["split"]
    tr, va, te = sp == 0, sp == 1, sp == 2
    Xc, Xn, y = z["X_cat"], z["X_num"], z["y"].astype(int)
    Xtr_t = build_tabular(Xn[tr], Xc[tr]); Xva_t = build_tabular(Xn[va], Xc[va]); Xte_t = build_tabular(Xn[te], Xc[te])
    train_boosters_binary("session_bounce", Xtr_t, y[tr], Xva_t, y[va], Xte_t, y[te])
    train_transformer("session_bounce", "binary", Xc[tr], Xn[tr], y[tr], Xc[va], Xn[va], y[va],
                      Xc[te], Xn[te], y[te], n_cat=500, n_out=1, epochs=4)
    log("=== BOUNCE 완료 ===")


def run_category():
    log("=== CATEGORY 학습 시작 ===")
    z = np.load(ROOT / "data/processed/next_category/category_gru_v1.npz", allow_pickle=True)
    Xc_tr, Xn_tr, y_tr = z["X_cat_train"], z["X_num_train"], z["y_train"].astype(int)
    Xc_va, Xn_va, y_va = z["X_cat_val"], z["X_num_val"], z["y_val"].astype(int)
    Xc_te, Xn_te, y_te = z["X_cat_test"], z["X_num_test"], z["y_test"].astype(int)
    Xtr_t = build_tabular(Xn_tr, Xc_tr); Xva_t = build_tabular(Xn_va, Xc_va); Xte_t = build_tabular(Xn_te, Xc_te)
    train_boosters_multiclass("next_category", Xtr_t, y_tr, Xva_t, y_va, Xte_t, y_te)
    train_transformer("next_category", "multiclass", Xc_tr, Xn_tr, y_tr, Xc_va, Xn_va, y_va,
                      Xc_te, Xn_te, y_te, n_cat=516, n_out=516, epochs=4)
    log("=== CATEGORY 완료 ===")


if __name__ == "__main__":
    import sys
    t0 = time.time()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "bounce"):
        run_bounce()
    if which in ("all", "category"):
        run_category()
    log(f"전체 완료 — {round(time.time()-t0,1)}s")
