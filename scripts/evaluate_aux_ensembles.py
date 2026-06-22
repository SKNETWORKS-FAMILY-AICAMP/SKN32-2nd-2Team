# -*- coding: utf-8 -*-
"""보조 태스크 앙상블 평가 — 동일 테스트셋에서 모델별 + 앙상블 성능 산출(리포트용).

bounce(이진): GRU(기존) + LightGBM·XGBoost·CatBoost·Transformer(신규) → 동일타깃 5종 앙상블.
              (기존 LogReg 데모는 event-level churn30 = 다른 타깃이라 블렌딩 제외, 별도 표기)
category(다중분류): GRU(기존) + LightGBM·XGBoost·CatBoost·Transformer(신규) → 앙상블.

앙상블 = 모델별 확률 평균(soft-voting). 산출 → data/processed/evaluation/{task}/ensemble_summary.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "3"
import sys
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"
EVAL = ROOT / "data" / "processed" / "evaluation"
from scripts.train_aux_ensembles import build_tabular, binary_metrics, multiclass_metrics  # 동일 피처


def _load_seq_transformer(cfg, path):
    import torch
    import torch.nn as nn

    class SeqTransformer(nn.Module):
        def __init__(self, n_cat, n_num, n_out, seq_len, d=64, layers=2, heads=4):
            super().__init__()
            self.emb = nn.Embedding(n_cat, d, padding_idx=0)
            self.num = nn.Linear(n_num, d)
            self.pos = nn.Parameter(torch.zeros(1, seq_len, d))
            enc = nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.1, batch_first=True)
            self.tr = nn.TransformerEncoder(enc, layers)
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_out))

        def forward(self, xc, xn):
            h = self.emb(xc) + self.num(xn) + self.pos
            mask = (xc == 0)
            h = self.tr(h, src_key_padding_mask=mask)
            pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)
            return self.head(pooled)

    m = SeqTransformer(cfg["n_cat"], cfg["n_num"], cfg["n_out"], cfg["seq_len"],
                       cfg.get("d", 64), cfg.get("layers", 2), cfg.get("heads", 4))
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def _transformer_proba(task, Xc, Xn, binary):
    import torch
    d = MODELS / task / "transformer"
    if not (d / "model.pt").exists():
        return None
    cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
    m = _load_seq_transformer(cfg, d / "model.pt")
    outs = []
    with torch.no_grad():
        for i in range(0, len(Xc), 4096):
            o = m(torch.tensor(Xc[i:i + 4096], dtype=torch.long),
                  torch.tensor(Xn[i:i + 4096], dtype=torch.float32))
            outs.append(o)
    out = torch.cat(outs)
    return torch.sigmoid(out.squeeze(-1)).numpy() if binary else torch.softmax(out, 1).numpy()


def _booster_proba(task, name, Xtab, vocab=None):
    import joblib
    p = MODELS / task / name / "model.joblib"
    if not p.exists():
        return None
    b = joblib.load(p)
    if b.get("framework_api") == "xgb_booster":      # 저수준 xgb.Booster(다중분류)
        import xgboost as xgb
        proba = b["model"].predict(xgb.DMatrix(Xtab))
    else:
        proba = b["model"].predict_proba(Xtab)
    if b["kind"] == "binary":
        return proba[:, 1]
    # multiclass: proba 컬럼 → vocab 공간 정렬. 컬럼은 모델이 train에서 본 클래스만(부분집합)일 수 있음.
    classes = np.array(b["classes"])                  # LabelEncoder 원라벨(전체 인코딩 공간)
    mc = getattr(b["model"], "classes_", None)        # 모델이 실제 학습한 인코딩 라벨(=proba 컬럼 순서)
    cols = classes[np.asarray(mc, dtype=int)] if mc is not None else classes[:proba.shape[1]]
    full = np.zeros((len(Xtab), vocab), dtype=np.float32)
    full[:, cols.astype(int)] = proba
    return full


def eval_bounce():
    import torch
    from src.models.session_bounce.gru_model import SessionBounceGRU
    z = np.load(ROOT / "data/processed/session_bounce/gru/dataset.npz", allow_pickle=True)
    te = z["split"] == 2
    Xc, Xn, y = z["X_cat"][te], z["X_num"][te], z["y"][te].astype(int)
    Xtab = build_tabular(Xn, Xc)
    cfg = json.loads((MODELS / "session_bounce/gru/model_config.json").read_text(encoding="utf-8"))
    probs = {}

    # GRU(기존)
    try:
        g = SessionBounceGRU(n_categories=cfg["n_categories_incl_pad"], n_num=cfg["n_num"],
                             cat_emb=cfg["cat_emb"], num_proj=cfg["num_proj"], hidden=cfg["hidden"],
                             layers=cfg["layers"], dropout=cfg["dropout"])
        g.load_state_dict(torch.load(MODELS / "session_bounce/gru/model.pt", map_location="cpu", weights_only=True))
        g.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xc), 4096):
                o = g(torch.tensor(Xn[i:i + 4096], dtype=torch.float32),
                      torch.tensor(Xc[i:i + 4096], dtype=torch.long))
                outs.append(torch.sigmoid(o))
        probs["GRU"] = torch.cat(outs).numpy()
    except Exception as e:
        print("GRU(bounce) 로드 실패:", e)

    for nm in ("lightgbm", "xgboost", "catboost"):
        pr = _booster_proba("session_bounce", nm, Xtab)
        if pr is not None:
            probs[nm] = pr
    tp = _transformer_proba("session_bounce", Xc, Xn, binary=True)
    if tp is not None:
        probs["transformer"] = tp

    per_model = {k: binary_metrics(y, v) for k, v in probs.items()}
    ens = np.mean(list(probs.values()), axis=0)
    out = {"task": "session_bounce", "metric": "auc", "n_models": len(probs),
           "members": list(probs.keys()), "per_model": per_model,
           "ensemble": binary_metrics(y, ens),
           "note": "기존 LogReg는 event-level churn30(다른 타깃)이라 블렌딩 제외 — 별도 6번째 자료로 보유"}
    (EVAL / "session_bounce").mkdir(parents=True, exist_ok=True)
    (EVAL / "session_bounce/ensemble_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


def eval_category():
    import torch
    from src.models.next_category.gru_model import CategoryGRU
    z = np.load(ROOT / "data/processed/next_category/category_gru_v1.npz", allow_pickle=True)
    Xc, Xn, y = z["X_cat_test"], z["X_num_test"], z["y_test"].astype(int)
    lengths = z["lengths_test"]
    Xtab = build_tabular(Xn, Xc)
    cfg = json.loads((MODELS / "next_category/gru/model_config.json").read_text(encoding="utf-8"))
    hp = cfg["hyperparameters"]
    vocab = 516
    probs = {}

    # GRU(기존)
    try:
        g = CategoryGRU(vocab_size=vocab, numeric_size=Xn.shape[2], embedding_dim=hp["embedding_dim"],
                        numeric_dim=hp["numeric_dim"], hidden_size=hp["hidden_size"],
                        num_layers=hp["num_layers"], dropout=hp["dropout"])
        g.load_state_dict(torch.load(MODELS / "next_category/gru/model.pt", map_location="cpu", weights_only=True))
        g.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Xc), 4096):
                o = g(torch.tensor(Xc[i:i + 4096], dtype=torch.long),
                      torch.tensor(Xn[i:i + 4096], dtype=torch.float32),
                      torch.tensor(lengths[i:i + 4096], dtype=torch.long))
                outs.append(torch.softmax(o, 1))
        probs["GRU"] = torch.cat(outs).numpy()
    except Exception as e:
        print("GRU(category) 로드 실패:", e)

    for nm in ("lightgbm", "xgboost", "catboost"):
        pr = _booster_proba("next_category", nm, Xtab, vocab=vocab)
        if pr is not None:
            probs[nm] = pr
    tp = _transformer_proba("next_category", Xc, Xn, binary=False)
    if tp is not None:
        probs["transformer"] = tp

    cls = np.arange(vocab)
    per_model = {k: multiclass_metrics(y, v, cls) for k, v in probs.items()}
    keys = list(probs.keys())
    mats = [probs[k] for k in keys]
    ens = np.mean(mats, axis=0)                       # 단순평균(soft-voting)
    # 가중 앙상블: 모델별 top1 비례 가중(시퀀스 약한 부스터가 끌어내리는 것 보정)
    w = np.array([max(per_model[k]["top1_acc"], 1e-3) for k in keys])
    w = w / w.sum()
    ens_w = sum(wi * mi for wi, mi in zip(w, mats))
    out = {"task": "next_category", "metric": "top1/top5", "n_models": len(probs),
           "members": keys, "per_model": per_model,
           "ensemble_mean": multiclass_metrics(y, ens, cls),
           "ensemble_weighted": multiclass_metrics(y, ens_w, cls),
           "weights": {k: round(float(wi), 3) for k, wi in zip(keys, w)},
           "note": "부스팅은 시퀀스 순서를 못 살려 next-category에 약함 → 시퀀스 모델(GRU·Transformer) 가중. "
                   "XGBoost/CatBoost 474클래스는 학습비용/메모리 제약으로 제외(LightGBM이 부스팅 대표)."}
    (EVAL / "next_category").mkdir(parents=True, exist_ok=True)
    (EVAL / "next_category/ensemble_summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "bounce"):
        eval_bounce()
    if which in ("all", "category"):
        eval_category()
