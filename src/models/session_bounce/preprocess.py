"""Session Bounce GRU 전처리 (GRU_분리모델_작업분담_및_통합계획 §3.2~§3.4).

입력: data2/clean/all_months_clean_core.csv  (5개월 이벤트 레벨, 세션+타임스탬프)
산출:
  data/processed/session_bounce/gru/dataset.npz
    X_num (N,10,6) float32  = [is_view, is_cart, is_remove, is_purchase, gap_log, price_log]
    X_cat (N,10)   int32    = 카테고리 인덱스(0=padding)
    y     (N,)     int8     = churn30 (이 행동 후 30분 무활동=1)
    user_id (N,)   int64
    split (N,)     int8     = 0 train / 1 val / 2 test  (사용자 단위 group split)
  models/session_bounce/gru/category_index_map.json   (category_id -> index, 0=padding)
  models/session_bounce/gru/feature_schema.json
  data/processed/session_bounce/gru/preprocess_meta.json

라벨/censoring (§3.3):
  각 이벤트 기준 같은 사용자의 다음 이벤트까지 간격으로 라벨링.
    next_gap <= 1800s            -> churn30 = 0 (30분 내 재행동)
    next_gap >  1800s            -> churn30 = 1 (다음 행동이 있으나 30분 초과 = 바운스)
    다음 이벤트 없음(유저 마지막):
        event_time + 1800 <= 관측종료  -> churn30 = 1 (30분 무활동 확정)
        그 외(관측 경계)               -> censored, 학습 제외
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC_CSV = ROOT / "data2" / "clean" / "all_months_clean_core.csv"
DATA_DIR = ROOT / "data" / "processed" / "session_bounce" / "gru"
MODEL_DIR = ROOT / "models" / "session_bounce" / "gru"

SEED = 42
SEQ_LEN = 10
BOUNCE_SEC = 1800  # 30분
NUM_COLS = ["is_view", "is_cart", "is_remove", "is_purchase", "gap_log", "price_log"]
EVENT_TYPES = {"view": 0, "cart": 1, "remove_from_cart": 2, "purchase": 3}


def _load_sampled(n_users, chunksize=2_000_000):
    """전체에서 user_id를 표본추출한 뒤 해당 유저 행만 청크로 읽어 모은다."""
    rng = np.random.default_rng(SEED)
    # 1패스: 전체 user_id 집합
    all_users = set()
    for ch in pd.read_csv(SRC_CSV, usecols=["user_id"], chunksize=chunksize):
        all_users.update(ch["user_id"].unique().tolist())
    all_users = np.array(sorted(all_users), dtype=np.int64)
    if n_users and n_users < len(all_users):
        sel = rng.choice(all_users, size=n_users, replace=False)
    else:
        sel = all_users
    sel_set = set(sel.tolist())
    print(f"[prep] users total={len(all_users):,} sampled={len(sel_set):,}")

    # 2패스: 선택 유저 행만 수집
    cols = ["event_time", "event_type", "category_id", "price", "user_id"]
    parts = []
    for ch in pd.read_csv(SRC_CSV, usecols=cols, chunksize=chunksize):
        parts.append(ch[ch["user_id"].isin(sel_set)])
    df = pd.concat(parts, ignore_index=True)
    print(f"[prep] collected rows={len(df):,}")
    return df, sel


def build(n_users=150_000, cap_per_user=100):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    df, sel_users = _load_sampled(n_users)
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"]).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0.0).clip(lower=0)
    obs_end = df["event_time"].max()
    print(f"[prep] obs window {df['event_time'].min()} ~ {obs_end}")

    # 카테고리 사전 (0=padding)
    cats = df["category_id"].dropna().unique()
    cat_index = {int(c): i + 1 for i, c in enumerate(sorted(int(x) for x in cats))}
    df["cat_idx"] = df["category_id"].map(lambda c: cat_index.get(int(c), 0) if pd.notna(c) else 0).astype(np.int32)

    # 이벤트 타입 원핫 + price_log
    et = df["event_type"].map(EVENT_TYPES).fillna(0).astype(int).to_numpy()
    onehot = np.zeros((len(df), 4), dtype=np.float32)
    onehot[np.arange(len(df)), et] = 1.0
    df["price_log"] = np.log1p(df["price"].to_numpy()).astype(np.float32)

    # 사용자 단위 그룹 split (70/15/15)
    u = sel_users.copy()
    rng.shuffle(u)
    n = len(u)
    split_of = {}
    for i, uid in enumerate(u):
        split_of[int(uid)] = 0 if i < 0.70 * n else (1 if i < 0.85 * n else 2)

    # 사용자별 정렬
    df = df.sort_values(["user_id", "event_time"], kind="mergesort").reset_index(drop=True)
    ts = df["event_time"].astype("int64").to_numpy() // 1_000_000_000  # 초
    obs_end_s = int(obs_end.value // 1_000_000_000)
    uid_arr = df["user_id"].to_numpy()
    cat_arr = df["cat_idx"].to_numpy()
    price_log = df["price_log"].to_numpy()

    # 유저 경계
    bounds = np.flatnonzero(np.r_[True, uid_arr[1:] != uid_arr[:-1]])
    starts = bounds
    ends = np.r_[bounds[1:], len(df)]

    # 출력 누적
    Xn_list, Xc_list, y_list, uid_list, sp_list = [], [], [], [], []
    for s, e in zip(starts, ends):
        n_ev = e - s
        uid = int(uid_arr[s])
        t = ts[s:e]
        gap_prev = np.zeros(n_ev, dtype=np.float32)
        gap_prev[1:] = np.clip(t[1:] - t[:-1], 0, None)
        gap_log = np.log1p(gap_prev).astype(np.float32)

        # 다음 이벤트까지 간격 / 라벨 + censor
        next_gap = np.full(n_ev, -1, dtype=np.int64)
        next_gap[:-1] = t[1:] - t[:-1]
        y = np.full(n_ev, -1, dtype=np.int8)  # -1 = censored/drop
        has_next = np.arange(n_ev) < n_ev - 1
        y[has_next & (next_gap <= BOUNCE_SEC)] = 0
        y[has_next & (next_gap > BOUNCE_SEC)] = 1
        last_idx = n_ev - 1
        if t[last_idx] + BOUNCE_SEC <= obs_end_s:
            y[last_idx] = 1  # 30분 무활동 확정
        # else: 관측 경계 -> -1 유지(drop)

        feat = np.column_stack([
            onehot[s:e], gap_log, price_log[s:e]
        ]).astype(np.float32)  # (n_ev, 6)
        cu = cat_arr[s:e]

        valid = np.flatnonzero(y >= 0)
        if len(valid) == 0:
            continue
        if len(valid) > cap_per_user:
            valid = np.sort(rng.choice(valid, size=cap_per_user, replace=False))

        sp = split_of[uid]
        for j in valid:
            w0 = max(0, j - SEQ_LEN + 1)
            L = j - w0 + 1
            xn = np.zeros((SEQ_LEN, 6), dtype=np.float32)
            xc = np.zeros(SEQ_LEN, dtype=np.int32)
            xn[SEQ_LEN - L:] = feat[w0:j + 1]
            xc[SEQ_LEN - L:] = cu[w0:j + 1]
            Xn_list.append(xn)
            Xc_list.append(xc)
            y_list.append(int(y[j]))
            uid_list.append(uid)
            sp_list.append(sp)

    X_num = np.stack(Xn_list).astype(np.float32)
    X_cat = np.stack(Xc_list).astype(np.int32)
    y = np.array(y_list, dtype=np.int8)
    user_id = np.array(uid_list, dtype=np.int64)
    split = np.array(sp_list, dtype=np.int8)

    np.savez_compressed(
        DATA_DIR / "dataset.npz",
        X_num=X_num, X_cat=X_cat, y=y, user_id=user_id, split=split,
    )
    (MODEL_DIR / "category_index_map.json").write_text(
        json.dumps({"padding": 0, "map": cat_index}, ensure_ascii=False), encoding="utf-8")
    (MODEL_DIR / "feature_schema.json").write_text(json.dumps({
        "seq_len": SEQ_LEN,
        "num_features": NUM_COLS,
        "cat_feature": "category_id (index, 0=padding)",
        "n_categories_incl_pad": len(cat_index) + 1,
        "event_types": EVENT_TYPES,
        "label": "churn30 (이 행동 후 30분 무활동=1)",
        "bounce_seconds": BOUNCE_SEC,
        "seed": SEED,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    def dist(mask):
        yy = y[mask]
        return {"n": int(mask.sum()), "bounce_rate": round(float(yy.mean()), 4) if len(yy) else 0.0}

    meta = {
        "source": str(SRC_CSV.relative_to(ROOT)),
        "n_users_sampled": int(len(sel_users)),
        "cap_per_user": cap_per_user,
        "n_sequences": int(len(y)),
        "overall_bounce_rate": round(float(y.mean()), 4),
        "splits": {
            "train": dist(split == 0),
            "val": dist(split == 1),
            "test": dist(split == 2),
        },
        "n_categories_incl_pad": len(cat_index) + 1,
        "seq_len": SEQ_LEN,
        "seed": SEED,
    }
    (DATA_DIR / "preprocess_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[prep] DONE", json.dumps(meta, ensure_ascii=False))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-users", type=int, default=150_000)
    ap.add_argument("--cap-per-user", type=int, default=100)
    args = ap.parse_args()
    build(n_users=args.n_users, cap_per_user=args.cap_per_user)
