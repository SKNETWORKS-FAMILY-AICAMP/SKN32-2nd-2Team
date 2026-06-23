# -*- coding: utf-8 -*-
"""infrastructure/files — 운영 조회용 데이터셋 reader(유저 대시보드·추천·데이터분포).
eval_predictions/recommendation 카탈로그를 읽어 chart-ready로 변환. 캐시(lru)."""
import json
from functools import lru_cache
from app.config import EVAL_DIR, REC_DIR, DATA_DIR, SB_DIR

CHURN_DIR = DATA_DIR / "churn"
RISK_HIGH, RISK_LOW = 0.65, 0.35
FEATURE_ORDER_10 = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart",
                    "n_remove_from_cart", "n_purchase", "avg_price", "purch_amt"]


@lru_cache(maxsize=1)
def _preds():
    import pandas as pd
    p = EVAL_DIR / "eval_predictions.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@lru_cache(maxsize=1)
def _metrics():
    p = EVAL_DIR / "metrics_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def baseline_comparison():
    """모델 비교(베이스라인): per-model 산출물(모델팀 우선·폴백) → 행리스트."""
    from app.infrastructure.files import eval_artifacts as ea
    rows = [{"model_name": k, "roc_auc": v["auc"], "pr_auc": v.get("pr_auc"), "f1": v.get("f1")}
            for k, v in ea.all_metrics().items()]
    rows.sort(key=lambda r: r["roc_auc"], reverse=True)
    return rows


def data_distribution(feature="recency_days", nbins=20):
    """canonical tabular의 피처 히스토그램(없으면 빈 리스트)."""
    import pandas as pd, numpy as np
    p = CHURN_DIR / "train_tabular_v2.parquet"
    if not p.exists():
        return []
    df = pd.read_parquet(p, columns=[feature]) if feature else pd.read_parquet(p)
    if feature not in df.columns:
        return []
    s = df[feature].astype(float)
    lo, hi = float(s.quantile(0.01)), float(s.quantile(0.99))
    if hi <= lo:
        hi = lo + 1
    edges = np.linspace(lo, hi, nbins + 1)
    cnt, _ = np.histogram(s.clip(lo, hi), bins=edges)
    return [{"feature": feature, "bin": round(float(edges[i]), 3), "count": int(cnt[i])} for i in range(nbins)]


def cohort_retention(weeks=8):
    """잔존(retention) 곡선 — tabular_v2의 tenure_days 기반 생존곡선(실데이터).
    week_index k 잔존율 = tenure_days >= 7k 인 유저 비율(관측기간 내 존속). 이탈 현상 설명용."""
    import pandas as pd
    p = CHURN_DIR / "train_tabular_v2.parquet"
    if not p.exists():
        return []
    cols = pd.read_parquet(p).columns
    if "tenure_days" not in cols:
        return []
    t = pd.read_parquet(p, columns=["tenure_days"])["tenure_days"].astype(float)
    n = int(len(t))
    if n == 0:
        return []
    rows = []
    for k in range(weeks + 1):
        retained = int((t >= 7 * k).sum())
        rows.append({"cohort": "전체", "week_index": k, "users": n,
                     "retained_users": retained, "retention_rate": round(retained / n, 4)})
    return rows


def _risk(p):
    return "high" if p >= RISK_HIGH else ("medium" if p >= RISK_LOW else "low")


def user_dashboard(user_id, model="CatBoost"):
    """유저 최신 이탈확률(active 모델 기준) — eval_predictions에서 조회."""
    df = _preds()
    if df.empty:
        return None
    try:
        uid = int(user_id)
    except (ValueError, TypeError):
        uid = user_id
    sub = df[(df.user_id == uid) & (df.model_name == model)]
    if sub.empty:
        sub = df[df.user_id == uid]
    if sub.empty:
        return None
    r = sub.iloc[0]
    p = float(r.y_score)
    return {"user_id": user_id, "model": str(r.model_name),
            "latest_prediction": {"churn_probability": round(p, 4), "risk_level": _risk(p),
                                  "top_category": int(r.get("top_category", 0)) if "top_category" in sub.columns else None,
                                  "top_brand": (str(r.top_brand) if "top_brand" in sub.columns else None)}}


def model_summary_stats(model):
    """모델별 운영 요약(eval_predictions 기준): 총 예측·고위험수·평균 이탈확률. 없으면 None."""
    df = _preds()
    if df.empty or "model_name" not in df.columns:
        return None
    sub = df[df.model_name == model]
    if sub.empty:
        return None
    p = sub["y_score"].astype(float)
    return {"total": int(len(sub)),
            "high_risk": int((p >= RISK_HIGH).sum()),
            "avg": round(float(p.mean()), 4),
            "latest_at": None}


def model_names():
    """대시보드 드롭다운용 모델명(per-model 산출물 기준, 모델팀 우선·폴백)."""
    from app.infrastructure.files import eval_artifacts as ea
    m = ea.all_metrics()
    best = max(m, key=lambda k: m[k]["auc"], default=None)
    # model_name 동시 제공(대시보드가 model_name/model 둘 다 읽어도 동작 — 드롭다운 None 방지)
    out = [{"model": k, "model_name": k, "is_best": k == best, "auc": v["auc"]}
           for k, v in m.items()]
    out.sort(key=lambda x: x["auc"], reverse=True)
    return out


def sample_users(model="CatBoost", n=60):
    """고객 선택 드롭다운 샘플(고/중/저위험 섞음). chart_service.sample_user_ids 포팅."""
    import pandas as pd
    df = _preds()
    if df.empty:
        return []
    d = df[df.model_name == model].sort_values("y_score", ascending=False)
    if d.empty:
        return []
    mix = pd.concat([d.head(n // 2), d.tail(n // 4),
                     d.iloc[len(d) // 2: len(d) // 2 + n // 4]])
    return mix["user_id"].astype(str).unique().tolist()[:n]


def aux_ensemble_summary():
    """보조 태스크(bounce·category) 앙상블 요약 — 모델별 + 합산 성능(발표/대시보드 '앙상블 현황')."""
    out = {}
    for task in ("session_bounce", "next_category"):
        p = EVAL_DIR / task / "ensemble_summary.json"
        if p.exists():
            out[task] = json.loads(p.read_text(encoding="utf-8"))
    return out


def session_bounce():
    """실시간 세션 바운스 메타 + 샘플 세션(파일 서빙)."""
    meta_p, samp_p = SB_DIR / "meta.json", SB_DIR / "sample_sessions.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    samples = json.loads(samp_p.read_text(encoding="utf-8")) if samp_p.exists() else {}
    return {"meta": meta, "samples": samples}


@lru_cache(maxsize=1)
def _tabular():
    """실시간 추론용 피처 룩업 — train+test 정본 결합(샘플 유저는 test셋이라 둘 다 필요)."""
    import pandas as pd
    frames = []
    for fn in ("train_tabular_v2.parquet", "test_tabular_v2.parquet"):
        p = CHURN_DIR / fn
        if p.exists():
            frames.append(pd.read_parquet(p))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def user_features(user_id):
    """실시간 추론용: 유저의 v2 피처 1행(없으면 None)."""
    df = _tabular()
    if df.empty:
        return None
    try:
        uid = int(user_id)
    except (ValueError, TypeError):
        uid = user_id
    sub = df[df.user_id == uid]
    if sub.empty:
        return None
    feat_cols = [c for c in df.columns if c not in ("user_id", "churn",
                 "churn_no_purchase", "cohort_recency7", "last_brand", "last_cat_id",
                 "top_brand", "top_category_id")]
    return sub.iloc[[0]][feat_cols]


def recommendations(user_id, model="CatBoost", topn=5):
    """유저 관심 카테고리(eval_predictions.top_category) → category_similar 유사 카테고리."""
    import pandas as pd
    df = _preds()
    sim_p = REC_DIR / "category_similar.parquet"
    cat_p = REC_DIR / "category_catalog.parquet"
    if df.empty or not sim_p.exists():
        return {"items": [], "categories": []}
    try:
        uid = int(user_id)
    except (ValueError, TypeError):
        uid = user_id
    sub = df[df.user_id == uid]
    if sub.empty or "top_category" not in sub.columns:
        return {"items": [], "categories": []}
    seed = int(sub.iloc[0].top_category)
    sim = pd.read_parquet(sim_p)
    rows = sim[sim.category_id == seed].head(topn)
    cats = []
    catalog = pd.read_parquet(cat_p) if cat_p.exists() else pd.DataFrame()
    from app.infrastructure.files import catalog_store as cat
    for _, r in rows.iterrows():
        cid = int(r.get("similar_category_id", 0))
        name = cat.name_of(cid)        # 매핑 우선(REES46 이름 결측 → 브랜드/가격·코드 라벨)
        if name is None and not catalog.empty and "category_id" in catalog.columns:
            hit = catalog[catalog.category_id == cid]
            if not hit.empty and "category_code" in hit.columns:
                v = hit.iloc[0].get("category_code")
                if v is not None and str(v) != "nan":
                    name = str(v)
        cats.append({"category_id": cid, "name": name or f"category_{cid}",
                     "score": round(float(r.get("cosine", 0)), 4)})
    return {"items": [], "categories": cats, "seed_category": seed}
