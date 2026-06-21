# -*- coding: utf-8 -*-
"""chart_service — 검증 산출물(eval_predictions·metrics·curves)을 읽어 15시각화 데이터/차트 생성.
의존성: streamlit 번들 altair + pandas/numpy/joblib (requirements 내). 데모는 백엔드 없이 직접 읽기.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

try:
    import streamlit as st
    cache = st.cache_data
except Exception:                       # 비-streamlit 환경(테스트)에서도 import 되게
    def cache(f=None, **k):
        return f if f else (lambda g: g)

ROOT = Path(__file__).resolve().parents[2]          # 가지마 루트
EVAL = ROOT / "data" / "processed" / "evaluation"
REC = ROOT / "data" / "processed" / "recommendation"
CHURN = ROOT / "data" / "processed" / "churn"
SB = ROOT / "data" / "processed" / "session_bounce"
MODELS_TAB = ["DecisionTree", "RandomForest", "LogReg", "XGBoost", "LightGBM", "CatBoost"]
RISK_HIGH, RISK_LOW = 0.65, 0.35


# ---------- 로더 ----------
@cache
def metrics():
    p = EVAL / "metrics_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

@cache
def curves():
    p = EVAL / "curves.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

@cache
def preds():
    p = EVAL / "eval_predictions.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()

@cache
def shap_summary():
    p = EVAL / "shap_summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

@cache
def category_catalog():
    p = REC / "category_catalog.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def model_list():
    m = metrics()
    return [k for k in MODELS_TAB if k in m]


# ---------- 메인: KPI + 비교표 ----------
def kpi():
    m, ep = metrics(), preds()
    tab = {k: v for k, v in m.items() if "auc" in v}
    best = max(tab, key=lambda k: tab[k]["auc"]) if tab else None
    out = {"best_model": best, "best_auc": tab.get(best, {}).get("auc") if best else None}
    if not ep.empty and best:
        d = ep[ep.model_name == best]
        high = d[d.y_score >= RISK_HIGH]
        out["n_users"] = int(d.user_id.nunique())
        out["n_high"] = int(len(high))
        out["rev_at_risk"] = round(float((high.y_score * high.revenue.clip(lower=0)).sum()), 0)
    return out


def comparison_table():
    m = metrics()
    rows = []
    for k in MODELS_TAB:
        if k in m and "auc" in m[k]:
            v = m[k]; rows.append({"model": k, "ROC_AUC": v["auc"], "PR_AUC": v["pr_auc"],
                                   "Recall": v.get("recall_at_thr") or _recall(k), "F1": v["f1"],
                                   "Brier": v["brier"], "ECE": v.get("ece")})
    df = pd.DataFrame(rows)
    return df.sort_values("ROC_AUC", ascending=False).reset_index(drop=True) if not df.empty else df


def _recall(model):
    ep = preds()
    if ep.empty: return None
    d = ep[ep.model_name == model]
    return round(float(d.loc[d.y_pred == 1, "y_true"].sum() / max(d.y_true.sum(), 1)), 4)


# ---------- 차트(altair) ----------
def _alt():
    import altair as alt
    return alt


def chart_auc_bar():
    alt = _alt(); df = comparison_table()
    if df.empty: return None
    return alt.Chart(df).mark_bar().encode(
        x=alt.X("ROC_AUC:Q", scale=alt.Scale(domain=[0.7, 0.8])),
        y=alt.Y("model:N", sort="-x"),
        color=alt.condition(alt.datum.ROC_AUC >= df.ROC_AUC.max(), alt.value("#2F5496"), alt.value("#9DB7E0")),
        tooltip=list(df.columns)).properties(height=220, title="모델별 ROC-AUC (분별력)")


def chart_radar_like():
    """레이더 대용: 5지표 정규화 그룹막대."""
    alt = _alt(); df = comparison_table()
    if df.empty: return None
    mets = ["ROC_AUC", "PR_AUC", "Recall", "F1", "Brier"]
    long = df.melt(id_vars="model", value_vars=mets, var_name="metric", value_name="value")
    # Brier는 낮을수록 좋음 → 역수 정규화
    long.loc[long.metric == "Brier", "value"] = 1 - long.loc[long.metric == "Brier", "value"]
    return alt.Chart(long).mark_bar().encode(
        x=alt.X("model:N", title=None), xOffset="metric:N",
        y=alt.Y("value:Q"), color="metric:N", tooltip=["model", "metric", "value"]).properties(height=240, title="5지표 비교(Brier는 1-값)")


def chart_roc(models):
    alt = _alt(); c = curves(); frames = []
    for mo in models:
        if mo in c:
            r = c[mo]["roc"]; frames.append(pd.DataFrame({"fpr": r["fpr"], "tpr": r["tpr"], "model": mo}))
    if not frames: return None
    df = pd.concat(frames)
    line = alt.Chart(df).mark_line().encode(x="fpr:Q", y="tpr:Q", color="model:N", tooltip=["model", "fpr", "tpr"])
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
    return (line + diag).properties(height=300, title="ROC 곡선")


def chart_pr(models):
    alt = _alt(); c = curves(); frames = []
    for mo in models:
        if mo in c:
            r = c[mo]["pr"]; frames.append(pd.DataFrame({"recall": r["recall"], "precision": r["precision"], "model": mo}))
    if not frames: return None
    return alt.Chart(pd.concat(frames)).mark_line().encode(x="recall:Q", y="precision:Q", color="model:N").properties(height=300, title="PR 곡선")


def chart_threshold(model):
    alt = _alt(); c = curves()
    if model not in c: return None
    df = pd.DataFrame(c[model]["threshold"]).melt("t", ["precision", "recall", "f1"], var_name="metric", value_name="value")
    return alt.Chart(df).mark_line().encode(x="t:Q", y="value:Q", color="metric:N").properties(height=300, title=f"Threshold P/R/F1 — {model}")


def chart_calibration(models):
    alt = _alt(); c = curves(); frames = []
    for mo in models:
        if mo in c:
            cc = c[mo]["calibration"]; frames.append(pd.DataFrame({"pred": cc["pred"], "true": cc["true"], "model": mo}))
    if not frames: return None
    line = alt.Chart(pd.concat(frames)).mark_line(point=True).encode(x="pred:Q", y="true:Q", color="model:N")
    diag = alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
    return (line + diag).properties(height=300, title="Calibration (보정)")


def chart_score_dist(model):
    alt = _alt(); ep = preds()
    if ep.empty: return None
    d = ep[ep.model_name == model].copy(); d["label"] = d.y_true.map({1: "이탈", 0: "유지"})
    return alt.Chart(d.sample(min(len(d), 20000), random_state=0)).mark_bar(opacity=0.6).encode(
        x=alt.X("y_score:Q", bin=alt.Bin(maxbins=40)), y="count()", color="label:N").properties(height=280, title=f"이탈 스코어 분포 — {model}")


def confusion(model):
    ep = preds()
    if ep.empty: return None
    d = ep[ep.model_name == model]
    tp = int(((d.y_pred == 1) & (d.y_true == 1)).sum()); fp = int(((d.y_pred == 1) & (d.y_true == 0)).sum())
    fn = int(((d.y_pred == 0) & (d.y_true == 1)).sum()); tn = int(((d.y_pred == 0) & (d.y_true == 0)).sum())
    return pd.DataFrame([[tn, fp], [fn, tp]], index=["실제 유지", "실제 이탈"], columns=["예측 유지", "예측 이탈"])


def chart_lift(model):
    alt = _alt(); ep = preds()
    if ep.empty: return None
    d = ep[ep.model_name == model].sort_values("y_score", ascending=False).reset_index(drop=True)
    d["decile"] = (np.arange(len(d)) / len(d) * 10).astype(int).clip(0, 9) + 1
    base = d.y_true.mean()
    g = d.groupby("decile").y_true.mean().reset_index(); g["lift"] = g.y_true / base
    return alt.Chart(g).mark_bar().encode(x="decile:O", y=alt.Y("lift:Q", title="Lift"), tooltip=["decile", "lift"]).properties(height=280, title=f"Lift(십분위) — {model}")


def chart_var_treemap(model, topn=12):
    """VaR: 카테고리별 위험매출(고위험 유저 revenue×score)."""
    alt = _alt(); ep = preds()
    if ep.empty: return None
    d = ep[ep.model_name == model].copy()
    d["risk_rev"] = d.y_score * d.revenue.clip(lower=0)
    g = d.groupby("top_category", as_index=False).risk_rev.sum().sort_values("risk_rev", ascending=False).head(topn)
    g["top_category"] = g.top_category.astype(str)
    return alt.Chart(g).mark_bar().encode(x=alt.X("risk_rev:Q", title="위험매출"), y=alt.Y("top_category:N", sort="-x"),
                                          tooltip=["top_category", "risk_rev"]).properties(height=320, title=f"Value at Risk(카테고리별) — {model}")


def revenue_recovery(model, campaign_cost=1.0, success_rate=0.2):
    """가정 명시 시뮬: 고위험 타깃 캠페인 → 회수 기대매출 − 비용."""
    ep = preds()
    if ep.empty: return None
    d = ep[ep.model_name == model]
    high = d[d.y_score >= RISK_HIGH]
    expected_recover = float((high.y_score * high.revenue.clip(lower=0)).sum() * success_rate)
    cost = float(len(high) * campaign_cost)
    return {"가정": {"성공률": success_rate, "1인캠페인비용": campaign_cost, "타깃(고위험)": int(len(high))},
            "기대회수매출": round(expected_recover, 0), "캠페인비용": round(cost, 0), "순효과": round(expected_recover - cost, 0)}


def chart_data_distribution(col="recency_days"):
    alt = _alt()
    p = CHURN / "train_tabular_v2.parquet"
    if not p.exists(): return None
    df = pd.read_parquet(p, columns=[col, "churn"]).sample(20000, random_state=0) if p.exists() else None
    df["label"] = df.churn.map({1: "이탈", 0: "유지"})
    return alt.Chart(df).mark_bar(opacity=0.6).encode(x=alt.X(f"{col}:Q", bin=alt.Bin(maxbins=40)), y="count()", color="label:N").properties(height=280, title=f"데이터 분포 — {col}")


def chart_shap(model):
    alt = _alt(); sh = shap_summary()
    if model not in sh: return None
    df = pd.DataFrame([{"feature": k, "mean_abs_shap": v} for k, v in sh[model].items()])
    return alt.Chart(df).mark_bar().encode(x="mean_abs_shap:Q", y=alt.Y("feature:N", sort="-x")).properties(height=300, title=f"SHAP 중요도 — {model}")


# ---------- 고객 이탈 조회 + 추천 (교육과제 ④⑤) ----------
def risk_level(p):
    return "high" if p >= RISK_HIGH else ("medium" if p >= RISK_LOW else "low")


def retention_action(p):
    r = risk_level(p)
    return ("쿠폰 발송 + 재방문 알림(고위험)" if r == "high" else
            "장바구니 리마인드/맞춤추천(중위험)" if r == "medium" else "정상 유지")


@cache
def sample_user_ids(model="CatBoost", n=60):
    ep = preds()
    if ep.empty: return []
    d = ep[ep.model_name == model].sort_values("y_score", ascending=False)
    # 고위험·중위험·저위험 섞어서 샘플
    ids = pd.concat([d.head(n // 2), d.tail(n // 4), d.iloc[len(d)//2: len(d)//2 + n//4]])["user_id"].astype(str).unique().tolist()
    return ids[:n]


def user_churn(user_id, model):
    ep = preds()
    if ep.empty: return None
    d = ep[(ep.model_name == model) & (ep.user_id.astype(str) == str(user_id))]
    if d.empty: return None
    p = float(d.iloc[0].y_score)
    return {"user_id": str(user_id), "model": model, "churn_prob": round(p, 4), "risk": risk_level(p),
            "action": retention_action(p), "revenue": float(d.iloc[0].revenue),
            "top_category": str(d.iloc[0].top_category), "top_brand": str(d.iloc[0].top_brand)}


def recommend_for(user_id, model, topk=5):
    """고위험 고객용: 관심 카테고리 → 유사 카테고리 추천(행동 동시출현)."""
    u = user_churn(user_id, model)
    if u is None: return None
    from pathlib import Path
    sim_p = Path(REC) / "category_similar.parquet"; cat = category_catalog()
    recs = []
    if sim_p.exists() and not cat.empty:
        sim = pd.read_parquet(sim_p)
        try:
            seed = int(u["top_category"])          # 18자리 → float 거치면 정밀도 손실, int() 직접
            rows = sim[sim.category_id == seed].merge(cat, left_on="similar_category_id", right_on="category_id", suffixes=("", "_s")).head(topk)
            recs = rows[["similar_category_id", "category_code", "top_brand", "price_median"]].to_dict("records")
        except Exception:
            recs = []
    return {"user_id": u["user_id"], "risk": u["risk"], "action": u["action"],
            "top_category": u["top_category"], "recommendations": recs}


# ---------- SB: 실시간 세션/바운스 ----------
@cache
def session_samples():
    p = SB / "sample_sessions.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@cache
def session_bounce_meta():
    p = SB / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def chart_session_replay(events_upto):
    """리플레이: step별 bounce_prob 라인(현재 step까지)."""
    alt = _alt()
    if not events_upto: return None
    df = pd.DataFrame(events_upto)
    line = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X("step:O", title="세션 내 행동 순서"),
        y=alt.Y("bounce_prob:Q", scale=alt.Scale(domain=[0, 1]), title="이탈(바운스) 확률"),
        tooltip=["step", "event", "bounce_prob"])
    rule = alt.Chart(pd.DataFrame({"y": [0.65]})).mark_rule(strokeDash=[4, 4], color="red").encode(y="y")
    return (line + rule).properties(height=260, title="실시간 세션 이탈 확률(누적)")
