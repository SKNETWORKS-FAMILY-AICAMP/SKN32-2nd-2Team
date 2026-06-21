# -*- coding: utf-8 -*-
"""Anchor 대시보드 — 개요(모델 비교) + 모델 진단(15시각화) + 추천 + 고객조회 + 관리자로그. chart_service가 eval 산출물 렌더."""
import streamlit as st
import pandas as pd
from app.services import chart_service as cs
from app import db as adb


def _show(chart, msg="데이터 없음(eval 산출물 확인)"):
    if chart is None:
        st.info(msg)
    else:
        st.altair_chart(chart, use_container_width=True)


def render_overview():
    st.subheader("개요 — 7모델 한눈 비교")
    k = cs.kpi()
    c1, c2, c3 = st.columns(3)
    c1.metric("최고 모델 (ROC-AUC)", f"{k.get('best_model','-')}", f"{k.get('best_auc','-')}")
    c2.metric("고위험 고객 (p≥0.65)", f"{k.get('n_high','-'):,}" if k.get('n_high') is not None else "-",
              f"/ {k.get('n_users','-'):,}명" if k.get('n_users') else "")
    c3.metric("기대 이탈매출 (위험)", f"{k.get('rev_at_risk','-'):,.0f}" if k.get('rev_at_risk') is not None else "-")

    st.markdown("**모델 비교표** (ROC-AUC 정렬 · Brier/ECE 낮을수록 좋음)")
    df = cs.comparison_table()
    if not df.empty:
        st.dataframe(df.style.highlight_max(subset=["ROC_AUC", "PR_AUC", "Recall", "F1"], color="#D7E3F4")
                     .highlight_min(subset=["Brier", "ECE"], color="#D7E3F4"), use_container_width=True)
    a, b = st.columns(2)
    with a: _show(cs.chart_auc_bar())
    with b: _show(cs.chart_radar_like())
    st.markdown("**선형 그래프 — ROC / PR 곡선 (모델 비교)**")
    a, b = st.columns(2)
    with a: _show(cs.chart_roc(cs.model_list()))
    with b: _show(cs.chart_pr(cs.model_list()))
    st.caption("불균형(이탈 82%)이라 PR-AUC는 높게 나옴 → 분별력은 ROC-AUC가 주지표. Recall은 운영 임계값 기준. "
               "Transformer(시퀀스)는 val 기준이라 ROC/PR 곡선은 정형 6모델만 표시.")


def render_diagnostics():
    st.subheader("Model Diagnostics — 15 시각화")
    models = cs.model_list()
    sel = st.multiselect("비교할 모델(곡선)", models, default=models[:3])
    one = st.selectbox("단일 모델(분포·혼동행렬·Lift 등)", models, index=len(models) - 1)

    st.markdown("#### 6·7. PR / ROC 곡선")
    a, b = st.columns(2)
    with a: _show(cs.chart_pr(sel))
    with b: _show(cs.chart_roc(sel))

    st.markdown("#### 12·9. Calibration / Threshold P·R·F1")
    a, b = st.columns(2)
    with a: _show(cs.chart_calibration(sel))
    with b: _show(cs.chart_threshold(one))

    st.markdown("#### 8·11. 이탈 스코어 분포 / Lift")
    a, b = st.columns(2)
    with a: _show(cs.chart_score_dist(one))
    with b: _show(cs.chart_lift(one))

    st.markdown("#### 10. Confusion Matrix (운영 임계값)")
    cm = cs.confusion(one)
    if cm is not None: st.table(cm)

    st.markdown("#### 2. 데이터 분포")
    col = st.selectbox("피처", ["recency_days", "n_events", "avg_price", "n_categories", "brand_loyalty"], index=0)
    _show(cs.chart_data_distribution(col))

    st.markdown("#### 13. 피처 중요도 (SHAP 대안)")
    _show(cs.chart_shap(one), "중요도 산출물 없음 — pp_feature_importance 실행 필요")
    st.caption("SHAP은 numpy 버전 충돌로 보류 → 모델 네이티브 피처 중요도(트리=gain, 선형=|coef|)로 대체. 상세 리포트 26-8.")

    st.markdown("#### 14·15. Value at Risk / Revenue Recovery 시뮬")
    a, b = st.columns([2, 1])
    with a: _show(cs.chart_var_treemap(one))
    with b:
        st.caption("가정 조정")
        sr = st.slider("캠페인 성공률", 0.05, 0.6, 0.3, 0.05)
        cost = st.slider("1인 캠페인 비용", 0.0, 2.0, 0.1, 0.05)
        rr = cs.revenue_recovery(one, campaign_cost=cost, success_rate=sr)
        if rr:
            st.metric("기대 회수매출", f"{rr['기대회수매출']:,.0f}")
            st.metric("캠페인 비용", f"{rr['캠페인비용']:,.0f}")
            st.metric("순효과", f"{rr['순효과']:,.0f}", delta_color="normal")
            st.caption(f"타깃(고위험) {rr['가정']['타깃(고위험)']:,}명")

    st.markdown("#### 1·3·4·5. 아키텍처 / Cohort / 베이스라인 / Loss")
    st.info("1 아키텍처=정적 다이어그램(reports), 3 Cohort Retention=canonical 추가 시, "
            "4 베이스라인 PR-AUC=Last-cat·GBM 대비(metrics), 5 Train/Val Loss=부스팅 per-iter·DL epoch(트리/LogReg=해당없음).")


def render_recommendation():
    st.subheader("추천 — 유사 카테고리(행동 동시출현)")
    cat = cs.category_catalog()
    if cat.empty:
        st.info("recommendation 산출물 없음"); return
    import pandas as pd
    from pathlib import Path
    sim_p = Path(cs.REC) / "category_similar.parquet"
    if not sim_p.exists():
        st.info("category_similar 없음"); return
    sim = pd.read_parquet(sim_p)
    catx = cat.copy()
    named = catx[catx.category_code.notna()] if "category_code" in catx else catx
    seed = st.selectbox("기준 카테고리", named.category_id.head(30).tolist(),
                        format_func=lambda c: f"{c} ({named.set_index('category_id').loc[c,'category_code'] if c in named.category_id.values else ''})")
    rows = sim[sim.category_id == seed].merge(cat, left_on="similar_category_id", right_on="category_id", suffixes=("", "_s"))
    if not rows.empty:
        st.dataframe(rows[["rank", "cosine", "similar_category_id", "category_code", "top_brand", "price_median"]], use_container_width=True)
    else:
        st.info("유사 카테고리 없음")


def render_customer_churn(role="customer", login_user="demo"):
    """고객 이탈 조회 + 고위험 자동추천 (교육과제 ④ 이탈저장 · ⑤ 추천)."""
    st.subheader("고객 이탈 조회 · 자동 추천")
    models = cs.model_list()
    model = st.selectbox("모델", models, index=len(models) - 1, key="cc_model")
    c1, c2 = st.columns(2)
    uid_sel = c1.selectbox("고객 선택(user_id)", cs.sample_user_ids(model), key="cc_user")
    uid_manual = c2.text_input("또는 user_id 직접 입력", "", key="cc_user_manual")  # 아이디 입력칸
    uid = uid_manual.strip() or uid_sel
    db_on = adb.available()
    if not db_on:
        st.warning("MySQL 미연결 — 조회는 되나 저장(과제④⑤)은 configs/.env 설정 후 활성됩니다.")
    if st.button("이탈 분석 실행"):
        u = cs.user_churn(uid, model)
        if not u:
            st.info("해당 고객 데이터 없음"); return
        c1, c2, c3 = st.columns(3)
        c1.metric("이탈 확률", f"{u['churn_prob']:.3f}")
        c2.metric("위험 등급", {"high": "고위험", "medium": "중위험", "low": "정상"}[u["risk"]])
        c3.metric("관심 브랜드", u["top_brand"])
        st.write(f"**추천 액션**: {u['action']}")
        rec = cs.recommend_for(uid, model) if u["risk"] in ("high", "medium") else None
        if rec and rec["recommendations"]:
            st.markdown("**고위험 자동 추천(유사 카테고리)**")
            st.dataframe(pd.DataFrame(rec["recommendations"]), use_container_width=True)
        if db_on:
            try:
                adb.save_prediction(uid, model, u["churn_prob"], u["risk"], u["action"])   # 과제④
                if rec is not None:
                    adb.save_recommendation(uid, rec or {"note": "no-sim"})                 # 과제⑤
                st.success("이탈예측·추천을 MySQL에 저장했습니다. (과제 ④⑤)")
            except Exception as e:
                st.warning(f"DB 저장 실패: {e}")
        else:
            st.info("MySQL 연결 시 이탈예측·추천이 DB에 저장됩니다(과제 ④⑤).")


def render_admin_logs():
    st.subheader("관리자 — 로그 / 예측 이력")
    if not adb.available():
        st.warning("MySQL 미연결 — configs/.env 설정 시 로그/이력 표시."); return
    a, b = st.columns(2)
    with a:
        st.markdown("**로그인 로그(최신, 최대 200 보존)**")
        st.dataframe(pd.DataFrame(adb.recent_logins(20)), use_container_width=True)
    with b:
        st.markdown("**이탈예측 이력**")
        st.dataframe(pd.DataFrame(adb.recent_predictions(20)), use_container_width=True)


def render_realtime_bounce():
    """SB — 실시간 세션 이탈/바운스(집계 churn이 못 잡는 단발·즉시이탈)."""
    st.subheader("실시간 세션 이탈 / 바운스")
    meta = cs.session_bounce_meta(); samples = cs.session_samples()
    if not samples:
        st.info("세션 바운스 산출물 없음(pp_session_bounce 실행 필요)"); return
    st.caption(f"라벨: {meta.get('label','-')} · 모델 {meta.get('model','-')} · AUC {meta.get('auc','-')} "
               f"(집계 churn 0.79보다 높음 — 세션 진행상태가 더 예측적)")
    label = {"BOUNCE": "바운스(단발)", "CART_ABANDON": "장바구니 이탈", "PURCHASE": "구매 세션"}
    tag = st.selectbox("시뮬 세션 유형", list(samples), format_func=lambda t: label.get(t, t))
    ev = samples[tag]
    step = st.slider("실시간 재생(행동 순서)", 1, len(ev), 1, help="시뮬 사이트에서 이벤트가 들어오는 상황")
    upto = ev[:step]
    cur = upto[-1]
    c1, c2, c3 = st.columns(3)
    c1.metric("현재 이탈(바운스) 확률", f"{cur['bounce_prob']:.3f}")
    c2.metric("현재 행동", cur["event"])
    c3.metric("위험", "고위험" if cur["bounce_prob"] >= 0.65 else "관찰" if cur["bounce_prob"] >= 0.35 else "안정")
    _show(cs.chart_session_replay(upto))
    st.dataframe(pd.DataFrame(upto)[["step", "event", "brand", "price", "dt_prev_s", "bounce_prob"]], use_container_width=True)
    if cur["bounce_prob"] >= 0.65:
        st.warning("⚠ 이 세션은 이탈 위험 높음 → 실시간 쿠폰/리마인드 권고 (이탈예측→이탈방지)")


def render_dashboard(role="customer", login_user="demo"):
    try:
        adb.init_db()
    except Exception:
        pass  # MySQL 미연결 — 차트는 표시, DB기능은 각 패널에서 경고
    # 스크롤 수정: 랜딩용 .block-container(560px·flex중앙)를 대시보드에선 전체폭·일반흐름으로 복원
    st.markdown("""<style>
      .block-container{max-width:96% !important;min-height:auto !important;display:block !important;
        align-items:initial !important;justify-content:initial !important;padding:24px 28px !important;}
    </style>""", unsafe_allow_html=True)
    st.title("Anchor — 실시간 고객 이탈 분석")
    base = ["개요(Overview)", "고객 이탈 조회", "실시간 세션/바운스", "추천(Recommendation)"]
    admin_tabs = ["모델 진단(Diagnostics)", "관리자 로그/이력"]
    names = base + (admin_tabs if role == "admin" else [])
    tabs = st.tabs(names)
    with tabs[0]:
        render_overview()
    with tabs[1]:
        render_customer_churn(role, login_user)
    with tabs[2]:
        render_realtime_bounce()
    with tabs[3]:
        render_recommendation()
    if role == "admin":
        with tabs[4]:
            render_diagnostics()
        with tabs[5]:
            render_admin_logs()
    else:
        st.caption("ℹ 모델 진단(15시각화)·로그 이력은 **관리자 전용**입니다. (교육과제 ③)")
