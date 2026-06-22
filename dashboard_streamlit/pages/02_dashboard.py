import streamlit as st
import pandas as pd
import time

# 계약서 기반 서비스 및 컴포넌트 임포트
from components.layout import load_css, render_brand_header, render_sidebar_menu
from components.error_state import render_error, render_empty
from components.charts import render_chart_payload  # 차트 JSON 공통 wrapper 렌더러

from services import (
    dashboard_service as dsvc,
    prediction_service as psvc,
    recommendation_service as rsvc,
    chart_service as csvc
)


# [계약서 요구사항] 위험 등급 한글화 매핑 함수
def translate_risk_level(level: str) -> str:
    mapping = {
        "high": "🚨 고위험",
        "medium": "⚠️ 중위험",
        "low": "✅ 정상"
    }
    return mapping.get(level.lower(), level)


def main() -> None:
    st.set_page_config(page_title="GAJIMA BI Dashboard", page_icon="📊", layout="wide")
    load_css("styles/main.css")
    render_sidebar_menu()

    # 로그인 상태 방어 코드
    if not st.session_state.get("is_logged_in", False):
        st.warning("로그인이 필요한 페이지입니다. 얼굴 로그인 페이지로 이동합니다.")
        time.sleep(1.5)
        st.switch_page("pages/01_face_login.py")
        return

    render_brand_header(
        f"Welcome back, {st.session_state.get('display_name', 'User')}님",
        f"Role: {st.session_state.get('role', 'customer')} | 실시간 Churn 예측 및 세션 바운스 모니터링 시스템"
    )

    # 3개 탭 구성
    personal_tab, operation_tab, diagnostic_tab = st.tabs([
        "👤 개인 — 고객 이탈 진단",
        "🏢 운영 — 모델 요약 / 고위험 고객",
        "🔬 모델 진단 — 차트 분석"
    ])

    # ==========================================
    # 탭 1: 개인 — 고객 이탈 진단
    # ==========================================
    with personal_tab:
        st.subheader("개인 맞춤형 실시간 이탈 위험 진단")

        # 유저 선택: ① 목록에서 선택 ② 직접 입력 ③ 서버에서 랜덤 수신
        _su = dsvc.get_sample_users(n=20)
        sample_ids = ([str(u) for u in _su["data"]["users"]]
                      if _su.get("ok") and isinstance(_su.get("data"), dict) and _su["data"].get("users") else [])
        sel_mode = st.radio("유저 선택 방식", ["목록에서 선택", "직접 입력", "랜덤 수신"], horizontal=True)
        if sel_mode == "목록에서 선택":
            target_user_id = st.selectbox("실제 유저 ID 선택", sample_ids or ["(샘플 없음)"])
        elif sel_mode == "직접 입력":
            target_user_id = st.text_input("유저 ID 직접 입력", value=st.session_state.get("user_id", ""))
            if sample_ids:
                st.caption("💡 실제 ID 예시: " + ", ".join(sample_ids[:5]) + " · 임의 ID는 대표 고객으로 매핑됩니다.")
        else:  # 랜덤 수신 — 서버 샘플에서 하나 무작위 수신
            if st.button("🎲 서버에서 랜덤 유저 받아오기") and sample_ids:
                import random
                st.session_state["_rand_uid"] = random.choice(sample_ids)
            target_user_id = st.session_state.get("_rand_uid", "")
            if target_user_id:
                st.success(f"받아온 랜덤 유저: **{target_user_id}**")
        target_user_id = (target_user_id or "").strip()

        if st.button("실시간 진단하기", type="primary", use_container_width=True) and target_user_id:
            with st.spinner("백엔드에서 실시간 추론 중..."):
                # ── 카드A: 과거 7일 이탈(REES46 학습모델). 행동이력 없으면 대표 고객(shadow)로 대체 ──
                pred_resp = psvc.run_realtime_prediction(target_user_id)
                shadow_id = None
                if not pred_resp.get("ok"):
                    users = (_su["data"]["users"] if _su.get("ok") and isinstance(_su.get("data"), dict) else [])
                    if users:                                # 가입/임의 ID → 결정적으로 대표 고객 매핑(재기동에도 고정)
                        import hashlib
                        _idx = int(hashlib.md5(target_user_id.encode()).hexdigest(), 16) % len(users)
                        shadow_id = str(users[_idx])
                        pred_resp = psvc.run_realtime_prediction(shadow_id)
                ref_id = shadow_id or target_user_id
                # ── 카드B: 실시간 세션 이탈(시뮬 사이트 라이브) ──
                sim_resp = psvc.get_sim_user_score(target_user_id)
                reco_resp = rsvc.get_recommendations(ref_id)

            pred_data = pred_resp["data"] if pred_resp.get("ok") and pred_resp.get("data") else {}
            sim_data = sim_resp["data"] if sim_resp.get("ok") and sim_resp.get("data") else {}

            # ===== 두 관점 카드 =====
            st.markdown("### 📌 이탈 진단 — 두 관점")
            cardA, cardB = st.columns(2)
            with cardA:
                st.markdown("#### 📊 과거 7일 이탈 (학습 모델)")
                if pred_data:
                    if shadow_id:
                        st.caption(f"⚠️ '{target_user_id}'는 행동이력이 없어 대표 고객 **{shadow_id}** 기준 표시")
                    st.metric("이탈 확률", f"{pred_data.get('churn_probability', 0) * 100:.1f}%",
                              help=f"모델: {pred_data.get('model_name')}")
                    st.write("위험 등급:", translate_risk_level(pred_data.get("risk_level", "low")))
                    rd = pred_data.get("recency_days")
                    if rd is not None:
                        st.caption(f"🕒 마지막 활동 후 **{rd:.0f}일** 만의 방문" if rd > 0 else "🕒 방금 활동(0일)")
                    st.info(f"권장 액션: {pred_data.get('recommended_action', '-')}")
                else:
                    st.info("과거 모델 예측 불가(행동이력 없음).")
            with cardB:
                st.markdown("#### ⚡ 실시간 세션 이탈 (시뮬 라이브)")
                if sim_data:
                    if sim_data.get("attributed"):
                        st.caption("ℹ️ 가정: 현재 시뮬 사이트의 최근 활동을 이 유저의 실시간으로 간주")
                    st.metric("이탈 확률", f"{sim_data.get('churn_probability', 0) * 100:.1f}%",
                              help=f"source: {sim_data.get('source')}")
                    st.write("위험 등급:", translate_risk_level(sim_data.get("risk_level", "low")))
                    act = sim_data.get("recommended_action")
                    msg = act.get("message") if isinstance(act, dict) else act
                    if msg:
                        st.info(f"실시간 액션: {msg}")
                else:
                    st.caption(f"시뮬 활동 없음 — '{target_user_id}'가 시뮬 사이트(:3000)에서 둘러보면 실시간 표시됩니다.")

            st.divider()

            # ===== 상세: 추천 + 리텐션 (카드A 모델 기준) =====
            if pred_data:
                left_col, right_col = st.columns([1, 1])
                with left_col:
                    st.markdown("### 🎁 개인화 추천 제안")
                    r_data = reco_resp["data"] if reco_resp.get("ok") and reco_resp.get("data") else {}
                    if r_data.get("top_categories"):
                        st.write("**💡 추천 카테고리**")
                        st.dataframe(pd.DataFrame(r_data["top_categories"]), use_container_width=True)
                    if r_data.get("recommendations"):
                        st.write("**🛍️ 추천 상품 목록**")
                        st.dataframe(pd.DataFrame(r_data["recommendations"]), use_container_width=True)
                    if not r_data.get("top_categories") and not r_data.get("recommendations"):
                        st.info("추천 데이터가 존재하지 않습니다.")
                with right_col:
                    st.markdown("### 🎯 시스템 권장 리텐션 조치")
                    action_message = pred_data.get("recommended_action", "특별 조치 없음")
                    st.info(f"**권장 액션:** {action_message}")
                    if st.button("🔥 리텐션 액션 즉시 실행 (쿠폰/푸시 발송)", use_container_width=True):
                        with st.spinner("백엔드로 조치 결과 기록 중..."):
                            action_resp = rsvc.create_retention_action(
                                user_id=ref_id,
                                prediction_id=pred_data.get("prediction_id", 0),
                                action_type="discount_coupon" if "쿠폰" in str(action_message) else "remind_push",
                                message=action_message)
                        if action_resp["ok"]:
                            st.success("✅ 리텐션 로그가 백엔드 `retention_action_log`에 기록되었습니다!")
                        else:
                            render_error(action_resp)

    # ==========================================
    # 탭 2: 운영 — 모델 요약 / 고위험 고객
    # ==========================================
    with operation_tab:
        st.subheader("전체 비즈니스 운영 메트릭 및 고위험군 통합 관리")

        # 1. 대시보드 요약 정보 조회 (GET /dashboard/summary)
        summary_resp = dsvc.get_summary()
        if summary_resp["ok"]:
            s_data = summary_resp["data"]
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("📊 현재 운영 모델 (Active)", s_data.get("active_model", "N/A"))
            with sc2:
                st.metric("👥 전체 누적 예측 건수", f"{s_data.get('total_predictions', 0):,}")
            with sc3:
                st.metric("🚨 집중 케어 고위험 고객", f"{s_data.get('high_risk_count', 0):,}명")
            with sc4:
                st.metric("💰 회복 예상 매출액", f"₩{s_data.get('expected_revenue_recovery', 0):,}")
        else:
            render_error(summary_resp)

        st.divider()

        # 2. 고위험 고객 목록 테이블 (GET /predictions/top-risk)
        st.markdown("### 🛑 실시간 이탈 고위험 고객 Top 리스트")
        top_risk_resp = psvc.get_top_risk()
        if top_risk_resp["ok"]:
            if top_risk_resp["data"]:
                df_risk = pd.DataFrame(top_risk_resp["data"])
                # 계약서 가이드라인에 맞춰 가독성 필터링 및 확률 변환
                if "churn_probability" in df_risk.columns:
                    df_risk["churn_probability"] = (df_risk["churn_probability"] * 100).map("{:.1f}%".format)
                if "risk_level" in df_risk.columns:
                    df_risk["risk_level"] = df_risk["risk_level"].map(translate_risk_level)

                st.dataframe(df_risk, use_container_width=True)
            else:
                st.info("현재 위험군으로 분류된 고객이 없습니다.")
        else:
            render_error(top_risk_resp)

    # ==========================================
    # 탭 3: 모델 진단 — 차트 분석 (핵심 8개 우선 노출)
    # ==========================================
    with diagnostic_tab:
        st.subheader("MLOps 인공지능 모델 평가 및 검증 차트")

        # 1. 모델 목록 자동 조회 (GET /models/active)
        models_resp = csvc.get_active_models()
        model_options = []

        if models_resp["ok"] and models_resp["data"]:
            data = models_resp["data"]
            # 🚨 [버그 해결] 계약서 규격인 {"models": [...]} 구조를 안전하게 분해(Unwrap)합니다.
            rows = data.get("models", []) if isinstance(data, dict) else data

            for m in rows:
                if isinstance(m, dict):
                    # 🚨 [피드백 반영] 가독성을 위해 model_name을 최우선 노출하고, 없을 경우 model_id로 폴백합니다.
                    model_options.append(m.get("model_name") or m.get("model_id"))
                elif isinstance(m, str):
                    model_options.append(m)

        # 만약 가져온 모델 목록이 비어있다면 폴백 기본값 사용
        if not model_options:
            model_options = ["CatBoost_Churn_v2", "XGBoost_Baseline"]

        # 중복을 제거하고 한 번만 깔끔하게 선언합니다.
        selected_model = st.selectbox("진단 및 비교할 모델을 선택하세요", options=model_options, index=0)
        st.session_state.active_model_id = selected_model

        # [계약서 요구사항] 핵심 8개 및 전체 15개 후보 차트 멀티셀렉트 기본값 세팅
        chart_candidates = {
            "System Architecture": "system-architecture",
            "Cohort Retention": "cohort-retention",
            "Baseline Comparison": "baseline-comparison",
            "PR-AUC Curve": "pr-auc",
            "Threshold P/R/F1": "threshold",
            "Lift Chart": "lift",
            "Calibration Curve": "calibration",
            "Revenue Recovery": "revenue-recovery"
        }

        selected_charts = st.multiselect(
            "시각화할 분석 차트를 선택하세요 (계약서 지정 핵심 8개 기본 로드)",
            options=list(chart_candidates.keys()),
            default=list(chart_candidates.keys())
        )

        st.write(f"#### 📉 [Model: {selected_model}] 심층 진단 분석 대시보드")

        # 선택된 차트 루프 돌며 공통 렌더러 컴포넌트로 전달
        if selected_charts:
            for chart_label in selected_charts:
                chart_slug = chart_candidates[chart_label]

                # API 호출 분기 (공통 시스템 영역 vs 특정 모델 평가 영역)
                if chart_slug in ["system-architecture", "cohort-retention", "baseline-comparison"]:
                    # endpoint 구조: /dashboard/charts/{slug}
                    chart_resp = csvc.get_system_chart(chart_slug)
                else:
                    # endpoint 구조: /models/{modelId}/charts/{slug}
                    chart_resp = csvc.get_model_chart(selected_model, chart_slug)

                # [계약서 요구사항] 공통 Wrapper 구조 검증 후 렌더링
                if chart_resp["ok"] and chart_resp["data"]:
                    st.markdown(f"##### 📍 {chart_label}")
                    # components/charts.py에 준비된 공통 객체 지향 렌더러 활용
                    render_chart_payload(chart_resp["data"])
                    st.divider()
                else:
                    st.caption(f"⚠️ {chart_label} 데이터를 불러오지 못했거나 폴백을 수행합니다.")


if __name__ == "__main__":
    main()