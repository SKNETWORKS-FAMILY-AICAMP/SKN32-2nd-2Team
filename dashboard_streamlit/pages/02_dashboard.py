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

        # 기본값은 로그인한 사용자의 ID로 세팅
        default_user_id = st.session_state.get("user_id", "")
        target_user_id = st.text_input("조회할 유저 ID를 입력하세요", value=default_user_id).strip()

        if st.button("실시간 진단하기", type="primary", use_container_width=True) and target_user_id:
            with st.spinner("백엔드 분석 서버로부터 실시간 데이터 동기화 중..."):
                # 1. 최신 예측 데이터 조회 (GET /predictions/latest)
                pred_resp = psvc.get_latest_prediction(target_user_id)
                # 2. 유저 개인 프로필/대시보드 정보 조회 (GET /dashboard/user/:userId)
                user_resp = dsvc.get_user_dashboard(target_user_id)
                # 3. 추천 카테고리/상품 조회 (GET /recommendations/:userId)
                reco_resp = rsvc.get_recommendations(target_user_id)
                # 4. 세션 바운스 위험 조회 (GET /session-bounce/latest)
                bounce_resp = psvc.get_session_bounce(session_id="s-001")  # 임시 세션 ID 매핑

            # [계약서 요구사항] 응답 봉투(ok) 및 Null 예외 처리
            if pred_resp["ok"] and user_resp["ok"]:
                pred_data = pred_resp["data"]
                user_data = user_resp["data"]

                if not pred_data:
                    render_empty(f"유저 [{target_user_id}]의 예측 데이터가 존재하지 않습니다.")
                else:
                    # 상단 유저 정보 요약 스트립
                    st.markdown(
                        f"#### 👤 {user_data.get('user_name', '고객')}님 진단 리포트 (관심 브랜드: {user_data.get('favorite_brand', '미정')})")
                    st.caption(f"적용 인공지능 모델: **{pred_data.get('model_name')}**")

                    # 1줄 KPI 카드 배치
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # [계약서 요구사항] 확률 단위 변환 (* 100)
                        prob_pct = pred_data.get("churn_probability", 0.0) * 100
                        st.metric(label="이탈 확률 (Churn Prob)", value=f"{prob_pct:.1f}%")
                    with col2:
                        # [계약서 요구사항] 위험 등급 한글화
                        translated_risk = translate_risk_level(pred_data.get("risk_level", "low"))
                        st.metric(label="위험 등급", value=translated_risk)
                    with col3:
                        # 세션 바운스 신규 계약 요소 반영
                        if bounce_resp["ok"] and bounce_resp["data"]:
                            bounce_pct = bounce_resp["data"].get("bounce_probability", 0.0) * 100
                            st.metric(label="⚡ 실시간 세션 바운스 위험", value=f"{bounce_pct:.1f}%")
                        else:
                            st.metric(label="⚡ 실시간 세션 바운스 위험", value="데이터 없음")

                    st.divider()

                    # 추천 카드 및 리텐션 액션 연동
                    left_col, right_col = st.columns([1, 1])
                    with left_col:
                        st.markdown("### 🎁 개인화 추천 제안")
                        if reco_resp["ok"] and reco_resp["data"]:
                            r_data = reco_resp["data"]
                            # top_categories 테이블 표출
                            if "top_categories" in r_data:
                                st.write("**💡 추천 카테고리**")
                                st.dataframe(pd.DataFrame(r_data["top_categories"]), use_container_width=True)
                            # recommendations 상품 테이블 표출
                            if "recommendations" in r_data:
                                st.write("**🛍️ 추천 상품 목록**")
                                st.dataframe(pd.DataFrame(r_data["recommendations"]), use_container_width=True)
                        else:
                            st.info("추천 데이터가 존재하지 않습니다.")

                    with right_col:
                        st.markdown("### 🎯 시스템 권장 리텐션 조치")
                        action_message = pred_data.get("recommended_action", "특별 조치 없음")
                        st.info(f"**권장 액션:** {action_message}")

                        # [계약서 요구사항] 리텐션 액션 실행 버튼 (POST /retention-actions)
                        if st.button("🔥 리텐션 액션 즉시 실행 (쿠폰/푸시 발송)", use_container_width=True):
                            with st.spinner("백엔드로 조치 결과 기록 중..."):
                                action_resp = rsvc.create_retention_action(
                                    user_id=target_user_id,
                                    prediction_id=pred_data.get("prediction_id", 0),
                                    action_type="discount_coupon" if "쿠폰" in action_message else "remind_push",
                                    message=action_message
                                )
                            if action_resp["ok"]:
                                st.success("✅ 리텐션 로그가 백엔드 `retention_action_log`에 안전하게 기록되었습니다!")
                            else:
                                render_error(action_resp)
            else:
                # 에러 공통 컴포넌트 처리
                render_error(pred_resp if not pred_resp["ok"] else user_resp)

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
            # 데이터 타입이 딕셔너리인지 문자열인지 판별하여 안전하게 리스트 추출
            for m in models_resp["data"]:
                if isinstance(m, dict):
                    model_options.append(m.get("model_id"))
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