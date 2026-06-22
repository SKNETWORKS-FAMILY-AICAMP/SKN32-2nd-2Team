import streamlit as st
import pandas as pd
from services import dashboard_service as dsvc
from services import prediction_service as psvc
from services import recommendation_service as rsvc
from services import chart_service as csvc
from components import charts, error_state, kpi_cards, risk_table
from components.layout import init_session_state, load_css, render_brand_header, render_sidebar_menu


def main() -> None:
    # 1. 초기 설정
    st.set_page_config(page_title="Anchor Dashboard", page_icon="A", layout="centered")
    load_css("styles/main.css")
    init_session_state()
    render_sidebar_menu()

    # 2. 로그인 상태가 아닐 때의 처리
    if not st.session_state.get("is_logged_in", False):
        st.error("로그인이 필요한 페이지입니다.")
        if st.button("로그인 페이지로 이동"):
            st.switch_page("pages/01_face_login.py")
        st.stop()  # 아래 코드가 실행되지 않도록 여기서 중단

    # 3. 로그인된 경우에만 실행될 대시보드 레이아웃 헤더
    display_name = st.session_state.get("display_name", st.session_state.get("user_id", "사용자"))
    role = st.session_state.get("role", "customer")
    render_brand_header("Dashboard", f"{display_name} / {role}")

    # 3개 메인 탭 생성
    user_tab, ops_tab, model_tab = st.tabs(["개인", "운영", "모델 진단"])

    # -------------------------------------------------------------
    # 3.1 [개인] 탭 — 유저 이탈 진단
    # -------------------------------------------------------------
    with user_tab:
        st.subheader("개인 — 고객 이탈 진단")
        uid = st.text_input("user_id", value=st.session_state.get("user_id", ""), key="user_uid")

        # 버튼을 누르거나 엔터를 쳤을 때 작동
        if uid.strip():
            uid = uid.strip()

            # 1) 최신 예측 데이터 가져오기 및 null 분기 처리
            lp = psvc.get_latest_prediction(uid)
            if not lp["ok"]:
                error_state.render_error(lp)
            else:
                pred = lp["data"].get("prediction")
                if not pred:
                    error_state.render_empty("해당 고객의 예측 데이터(Prediction)가 존재하지 않습니다.")
                else:
                    risk_kr = {"high": "고위험", "medium": "중위험", "low": "정상"}.get(pred["risk_level"], "-")

                    c1, c2, c3 = st.columns(3)
                    # 확률 단위 0~1 실수를 % 단위로 변환 (*100)
                    c1.metric("최신 이탈 확률", f"{pred['churn_probability'] * 100:.1f}%")
                    c2.metric("위험 등급", risk_kr)
                    c3.metric("예측 기준", f"{pred['horizon_days']}일")
                    st.caption(
                        f"💡 **권장 액션:** {pred.get('recommended_action', '-')}  ·  **생성일시:** {pred.get('created_at', '')}")

            # 2) 유저 대시보드 (관심 브랜드 / 분석 모델 정보)
            ud = dsvc.get_user_dashboard(uid)
            if ud["ok"]:
                lpv = ud["data"].get("latest_prediction", {})
                st.write(f"📊 **관심 브랜드:** {lpv.get('top_brand', '-')}  ·  **적용 모델:** {ud['data'].get('model', '-')}")

            # 3) 추천 카테고리 정보 출력
            rec = rsvc.get_recommendations(uid)
            if rec["ok"] and rec["data"].get("categories"):
                st.markdown("#### 🎯 개인화 추천 카테고리")
                st.dataframe(pd.DataFrame(rec["data"]["categories"]), use_container_width=True, hide_index=True)
            elif rec["ok"]:
                error_state.render_empty("추천할 카테고리 리스트가 비어있습니다.")

    # -------------------------------------------------------------
    # 3.2 [운영] 탭 — 요약 + 고위험 목록
    # -------------------------------------------------------------
    with ops_tab:
        st.subheader("운영 — 모델 요약 / 고위험 고객")

        # 1) 전체 모델 요약 통계 정보
        summ = dsvc.get_dashboard_summary()
        if not summ["ok"]:
            error_state.render_error(summ)
        else:
            d = summ["data"]
            c1, c2 = st.columns(2)
            c1.metric("Best 모델", d.get("best_model", "-"))
            c2.metric("Best ROC-AUC", f"{d.get('best_auc', 0):.4f}")
            st.caption(d.get("title", ""))

            if d.get("models"):
                st.markdown("##### 📈 모델 성능 비교 표준 데이터셋")
                st.dataframe(pd.DataFrame(d["models"]), use_container_width=True, hide_index=True)

        st.divider()

        # 2) 실시간 이탈 위험 고위험 탑 리스트 목록
        tr = psvc.get_top_risk()
        if not tr["ok"]:
            error_state.render_error(tr)
        else:
            users = tr["data"].get("users", [])
            st.metric("집계된 고위험 고객 수", tr["data"].get("count", len(users)))

            if users:
                # 명세서 권장 혹은 재사용 컴포넌트 활용 대안 적용
                # risk_table.render_risk_table(users) 함수가 정의되어 있다면 아래 한 줄로 대체 가능합니다.
                cols = ["user_id", "churn_probability", "risk_level", "recommended_action", "created_at"]
                filtered_users = [
                    {c: user.get(c) for c in cols if c in user} for user in users
                ]
                df_users = pd.DataFrame(filtered_users)
                # 확률 가시성 향상을 위해 컬럼 포맷 설정 가능
                st.dataframe(df_users, use_container_width=True, hide_index=True)
            else:
                error_state.render_empty("현재 탐지된 고위험 상태의 고객이 없습니다.")

    # -------------------------------------------------------------
    # 3.3 [모델 진단] 탭 — 차트 시각화
    # -------------------------------------------------------------
    with model_tab:
        st.subheader("모델 진단 — 차트 성능 분석")

        # 명세서 주의사항(빈틈 1): 서비스 함수 부재 가능성에 대비한 안전 분기 구현
        try:
            mn = dsvc.get_model_names()
            if mn["ok"] and mn["data"].get("models"):
                model_list = [m["model"] for m in mn["data"]["models"]]
                default_model = next((m["model"] for m in mn["data"]["models"] if m.get("is_best")), model_list[0])
            else:
                model_list = ["CatBoost", "XGBoost", "LightGBM"]
                default_model = "CatBoost"
        except AttributeError:
            # 명세서 가이드대로 아직 dashboard_service.py에 코드가 추가되지 않았을 때의 Fallback 예외 처리
            model_list = ["CatBoost", "XGBoost", "LightGBM"]
            default_model = "CatBoost"

        model = st.selectbox("진단 대상 모델 선택", model_list,
                             index=model_list.index(default_model) if default_model in model_list else 0)

        # 11종~15종 서포트 차트 멀티 셀렉트 구성
        pick = st.multiselect("출력할 진단 지표 차트 선택", csvc.MODEL_CHARTS,
                              default=["roc-auc", "pr-auc", "confusion-matrix"])

        for chart in pick:
            resp = csvc.get_model_chart(model, chart)
            if resp["ok"]:
                st.write(f"**📈 {chart.upper()} Metric Result**")
                charts.render_chart_payload(resp["data"])

                # [명세서 5번 참고] 표 기반 외에 실제 시각화 그래프 선/바 형태 렌더링이 필요할 때의 유연한 확장 코드
                df_chart = pd.DataFrame(resp["data"].get("data", []))
                x_axis = resp["data"].get("x")
                if not df_chart.empty and x_axis in df_chart.columns:
                    with st.expander(f"{chart} 시각화 그래프 보기"):
                        st.line_chart(df_chart.set_index(x_axis))
            else:
                error_state.render_error(resp, fallback=f"{chart} 차트 데이터를 불러올 수 없습니다.")

    # 로그아웃 버튼 영역
    if st.button("로그아웃", key="dashboard_logout"):
        st.session_state.is_logged_in = False
        st.session_state.user_id = ""
        st.session_state.display_name = ""
        st.session_state.role = "customer"
        st.rerun()


if __name__ == "__main__":
    main()