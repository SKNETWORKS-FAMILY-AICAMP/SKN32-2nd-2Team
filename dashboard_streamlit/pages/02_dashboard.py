import streamlit as st

from components.layout import init_session_state, load_css, render_brand_header, render_sidebar_menu

def main() -> None:
    # 1. 초기 설정
    load_css("styles/main.css")
    init_session_state()
    render_sidebar_menu()

    # 2. 로그인 상태가 아닐 때의 처리 (핵심!)
    if not st.session_state.get("is_logged_in", False):
        st.error("로그인이 필요한 페이지입니다.")
        # 버튼을 눌러야만 이동하는 방식도 좋지만,
        # 즉시 이동이 필요하다면 아래와 같이 st.stop()과 함께 배치합니다.
        if st.button("로그인 페이지로 이동"):
            st.switch_page("pages/01_face_login.py")
        st.stop()  # 아래 코드가 실행되지 않도록 여기서 중단합니다.

    # 3. 로그인된 경우에만 실행될 대시보드 로직
    render_brand_header("Dashboard", f"{st.session_state.display_name} / {st.session_state.role}")

    user_tab, ops_tab, model_tab = st.tabs(["개인", "운영", "모델 진단"])
    with user_tab:
        st.metric("최신 이탈 확률", "API 연결 대기")
        st.info("`/dashboard/user/:userId`, `/predictions/latest`, `/recommendations/:userId` 연동 영역입니다.")

    with ops_tab:
        st.metric("고위험 고객", "API 연결 대기")
        st.info("`/dashboard/summary`, `/predictions/top-risk` 연동 영역입니다.")

    with model_tab:
        st.info("15개 chart-ready JSON을 받아 렌더링할 영역입니다.")

    if st.button("로그아웃"):
        st.session_state.is_logged_in = False
        st.session_state.user_id = ""
        st.session_state.display_name = ""
        st.rerun()  # 현재 페이지를 새로고침하여 로그인 체크 로직을 다시 태움


if __name__ == "__main__":
    main()
