import streamlit as st

from components.layout import init_session_state, load_css, render_brand_header, render_sidebar_menu
from services.auth_service import check_user_id_available, login_face, register_face
from services.face_utils import detect_largest_face


def _render_face_preview(camera_file, mode: str):
    if camera_file is None:
        st.info("카메라로 얼굴을 정면에서 촬영해주세요.")
        return None

    image_bytes = camera_file.getvalue()
    result = detect_largest_face(image_bytes)
    if not result.detected:
        st.error("얼굴을 찾지 못했습니다. 밝은 곳에서 얼굴 전체가 보이게 다시 촬영해주세요.")
        return None

    st.image(result.preview_bytes, caption=f"{mode} 얼굴 검출 완료", use_container_width=True)
    st.success("OpenCV 얼굴 검출 성공. 백엔드에서 512d 임베딩, L2 정규화, 저장/비교를 진행합니다.")
    return result


def _apply_login_session(data: dict) -> None:
    st.session_state.is_logged_in = True
    st.session_state.user_id = data.get("user_id", "")
    st.session_state.display_name = data.get("display_name") or data.get("user_id", "")
    st.session_state.role = data.get("role") or "customer"
    st.session_state.access_token = data.get("access_token")


def render_register() -> None:
    st.subheader("얼굴 등록")
    user_id = st.text_input("ID", key="register_user_id", max_chars=64)
    display_name = st.text_input("이름", key="register_display_name", max_chars=100)
    role = st.selectbox("역할", ["customer", "admin"], index=0)

    if st.button("ID 중복 확인", use_container_width=True, disabled=not user_id.strip()):
        response = check_user_id_available(user_id.strip())
        if response["ok"]:
            data = response["data"]
            available = bool(data.get("available", not data.get("exists", True)))
            st.session_state.register_checked_user_id = user_id.strip()
            st.session_state.register_id_checked = available
            if st.session_state.register_id_checked:
                st.success("사용 가능한 ID입니다.")
            else:
                st.error("이미 등록된 ID입니다.")
        else:
            st.error(response["error"]["message"])

    if "register_id_checked" not in st.session_state:
        st.session_state.register_id_checked = False
    if st.session_state.get("register_checked_user_id") != user_id.strip():
        st.session_state.register_id_checked = False

    camera_file = st.camera_input("등록할 얼굴 촬영", key="register_camera")
    face = _render_face_preview(camera_file, "등록")

    disabled = not (user_id.strip() and display_name.strip() and st.session_state.register_id_checked and face)
    if st.button("얼굴 등록하기", type="primary", use_container_width=True, disabled=disabled):
        response = register_face(
            user_id=user_id.strip(),
            display_name=display_name.strip(),
            role=role,
            image_bytes=camera_file.getvalue(),
            face_bbox=face.bbox,
        )
        if response["ok"]:
            st.success("등록 완료. 이제 얼굴 로그인으로 진입할 수 있습니다.")
            st.session_state.register_id_checked = False
        else:
            st.error(response["error"]["message"])


def render_login() -> None:
    st.subheader("얼굴 로그인")
    camera_file = st.camera_input("로그인 얼굴 촬영", key="login_camera")
    face = _render_face_preview(camera_file, "로그인")

    if st.button("얼굴로 로그인", type="primary", use_container_width=True, disabled=not face):
        response = login_face(image_bytes=camera_file.getvalue(), face_bbox=face.bbox)
        if response["ok"]:
            _apply_login_session(response["data"])
            st.success(f"{st.session_state.display_name}님, 로그인되었습니다.")
            st.switch_page("pages/02_dashboard.py")
        else:
            st.error(response["error"]["message"])


def main() -> None:
    st.set_page_config(page_title="GAJIMA Face Login", page_icon="G", layout="centered")
    load_css("styles/main.css")
    init_session_state()
    render_sidebar_menu()
    render_brand_header("Face Login", "OpenCV detection + backend face embedding")

    register_tab, login_tab = st.tabs(["등록", "로그인"])
    with register_tab:
        render_register()
    with login_tab:
        render_login()


if __name__ == "__main__":
    main()
