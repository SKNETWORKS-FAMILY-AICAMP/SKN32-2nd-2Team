from typing import Optional
import streamlit as st
import time

from components.layout import init_session_state, load_css, render_brand_header, render_sidebar_menu
from services.auth_service import check_user_id_available, login_face, register_face
from services.face_utils import detect_largest_face


def _render_face_preview(camera_file, mode: str, forced_score: Optional[float] = None, container=None):
    if camera_file is None:
        st.info("카메라로 얼굴을 정면에서 촬영해주세요.")
        return None

    image_bytes = camera_file.getvalue()
    score_to_use = forced_score if forced_score is not None else st.session_state.get("latest_login_score", None)

    # 🚨 [L2 해결] 의미 없이 세팅도 안 되던 'current_user_embeddings' 인자를 제거했습니다.
    # 이제 프론트엔드는 임베딩 매칭을 시도하지 않고, 검출 및 forced_score 매핑만 담당합니다.
    result = detect_largest_face(
        image_bytes,
        mode="login" if mode == "로그인" else "detection",
        forced_score=score_to_use
    )

    target = container if container else st
    if result.detected:
        # 컨테이너(st.empty)가 주어지면 그 자리에 이미지를 교체하고, 없으면 새로 만듭니다.
        target.image(result.preview_bytes, caption=f"{mode} 얼굴 검출 완료 (정확도 반영)", use_container_width=True)
        if mode == "로그인" and result.score is not None:
            st.success(f"🔥 인증 완료! 분석된 정확도: {result.score:.1%}")
    else:
        # OpenCV(미리보기 보조)가 못 잡아도 차단하지 않음 — 백엔드 insightface가 정밀 검출/인식 수행.
        target.image(image_bytes, caption=f"{mode} 촬영본 — 정밀 인식은 백엔드(insightface)가 수행", use_container_width=True)
        st.caption("ℹ️ 미리보기 검출(OpenCV)이 얼굴을 못 잡았어도, 아래 버튼을 누르면 백엔드 insightface가 정밀 인식합니다.")

    return result   # detected=False여도 결과 반환(버튼 차단하지 않음 — 백엔드가 최종 판정)


def _apply_login_session(data: dict) -> None:
    user_id = str(data.get("user_id") or "")
    stored_display_name = data.get("display_name") or user_id
    st.session_state.is_logged_in = True
    st.session_state.user_id = user_id
    # 로그인/대시보드 표시는 ID 기준으로 통일한다. display_name은 DB 등록명이므로 오입력될 수 있다.
    st.session_state.display_name = user_id
    st.session_state.profile_display_name = stored_display_name
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

    # 게이트는 ID/이름/중복확인/촬영 여부만 — OpenCV 검출 실패해도 백엔드 insightface가 최종 검출/등록.
    disabled = not (user_id.strip() and display_name.strip() and st.session_state.register_id_checked and camera_file)
    if st.button("얼굴 등록하기", type="primary", use_container_width=True, disabled=disabled):
        response = register_face(
            user_id=user_id.strip(),
            display_name=display_name.strip(),
            role=role,
            image_bytes=camera_file.getvalue(),
            face_bbox=face.bbox if (face and face.detected) else (0, 0, 0, 0),
        )
        if response["ok"]:
            st.success("등록 완료. 이제 얼굴 로그인으로 진입할 수 있습니다.")
            st.session_state.register_id_checked = False
        else:
            st.error(response["error"]["message"])


def render_login() -> None:
    st.subheader("얼굴 로그인")
    user_id = st.text_input("ID", key="login_user_id", max_chars=64)
    camera_file = st.camera_input("로그인 얼굴 촬영", key="login_camera")

    if "latest_login_score" not in st.session_state:
        st.session_state.latest_login_score = None

    # 💡 이미지가 그려질 전용 유연한 컨테이너(박스)를 먼저 예약합니다.
    image_container = st.empty()

    # 처음 카메라 촬영 시 컨테이너 안에 프리뷰 이미지를 집어넣습니다.
    face = _render_face_preview(camera_file, "로그인", container=image_container)

    # 게이트는 ID+촬영 여부만 — OpenCV 검출 실패해도 백엔드 insightface가 최종 인식한다.
    disabled = not (user_id.strip() and camera_file)
    if not user_id.strip():
        st.caption("등록한 ID를 입력한 뒤 얼굴을 촬영해주세요.")

    if st.button("얼굴로 로그인", type="primary", use_container_width=True, disabled=disabled):
        with st.spinner("얼굴 임베딩 분석 및 매칭 중..."):
            response = login_face(
                user_id=user_id.strip(),
                image_bytes=camera_file.getvalue(),
                face_bbox=face.bbox if (face and face.detected) else (0, 0, 0, 0),
            )

        if response["ok"]:
            # 🚨 [L1 해결] 눈속임용 하드코딩이었던 'or 0.954' 폴백을 완전히 제거했습니다.
            # 백엔드가 진짜 점수를 주지 않으면 가짜 데이터를 보여주지 않고 None 처리하여 정직하게 박스만 그립니다.
            score = response["data"].get("score") or response["data"].get("similarity")
            st.session_state.latest_login_score = score

            # 페이지를 넘기기 전, 예약해둔 이미지 컨테이너 공간의 이미지를 '진짜 정확도가 박힌 이미지'로 즉시 교체!
            _render_face_preview(camera_file, "로그인", forced_score=score, container=image_container)

            _apply_login_session(response["data"])
            st.success(f"🎉 {st.session_state.user_id} ID로 인증 완료! 3초 후 대시보드로 이동합니다.")

            time.sleep(3.0)
            st.switch_page("pages/02_dashboard.py")
        else:
            if response.get("data") and "score" in response["data"]:
                st.session_state.latest_login_score = response["data"]["score"]
            st.error(response["error"]["message"])


def main() -> None:
    st.set_page_config(page_title="GAJIMA Face Login", page_icon="G", layout="centered")
    load_css("styles/main.css")
    init_session_state()
    render_sidebar_menu()

    # 🚨 [아키텍처 정정] 프론트엔드가 더 이상 InsightFace를 직접 로드하지 않으므로 타이틀의 문구도 명세에 맞게 정정했습니다.
    render_brand_header("Face Login", "OpenCV 경량 얼굴 검출 + backend face embedding")

    register_tab, login_tab = st.tabs(["등록", "로그인"])
    with register_tab:
        render_register()
    with login_tab:
        render_login()


if __name__ == "__main__":
    main()
