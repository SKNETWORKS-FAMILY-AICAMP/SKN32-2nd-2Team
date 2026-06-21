from pathlib import Path

import streamlit as st


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_css(path) -> None:
    css_path = Path(path)
    if not css_path.is_absolute():
        css_path = _project_root() / css_path
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    defaults = {
        "is_logged_in": False,
        "user_id": "",
        "display_name": "",
        "role": "customer",
        "access_token": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_brand_header(title: str, subtitle: str) -> None:
    logo = (_project_root() / "assets" / "logo.svg").read_text(encoding="utf-8")
    st.markdown(
        f"""
        <section class="brand-shell">
            <div class="brand-logo">{logo}</div>
            <div>
                <h1>{title}</h1>
                <p>{subtitle}</p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar_menu() -> None:
    if st.session_state.get("is_logged_in", False):
        st.sidebar.markdown(f"**User ID:** `{st.session_state.get('user_id', '-')}`")
        st.sidebar.markdown(f"**Role:** `{st.session_state.get('role', '-')}`")
        st.sidebar.divider()

    st.sidebar.markdown("### [Menu]")
    st.sidebar.page_link("app.py", label="Home")
    st.sidebar.page_link("pages/01_face_login.py", label="face login")
    st.sidebar.page_link("pages/02_dashboard.py", label="dashboard")