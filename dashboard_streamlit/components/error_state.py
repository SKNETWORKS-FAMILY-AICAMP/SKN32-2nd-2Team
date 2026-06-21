import streamlit as st


def render_error(response: dict, fallback: str = "요청을 처리하지 못했습니다.") -> None:
    error = response.get("error") or {}
    st.error(error.get("message") or fallback)


def render_empty(message: str = "표시할 데이터가 없습니다.") -> None:
    st.info(message)
