import pandas as pd
import streamlit as st


def render_chart_payload(payload: dict) -> None:
    title = payload.get("title") or payload.get("chart_name", "chart")
    data = payload.get("data") or []
    st.subheader(title)
    if not data:
        st.info("차트 데이터가 없습니다.")
        return
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
