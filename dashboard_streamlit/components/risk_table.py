import pandas as pd
import streamlit as st


def render_risk_table(rows: list[dict]) -> None:
    if not rows:
        st.info("고위험 고객 데이터가 없습니다.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
