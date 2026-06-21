import streamlit as st


def render_kpis(kpi: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("전체 고객", f"{kpi.get('n_users', 0):,}")
    cols[1].metric("고위험 고객", f"{kpi.get('n_high_risk', 0):,}")
    cols[2].metric("평균 이탈 확률", f"{kpi.get('avg_churn_probability', 0):.2%}")
    cols[3].metric("예상 회복 매출", f"{kpi.get('expected_recovery', 0):,}")
