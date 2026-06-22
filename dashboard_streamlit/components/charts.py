import pandas as pd
import streamlit as st


def render_chart_payload(payload: dict) -> None:
    title = payload.get("title") or payload.get("chart_name", "chart")
    st.subheader(title)
    data = payload.get("data")
    chart_type = payload.get("chart_type")
    # 이미지/단일 객체 차트(예: system-architecture → {"asset_path": ...})는 행리스트가 아니라
    # dict 라서 pd.DataFrame 에 넣으면 'all scalar values' 오류. chart_type/형태로 분기한다.
    if chart_type == "image" or isinstance(data, dict):
        asset = data.get("asset_path") if isinstance(data, dict) else None
        if asset:
            import os
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # dashboard_streamlit/
            path = os.path.join(base, asset)
            if os.path.exists(path) and path.lower().endswith(".svg"):
                import base64
                with open(path, "rb") as fp:
                    b64 = base64.b64encode(fp.read()).decode()
                st.markdown(f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%"/>',
                            unsafe_allow_html=True)
            elif os.path.exists(path):
                st.image(path, use_container_width=True)
            else:
                st.caption(f"🖼️ 다이어그램 asset 없음: `{asset}`")
        elif data:
            st.json(data)
        else:
            st.info("차트 데이터가 없습니다.")
        return
    if not data:
        st.info("차트 데이터가 없습니다.")
        return

    df = pd.DataFrame(data)
    # 점이 너무 많으면 다운샘플(렌더 성능) — 곡선 모양은 보존
    if len(df) > 3000:
        df = df.iloc[:: max(1, len(df) // 1500)].reset_index(drop=True)

    x = payload.get("x")
    y = payload.get("y")
    ycols = [c for c in (y if isinstance(y, list) else [y]) if c and c in df.columns]

    try:
        if chart_type == "line" and x in df.columns and ycols:
            st.line_chart(df, x=x, y=ycols)
        elif chart_type in ("bar", "histogram") and x in df.columns and ycols:
            st.bar_chart(df, x=x, y=ycols)
        elif chart_type in ("bar", "histogram") and x in df.columns:
            st.bar_chart(df.set_index(x))
        else:
            # matrix/treemap/heatmap 등은 표로 표시(시각화 미지원 형태)
            st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)
