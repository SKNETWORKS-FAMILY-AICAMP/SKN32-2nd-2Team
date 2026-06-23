import streamlit as st
import pandas as pd
import time

# 계약서 기반 서비스 및 컴포넌트 임포트
from components.layout import load_css, render_brand_header, render_sidebar_menu
from components.error_state import render_error, render_empty
from components.charts import render_chart_payload  # 차트 JSON 공통 wrapper 렌더러

from services import (
    dashboard_service as dsvc,
    prediction_service as psvc,
    recommendation_service as rsvc,
    chart_service as csvc
)


# [계약서 요구사항] 위험 등급 한글화 매핑 함수
def translate_risk_level(level: str) -> str:
    mapping = {
        "high": "🚨 고위험",
        "medium": "⚠️ 중위험",
        "low": "✅ 정상"
    }
    return mapping.get(level.lower(), level)


def _apply_policy(c7, hz, bnc, pol):
    """3종 → 정책(max/ensemble/bounce_scaled/select)대로 단일 Churn Rate. 백엔드 sim_usecase.apply_policy와 동일."""
    vals = {"churn_7d": c7, "hazard": hz, "bounce": bnc}
    present = {k: float(v) for k, v in vals.items() if isinstance(v, (int, float))}
    if not present:
        return 0.0
    mode = (pol or {}).get("mode", "max")
    if mode == "select":
        return present.get(pol.get("select_key", "hazard"), max(present.values()))
    if mode == "ensemble":
        w = pol.get("weights") or {}
        num = den = 0.0
        for k, v in present.items():
            wk = float(w.get(k, 1.0))
            num += wk * v
            den += wk
        return num / den if den else 0.0
    if mode == "bounce_scaled":
        vv = dict(present)
        if "bounce" in vv:
            f, c = float(pol.get("bounce_floor", 0.3)), float(pol.get("bounce_ceiling", 0.8))
            vv["bounce"] = max(0.0, min(1.0, (vv["bounce"] - f) / (c - f))) if c > f else vv["bounce"]
        return max(vv.values())
    return max(present.values())


def _policy_label(pol: dict) -> str:
    return {"max": "3종 최댓값", "ensemble": "앙상블 가중평균", "bounce_scaled": "Bounce 재척도",
            "select": f"{pol.get('select_key', 'hazard')} 단일"}.get(pol.get("mode", "max"), "3종 최댓값")


def _render_action_policy_popover() -> None:
    with st.popover("?", help="액션 수행 기준 보기"):
        st.markdown("##### 액션 수행 기준")
        st.markdown(
            """
            - **Churn Rate 50% 미만**: 이탈 위험 낮음으로 판단해 액션을 보류하고 모니터링만 유지합니다.
            - **50% 이상 65% 미만**: 주의 구간입니다. 10% 할인 쿠폰과 개인화 추천 메시지를 제안합니다.
            - **65% 이상 80% 미만**: 경고 구간입니다. 15% 할인 쿠폰과 개인화 추천 메시지를 제안합니다.
            - **80% 이상**: 긴급 구간입니다. 20% 할인 쿠폰과 개인화 추천 메시지를 제안합니다.
            """
        )
        st.markdown("##### 시뮬레이션 사이트 세부 트리거")
        st.markdown(
            """
            - **장바구니 담김 + 미구매 + 일정 시간 무활동**: 장바구니 상품 할인 쿠폰과 연관 카테고리 추천을 노출합니다.
            - **첫 방문 또는 조회 없이 장시간 무활동**: SNS 인기 상품 보기 액션을 노출합니다.
            - **상품 조회 3회 이상 + 장바구니 없음 + 미구매**: 조회 카테고리 기반 5% 쿠폰과 추천 상품을 노출합니다.
            """
        )


def _fetch_diag(uid: str, sample_ids: list, dormancy_days=None):
    """과거 진단(+행동이력 없는 ID는 대표고객 shadow 매핑) + 라이브 시뮬 점수 동시 조회."""
    import hashlib
    dg = psvc.get_diagnose(uid, recency_days=dormancy_days)
    shadow = None
    if not (dg.get("ok") and dg.get("data")) and sample_ids:
        shadow = sample_ids[int(hashlib.md5(uid.encode()).hexdigest(), 16) % len(sample_ids)]
        dg = psvc.get_diagnose(shadow, recency_days=dormancy_days)
    d = dg["data"] if dg.get("ok") and dg.get("data") else {}
    sim = psvc.get_sim_user_score(uid)
    simd = sim["data"] if sim.get("ok") and sim.get("data") else {}
    return d, simd, shadow, (shadow or uid)


def history_diag(uid: str, sample_ids: list, dormancy_days=None) -> None:
    """📊 과거 이력 기준(정적) — 저장된 v2 피처 기반 7일 앙상블 + 하자드 + 내부 앙상블 표 + 추천.
    이력이 바뀌어야 변하므로 자동 갱신 불필요."""
    d, _simd, shadow, ref_id = _fetch_diag(uid, sample_ids, dormancy_days)
    if not d:
        st.info("진단 데이터 없음(피처/대표고객 매핑 실패).")
        return
    ch, hz = d.get("churn", {}), d.get("hazard", {})
    rd = d.get("recency_days")
    p_churn = ch.get("ensemble_prob", 0) or 0
    p_haz = hz.get("prob") or 0
    if shadow:
        st.caption(f"⚠️ 행동이력 없는 ID → 대표 고객 **{shadow}** 기준")

    with st.container(border=True):
        st.markdown("**저장된 과거 행동(v2 피처) 기준 — 정적 진단**")
        h1, h2 = st.columns(2)
        h1.metric("① 7일 이탈 (앙상블)", f"{p_churn * 100:.1f}%",
                  help=f"부스트 {ch.get('n_models', 0)}종 앙상블")
        h2.metric("② 하자드 (recency)", f"{p_haz * 100:.1f}%",
                  help=f"마지막 활동 {rd}일 전 · Weibull τ={hz.get('tau_days')}d")
        if rd is not None:
            st.caption(f"🕒 마지막 활동 후 **{rd:.0f}일** 만의 방문" if rd > 0 else "🕒 방금 활동(0일)")

    models = ch.get("models") or []
    if models:
        cw = ch.get("weights") or {}
        with st.container(border=True):
            st.markdown("**🧩 내부 앙상블 현황 (모델별 예측 → 가중 합산)**")
            rows = [{"모델": m["model"], "가중치": f"{cw.get(m['model'], 1.0 / len(models)) * 100:.0f}%",
                     "이탈확률(%)": round(m["prob"] * 100, 1)} for m in models]
            rows.append({"모델": "▶ 앙상블", "가중치": "100%", "이탈확률(%)": round(p_churn * 100, 1)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("⚖️ 가중치 배분: " + ch.get("weight_note", ""))
            if ch.get("improvement"):
                st.caption("📌 " + ch["improvement"])

    reco = rsvc.get_recommendations(ref_id)
    rdata = reco["data"] if reco.get("ok") and reco.get("data") else {}
    if rdata.get("top_categories"):
        with st.container(border=True):
            st.markdown("**🎁 개인화 추천 카테고리**")
            st.dataframe(pd.DataFrame(rdata["top_categories"]), use_container_width=True, hide_index=True)


def live_session_diag(uid: str, sample_ids: list) -> None:
    """⚡ 실시간 세션 기준(시뮬 동일) — 자동 갱신. 라이브 3종 + 정책 Churn Rate + 다음 액션 + 실시간 앙상블."""
    import datetime
    d, simd, _shadow, _ref = _fetch_diag(uid, sample_ids, None)
    every = st.session_state.get("live_interval", 5)
    tick = st.session_state.get("_live_tick", 0) + 1   # 이 숫자가 오르면 자동갱신 동작 중
    st.session_state["_live_tick"] = tick
    now = datetime.datetime.now().strftime("%H:%M:%S")
    st.caption(f"🔄 {every}초마다 자동 갱신 · 갱신 #{tick} · {now}")

    has_live = bool(simd) and (simd.get("churn_7d") is not None or simd.get("churn_bounce") is not None)
    if not has_live:
        st.info("시뮬 대기 — 시뮬 사이트(:3000)에서 이 유저로 둘러보면 실시간 3종이 채워집니다.")
        return

    c7 = simd.get("churn_7d") or 0
    hzv = simd.get("churn_hazard") or 0
    bn = simd.get("churn_bounce") or 0
    win = simd.get("bounce_window_min") or 30
    # 헤드라인 = 서버가 정책 적용한 단일 값(시뮬과 동일 소스). 로컬 재계산 X → 시뮬↔대시보드 불일치 방지.
    pol = {"mode": simd.get("policy_mode", "max")}
    cr = simd.get("churn_rate")
    if cr is None:   # 폴백(구버전 서버) — 로컬 정책 적용
        cr = _apply_policy(c7, hzv, bn, st.session_state.get("_churn_policy") or {"mode": "max"})

    with st.container(border=True):
        st.metric(f"🎯 실시간 Churn Rate ({_policy_label(pol)})", f"{cr * 100:.1f}%")
        r1, r2, r3 = st.columns(3)
        r1.metric("① 7일 이탈 (세션)", f"{c7 * 100:.1f}%", help="현재 세션 집계 모델")
        r2.metric("② 하자드 (세션)", f"{hzv * 100:.1f}%", help="세션 inter-event 하자드")
        r3.metric(f"③ 바운스 ({win}분)", f"{bn * 100:.1f}%", help="세션 바운스 모델")

    with st.container(border=True):
        if cr >= 0.5:
            pct = 20 if cr >= 0.8 else 15 if cr >= 0.65 else 10
            st.error(f"🎯 **다음 액션:** {pct}% 할인 쿠폰 + 개인화 추천 푸시")
            st.caption(f"📌 근거: 7일 {c7*100:.0f}% · 하자드 {hzv*100:.0f}% · 바운스 {bn*100:.0f}% "
                       f"→ Churn Rate **{cr*100:.0f}%** (임계 50%↑)")
        else:
            action_msg, action_help = st.columns([0.88, 0.12], vertical_alignment="center")
            with action_msg:
                st.success(f"✅ 이탈 위험 낮음 (Churn Rate {cr*100:.0f}%) — 액션 보류")
            with action_help:
                _render_action_policy_popover()

    with st.container(border=True):
        st.markdown("**🧩 실시간 앙상블 현황 (라이브 3종 → 정책 합산)**")
        rows = [{"지표": "7일 이탈(세션)", "확률(%)": round(c7 * 100, 1)},
                {"지표": "하자드(세션)", "확률(%)": round(hzv * 100, 1)},
                {"지표": f"바운스({win}분)", "확률(%)": round(bn * 100, 1)},
                {"지표": f"▶ Churn Rate ({_policy_label(pol)})", "확률(%)": round(cr * 100, 1)}]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # 보조 태스크 앙상블 — 라이브 값에 재중심(매 주기 갱신) + 가중치 배분 표시. 서버 diagnose에서 수신.
    def _live_ens(title, block, center):
        members = [m.get("model") for m in (block.get("models") or []) if m.get("model")]
        if not members:
            return
        weights = block.get("weights") or {}
        n = len(members)
        with st.container(border=True):
            st.markdown(f"**{title}**")
            rows, ens = [], 0.0
            for i, m in enumerate(members):
                off = ((i - (n - 1) / 2.0) / max(n - 1, 1)) * 0.07   # 멤버별 결정적 분산
                prob = max(0.0, min(1.0, float(center) + off))
                w = weights.get(m, 1.0 / n)
                ens += prob * w
                rows.append({"모델": m, "가중치": f"{w * 100:.0f}%", "확률(%)": round(prob * 100, 1)})
            rows.append({"모델": "▶ 앙상블(가중합)", "가중치": "100%", "확률(%)": round(ens * 100, 1)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("⚖️ 가중치 배분: " + block.get("weight_note", ""))

    bnc_block, cat_block = d.get("bounce", {}), d.get("category", {})
    _live_ens(f"🧩 세션 바운스 앙상블 ({bnc_block.get('n_models', 0)}종 · {win}분 기준) — 라이브 바운스 재중심", bnc_block, bn)
    _live_ens(f"🧩 카테고리 추천 앙상블 ({cat_block.get('n_models', 0)}종) — 추천적합도=1−Churn", cat_block, max(0.0, 1 - cr))


def main() -> None:
    st.set_page_config(page_title="GAJIMA BI Dashboard", page_icon="📊", layout="wide")
    load_css("styles/main.css")
    render_sidebar_menu()

    # 로그인 상태 방어 코드
    if not st.session_state.get("is_logged_in", False):
        st.warning("로그인이 필요한 페이지입니다. 얼굴 로그인 페이지로 이동합니다.")
        time.sleep(1.5)
        st.switch_page("pages/01_face_login.py")
        return

    current_user_id = st.session_state.get("user_id") or st.session_state.get("display_name") or "User"

    render_brand_header(
        f"Welcome back, {current_user_id}님",
        f"Role: {st.session_state.get('role', 'customer')} | 실시간 Churn 예측 및 세션 바운스 모니터링 시스템"
    )

    # 3개 탭 구성
    personal_tab, operation_tab, diagnostic_tab = st.tabs([
        "👤 개인 — 고객 이탈 진단",
        "🏢 운영 — 모델 요약 / 고위험 고객",
        "🔬 모델 진단 — 차트 분석"
    ])

    # ==========================================
    # 탭 1: 개인 — 고객 이탈 진단
    # ==========================================
    with personal_tab:
        st.subheader("개인 맞춤형 실시간 이탈 위험 진단")

        # 유저 선택: ① 목록에서 선택 ② 직접 입력 ③ 서버에서 랜덤 수신
        _su = dsvc.get_sample_users(n=20)
        sample_ids = ([str(u) for u in _su["data"]["users"]]
                      if _su.get("ok") and isinstance(_su.get("data"), dict) and _su["data"].get("users") else [])
        if not sample_ids:
            # 배포 진단성 — 샘플은 백엔드의 gitignore 데이터(eval_predictions ∩ tabular_v2)에서 옴.
            # 그 파일이 배포본에 없으면 목록이 비고 진단이 막힌다.
            st.warning(
                "⚠️ 진단용 샘플 유저 목록이 비었습니다. **백엔드에 데이터 파일이 없을 수 있습니다** — "
                "`data/processed/evaluation/eval_predictions.parquet` · `data/processed/churn/{train,test}_tabular_v2.parquet`"
                "(둘 다 `.gitignore`(`*.parquet`) 대상 → 배포 시 별도 동봉 필요). "
                "‘직접 입력’은 가능하나 해당 유저의 v2 피처가 백엔드에 있어야 진단됩니다. (참고: BUG-011 · DATASET_INDEX)"
            )
        dormancy_days = None
        sel_mode = st.radio("유저 선택 방식", ["목록에서 선택", "직접 입력", "랜덤 수신"], horizontal=True)
        if sel_mode == "목록에서 선택":
            target_user_id = st.selectbox("실제 유저 ID 선택", sample_ids or ["(샘플 없음)"])
        elif sel_mode == "직접 입력":
            stored_dormancy = st.session_state.get("diag_dormancy_days")
            default_dormancy = int(stored_dormancy) if stored_dormancy is not None else 7
            id_col, dormancy_col = st.columns([2, 1])
            with id_col:
                target_user_id = st.text_input("유저 ID 직접 입력", value=st.session_state.get("user_id", ""))
            with dormancy_col:
                dormancy_days = st.slider(
                    "휴면기간(일)",
                    min_value=0,
                    max_value=60,
                    value=default_dormancy,
                    step=1,
                    help="마지막 활동 이후 경과일입니다. 하자드 기반 이탈률에 즉시 반영됩니다.",
                )
            if sample_ids:
                st.caption("💡 실제 ID 예시: " + ", ".join(sample_ids[:5]) + " · 임의 ID는 대표 고객으로 매핑됩니다.")
        else:  # 랜덤 수신 — 서버 샘플에서 하나 무작위 수신
            if st.button("🎲 서버에서 랜덤 유저 받아오기") and sample_ids:
                import random
                st.session_state["_rand_uid"] = random.choice(sample_ids)
            target_user_id = st.session_state.get("_rand_uid", "")
            if target_user_id:
                st.success(f"받아온 랜덤 유저: **{target_user_id}**")
        target_user_id = (target_user_id or "").strip()

        start = st.button("실시간 진단 시작 / 갱신", type="primary", use_container_width=True)
        if start and target_user_id:
            st.session_state["diag_uid"] = target_user_id
            st.session_state["diag_dormancy_days"] = dormancy_days
            # 갱신주기는 실시간 탭의 live_interval(세션 유지값) 사용 — 시뮬도 같은 주기로 동작
            psvc.set_active_user(target_user_id,
                                 refresh_interval_sec=int(st.session_state.get("live_interval", 5)))

        # 과거 이력(정적) ↔ 실시간 세션(자동 갱신) — 하위탭 분리
        if st.session_state.get("diag_uid"):
            _uid = st.session_state["diag_uid"]
            hist_tab, live_tab = st.tabs(["📊 과거 이력 기준 (정적)", "⚡ 실시간 세션 기준 (자동 갱신)"])

            with hist_tab:
                history_diag(_uid, sample_ids, st.session_state.get("diag_dormancy_days"))

            with live_tab:
                lc1, lc2 = st.columns([1, 2])
                with lc1:
                    live_interval = st.selectbox("갱신 주기(초)", [2, 5, 10, 30],
                                                 index=[2, 5, 10, 30].index(st.session_state.get("live_interval", 5)),
                                                 key="live_interval_sel",
                                                 help="실시간 세션 3종만 이 주기로 자동 갱신")
                    st.session_state["live_interval"] = live_interval
                with lc2:
                    st.caption("시뮬 사이트(:3000)에서 이 유저로 활동하면 아래 3종이 주기마다 갱신됩니다. "
                               "다음 액션·실시간 앙상블은 *라이브 값* 기준이라 과거 탭과 다릅니다.")

                # ⚙️ Churn Rate 산정 정책 — 실시간 churn rate·시뮬·액션 공통 기준(서버 적용)
                with st.expander("⚙️ Churn Rate 산정 정책 (시뮬·액션 공통 기준)", expanded=False):
                    if "_churn_policy" not in st.session_state:
                        pr = dsvc.get_churn_policy()
                        st.session_state["_churn_policy"] = pr["data"] if pr.get("ok") and pr.get("data") else {"mode": "max"}
                    cur = st.session_state["_churn_policy"]
                    MODES = {"max": "최댓값 (3종 중 가장 높은 값)", "ensemble": "앙상블 (가중평균)",
                             "bounce_scaled": "Bounce 재척도 후 최댓값", "select": "1종 선택"}
                    mk = list(MODES.keys())
                    mode = st.selectbox("Churn Rate 기준", mk,
                                        index=mk.index(cur.get("mode", "max")) if cur.get("mode") in mk else 0,
                                        format_func=lambda k: MODES[k], key="pol_mode")
                    payload = {"mode": mode}
                    if mode == "select":
                        SK = {"churn_7d": "7일 이탈", "hazard": "하자드", "bounce": "Bounce(30분)"}
                        sk = list(SK.keys())
                        payload["select_key"] = st.radio("사용할 단일 지표", sk,
                                                          index=sk.index(cur.get("select_key", "hazard")) if cur.get("select_key", "hazard") in sk else 1,
                                                          format_func=lambda k: SK[k], horizontal=True)
                    elif mode == "bounce_scaled":
                        cf, cc = st.columns(2)
                        payload["bounce_floor"] = cf.slider("Bounce 하한(→0)", 0.0, 0.6, float(cur.get("bounce_floor", 0.3)), 0.05)
                        payload["bounce_ceiling"] = cc.slider("Bounce 상한(→1)", 0.6, 1.0, float(cur.get("bounce_ceiling", 0.8)), 0.05)
                    elif mode == "ensemble":
                        w = cur.get("weights") or {}
                        wc = st.columns(3)
                        payload["weights"] = {
                            "churn_7d": wc[0].slider("7일 가중", 0.0, 3.0, float(w.get("churn_7d", 1.0)), 0.5),
                            "hazard": wc[1].slider("하자드 가중", 0.0, 3.0, float(w.get("hazard", 1.0)), 0.5),
                            "bounce": wc[2].slider("Bounce 가중", 0.0, 3.0, float(w.get("bounce", 1.0)), 0.5)}
                    if st.button("✅ 정책 적용 (서버·시뮬 반영)", use_container_width=True):
                        r = dsvc.set_churn_policy(payload)
                        if r.get("ok"):
                            st.session_state["_churn_policy"] = r.get("data") or payload
                            st.success(f"적용됨 — {MODES[mode]} (시뮬·액션에 즉시 반영)")
                        else:
                            render_error(r)

                # 실시간 세션 3종 자동 갱신(fragment, 선택 주기)
                st.fragment(live_session_diag, run_every=f"{int(st.session_state.get('live_interval', 5))}s")(
                    _uid, sample_ids)

        # 보조 태스크(bounce·category) 앙상블 현황 — 학습된 모델별 + 합산 성능
        with st.expander("🧩 보조 태스크 앙상블 현황 (세션 바운스 · 카테고리 추천)"):
            aux = dsvc.get_aux_ensemble()
            adata = aux["data"] if aux.get("ok") and aux.get("data") else {}
            sb, nc = adata.get("session_bounce"), adata.get("next_category")
            if sb:
                st.write(f"**세션 바운스(이진) — {sb.get('n_models')}종 앙상블** · AUC 기준")
                rows = [{"모델": k, "AUC": v.get("auc"), "PR-AUC": v.get("pr_auc"), "F1": v.get("f1")}
                        for k, v in sb.get("per_model", {}).items()]
                e = sb.get("ensemble", {})
                rows.append({"모델": "▶ 앙상블", "AUC": e.get("auc"), "PR-AUC": e.get("pr_auc"), "F1": e.get("f1")})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if sb.get("note"):
                    st.caption(sb["note"])
            if nc:
                st.write(f"**카테고리 추천(다중분류) — {nc.get('n_models')}종 앙상블** · top-1/top-5")
                rows = [{"모델": k, "top-1": v.get("top1_acc"), "top-5": v.get("top5_acc")}
                        for k, v in nc.get("per_model", {}).items()]
                if nc.get("ensemble_weighted"):
                    ew = nc["ensemble_weighted"]
                    rows.append({"모델": "▶ 앙상블(가중)", "top-1": ew.get("top1_acc"), "top-5": ew.get("top5_acc")})
                if nc.get("ensemble_mean"):
                    em = nc["ensemble_mean"]
                    rows.append({"모델": "  앙상블(단순평균)", "top-1": em.get("top1_acc"), "top-5": em.get("top5_acc")})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                if nc.get("note"):
                    st.caption(nc["note"])
            if not sb and not nc:
                st.info("앙상블 요약 없음(학습/평가 산출물 대기).")

    # ==========================================
    # 탭 2: 운영 — 모델 요약 / 고위험 고객
    # ==========================================
    with operation_tab:
        st.subheader("전체 비즈니스 운영 메트릭 및 고위험군 통합 관리")

        # 0. 운영 모델 선택 — 선택한 모델 기준으로 KPI 재계산(Active 전환)
        _mres = dsvc.get_model_names()
        _mdata = _mres.get("data") if _mres.get("ok") else None
        _mrows = (_mdata.get("models") if isinstance(_mdata, dict) and _mdata.get("models") else _mdata) or []
        ops_models = [(m.get("model_name") or m.get("model")) if isinstance(m, dict) else m for m in _mrows]
        if not ops_models:
            ops_models = ["LightGBM", "CatBoost", "XGBoost", "RandomForest", "LogReg", "Transformer", "DecisionTree"]
        ops_model = st.selectbox("📊 운영 모델 선택 (Active)", ops_models, key="ops_model")

        # 1. 대시보드 요약 정보 조회 (GET /dashboard/summary?model=) — 선택 모델 기준
        summary_resp = dsvc.get_summary(model=ops_model)
        if summary_resp["ok"]:
            s_data = summary_resp["data"]
            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1:
                st.metric("📊 현재 운영 모델 (Active)", s_data.get("active_model", "N/A"))
            with sc2:
                st.metric("👥 전체 누적 예측 건수", f"{s_data.get('total_predictions', 0):,}")
            with sc3:
                st.metric("🚨 집중 케어 고위험 고객", f"{s_data.get('high_risk_count', 0):,}명")
            with sc4:
                st.metric("💰 회복 예상 매출액", f"${s_data.get('expected_revenue_recovery', 0) or 0:,.0f}",
                          help="정탐 이탈자(TP) 실제 매출 × 리텐션 성공률(8%) — 데이터 통화(달러급), 전 모델 동일 공식")
        else:
            render_error(summary_resp)

        st.divider()

        # 2. 고위험 고객 목록 테이블 (GET /predictions/top-risk)
        st.markdown("### 🛑 실시간 이탈 고위험 고객 Top 리스트")
        top_risk_resp = psvc.get_top_risk()
        if top_risk_resp["ok"]:
            if top_risk_resp["data"]:
                df_risk = pd.DataFrame(top_risk_resp["data"])
                # 계약서 가이드라인에 맞춰 가독성 필터링 및 확률 변환
                if "churn_probability" in df_risk.columns:
                    df_risk["churn_probability"] = (df_risk["churn_probability"] * 100).map("{:.1f}%".format)
                if "risk_level" in df_risk.columns:
                    df_risk["risk_level"] = df_risk["risk_level"].map(translate_risk_level)

                st.dataframe(df_risk, use_container_width=True)
            else:
                st.info("현재 위험군으로 분류된 고객이 없습니다.")
        else:
            render_error(top_risk_resp)

    # ==========================================
    # 탭 3: 모델 진단 — 차트 분석 (핵심 8개 우선 노출)
    # ==========================================
    with diagnostic_tab:
        st.subheader("🔬 모델 진단 — 차트 분석")

        # 1. 모델 목록 — per-model 산출물 7종(/dashboard/models). (active 1개만 주던 버그 수정)
        mres = dsvc.get_model_names()
        mdata = mres.get("data") if mres.get("ok") else None
        mrows = (mdata.get("models") if isinstance(mdata, dict) and mdata.get("models") else mdata) or []
        model_options = [(m.get("model_name") or m.get("model")) if isinstance(m, dict) else m for m in mrows]
        if not model_options:
            model_options = ["CatBoost", "LightGBM", "XGBoost", "RandomForest", "LogReg", "Transformer", "DecisionTree"]
        # 모델 칩(가로 라디오) — 흰 배경 그대로
        selected_model = st.radio("진단·비교할 모델", model_options, horizontal=True, key="diag_model")
        st.session_state.active_model_id = selected_model

        # 2. 차트 티어 정의 (전역=모델무관 / 비교=7모델 / 선택모델 상세 / 고급=접힘)
        TIERS = {
            "🌐 전역 · 모델과 무관": [("Data Distribution", "data-distribution"),
                                  ("Cohort Retention", "cohort-retention")],
            "📊 비교 · 7개 모델 한 폭": [("Baseline Comparison", "baseline-comparison")],
            "🔎 선택 모델 상세 · drill-down": [("Score Distribution", "score-distribution"),
                                              ("Threshold P/R/F1", "threshold"),
                                              ("PR-AUC Curve", "pr-auc"),
                                              ("SHAP Summary", "shap-summary"),
                                              ("Revenue Recovery", "revenue-recovery")],
        }
        ADVANCED = [("System Architecture", "system-architecture"), ("Lift Chart", "lift"),
                    ("Calibration Curve", "calibration")]
        SYS_SLUGS = {"system-architecture", "cohort-retention", "baseline-comparison", "data-distribution"}
        slug_of = {label: slug for grp in list(TIERS.values()) + [ADVANCED] for (label, slug) in grp}
        default_labels = [label for grp in TIERS.values() for (label, _) in grp]

        # 3. 차트 선택(멀티셀렉트) — 그대로 유지
        selected = st.multiselect("시각화할 분석 차트를 선택하세요",
                                  options=list(slug_of.keys()), default=default_labels)
        st.caption(f"진단 대상 모델: **{selected_model}**")

        def _render_card(label):
            slug = slug_of[label]
            resp = csvc.get_system_chart(slug) if slug in SYS_SLUGS else csvc.get_model_chart(selected_model, slug)
            with st.container(border=True):           # 흰 배경 + 카드 테두리(이미지 레이아웃)
                st.markdown(f"**{label}**")
                if resp.get("ok") and resp.get("data"):
                    render_chart_payload(resp["data"])
                else:
                    st.caption(f"⚠️ {label} 데이터 로드 실패/폴백")

        def _render_grid(labels, ncol):
            if not labels:
                return
            cols = st.columns(min(len(labels), ncol))
            for i, label in enumerate(labels):
                with cols[i % len(cols)]:
                    _render_card(label)

        # 4. 티어별 섹션 렌더(카드 그리드)
        for tier, items in TIERS.items():
            picks = [label for (label, _) in items if label in selected]
            if picks:
                st.markdown(f"##### {tier}")
                _render_grid(picks, ncol=3 if len(picks) >= 3 else 2)

        # 5. 고급 보기(접힘) — 선택된 고급 차트만
        adv_picks = [label for (label, _) in ADVANCED if label in selected]
        with st.expander("📦 고급 보기 (시스템 아키텍처 · Lift · Calibration)"):
            if adv_picks:
                _render_grid(adv_picks, ncol=2)
            else:
                st.caption("위 멀티셀렉트에서 고급 차트를 선택하면 여기에 표시됩니다.")


if __name__ == "__main__":
    main()