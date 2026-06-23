# -*- coding: utf-8 -*-
"""application — 시뮬 사이트 실시간 이탈예측 루프(26-9 P2, 19-2 §9.2).
유저 세션 행동(view/cart/remove/purchase) → v2 피처 집계 → 활성 모델 직접 추론 → 위험등급 →
리텐션/추천 push. 세션은 메모리(점수 권위), 이벤트는 sim_event_log·예측은 prediction_log에 영속.
피처는 응답에 그대로 노출(전처리 투명성)."""
from app.domain.risk_level import risk_level, retention_action
from app.domain import session_hazard
from app.config import MODELS_DIR, DATA_DIR
from app.infrastructure.files import catalog_store as cat
from app.infrastructure.model_inference import python_model_loader as loader
from app.infrastructure.mysql.session import (sim_event_repository, prediction_repository,
                                              model_repository)

FEATURE_ORDER = ["recency_days", "tenure_days", "ndays", "n_events", "n_view", "n_cart",
                 "n_remove_from_cart", "n_purchase", "avg_price", "purch_amt",
                 "min_price", "max_price", "std_price", "purchase_avg_price",
                 "remove_ratio", "cart_purchase_ratio", "n_categories", "cat_entropy",
                 "n_brands", "brand_loyalty", "n_sessions", "events_per_session"]   # v2 22피처 전체

# 프로필 베이스라인(recency_days, tenure_days, ndays, n_sessions). recency가 7일 이탈을 지배 → 프로필로 시드.
PROFILES = {
    "new":       {"recency_days": 0, "tenure_days": 1,   "ndays": 1,  "n_sessions": 1},
    "returning": {"recency_days": 1, "tenure_days": 30,  "ndays": 6,  "n_sessions": 6},
    "loyal":     {"recency_days": 0, "tenure_days": 180, "ndays": 42, "n_sessions": 42},
    "lapsing":   {"recency_days": 6, "tenure_days": 90,  "ndays": 9,  "n_sessions": 9},  # 식어가는 유저(고위험 베이스)
}

import math
import time
from collections import OrderedDict
_SESSIONS = OrderedDict()   # session_id -> {user_id, profile, events:[...], _ts}  (LRU + idle TTL)
_LAST_SIM = OrderedDict()   # user_id -> 최신 시뮬 세션 점수(대시보드 개인진단 카드B용, LRU 1000)
_ACTIVE_USER = {"user_id": None, "refresh_interval_sec": 4}   # 대시보드에서 설정한 '현재 진단 대상' + 시뮬 갱신주기
BOUNCE_WINDOW_MIN = 30      # 바운스 기준시간(이 행동 후 30분 무활동 = 이탈)
MAX_SESSIONS = 500          # 동시 세션 상한(초과 시 가장 오래된 것 evict)


def set_active_user(user_id, refresh_interval_sec=None):
    """대시보드가 현재 진단 대상 유저를 설정 → 시뮬 사이트가 동일 유저로 표시/동작."""
    _ACTIVE_USER["user_id"] = str(user_id) if user_id else None
    if refresh_interval_sec is not None:
        try:
            _ACTIVE_USER["refresh_interval_sec"] = max(1, min(60, int(refresh_interval_sec)))
        except Exception:
            pass
    return dict(_ACTIVE_USER)


def get_active_user():
    """현재 진단 대상 유저(없으면 user_id=None)."""
    return dict(_ACTIVE_USER)


# ── Churn Rate 정책(대시보드가 설정 → 서버가 적용 → 시뮬은 받은 값만 표시) ──────────
# mode: max(3종 최댓값) | ensemble(가중평균) | bounce_scaled(bounce 재척도 후 max) | select(1종 선택)
_CHURN_POLICY = {"mode": "max", "select_key": "hazard",
                 "bounce_floor": 0.3, "bounce_ceiling": 0.8,
                 "weights": {"churn_7d": 1.0, "hazard": 1.0, "bounce": 1.0}}
_POLICY_PATH = DATA_DIR / "realtime" / "churn_policy.json"


def _load_churn_policy():
    """재기동에도 정책 유지(파일 영속). 없으면 기본값."""
    try:
        import json as _json
        if _POLICY_PATH.exists():
            _CHURN_POLICY.update(_json.loads(_POLICY_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass


def set_churn_policy(**kw):
    for k in ("mode", "select_key", "bounce_floor", "bounce_ceiling", "weights"):
        if kw.get(k) is not None:
            _CHURN_POLICY[k] = kw[k]
    try:                                   # 파일 영속 → 백엔드 재기동에도 유지
        import json as _json
        _POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _POLICY_PATH.write_text(_json.dumps(_CHURN_POLICY, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass
    return dict(_CHURN_POLICY)


_load_churn_policy()


def get_churn_policy():
    return dict(_CHURN_POLICY)


def _bounce_rescale(b, floor, ceil):
    """bounce 높은 기저(예 0.3~0.8)를 0~1로 선형 재척도 → 덜 지배적으로."""
    if not isinstance(b, (int, float)) or ceil <= floor:
        return b
    return max(0.0, min(1.0, (b - floor) / (ceil - floor)))


def apply_policy(churn_7d, hazard, bounce, policy=None):
    """3종(7일·하자드·bounce) → 정책에 따른 단일 Churn Rate(0~1). 헤드라인·액션 공통 기준."""
    p = policy or _CHURN_POLICY
    vals = {"churn_7d": churn_7d, "hazard": hazard, "bounce": bounce}
    present = {k: float(v) for k, v in vals.items() if isinstance(v, (int, float))}
    if not present:
        return 0.0
    mode = p.get("mode", "max")
    if mode == "select":
        return present.get(p.get("select_key", "hazard"), max(present.values()))
    if mode == "ensemble":
        w = p.get("weights") or {}
        num = den = 0.0
        for k, v in present.items():
            wk = float(w.get(k, 1.0))
            num += wk * v
            den += wk
        return num / den if den else 0.0
    if mode == "bounce_scaled":
        vv = dict(present)
        if "bounce" in vv:
            vv["bounce"] = _bounce_rescale(vv["bounce"], p.get("bounce_floor", 0.3), p.get("bounce_ceiling", 0.8))
        return max(vv.values())
    return max(present.values())   # max(기본)


MAX_EVENTS = 300            # 세션당 이벤트 상한(최근 것만 유지)
SESSION_IDLE_SEC = 1800     # 세션 idle 타임아웃(30분 무활동 시 만료)


def sweep_sessions(idle_sec=SESSION_IDLE_SEC) -> int:
    """idle 초과 세션 제거(스케줄러가 주기 호출). 제거 수 반환."""
    now = time.monotonic()
    stale = [k for k, v in _SESSIONS.items() if now - v.get("_ts", now) > idle_sec]
    for k in stale:
        _SESSIONS.pop(k, None)
    return len(stale)


def _active_model():
    try:
        rows = model_repository.active()
        if rows:
            return str(rows[0]["model_name"]).replace("_Churn_v2", "").replace("_v2", "")
    except Exception:
        pass
    return "CatBoost"


def _session(session_id, user_id=None, profile="returning"):
    s = _SESSIONS.get(session_id)
    if not s:
        while len(_SESSIONS) >= MAX_SESSIONS:        # 가장 오래된 세션 evict(LRU)
            _SESSIONS.popitem(last=False)
        s = {"user_id": user_id or session_id, "profile": profile, "events": [], "_ts": time.monotonic()}
        _SESSIONS[session_id] = s
    else:
        _SESSIONS.move_to_end(session_id)            # 최근 사용 표시
    s["_ts"] = time.monotonic()                       # idle TTL 갱신
    return s


def reset(session_id):
    _SESSIONS.pop(session_id, None)
    return {"session_id": session_id, "reset": True}


def _entropy(items):
    """카테고리 분포 Shannon 엔트로피(자연로그)."""
    if not items:
        return 0.0
    from collections import Counter
    n = len(items)
    return float(-sum((c / n) * math.log(c / n) for c in Counter(items).values()))


def _max_share(items):
    """최빈 항목 점유율(브랜드 충성도 proxy)."""
    if not items:
        return 0.0
    from collections import Counter
    return max(Counter(items).values()) / len(items)


def _features(s):
    """세션 이벤트 → v2 22피처 집계(모델 기대치 전체). 누락=NaN 방지 위해 22개 모두 채운다."""
    ev = s["events"]
    n_view = sum(1 for e in ev if e["type"] == "view")
    n_cart = sum(1 for e in ev if e["type"] == "cart")
    n_remove = sum(1 for e in ev if e["type"] == "remove")
    n_purchase = sum(1 for e in ev if e["type"] == "purchase")
    prices = [float(e["price"]) for e in ev if e.get("price")]
    purch = [float(e["price"]) for e in ev if e["type"] == "purchase" and e.get("price")]
    cats = [e["category_id"] for e in ev if e.get("category_id")]
    brands = [e["brand"] for e in ev if e.get("brand")]

    base = dict(PROFILES.get(s["profile"], PROFILES["returning"]))
    if n_purchase:                                   # 구매=강한 인게이지먼트 → 방금 활동
        base["recency_days"] = 0

    n_events = len(ev)
    n_sessions = base.get("n_sessions", 1) or 1
    avg_price = sum(prices) / len(prices) if prices else 0.0
    std_price = (sum((p - avg_price) ** 2 for p in prices) / len(prices)) ** 0.5 if len(prices) > 1 else 0.0

    return {
        "recency_days": base["recency_days"], "tenure_days": base["tenure_days"], "ndays": base["ndays"],
        "n_events": n_events, "n_view": n_view, "n_cart": n_cart,
        "n_remove_from_cart": n_remove, "n_purchase": n_purchase,
        "avg_price": round(avg_price, 2), "purch_amt": round(sum(purch), 2),
        "min_price": round(min(prices), 2) if prices else 0.0,
        "max_price": round(max(prices), 2) if prices else 0.0,
        "std_price": round(std_price, 2),
        "purchase_avg_price": round(sum(purch) / len(purch), 2) if purch else 0.0,
        "remove_ratio": round(n_remove / n_cart, 4) if n_cart else 0.0,
        "cart_purchase_ratio": round(n_purchase / n_cart, 4) if n_cart else 0.0,
        "n_categories": len(set(cats)), "cat_entropy": round(_entropy(cats), 4),
        "n_brands": len(set(brands)), "brand_loyalty": round(_max_share(brands), 4),
        "n_sessions": n_sessions, "events_per_session": round(n_events / n_sessions, 4),
    }


def _top_category(s):
    cnt = {}
    for e in s["events"]:
        if e.get("category_id"):
            cnt[e["category_id"]] = cnt.get(e["category_id"], 0) + 1
    return max(cnt, key=cnt.get) if cnt else None


SNS_URL = "/sns"   # 시뮬 프론트 SNS 연동 페이지(클릭=view 이벤트 → 이탈률↓ 기대)
SNS_IDLE_SEC = 30  # 마지막 이벤트 후 무활동 이 초 이상이면 'SNS 둘러보기'(첫 접속 외). 활동 중엔 미노출.
CART_IDLE_SEC = 5  # 장바구니/조회 쿠폰 제안도 무활동 이 초 이상일 때만. view 발생(재참여) 시 idle 리셋→제안 사라짐.
ACTION_P = 0.5     # 이탈확률 이 이상이면 무활동(idle) 없이도 즉시 액션(대시보드 '다음 액션'과 동일 기준)


def coupon_grade(p):
    """이탈확률 → 쿠폰 등급/할인율 4단계: 80%↑→20%(긴급), 65-80%→15%(경고), 50-65%→10%(주의), else 5%(관심).
    ※ p 는 3종(7일·하자드·bounce) 중 최댓값(헤드라인 동일) — hazard만 쓰면 cart 세션이 늘 5%로 떨어지던 문제 해소."""
    if p >= 0.8:
        return 20, "긴급"
    if p >= 0.65:
        return 15, "경고"
    if p >= 0.5:
        return 10, "주의"
    return 5, "관심"   # 담고 미구매면 낮은 확률이어도 최소 nudge


def recommend_for(tc, k=4):
    """관심 카테고리(tc) → 추천 카테고리 1개 + 그 카테고리 상품 k개(브랜드·가격). 시뮬 푸시 표시용.
    반환 {category_id, category_name, items:[{product_id,name,brand,price}×k]} 또는 None."""
    if not tc:
        return None
    sims = cat.similar_categories(tc, k=3)
    target = sims[0] if sims else None
    cid = target["category_id"] if target else str(tc)
    cname = (target["display_name"] if target else None) or cat.name_of(tc) or f"category_{tc}"
    items = [{"product_id": str(pr.get("product_id")), "name": pr.get("name"),
              "brand": pr.get("brand"), "price": pr.get("price")}
             for pr in cat.products(limit=k, category_id=cid)]
    return {"category_id": str(cid), "category_name": cname, "items": items}


def decide_action(p, f, rec):
    """현재 이벤트/이탈률 → 이탈방지 액션(사용자 명세 3 시나리오). rec=추천(카테고리+상품4).
    장바구니 2+ & 미구매 = 쿠폰 타게팅 대상 → 이탈확률 등급별 할인."""
    n_cart, n_purchase, n_view = f["n_cart"], f["n_purchase"], f["n_view"]
    n_events = f["n_events"]
    idle_sec = f.get("idle_sec", 0)
    cname = (rec or {}).get("category_name")
    hot = float(p or 0.0) >= ACTION_P          # 고이탈 = idle 없이도 트리거(활발히 둘러봐도 위험하면 노출)
    # ① 담았는데 미구매 + (무활동 or 고이탈) → 이탈확률 등급별 쿠폰 할인 + 추천(카테고리+상품4)
    if n_cart > 0 and n_purchase == 0 and (idle_sec >= CART_IDLE_SEC or hot):
        pct, grade = coupon_grade(p)
        msg = f"장바구니 상품 {pct}% 할인 쿠폰({grade})"
        if cname:
            msg += f" + '{cname}' 추천상품"
        return {"action_type": "discount_related", "trigger": "cart_no_purchase",
                "message": msg + "을 추천합니다.",
                "payload": {"discount_pct": pct, "coupon_grade": grade,
                            "coupon_target": n_cart >= 2, "recommendation": rec}}
    # ② 첫 접속(이벤트 없음) 또는 마지막 이벤트 후 장시간 무활동(이탈 상승) → SNS 연동(클릭=view → 이탈률↓)
    #    ※ 둘러보는 중(활동 직후)에는 뜨지 않음 — idle 임계 초과 시에만.
    if n_cart == 0 and n_purchase == 0 and (n_events == 0 or idle_sec >= SNS_IDLE_SEC):
        first = n_events == 0
        return {"action_type": "sns_view", "trigger": "first_visit" if first else "idle_high",
                "message": ("환영합니다! SNS에서 인기 상품을 둘러보세요." if first
                            else "오랜만이에요! SNS에서 인기 상품을 둘러보세요."),
                "payload": {"sns_url": SNS_URL, "as_view_event": True, "recommendation": rec}}
    # ③ 조회만(view≥3, 미담음·미구매) + (무활동 or 고이탈) → 할인 + 추천(카테고리+상품4)
    if n_view >= 3 and n_cart == 0 and n_purchase == 0 and (idle_sec >= CART_IDLE_SEC or hot):
        msg = "지금 보는 카테고리 한정 할인! 5% 쿠폰을 드려요."
        if cname:
            msg = f"'{cname}' 한정 5% 할인 쿠폰을 드려요."
        return {"action_type": "discount", "trigger": "view_only",
                "message": msg, "payload": {"discount_pct": 5, "recommendation": rec}}
    # ④ 시나리오 미매칭이지만 고이탈(예: 활동 적은데 bounce↑) → 등급별 쿠폰 폴백(대시보드와 일관)
    if hot:
        pct, grade = coupon_grade(p)
        msg = f"이탈 위험이 높아요. {pct}% 할인 쿠폰({grade})으로 지금 혜택을 받아보세요."
        if cname:
            msg += f" + '{cname}' 추천상품"
        return {"action_type": "discount", "trigger": "high_churn",
                "message": msg, "payload": {"discount_pct": pct, "coupon_grade": grade,
                                            "recommendation": rec}}
    return {"action_type": "none", "trigger": "ok", "message": "", "payload": {}}


def action_from_events(p, events):
    """원시 이벤트(event_type/category_id) + 이탈확률(0~1) → 액션. /api/churn/predict 어댑터용."""
    def et(e): return e.get("event_type") or e.get("type")
    n_view = sum(1 for e in events if et(e) == "view")
    n_cart = sum(1 for e in events if et(e) == "cart")
    n_purchase = sum(1 for e in events if et(e) == "purchase")
    cats = [e.get("category_id") for e in events if e.get("category_id")]
    tc = max(set(cats), key=cats.count) if cats else None
    rec = recommend_for(tc, k=4)
    # 마지막 이벤트 이후 무활동(idle) 초 — SNS '오랜만이에요' 트리거(첫 접속 or 장시간 무활동)
    from datetime import datetime, timezone
    ts = [t for t in (_parse_ts(e.get("timestamp")) for e in events) if t is not None]
    idle_sec = (datetime.now(timezone.utc).timestamp() - max(ts)) if ts else 0.0
    f = {"n_cart": n_cart, "n_purchase": n_purchase, "n_view": n_view,
         "n_events": len(events), "recency_days": 0, "idle_sec": idle_sec}
    return decide_action(float(p or 0.0), f, rec)


def score_session(session_id, model=None):
    s = _SESSIONS.get(session_id)
    if not s or not s["events"]:
        return {"_status": 404, "error": "세션 이벤트 없음(먼저 행동을 기록하세요)"}
    model = model or _active_model()
    if not loader.available(model):
        return {"_status": 503, "error": f"모델 번들 로드 불가: {model}"}
    import pandas as pd
    feats = _features(s)
    X = pd.DataFrame([feats])[FEATURE_ORDER]
    probs = loader.score(model, X)
    if not probs:
        return {"_status": 500, "error": "추론 실패"}
    p = round(float(probs[0]), 4)
    r = risk_level(p)
    act = retention_action(p)
    # 추천: 세션 최다 관심 카테고리 → 유사 카테고리 + 상품4(고위험일 때 리텐션 push)
    rec = recommend_for(_top_category(s), k=4)
    action = decide_action(p, feats, rec)             # 이탈방지 액션(3 시나리오)
    # 영속: 예측 로그(top-risk/대시보드 반영)
    prediction_repository.log({"model_id": None, "user_id": str(s["user_id"]),
                               "churn_probability": p, "risk_level": r,
                               "recommended_action": action["message"] or act["action_message"]})
    return {"session_id": session_id, "user_id": s["user_id"], "model": model,
            "churn_probability": p, "risk_level": r, "horizon_days": 7,
            "recommended_action": act["action_message"],
            "action": action,
            "push_retention": (r == "high"),
            "recommendations": recs, "features": feats, "event_count": len(s["events"]),
            "source": "live-session-inference"}


def score_from_events(session_id, user_id, events):
    """배치 이벤트로 세션을 채워 채점(시뮬 사이트 /api/churn/predict 어댑터용). sim_event_log엔 기록 안 함."""
    s = _session(session_id, user_id)
    for e in (events or []):
        s["events"].append({"type": e.get("event_type"), "price": e.get("price"),
                            "category_id": e.get("category_id"), "brand": e.get("brand")})
    if len(s["events"]) > MAX_EVENTS:
        s["events"][:] = s["events"][-MAX_EVENTS:]
    return score_session(session_id)


# ── 실시간 세션 이탈값: 3단 폴백 체인 (B-모델 → B-데이터백업 → A-하자드) ──────────
# B 는 모델팀(모델 브런치) 산출물 파일이 있을 때만 동작(임의 구현 X). 없으면 A(하자드).
_SESSION_MODEL_PATH = MODELS_DIR / "churn" / "_realtime" / "session_model.joblib"
_SESSION_DATA_PATH = DATA_DIR / "realtime" / "session_predictions.json"


def _parse_ts(s):
    """ISO 문자열 → epoch sec. 실패 시 None."""
    if not s:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _session_feature_row(evs):
    """세션 이벤트 → 흔한 세션 피처 dict(모델 tier 가 bundle['feat']로 골라 씀)."""
    import math
    n = len(evs)
    ts = [e["ts"] for e in evs if e.get("ts") is not None]
    last_gap = (ts[-1] - ts[-2]) if len(ts) >= 2 else 0.0
    prices = [float(e["price"]) for e in evs if e.get("price")]
    def c(t): return sum(1 for e in evs if e["type"] == t)
    return {
        "step": n, "dt_prev_log": math.log1p(max(last_gap, 0.0)),
        "n_view": c("view"), "n_cart": c("cart"), "n_purchase": c("purchase"),
        "n_view_sf": c("view"), "n_cart_sf": c("cart"), "n_purchase_sf": c("purchase"),
        "price_log": math.log1p(prices[-1]) if prices else 0.0,
        "price_mean_sf_log": math.log1p(sum(prices) / len(prices)) if prices else 0.0,
        "is_first": 1 if n <= 1 else 0,
    }


def _b_model_tier(evs):
    """B-주: 모델팀 세션모델(models/churn/_realtime/session_model.joblib). 없으면 None.
    계약: {pipeline, feat:[...]}. 파일 있을 때만 추론(임의 구현 아님). 불일치/오류 시 None→다음 tier."""
    if not _SESSION_MODEL_PATH.exists():
        return None
    try:
        import joblib, pandas as pd
        b = joblib.load(_SESSION_MODEL_PATH)
        feat = b.get("feat") or []
        row = _session_feature_row(evs)
        X = pd.DataFrame([{k: row.get(k, 0.0) for k in feat}])
        p = float(b["pipeline"].predict_proba(X)[:, 1][0])
        return {"p": round(p, 4), "risk_level": session_hazard._level(p), "source": "model"}
    except Exception:
        return None


def _b_data_tier(evs):
    """B-백업: data/ 폴더 precomputed(data/realtime/session_predictions.json). 없으면 None."""
    if not _SESSION_DATA_PATH.exists():
        return None
    try:
        import json
        data = json.loads(_SESSION_DATA_PATH.read_text(encoding="utf-8"))
        # 계약(활성화 시): step→확률 룩업 등. 미정의면 None.
        key = str(len(evs))
        if isinstance(data, dict) and key in data:
            p = float(data[key])
            return {"p": round(p, 4), "risk_level": session_hazard._level(p), "source": "data"}
    except Exception:
        pass
    return None


def realtime_session_score(session_id, user_id, events):
    """시뮬 실시간 세션 이탈값. B-모델 → B-데이터백업 → A-하자드 순 폴백."""
    evs = []
    for e in (events or []):
        evs.append({"type": e.get("event_type"), "price": e.get("price"),
                    "category_id": e.get("category_id"), "brand": e.get("brand"),
                    "ts": _parse_ts(e.get("timestamp"))})
    evs.sort(key=lambda x: (x["ts"] is None, x["ts"] or 0))   # 시간순(ts 없으면 뒤로)
    r = _b_model_tier(evs) or _b_data_tier(evs)
    if r is None:
        from datetime import datetime, timezone
        r = session_hazard.session_risk(evs, now_ts=datetime.now(timezone.utc).timestamp())
    out = {"churn_probability": r["p"], "risk_level": r["risk_level"],
           "source": r["source"], "detail": r.get("detail")}
    if user_id:                                  # 유저별 최신 시뮬 점수 캐시(개인진단 카드B용)
        _LAST_SIM[str(user_id)] = out
        _LAST_SIM.move_to_end(str(user_id))
        while len(_LAST_SIM) > 1000:
            _LAST_SIM.popitem(last=False)
    return out


_BOUNCE_PATH = MODELS_DIR / "sequence" / "session_bounce_model.joblib"
_bounce_cache = {}


def _bounce_bundle():
    """bounce(churn30·30분) 세션 시퀀스 모델. {pipeline, feat, gap_sec}. 없으면 None(캐시)."""
    if "b" not in _bounce_cache:
        try:
            import joblib, warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _bounce_cache["b"] = joblib.load(_BOUNCE_PATH) if _BOUNCE_PATH.exists() else None
        except Exception:
            _bounce_cache["b"] = None
    return _bounce_cache["b"]


def _bounce_score(evs):
    """세션 이벤트 → bounce(다음 행동까지 30분↑ 이탈) 확률. 모델/오류 시 None."""
    b = _bounce_bundle()
    if not b:
        return None
    try:
        import pandas as pd
        feat = b.get("feat") or []
        row = _session_feature_row(evs)
        X = pd.DataFrame([{k: row.get(k, 0.0) for k in feat}])
        return round(float(b["pipeline"].predict_proba(X)[:, 1][0]), 4)
    except Exception:
        return None


def churn_three(session_id, user_id, events):
    """시뮬 표시용 3종 이탈값: ①7일 churn(v2 집계모델) ②하자드(세션 실시간) ③bounce(30분 이탈).
    헤드라인/액션은 하자드(realtime_session_score, 기존 동작) 기준."""
    rt = realtime_session_score(session_id, user_id, events)   # 하자드(또는 모델 tier) + 캐시
    evs = []
    for e in (events or []):
        evs.append({"type": e.get("event_type"), "price": e.get("price"),
                    "category_id": e.get("category_id"), "brand": e.get("brand"),
                    "ts": _parse_ts(e.get("timestamp"))})
    evs.sort(key=lambda x: (x["ts"] is None, x["ts"] or 0))
    p_bounce = _bounce_score(evs)
    p_7d = None                                                # 7일 v2 집계모델(22피처)
    model = _active_model()
    if loader.available(model):
        try:
            import pandas as pd
            s = {"events": [{"type": e["type"], "price": e["price"],
                             "category_id": e["category_id"], "brand": e["brand"]} for e in evs],
                 "profile": "returning"}
            sc = loader.score(model, pd.DataFrame([_features(s)])[FEATURE_ORDER])
            p_7d = round(float(sc[0]), 4) if sc else None
        except Exception:
            p_7d = None
    # 정책 적용 헤드라인 — 단일 소스(서버). 시뮬·대시보드 모두 이 값을 그대로 표시(로컬 재계산 X).
    churn_rate = round(apply_policy(p_7d, rt["churn_probability"], p_bounce), 4)
    pmode = _CHURN_POLICY.get("mode", "max")
    three = {"churn_7d": p_7d, "churn_hazard": rt["churn_probability"], "churn_bounce": p_bounce,
             "churn_rate": churn_rate, "policy_mode": pmode,
             "risk_level": rt["risk_level"], "source": rt["source"], "model": model}
    if user_id:                                       # 실시간 3종 + 정책값 캐시(대시보드가 시뮬과 동일값 표시)
        _LAST_SIM[str(user_id)] = {"churn_probability": rt["churn_probability"],
                                   "churn_rate": churn_rate, "policy_mode": pmode,
                                   "risk_level": rt["risk_level"], "source": rt["source"],
                                   "churn_7d": p_7d, "churn_hazard": rt["churn_probability"],
                                   "churn_bounce": p_bounce, "bounce_window_min": BOUNCE_WINDOW_MIN}
        _LAST_SIM.move_to_end(str(user_id))
        while len(_LAST_SIM) > 1000:
            _LAST_SIM.popitem(last=False)
    return three


def latest_score_by_user(user_id):
    """유저의 최신 시뮬 세션 점수.
    - 대시보드 선택(active) 유저는 '지금 시뮬에서 활동 중인 가장 최신 세션'을 본인 실시간으로 간주
      → 시뮬을 둘러보면 (어느 세션이든) 대시보드 실시간 값이 계속 움직인다.
    - 그 외 유저는 본인 캐시, 없으면 전역 최신(attributed=True)."""
    uid = str(user_id)
    if _LAST_SIM and uid == _ACTIVE_USER.get("user_id"):
        latest = next(reversed(_LAST_SIM.values()))          # 가장 최근 갱신된 세션
        return {**latest, "attributed": _LAST_SIM.get(uid) is not latest}
    own = _LAST_SIM.get(uid)
    if own:
        return {**own, "attributed": False}
    if _LAST_SIM:
        return {**next(reversed(_LAST_SIM.values())), "attributed": True}
    return None


def session_analytics(session_id):
    """세션 이벤트 요약(시뮬 사이트 /api/analytics 어댑터용)."""
    s = _SESSIONS.get(session_id)
    evs = s["events"] if s else []
    bd = {}
    for e in evs:
        bd[e["type"]] = bd.get(e["type"], 0) + 1
    return {"total_events": len(evs), "event_breakdown": bd,
            "products_viewed": bd.get("view", 0), "products_in_cart": bd.get("cart", 0),
            "purchases": bd.get("purchase", 0)}


def record_event(session_id, user_id, event_type, product_id=None, category_id=None,
                 brand=None, price=None, profile="returning"):
    s = _session(session_id, user_id, profile)
    if profile:
        s["profile"] = profile
    s["events"].append({"type": event_type, "price": price, "category_id": category_id, "brand": brand})
    if len(s["events"]) > MAX_EVENTS:                 # 세션당 이벤트 상한
        s["events"][:] = s["events"][-MAX_EVENTS:]
    scored = score_session(session_id)
    # 이벤트 영속(점수 스냅샷 동반)
    sim_event_repository.log({
        "user_id": str(s["user_id"]), "session_id": session_id, "event_type": event_type,
        "product_id": product_id, "category_id": category_id, "brand": brand, "price": price,
        "churn_prob": scored.get("churn_probability"), "risk_level": scored.get("risk_level")})
    return scored
