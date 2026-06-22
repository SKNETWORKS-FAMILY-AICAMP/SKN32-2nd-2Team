# -*- coding: utf-8 -*-
"""쿠폰 타게팅 배치 산출물 생성 — 장바구니 2+ 고객을 LightGBM v2로 채점 → 이탈확률 등급 → coupon_targets.csv + 시각화.

등급(쿠폰 기능 스펙): ≥80%→20%(긴급), 60-80%→10%(주의), 50-60%→5%(관심). <50%=비대상.
입력: data/processed/churn/train_tabular_v2.parquet(user_id·n_cart·22피처·churn) + models/preprocessors/prep_LightGBM_v2.joblib
산출: data/processed/coupons/coupon_targets.csv, coupon_summary.json, coupon_dist.png
실행: python scripts/make_coupon_targets.py
"""
import os, sys, json, warnings
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, joblib

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))   # SKN32-2nd_GAJIMA_Dev (models/·data/ 위치)
TAB = os.path.join(ROOT, "data", "processed", "churn", "train_tabular_v2.parquet")
BUNDLE = os.path.join(ROOT, "models", "preprocessors", "prep_LightGBM_v2.joblib")
OUT = os.path.join(ROOT, "data", "processed", "coupons"); os.makedirs(OUT, exist_ok=True)
MIN_CART = 2


def grade(p):
    if p >= 0.8: return "20% 할인(긴급)", 20
    if p >= 0.6: return "10% 할인(주의)", 10
    if p >= 0.5: return "5% 할인(관심)", 5
    return None, 0


def main():
    b = joblib.load(BUNDLE)
    feat = b["feature_order"]; cal = b.get("calibrator") or b.get("model")
    df = pd.read_parquet(TAB)
    cart2 = df[df["n_cart"] >= MIN_CART].copy()
    X = np.nan_to_num(cart2[feat].values.astype(float))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cart2["churn_prob"] = cal.predict_proba(pd.DataFrame(X, columns=feat))[:, 1]
    g = cart2["churn_prob"].apply(grade)
    cart2["coupon_grade"] = g.apply(lambda x: x[0]); cart2["discount_pct"] = g.apply(lambda x: x[1])
    targets = cart2[cart2["discount_pct"] > 0].copy()
    out = targets[["user_id", "n_cart", "churn_prob", "coupon_grade", "discount_pct"]].sort_values("churn_prob", ascending=False)
    out.to_csv(os.path.join(OUT, "coupon_targets.csv"), index=False, encoding="utf-8-sig")
    counts = out["coupon_grade"].value_counts().to_dict()
    summary = {"cart2plus_customers": int(len(cart2)), "coupon_targets": int(len(out)),
               "grades": {k: int(counts.get(k, 0)) for k in ["20% 할인(긴급)", "10% 할인(주의)", "5% 할인(관심)"]},
               "model": "LightGBM_v2", "min_cart": MIN_CART}
    json.dump(summary, open(os.path.join(OUT, "coupon_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[쿠폰] 2+장바구니", summary["cart2plus_customers"], "→ 대상", summary["coupon_targets"])
    print("[등급]", summary["grades"])
    # 시각화(이탈확률 분포 + 등급별 인원)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].hist(cart2["churn_prob"], bins=40, color="#4f8cff"); ax[0].axvline(0.5, color="r", ls="--")
        ax[0].set_title("2+장바구니 고객 이탈확률 분포"); ax[0].set_xlabel("churn prob")
        order = ["5% 할인(관심)", "10% 할인(주의)", "20% 할인(긴급)"]
        ax[1].bar(["관심(5%)", "주의(10%)", "긴급(20%)"], [counts.get(k, 0) for k in order],
                  color=["#7bd88f", "#ffd166", "#ff6b6b"])
        ax[1].set_title("등급별 쿠폰 대상자 수")
        for i, k in enumerate(order): ax[1].text(i, counts.get(k, 0), f"{counts.get(k,0):,}", ha="center", va="bottom")
        plt.tight_layout(); plt.savefig(os.path.join(OUT, "coupon_dist.png"), dpi=110); plt.close()
        print("[시각화] coupon_dist.png 저장")
    except Exception as e:
        print("[시각화 생략]", e)
    print("→", OUT)


if __name__ == "__main__":
    main()
