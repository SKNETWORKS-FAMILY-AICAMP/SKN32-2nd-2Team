# -*- coding: utf-8 -*-
"""infrastructure/files — 쿠폰 타게팅 배치 산출물(coupon_targets.csv·coupon_summary.json) 읽기.
artifact-first: 파일 있으면 그대로, 없으면 빈 결과. (scripts/make_coupon_targets.py 가 생성)"""
import csv
import json
from app.config import DATA_DIR

CDIR = DATA_DIR / "coupons"   # DATA_DIR = .../data/processed (config), 산출물은 .../data/processed/coupons


def summary() -> dict:
    p = CDIR / "coupon_summary.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"cart2plus_customers": 0, "coupon_targets": 0, "grades": {}, "source": "none"}


def targets(grade: str = None, limit: int = 100) -> list:
    p = CDIR / "coupon_targets.csv"
    if not p.exists():
        return []
    out = []
    with open(p, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if grade and r.get("coupon_grade") != grade:
                continue
            out.append({"user_id": r.get("user_id"), "n_cart": r.get("n_cart"),
                        "churn_prob": round(float(r.get("churn_prob", 0)), 4),
                        "coupon_grade": r.get("coupon_grade"), "discount_pct": int(float(r.get("discount_pct", 0)))})
            if len(out) >= limit:
                break
    return out
