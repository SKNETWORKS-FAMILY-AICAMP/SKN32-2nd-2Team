# -*- coding: utf-8 -*-
"""infrastructure/files — 시뮬 사이트용 REES46 카탈로그 reader(26-9 P2).
simulation_site/neon/seed/*.csv 에서 상품·카테고리·유사도를 읽어 chart/UI-ready로 제공.
Neon 미사용 로컬 데모: 파일 직접. (Neon 적재 시 동일 스키마를 그대로 대체)"""
import csv
from functools import lru_cache
from app.config import GAJIMA_ROOT

SEED = GAJIMA_ROOT / "simulation_site" / "neon" / "seed"


def _rows(name):
    p = SEED / name
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def _categories():
    out = {}
    for r in _rows("categories.csv"):
        out[r["category_id"]] = {
            "category_id": r["category_id"],
            "display_name": r.get("display_name") or r["category_id"],
            "top_brand": r.get("top_brand"),
            "price_median": float(r.get("price_median") or 0),
            "n_events": int(float(r.get("n_events") or 0)),
        }
    return out


@lru_cache(maxsize=1)
def _products_sorted():
    cats = _categories()
    rows = []
    for r in _rows("products.csv"):
        cat = cats.get(r["category_id"], {})
        rows.append({
            "product_id": r["product_id"],
            "category_id": r["category_id"],
            "category_name": cat.get("display_name", r["category_id"]),
            "brand": r.get("brand") or "UNK",
            "price": float(r.get("price_median") or 0),
            "n_events": int(float(r.get("n_events") or 0)),
            "name": r.get("display_name") or f"Product {r['product_id']}",
        })
    rows.sort(key=lambda x: x["n_events"], reverse=True)
    return rows


@lru_cache(maxsize=1)
def _similarity():
    sim = {}
    for r in _rows("category_similarity.csv"):
        cid = r.get("category_id")
        if not cid:
            continue
        sim.setdefault(cid, []).append({
            "rank": int(float(r.get("rank") or 0)),
            "category_id": r.get("similar_category_id"),
            "cosine": float(r.get("cosine") or 0),
        })
    for cid in sim:
        sim[cid].sort(key=lambda x: x["rank"])
    return sim


def products(limit=24, category_id=None):
    rows = _products_sorted()
    if category_id:
        rows = [r for r in rows if r["category_id"] == category_id]
    return rows[:limit]


def search_products(limit=60, category_id=None, brand=None, q=None):
    """시뮬 상품목록용 — 인기순(n_events) + 카테고리/브랜드/검색 필터. (seed products.csv 직접)"""
    rows = _products_sorted()
    if category_id and category_id != "all":
        rows = [r for r in rows if r["category_id"] == category_id]
    if brand and brand != "all":
        rows = [r for r in rows if r["brand"] == brand]
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in str(r["product_id"]).lower() or ql in r["brand"].lower() or ql in r["name"].lower()]
    return rows[:limit]


def categories(limit=12):
    cs = sorted(_categories().values(), key=lambda x: x["n_events"], reverse=True)
    return cs[:limit]


@lru_cache(maxsize=1)
def _brand_counts():
    from collections import Counter
    return Counter(r["brand"] for r in _products_sorted())


def brands(limit=30):
    """상품 수 상위 브랜드(드롭다운 facet용)."""
    return [{"brand": b, "count": n} for b, n in _brand_counts().most_common(limit)]


def total_products():
    return len(_products_sorted())


def similar_categories(category_id, k=3):
    """유사 카테고리 top-k(추천용). display_name 부여."""
    cats = _categories()
    out = []
    for s in _similarity().get(str(category_id), [])[:k]:
        c = cats.get(s["category_id"], {})
        out.append({"category_id": s["category_id"],
                    "display_name": c.get("display_name", s["category_id"]),
                    "cosine": round(s["cosine"], 3),
                    "top_brand": c.get("top_brand")})
    return out
