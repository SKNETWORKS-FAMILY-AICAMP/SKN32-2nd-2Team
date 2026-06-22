# -*- coding: utf-8 -*-
"""infrastructure/files — 시뮬 사이트용 REES46 카탈로그 reader(26-9 P2).
simulation_site/neon/seed/*.csv 에서 상품·카테고리·유사도를 읽어 chart/UI-ready로 제공.
Neon 미사용 로컬 데모: 파일 직접. (Neon 적재 시 동일 스키마를 그대로 대체)"""
import csv
from functools import lru_cache
from app.config import GAJIMA_ROOT

SEED = GAJIMA_ROOT / "simulation_site" / "neon" / "seed"

# REES46 화장품은 상품명 결측 → 화장품 품목명 풀(웹서치 기반)로 임의 매핑(브랜드와 결합). id는 별도 표시.
ITEM_POOL = [
    "젤 폴리시", "매트 탑코트", "글리터 젤", "베이스 코트", "큐티클 오일", "퀵드라이 탑코트",
    "시럽 젤", "네일 스티커", "쉬머 폴리시", "카멜레온 젤", "네일 스트렝스너", "매니큐어",
    "벨벳 매트 립스틱", "글로우 립틴트", "수분 립밤", "립 플럼퍼", "매트 립크레용", "립오일",
    "쉬머 아이섀도우", "워터프루프 마스카라", "젤 아이라이너", "브로우 펜슬", "아이 프라이머",
    "글로우 파운데이션", "커버 쿠션", "래디언트 컨실러", "실키 프라이머", "매트 파우더",
    "크림 블러셔", "리퀴드 하이라이터", "하이드라 세럼", "나이아신아마이드 앰플", "히알루론 토너",
    "수분 크림", "수딩 마스크팩", "클렌징 오일", "젤 클렌저", "아이크림", "비타민C 세럼",
    "선크림 SPF50", "미스트 토너", "오 드 퍼퓸", "핸드크림", "바디로션", "헤어 세럼",
    "모이스처 샴푸", "화장솜", "메이크업 퍼프",
]


def _rows(name):
    p = SEED / name
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=1)
def _name_map():
    """category_id → 사람이 읽는 이름(make_category_names.py 생성). 없으면 빈 dict."""
    return {r["category_id"]: r["category_name"] for r in _rows("category_name_map.csv") if r.get("category_id")}


@lru_cache(maxsize=1)
def _categories():
    nm = _name_map()
    out = {}
    for r in _rows("categories.csv"):
        cid = r["category_id"]
        out[cid] = {
            "category_id": cid,
            # 이름 매핑 우선(REES46 화장품은 원본 이름 결측 → 브랜드·가격/코드 기반 라벨)
            "display_name": nm.get(cid) or r.get("display_name") or cid,
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
            "name": None,   # 아래에서 인기순으로 화장품 품목명 부여
        })
    rows.sort(key=lambda x: x["n_events"], reverse=True)
    for i, row in enumerate(rows):       # 인기순 → 화장품 품목명 + 브랜드(결정적)
        b = row["brand"]
        bn = b.title() if b and b != "UNK" else ""
        row["name"] = f"{bn} {ITEM_POOL[i % len(ITEM_POOL)]}".strip()
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


@lru_cache(maxsize=1)
def _by_id():
    return {r["product_id"]: r for r in _products_sorted()}


def product_by_id(product_id):
    """단건 상품(상세페이지용). 없으면 None."""
    return _by_id().get(str(product_id))


def categories(limit=12):
    cs = sorted(_categories().values(), key=lambda x: x["n_events"], reverse=True)
    return cs[:limit]


def name_of(category_id):
    """category_id → 매핑된 사람이 읽는 이름(없으면 None). 대시보드/추천 공용."""
    return _name_map().get(str(category_id))


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
