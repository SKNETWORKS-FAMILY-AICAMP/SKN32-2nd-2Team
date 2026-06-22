# -*- coding: utf-8 -*-
"""REES46 화장품 category_id → 사람이 읽는 화장품 카테고리 이름 매핑 생성.

데이터 현실(웹서치 확인): 화장품 데이터셋은 category_code가 거의 결측(525중 18만, 그나마 비화장품 노이즈).
→ 사용자 합의: "임의 매핑이어도 됨" — 실제 category_id에 **그럴듯한 화장품 카테고리명 + 실제 top_brand**를 부여.
   인기순(n_events)으로 정렬해 인기 카테고리에 대표 품목명을 배정(결정적·재현가능).
산출: seed/category_name_map.csv (category_id, category_name)
실행: python simulation_site/neon/make_category_names.py
"""
import os, csv

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = os.path.join(HERE, "seed")
SRC = os.path.join(SEED, "categories.csv")
OUT = os.path.join(SEED, "category_name_map.csv")

# 네일샵 중심(브랜드 runail/irisk/masura/grattol=네일) + 일반 화장품 품목
TYPES = [
    "네일 폴리시", "젤 네일", "베이스·탑코트", "네일 팁", "네일 아트 도구", "매니큐어 세트",
    "큐티클 케어", "페디큐어", "네일 리무버", "네일 스톤·파츠",
    "립스틱", "립틴트", "립밤", "아이섀도우", "아이라이너", "마스카라", "아이브로우",
    "파운데이션", "쿠션·팩트", "컨실러", "블러셔", "하이라이터", "프라이머",
    "토너", "에센스·세럼", "수분크림", "마스크팩", "클렌징폼", "클렌징오일", "선크림", "미스트",
    "향수", "핸드크림", "바디로션", "헤어에센스", "샴푸", "화장솜·퍼프",
]


def main():
    rows = []
    with open(SRC, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    # 인기순 — 인기 카테고리가 대표 품목명을 받도록
    rows.sort(key=lambda r: float(r.get("n_events") or 0), reverse=True)
    out = []
    for i, r in enumerate(rows):
        cid = r["category_id"]
        brand = (r.get("top_brand") or "").strip()
        t = TYPES[i % len(TYPES)]
        name = f"{t} · {brand.title()}" if brand and brand.upper() != "UNK" else t
        out.append((cid, name))
    with open(OUT, "w", encoding="utf-8", newline="") as f:   # BOM 없이(백엔드 _rows가 utf-8로 읽음)
        w = csv.writer(f); w.writerow(["category_id", "category_name"]); w.writerows(out)
    print(f"[카테고리 매핑] {len(out)}개 → {OUT}")
    print("  샘플(인기순):", [n for _, n in out[:8]])


if __name__ == "__main__":
    main()
