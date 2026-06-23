# -*- coding: utf-8 -*-
"""시뮬 Neon(Postgres)에 seed CSV 적재 — psql 없이 psycopg2로 COPY 업로드.

준비:  pip install psycopg2-binary
사용:  python simulation_site/neon/upload_neon.py "postgresql://USER:PASS@ep-xxx.neon.tech/DB?sslmode=require"
       (또는 DATABASE_URL 환경변수)
Neon 접속문자열: Neon 콘솔(project bold-meadow-77051786) → Connection Details → Connection string 복사.
"""
import os
import sys
import psycopg2

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "schema_neon.sql")
SEED = os.path.join(HERE, "..", "seed")   # 레거시 분리: seed/는 상위(neon/)의 활성 카탈로그 소스

# (CSV파일, 테이블, 컬럼순서) — catalog_import.sql과 동일
LOADS = [
    ("categories.csv", "sim_category_catalog",
     "category_id, category_code, display_name, top_brand, price_median, n_products, n_events, price_sum, source_dataset"),
    ("brands.csv", "sim_brand_catalog",
     "brand, n_products, n_categories, n_events, price_median, source_dataset"),
    ("products.csv", "sim_product_catalog",
     "product_id, category_id, brand, price_median, n_events, display_name, is_active, source_dataset"),
    ("category_similarity.csv", "sim_category_similarity",
     "category_id, rank, similar_category_id, cosine, source_dataset"),
]


def main():
    dsn = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATABASE_URL", "")
    if not dsn:
        sys.exit("Neon 접속문자열을 인자나 DATABASE_URL로 주세요. 예: python upload_neon.py \"postgresql://...neon.tech/db?sslmode=require\"")
    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    cur = conn.cursor()
    print("① 스키마 생성(schema_neon.sql)")
    cur.execute(open(SCHEMA, encoding="utf-8").read())
    for fname, table, cols in LOADS:
        path = os.path.join(SEED, fname)
        print(f"② 적재 {fname} → {table}")
        cur.execute(f"TRUNCATE {table};")          # 재적재 시 중복 방지
        with open(path, "r", encoding="utf-8") as f:
            cur.copy_expert(
                f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, HEADER true, ENCODING 'UTF8')", f)
    conn.commit()
    print("③ 행수 확인")
    for _, table, _ in LOADS:
        cur.execute(f"SELECT count(*) FROM {table};")
        print(f"   {table}: {cur.fetchone()[0]:,}")
    cur.close(); conn.close()
    print("=== Neon 적재 완료 ===")


if __name__ == "__main__":
    main()
