#!/usr/bin/env bash
# 시뮬 Neon(Postgres)에 카탈로그/브랜드/가격 CSV 적재.
# 사용:  DATABASE_URL="postgres://...neon..." bash simulation_site/neon/upload_neon.sh
# (반드시 GAJIMA 레포 루트 'SKN32-2nd_GAJIMA_Dev/' 에서 실행 — catalog_import.sql의 \copy 경로가 상대경로라서)
set -euo pipefail
: "${DATABASE_URL:?Neon connection string을 DATABASE_URL 환경변수로 주세요 (예: postgres://user:pass@ep-xxx.neon.tech/db?sslmode=require)}"
HERE="simulation_site/neon"
command -v psql >/dev/null || { echo "psql이 필요합니다(설치: postgresql-client)"; exit 1; }
echo "① 스키마 생성 (schema_neon.sql)"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$HERE/schema_neon.sql"
echo "② CSV 적재 (catalog_import.sql — categories/brands/products/similarity)"
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -f "$HERE/catalog_import.sql"
echo "③ 행수 확인"
psql "$DATABASE_URL" -c "select 'category' t, count(*) from sim_category_catalog
  union all select 'brand', count(*) from sim_brand_catalog
  union all select 'product', count(*) from sim_product_catalog
  union all select 'similarity', count(*) from sim_category_similarity;"
echo "=== Neon 적재 완료 ==="
