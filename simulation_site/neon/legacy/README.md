# Neon 레거시 (비활성)

이 폴더의 파일들은 **현재 운영에서 사용하지 않는** Neon(PostgreSQL) 전용 도구입니다.
프로젝트는 **MySQL 단일 + 백엔드 인메모리 세션 + 파일 기반 카탈로그**로 확정됐고, Neon은 채택하지 않았습니다.

| 파일 | 용도(과거) | 상태 |
| --- | --- | --- |
| `upload_neon.py` · `upload_neon.sh` | seed CSV → Neon 적재(psycopg2 COPY / psql) | 비활성(보존) |
| `schema_neon.sql` | Neon Postgres 스키마 | 비활성(보존) |
| `catalog_import.sql` | Neon \copy 임포트 SQL | 비활성(보존) |

- **런타임 의존 없음**: 백엔드는 Neon에 연결하지 않습니다(`psycopg2` import 없음, `NEON_URL`은 옵션 헬스 플래그뿐).
- **재적재 필요 시**: `python legacy/upload_neon.py "postgresql://...neon.tech/db?sslmode=require"` — `SEED`는 상위 `neon/seed/`(활성 카탈로그 소스)를 가리킵니다.
- 활성 카탈로그 소스(상위 `neon/seed/*.csv`)는 백엔드 `catalog_store`가 직접 읽습니다 — 이 폴더와 무관하게 동작.

> 상세: 「19-11 Neon 레거시 분리 계획서 및 완료 기록」(team_project_churn/reports).
