# -*- coding: utf-8 -*-
"""배포 자가검증 — 다른 PC에서 실행 전/후 깨질 지점을 사전 점검(서버 미기동).
사용: python scripts/check_deploy.py
종료코드 0=OK, 1=FAIL(필수 누락). 경고(WARN)는 0이지만 권고."""
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ok = warn = fail = 0


def check(cond, label, hard=True):
    global ok, warn, fail
    if cond:
        print(f"  [OK]   {label}"); ok += 1
    elif hard:
        print(f"  [FAIL] {label}"); fail += 1
    else:
        print(f"  [WARN] {label}"); warn += 1


def has_mod(name):
    try:
        importlib.import_module(name); return True
    except Exception:
        return False


print("== 1) 백엔드 런타임 의존성 ==")
for m in ["fastapi", "uvicorn", "pydantic", "multipart", "mysql.connector",
          "insightface", "onnxruntime", "cv2", "joblib", "sklearn", "numpy",
          "pandas", "pyarrow", "catboost", "xgboost", "lightgbm"]:
    check(has_mod(m), f"import {m}")

print("== 2) 모델 산출물(배포 누락 단골: gitignore 대상) ==")
prep = list((ROOT / "models" / "preprocessors").glob("prep_*_v2.joblib"))
check(len(prep) >= 1, f"models/preprocessors/prep_*_v2.joblib ({len(prep)}개)")
bdir = ROOT / "models" / "buffalo_l"
bzip = ROOT / "models" / "buffalo_l.zip"
check((bdir.exists() and any(bdir.glob("*.onnx"))) or bzip.exists(),
      "models/buffalo_l/*.onnx (또는 buffalo_l.zip) — 얼굴인식")

print("== 3) 데이터 산출물 ==")
check((ROOT / "data/processed/churn/train_tabular_v2.parquet").exists(), "churn/train_tabular_v2.parquet", hard=False)
# ★ BUG-011: test_tabular_v2 누락 시 개인진단 교집합 붕괴 → 명시 점검
check((ROOT / "data/processed/churn/test_tabular_v2.parquet").exists(), "churn/test_tabular_v2.parquet (개인진단 test 코호트)", hard=False)
# ★ BUG-011: 유저 선택 드롭다운(/samples/users)의 출처 — 없으면 목록 비고 진단 불가
check((ROOT / "data/processed/evaluation/eval_predictions.parquet").exists(), "evaluation/eval_predictions.parquet (유저 선택 드롭다운 출처)", hard=False)
check((ROOT / "data/processed/evaluation").exists(), "data/processed/evaluation/ (차트/메트릭)", hard=False)

print("== 4) 환경설정(.env) ==")
be = ROOT / "backend" / ".env"
de = ROOT / "dashboard_streamlit" / ".env"
check(be.exists(), "backend/.env 존재(없으면 memory 폴백·기본키)", hard=False)
check(de.exists(), "dashboard_streamlit/.env 존재", hard=False)
if de.exists():
    t = de.read_text(encoding="utf-8", errors="ignore")
    check("USE_MOCK=true" not in t.lower().replace(" ", ""), "대시보드 USE_MOCK!=true(실백엔드 사용)", hard=False)
    check("8090" in t or "DASHBOARD_API_BASE_URL" in t, "대시보드 API_BASE_URL 설정(포트 8090 권장)", hard=False)

print("== 5) 정적 자산 ==")
check((ROOT / "dashboard_streamlit" / "styles" / "main.css").exists(), "dashboard styles/main.css", hard=False)

print(f"\n== 요약: OK {ok} · WARN {warn} · FAIL {fail} ==")
if fail:
    print("FAIL 항목을 해결해야 배포 정상 동작(특히 의존성·모델 누락).")
    sys.exit(1)
print("필수 점검 통과. (WARN은 환경/네트워크 구성에 따라 확인 권장)")
sys.exit(0)
