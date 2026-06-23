# -*- coding: utf-8 -*-
"""infrastructure/files — 필수 산출물 매니페스트(존재 점검). BUG-011 재발 방지.
대용량 데이터/모델은 .gitignore 라 환경마다 실재가 다름 → '조용한 누락'을 기동 시 감지한다.
치명 처리 X(폴백 동작 유지), 누락은 WARN 로그로 노출."""
from app.config import DATA_DIR, MODELS_DIR

CHURN = DATA_DIR / "churn"

# (라벨, 경로, glob 여부) — glob=True면 패턴 매칭 1개 이상
REQUIRED = [
    ("churn train tabular", CHURN / "train_tabular_v2.parquet", False),
    ("churn test tabular",  CHURN / "test_tabular_v2.parquet", False),   # ★ BUG-011: 누락 시 진단 교집합 붕괴
    ("prep 번들(부스터)",   MODELS_DIR / "preprocessors", "prep_*_v2.joblib"),
    ("얼굴 모델 buffalo_l",  MODELS_DIR / "buffalo_l", "*.onnx"),
]


def required_status():
    """필수 산출물 존재 점검. 반환: [(label, path, ok)]."""
    out = []
    for label, path, pat in REQUIRED:
        if pat and pat is not False:
            ok = path.exists() and any(path.glob(pat))
            # buffalo_l 은 zip 동봉도 허용
            if not ok and path.name == "buffalo_l":
                ok = (path.parent / "buffalo_l.zip").exists()
        else:
            ok = path.exists()
        out.append((label, str(path), bool(ok)))
    return out


def missing():
    """누락된 필수 산출물 라벨 목록."""
    return [label for label, _p, ok in required_status() if not ok]
