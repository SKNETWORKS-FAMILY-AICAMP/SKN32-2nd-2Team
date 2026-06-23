# -*- coding: utf-8 -*-
"""infrastructure/face — 얼굴 이미지 → 512d 임베딩(insightface buffalo_l/ArcFace).
19-4 §1(대시보드=표시, 무거운 ML=백엔드) 원칙에 따라 임베딩은 백엔드에서 수행.
모델/라이브러리 미가용 시 None 반환(graceful) → 호출부가 422로 처리."""
import warnings
import zipfile

_APP = None
_TRIED = False
_LAST_ERR = None   # 배포 진단용(왜 로드 실패했는지)


def _ensure_bundle():
    """번들 모델 준비(배포 PC 오프라인 대비). models/buffalo_l 에 onnx 없고 buffalo_l.zip 있으면 추출.
    반환: (insightface root, 번들 사용가능 여부). insightface는 <root>/models/buffalo_l 에서 찾는다."""
    from app.config import MODELS_DIR
    bdir = MODELS_DIR / "buffalo_l"
    bzip = MODELS_DIR / "buffalo_l.zip"
    has_onnx = bdir.exists() and any(bdir.glob("*.onnx"))
    if not has_onnx and bzip.exists():
        bdir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(bzip) as zf:
            zf.extractall(bdir)                       # zip 루트에 onnx들 → models/buffalo_l/*.onnx
        has_onnx = any(bdir.glob("*.onnx"))
    return str(MODELS_DIR.parent), has_onnx           # MODELS_DIR.parent/models/buffalo_l


def _get_app():
    """insightface FaceAnalysis lazy 로드. **번들(models/buffalo_l) 우선** → 없으면 기본(~/.insightface 다운로드).
    실패 시 None(이유는 _LAST_ERR/로그). 배포 PC에서 자동 다운로드 의존하다 422 나던 문제 해소."""
    global _APP, _TRIED, _LAST_ERR
    if _APP is not None or _TRIED:
        return _APP
    _TRIED = True
    try:
        from insightface.app import FaceAnalysis
        root, has_bundle = _ensure_bundle()
        kw = dict(name="buffalo_l", allowed_modules=["detection", "recognition"],
                  providers=["CPUExecutionProvider"])   # 배포 PC GPU provider 부재 대비
        if has_bundle:
            kw["root"] = root                            # 번들 사용 → 런타임 다운로드 불필요(오프라인 OK)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            app = FaceAnalysis(**kw)
            app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 → CPU
        _APP = app
    except Exception as e:
        import traceback
        _LAST_ERR = f"{type(e).__name__}: {e}"
        print("[face.embedder] insightface 로드 실패 —", _LAST_ERR)   # silent 금지(배포 진단)
        traceback.print_exc()
        _APP = None
    return _APP


def available() -> bool:
    return _get_app() is not None


def last_error():
    """insightface 로드 실패 사유(배포 진단용). 성공/미시도면 None."""
    return _LAST_ERR


def embed_from_image_bytes(img_bytes: bytes):
    """이미지 bytes → L2 정규화된 512d 임베딩(list[float]). 얼굴 미검출/오류 시 None."""
    app = _get_app()
    if app is None or not img_bytes:
        return None
    try:
        import cv2
        import numpy as np
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        faces = app.get(img)
        if not faces:
            return None
        # 가장 큰 얼굴 선택
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        e = faces[0].normed_embedding.astype("float32")
        norm = float((e ** 2).sum() ** 0.5) + 1e-8
        return [float(x) for x in (e / norm)]
    except Exception:
        return None
