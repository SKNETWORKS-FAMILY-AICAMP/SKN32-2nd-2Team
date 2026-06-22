# -*- coding: utf-8 -*-
"""infrastructure/face — 얼굴 이미지 → 512d 임베딩(insightface buffalo_l/ArcFace).
19-4 §1(대시보드=표시, 무거운 ML=백엔드) 원칙에 따라 임베딩은 백엔드에서 수행.
모델/라이브러리 미가용 시 None 반환(graceful) → 호출부가 422로 처리."""
import warnings

_APP = None
_TRIED = False


def _get_app():
    """insightface FaceAnalysis lazy 로드(최초 buffalo_l 다운로드). 실패 시 None."""
    global _APP, _TRIED
    if _APP is not None or _TRIED:
        return _APP
    _TRIED = True
    try:
        from insightface.app import FaceAnalysis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 임베딩엔 detection+recognition 만 필요 → landmark/genderage 제외로 로드·추론 단축
            app = FaceAnalysis(name="buffalo_l",
                               allowed_modules=["detection", "recognition"])
            app.prepare(ctx_id=-1, det_size=(640, 640))   # ctx_id=-1 → CPU
        _APP = app
    except Exception:
        _APP = None
    return _APP


def available() -> bool:
    return _get_app() is not None


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
