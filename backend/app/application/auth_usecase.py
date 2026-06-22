# -*- coding: utf-8 -*-
"""application — 얼굴 인증 usecase(19-4 §5·§7.1).
19-4 §1(대시보드=표시, 무거운 ML=백엔드)에 따라 **임베딩은 백엔드(insightface)** 에서 수행.
프론트는 얼굴 이미지를 보내고(클라 OpenCV는 검출만), 백엔드가 임베딩·저장·매칭·로깅을 담당.
(구 JSON-embedding 버전 register_face/login_face는 하위호환 위해 유지.)"""
from app.domain.face_match import best_match
from app.infrastructure.mysql import face_repository as repo
from app.infrastructure.face import embedder


def check_id(user_id) -> dict:
    """ID 중복 확인(교육과제②). 프론트 /auth/face/check-id."""
    if not user_id:
        return {"_status": 400, "error": "user_id 필수"}
    exists = repo.get_user(user_id) is not None
    return {"user_id": user_id, "exists": exists, "available": not exists}


def register_face_image(user_id, display_name, role, img_bytes) -> dict:
    """얼굴 이미지 → 백엔드 임베딩 → 등록(19-4 §7.1)."""
    if not user_id:
        return {"_status": 400, "error": "user_id 필수"}
    if repo.get_user(user_id):
        return {"_status": 409, "error": f"이미 등록된 사용자 ID: {user_id}"}   # 과제②
    emb = embedder.embed_from_image_bytes(img_bytes)
    if emb is None:
        return {"_status": 422, "error": "얼굴 임베딩 실패(얼굴 미검출 또는 insightface 모델 미준비)"}
    repo.register(user_id, display_name, role or "customer", emb)
    return {"user_id": user_id, "display_name": display_name or user_id,
            "role": role or "customer", "has_face": True}


def login_face_image(img_bytes, user_id=None) -> dict:
    """얼굴 이미지 → 백엔드 임베딩 → 입력 ID의 등록 얼굴과 매칭(19-4 §7.1)."""
    if not user_id:
        return {"_status": 400, "error": "user_id 필수"}
    target = repo.get_user(user_id)
    if not target:
        repo.log_login(user_id, False, None, "user_not_found")
        return {"_status": 404, "error": f"등록되지 않은 사용자 ID: {user_id}"}
    if not target.get("embedding"):
        repo.log_login(user_id, False, None, "no_face_embedding")
        return {"_status": 409, "error": f"등록된 얼굴 임베딩이 없습니다: {user_id}"}

    emb = embedder.embed_from_image_bytes(img_bytes)
    if emb is None:
        repo.log_login(user_id, False, None, "embedding_failed")
        return {"_status": 422, "error": "얼굴 임베딩 실패(얼굴 미검출 또는 insightface 모델 미준비)"}
    user, sim = best_match(emb, [target])
    if user:
        repo.log_login(user["user_id"], True, round(sim, 3))                       # 과제①
        return {"user_id": user["user_id"], "display_name": user["display_name"],
                "role": user.get("role", "customer"), "similarity": round(sim, 3),
                "access_token": None}
    repo.log_login(user_id, False, round(sim, 3), "no_match")
    return {"_status": 401, "error": f"ID와 얼굴이 일치하지 않습니다(유사도 {sim:.2f})", "similarity": round(sim, 3)}


def register_face(user_id, display_name=None, role="customer", embedding=None) -> dict:
    if not user_id:
        return {"_status": 400, "error": "user_id 필수"}
    ok = repo.register(user_id, display_name, role or "customer", embedding)
    if not ok:
        return {"_status": 409, "error": f"이미 등록된 사용자 ID: {user_id}"}   # 과제②
    return {"user_id": user_id, "display_name": display_name or user_id,
            "role": role or "customer", "has_face": embedding is not None}


def login_face(embedding=None, user_id=None) -> dict:
    """얼굴 임베딩으로 매칭하거나(우선), user_id로 직접 로그인(폴백)."""
    if embedding:
        user, sim = best_match(embedding, repo.list_users())
        if user:
            repo.log_login(user["user_id"], True, round(sim, 3))                  # 과제①
            return {"user_id": user["user_id"], "display_name": user["display_name"],
                    "role": user.get("role", "customer"), "similarity": round(sim, 3)}
        repo.log_login(user_id or "unknown", False, round(sim, 3), "no_match")
        return {"_status": 401, "error": f"일치하는 얼굴 없음(유사도 {sim:.2f})", "similarity": round(sim, 3)}
    # user_id 직접 로그인(없으면 customer로 생성)
    if not user_id:
        return {"_status": 400, "error": "embedding 또는 user_id 필요"}
    u = repo.get_user(user_id)
    if not u:
        repo.register(user_id, user_id, "customer", None)
        u = repo.get_user(user_id)
    repo.log_login(user_id, True, None)
    return {"user_id": u["user_id"], "display_name": u["display_name"],
            "role": u.get("role", "customer"), "similarity": None}


def get_me(user_id) -> dict:
    u = repo.get_user(user_id)
    if not u:
        return {"_status": 404, "error": f"사용자 없음: {user_id}"}
    return {"user_id": u["user_id"], "display_name": u["display_name"],
            "role": u.get("role", "customer")}


def recent_logins(limit=20) -> dict:
    return {"logins": repo.recent_logins(limit)}
