import json
from typing import Any

from services.api_client import USE_MOCK, failure, request_json, success


_MOCK_USERS: dict[str, dict[str, Any]] = {}


def check_user_id_available(user_id: str) -> dict[str, Any]:
    if not user_id:
        return failure("INVALID_USER_ID", "ID를 입력해주세요.")
    if USE_MOCK:
        return success({"user_id": user_id, "available": user_id not in _MOCK_USERS})
    return request_json("GET", "/auth/face/check-id", params={"user_id": user_id})


def register_face(
    user_id: str,
    display_name: str,
    role: str,
    image_bytes: bytes,
    face_bbox: tuple[int, int, int, int],
) -> dict[str, Any]:
    if USE_MOCK:
        if user_id in _MOCK_USERS:
            return failure("DUPLICATE_USER_ID", "이미 등록된 ID입니다.")
        _MOCK_USERS[user_id] = {"user_id": user_id, "display_name": display_name, "role": role or "customer"}
        return success({**_MOCK_USERS[user_id], "embedding_status": "mocked"})

    files = {"image": ("face.jpg", image_bytes, "image/jpeg")}
    data = {
        "user_id": user_id,
        "display_name": display_name,
        "role": role or "customer",
        "face_bbox": json.dumps(face_bbox),
    }
    return request_json("POST", "/auth/face/register", files=files, data=data)


def login_face(user_id: str, image_bytes: bytes, face_bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    if not user_id:
        return failure("INVALID_USER_ID", "ID를 입력해주세요.")
    if USE_MOCK:
        if user_id in _MOCK_USERS:
            return success(_MOCK_USERS[user_id])
        if not _MOCK_USERS:
            return success({"user_id": user_id, "display_name": user_id, "role": "customer"})
        return failure("USER_NOT_FOUND", f"등록되지 않은 ID입니다: {user_id}")

    files = {"image": ("face.jpg", image_bytes, "image/jpeg")}
    data = {"user_id": user_id, "face_bbox": json.dumps(face_bbox)}
    return request_json("POST", "/auth/face/login", files=files, data=data)
