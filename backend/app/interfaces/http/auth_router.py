# -*- coding: utf-8 -*-
"""interfaces/http — 얼굴 인증(19-4 §5·§7.1). 봉투 적용.
프론트(대시보드)는 OpenCV로 검출만 하고 **얼굴 이미지를 multipart로 전송** → 백엔드가 insightface 임베딩.
(19-4 §1: 대시보드=표시, 무거운 ML=백엔드)"""
from fastapi import APIRouter, Depends, Query, UploadFile, File, Form
from app.interfaces.http.deps import require_api_key, unwrap
from app.interfaces.http.responses import ok
from app.application import auth_usecase as uc

router = APIRouter(tags=["auth"], dependencies=[Depends(require_api_key)])


@router.get("/auth/face/check-id")
async def check_id(user_id: str = Query(...)):
    """ID 중복 확인(교육과제②)."""
    return ok(unwrap(uc.check_id(user_id)))


@router.post("/auth/face/register")
async def face_register(
    user_id: str = Form(...),
    display_name: str = Form(""),
    role: str = Form("customer"),
    face_bbox: str = Form(None),          # 프론트 OpenCV 검출 bbox(JSON 문자열, 참고용)
    image: UploadFile = File(...),
):
    img = await image.read()
    return ok(unwrap(uc.register_face_image(user_id, display_name, role, img)))


@router.post("/auth/face/login")
async def face_login(
    user_id: str = Form(...),
    face_bbox: str = Form(None),
    image: UploadFile = File(...),
):
    img = await image.read()
    return ok(unwrap(uc.login_face_image(img, user_id)))


@router.get("/auth/me")
async def auth_me(user_id: str = Query(...)):
    return ok(unwrap(uc.get_me(user_id)))


@router.get("/auth/logins")
async def auth_logins(limit: int = Query(20, ge=1, le=200)):
    return ok(uc.recent_logins(limit))
