# -*- coding: utf-8 -*-
"""main — FastAPI 부트스트랩(19-2). 운영 백엔드: 레지스트리·평가 ingest·차트 API·예측 로그.
모든 응답은 {ok,data,meta,error} 봉투(19-4 §6). 추론(학습) 코드 없음.
실행: uvicorn app.main:app --port 8090."""
import asyncio
import contextlib
import os
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.responses import JSONResponse
from app.config import PORT
from app.infrastructure.mysql.session import mode
from app.infrastructure.mysql import retention
from app.application import sim_usecase
from app.interfaces.http.responses import err
from app.interfaces.http import (health_router, auth_router, models_router,
                                 predictions_router, dashboard_router, sim_router,
                                 sim_external_router)

CLEANUP_INTERVAL_SEC = int(os.environ.get("CLEANUP_INTERVAL_SEC", str(6 * 3600)))   # 기본 6시간


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """보존정책 스케줄러 + insightface 사전로드(첫 얼굴 요청 지연/타임아웃 방지)."""
    # 필수 산출물 존재 점검(BUG-011 재발 방지) — 누락 시 명확 경고(치명 X, 폴백 동작).
    try:
        from app.infrastructure.files import dataset_index
        miss = dataset_index.missing()
        if miss:
            print(f"[startup][WARN] 필수 산출물 누락: {miss} — 진단/예측/얼굴이 부분 동작할 수 있음. "
                  f"(대용량은 gitignore라 환경별 누락 가능 — 복원/재생성 필요)")
        else:
            print("[startup] 필수 산출물 점검 OK")
    except Exception as _e:
        print("[startup] 산출물 점검 스킵:", _e)

    async def prewarm_face():
        # 단일 워커 uvicorn에서 첫 임베딩 시 모델 로드(수 초)가 요청을 막아 타임아웃 →
        # 기동 시 백그라운드 스레드로 미리 로드해 둔다(블로킹 회피).
        try:
            from app.infrastructure.face import embedder
            await asyncio.to_thread(embedder._get_app)
        except Exception:
            pass
    asyncio.create_task(prewarm_face())

    async def loop():
        while True:
            try:
                await asyncio.to_thread(retention.cleanup_db)   # DB TTL 정리(블로킹 → 스레드)
                sim_usecase.sweep_sessions()                    # 메모리 idle 세션 정리
            except Exception:
                pass
            await asyncio.sleep(CLEANUP_INTERVAL_SEC)
    task = asyncio.create_task(loop())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="가지마 운영 백엔드 (FastAPI)", version="19-2", lifespan=lifespan,
              description="모델 레지스트리·평가 ingest·대시보드 차트 API·예측 로그. 학습 없음.")

# 시뮬 사이트(정적/별 오리진)에서 호출 허용 — 데모용 전체 허용
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"], allow_credentials=False)

app.include_router(health_router.router)
app.include_router(auth_router.router)
app.include_router(models_router.router)
app.include_router(predictions_router.router)
app.include_router(dashboard_router.router)
app.include_router(sim_router.router)
app.include_router(sim_external_router.router)   # 시뮬 사이트 외부 계약(/api/*)


# --- 에러도 봉투로 통일(19-4 §6) ---
@app.exception_handler(StarletteHTTPException)
async def http_exc(_: Request, exc: StarletteHTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content=err(f"HTTP_{exc.status_code}", str(exc.detail)))


@app.exception_handler(RequestValidationError)
async def validation_exc(_: Request, exc: RequestValidationError):
    msg = "; ".join(f"{'.'.join(str(x) for x in e.get('loc', []))}: {e.get('msg')}" for e in exc.errors())
    return JSONResponse(status_code=422, content=err("VALIDATION", msg or "검증 실패"))


@app.exception_handler(Exception)
async def unhandled_exc(_: Request, exc: Exception):
    return JSONResponse(status_code=500, content=err("INTERNAL", str(exc)))


@app.get("/")
async def root():
    return {"service": "gajima-backend", "framework": "fastapi", "db_mode": mode(),
            "docs": "/docs", "spec": "19-2/19-4"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=PORT, reload=False)
