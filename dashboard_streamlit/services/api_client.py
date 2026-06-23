import os
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_env_file()

# 🚨 [피드백 반영] .env 누락 시 시스템 오작동 및 401 에러를 막기 위해 올바른 기본값으로 정정했습니다.
API_BASE_URL = os.getenv("DASHBOARD_API_BASE_URL", "http://localhost:8090").rstrip("/")
API_KEY = os.getenv("DASHBOARD_API_KEY", "anchor-dev-key")
TIMEOUT_SEC = float(os.getenv("DASHBOARD_TIMEOUT_SEC", "10"))
USE_MOCK = os.getenv("DASHBOARD_USE_MOCK", "false").lower() == "true"


def success(data: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True, "data": data, "meta": {"source": "mock" if USE_MOCK else "backend"}, "error": None}


def failure(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "data": None, "meta": {}, "error": {"code": code, "message": message}}


def request_json(method: str, path: str, **kwargs) -> dict[str, Any]:
    headers = kwargs.pop("headers", {})
    headers.setdefault("X-API-Key", API_KEY)
    try:
        response = requests.request(
            method=method,
            url=f"{API_BASE_URL}{path}",
            headers=headers,
            timeout=TIMEOUT_SEC,
            **kwargs,
        )
        # 봉투(ok/error) 응답은 4xx여도 그대로 반환 — 백엔드 비즈니스 메시지
        # (예: "등록되지 않은 사용자 ID", "얼굴 임베딩 실패", "이미 등록된 ID")를 보존한다.
        # raise_for_status()를 먼저 호출하면 이 메시지가 "백엔드 API 호출 실패: 404"로 가려짐.
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict) and "ok" in payload:
            return payload
        response.raise_for_status()   # 봉투가 아닌 응답이 4xx/5xx면 진짜 실패로 처리
        return success(payload if isinstance(payload, dict) else {"value": payload})
    except requests.RequestException as exc:
        return failure("BACKEND_UNAVAILABLE", f"백엔드 API 호출 실패: {exc}")