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

API_BASE_URL = os.getenv("DASHBOARD_API_BASE_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.getenv("DASHBOARD_API_KEY", "dev-key")
TIMEOUT_SEC = float(os.getenv("DASHBOARD_TIMEOUT_SEC", "10"))
USE_MOCK = os.getenv("DASHBOARD_USE_MOCK", "true").lower() == "true"


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
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "ok" in payload:
            return payload
        return success(payload if isinstance(payload, dict) else {"value": payload})
    except requests.RequestException as exc:
        return failure("BACKEND_UNAVAILABLE", f"백엔드 API 호출 실패: {exc}")
    except ValueError:
        return failure("INVALID_RESPONSE", "백엔드 응답이 JSON 형식이 아닙니다.")
