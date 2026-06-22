# -*- coding: utf-8 -*-
"""서버(이 백엔드) 주소 탐지 — 시뮬 사이트가 우리 백엔드에 닿게 할 주소를 출력.

`ipconfig`로 보이는 IP는 **사설 LAN IP**(192.168.x / 10.x). 같은 와이파이/공유기 안에서만 닿는다.
인터넷(Vercel 등)에서 닿으려면 공유기 **포트포워딩**(8090) 또는 **터널**(ngrok/cloudflared)이 필요하다.

실행: python scripts/show_server_addr.py
"""
import socket
import sys

PORT = 8090


def lan_ip():
    """기본 라우트로 나가는 인터페이스의 사설 IP(소켓 트릭, 외부 전송 없음)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))   # 실제 전송 X, 라우팅만
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ip = lan_ip()
    print("=" * 56)
    print(f" 백엔드 LAN 주소 : http://{ip}:{PORT}")
    print(f" 로컬 주소       : http://127.0.0.1:{PORT}")
    print("=" * 56)
    print("\n[같은 네트워크(같은 와이파이) 데모]")
    print(f"  시뮬 .env → VITE_FASTAPI_URL=http://{ip}:{PORT}")
    print("\n[인터넷(Vercel 배포 시뮬)에서 닿게 하려면]")
    print("  ① 공유기 포트포워딩: 외부:8090 → " + f"{ip}:{PORT}  + 공인 IP 확인(whatismyip)")
    print("  ② 또는 터널:  cloudflared tunnel --url http://localhost:8090")
    print(f"     → 발급된 https 주소를 시뮬 VITE_FASTAPI_URL 에 사용(혼합콘텐츠 회피)")
    print("\n  ※ ipconfig의 IP는 사설(LAN)이라 인터넷에서 바로 못 닿음 — ①/② 중 하나 필요.")


if __name__ == "__main__":
    main()
