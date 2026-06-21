# GAJIMA Dashboard Streamlit

GAJIMA 프로젝트의 Streamlit 기반 대시보드 애플리케이션입니다.  
얼굴 등록/로그인 기능과 고객 이탈 분석 대시보드 화면을 제공합니다.

## 주요 기능

- Home 화면
- 얼굴 등록 및 얼굴 로그인 화면
- 로그인 사용자 전용 Dashboard 화면
- OpenCV 기반 얼굴 검출
- 백엔드 얼굴 인증 API 연동
- Streamlit 사이드바 커스텀 메뉴
- Mock 데이터 기반 대시보드/API 테스트 지원

## 프로젝트 구조
dashboard_streamlit/ 
├── app.py # Streamlit 메인 진입 파일 
├── pages/
│ ├── 01_face_login.py # 얼굴 등록 / 로그인 화면
│ └── 02_dashboard.py # 대시보드 화면
├── components/ # 공통 UI 컴포넌트
├── services/ # API, 인증, 얼굴 검출 등 서비스 로직
├── mocks/ # Mock 응답 데이터
├── assets/ # 로고 등 정적 리소스
├── styles/ # CSS 스타일
├── .streamlit/ # Streamlit 설정
├── .env.example # 환경 변수 예시
├── requirements.txt # Python 의존성 목록
└── README.md

## 실행 방법

꼭! dashboard_streamlit 폴더에 들어와서 아래 명령어를 실행합니다.
streamlit run app.py

명령어 작성 예시:
(.venv) PS C:\Python_workspace\dashboard_streamlit\dashboard_streamlit> streamlit run app.py


## 환경 변수 설정

`.env.example` 파일을 참고하여 필요한 환경 변수를 설정합니다.
dotenv DASHBOARD_API_BASE_URL=[http://localhost:8080](http://localhost:8080) DASHBOARD_API_KEY=dev-key DASHBOARD_USE_MOCK=true DASHBOARD_TIMEOUT_SEC=10

### 환경 변수 설명

| 변수명 | 설명 |
|---|---|
| `DASHBOARD_API_BASE_URL` | 백엔드 API 서버 주소 |
| `DASHBOARD_API_KEY` | 백엔드 API 호출 시 사용할 API Key |
| `DASHBOARD_USE_MOCK` | Mock 데이터 사용 여부 |
| `DASHBOARD_TIMEOUT_SEC` | API 요청 타임아웃 시간 |

## 얼굴 인증 흐름

### 얼굴 등록

1. 사용자가 ID를 입력합니다.
2. ID 중복 확인을 수행합니다.
3. 카메라로 얼굴을 촬영합니다.
4. OpenCV로 얼굴을 검출합니다.
5. 백엔드 `/auth/face/register` API로 얼굴 등록 요청을 보냅니다.

### 얼굴 로그인

1. 사용자가 카메라로 얼굴을 촬영합니다.
2. OpenCV로 얼굴을 검출합니다.
3. 백엔드 `/auth/face/login` API로 로그인 요청을 보냅니다.
4. 로그인 성공 시 사용자 정보를 세션에 저장합니다.
5. Dashboard 화면으로 이동합니다.

## 사용 중인 백엔드 API

### ID 중복 확인
http GET /auth/face/check-id?user_id={user_id}

### 얼굴 등록
http POST /auth/face/register


`multipart/form-data` 형식으로 전송합니다.

| 필드명 | 설명 |
|---|---|
| `image` | 얼굴 이미지 |
| `user_id` | 사용자 ID |
| `display_name` | 사용자 이름 |
| `role` | 사용자 역할 |
| `face_bbox` | OpenCV로 검출한 얼굴 좌표 |

### 얼굴 로그인
http POST /auth/face/login


`multipart/form-data` 형식으로 전송합니다.

| 필드명 | 설명 |
|---|---|
| `image` | 얼굴 이미지 |
| `face_bbox` | OpenCV로 검출한 얼굴 좌표 |

## 백엔드 처리 범위

얼굴 인증과 관련된 핵심 로직은 백엔드에서 처리합니다.

- 512차원 얼굴 임베딩 생성
- L2 정규화
- 얼굴 임베딩 암호화 및 저장
- `face_user` 테이블 저장
- 코사인 유사도 기반 얼굴 비교
- 인증 임계값 판단
- `face_login_log` 테이블 기록

## 화면 구성

### Home

서비스 소개 및 주요 화면으로 이동하는 버튼을 제공합니다.

### Face Login

얼굴 등록과 얼굴 로그인을 수행하는 화면입니다.

### Dashboard

로그인한 사용자만 접근할 수 있는 대시보드 화면입니다.  
고객 이탈 분석, 모델 진단, 운영 지표 등을 표시하는 영역으로 구성되어 있습니다.

## 사이드바 메뉴

Streamlit 기본 multipage 메뉴는 숨기고, 커스텀 사이드바 메뉴를 사용합니다.
표시 예시: text [Menu] Home face login dashboard

로그인 후에는 사용자 정보가 함께 표시됩니다.
text User ID: user001 Role: customer
[Menu] Home face login dashboard

## 개발 메모

- `app.py`는 Streamlit 메인 진입 파일이므로 삭제하면 안 됩니다.
- `pages/` 폴더명은 Streamlit multipage 구조에서 사용됩니다.
- 커스텀 사이드바 메뉴는 `components/layout.py`에서 관리합니다.
- 얼굴 검출 로직은 `services/face_utils.py`에서 관리합니다.
- 백엔드 API 호출 로직은 `services/` 하위 파일에서 관리합니다.