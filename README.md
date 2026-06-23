<div align="center">

# ⚓ Anchor
### 가입 고객 이탈 예측 시스템

![Python](https://img.shields.io/badge/Python-3.12.7-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.138-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-337AB7?style=for-the-badge)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?style=for-the-badge&logo=mysql&logoColor=white)

</div>

---

## 👥 팀 소개

<div align="center">

### 🏷️ Team **가지마** | Project **Anchor**

| 역할 | 이름 | 담당 |
|:---:|:---:|:---|
| 👑 팀장 | 문동원 | Face Recognition · UI/대시보드 · 시스템 통합 |
| 🔧 팀원 | 최연우 | Backend (FastAPI · DB 연동 · 실시간 추론 API) |
| 🤖 팀원 | 박수진 | ML 모델 학습 · 평가 |
| 🤖 팀원 | 이서은 | ML 모델 학습 · 평가 |
| 🤖 팀원 | 최상욱 | ML 모델 학습 · 평가 |

</div>

가입 고객의 행동 로그를 학습해 **이탈(churn)을 예측**하고, 시뮬레이션 쇼핑몰에서 **실시간으로 이탈률을 분석**하여
운영 대시보드에서 **고객 행동별 Action Plan**을 제시하는 이탈 예측 시스템입니다.

---

## 기술 스택 (요약)

| 영역 | 사용 기술 |
|------|-----------|
| Backend API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit |
| Simulation Frontend | React, Vite, TypeScript |
| Simulation Backend | Express, tRPC |
| ML | XGBoost, LightGBM, CatBoost, scikit-learn, PyTorch |
| Data Processing | pandas, NumPy, Parquet |
| Face Recognition | insightface |
| DB/Storage | MySQL(운영/옵션), Neon(PostgreSQL, 시뮬 로그), 파일기반 산출물(parquet/npz/model) |
| DevOps/Run | PowerShell, Bash, pnpm |

## WBS (구현 단계)

| 단계 | 작업 항목 | 산출물 |
|------|-----------|--------|
| 1. 기획/정의 | 문제 정의, 이탈 라벨 기준 확정(7일 무활동), 지표 정의 | 요구사항 정리, 라벨링 기준 |
| 2. 데이터 준비 | 대용량 이벤트 로그 정제, 월별 누적 집계, 피처 엔지니어링 | 학습용 전처리 데이터셋 |
| 3. 모델 개발 | ML/DL 후보 모델 학습 및 비교, 임계값 튜닝, 상위 3개 모델 선정 | 모델 아티팩트, 평가 리포트 |
| 4. 백엔드 구축 | FastAPI 추론/대시보드/시뮬 API, 인증/검증, 저장소 연동 | backend 서비스 (8090) |
| 5. 프론트/대시보드 구축 | Streamlit 대시보드, React 시뮬레이션 쇼핑몰 UI 구현 | dashboard(8501), simulation(3000) |
| 6. 실시간 연동 | 시뮬 사용자 이벤트 → 추론 API → 대시보드 반영 파이프라인 구성 | 실시간 churn 추적 흐름 |
| 7. 통합/검증 | 엔드투엔드 동작 점검, 성능/품질 확인, 실행 스크립트 정리 | run_all 스크립트, 최종 통합본 |

---

## 1. 주요 기능 (세부 구현)

1. **Kaggle dataset 기반 ML 모델 3개 학습** (평가 지표 상위 3종 선정)
2. **Face-login** — 얼굴 인식 로그인
3. **ML 모델 트레이닝 정보**를 대시보드에 표시 (모델 성능 진단)
4. 사용자가 ML 모델 3개 중 **하나를 선택**해 Simulation site의 고객 행동에 따른 **이탈률 실시간 분석**
5. **Dashboard / Simulation 2-사이트 구성**
   - **Dashboard**: Face login + ML 엔지니어가 개발한 모델의 성능 확인
   - **Simulation**: 이커머스 사이트로서 고객 행동에 따라 ML 모델이 **실시간 이탈률 예측**
   - 예측 값은 Dashboard에 표시되고, 고객 행동에 따른 **Action Plan을 Dashboard에서 제시**

---

## 2. 데이터

- **출처**: Kaggle — [eCommerce Events History in Cosmetics Shop](https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop?select=2020-Jan.csv)
- **규모**: 5개월 · 약 2천만 건 이벤트 로그

### Y값 정의
> **1주일간 어떠한 이벤트 로그도 없는 유저**를 이탈 유저로 정의

### X값 전처리

| 구분 | 기술/기법 | 단계 | 목적 및 내용 |
|------|-----------|:----:|------|
| 데이터 준비 | 시계열 라벨링 | Prep | 관찰/결과 기간 분리, 7일 무활동 시 이탈 정의, 데이터 누수 방지 |
| 데이터 준비 | 월별 순차 처리 | Prep | 5개월 2천만 건 데이터 월별 처리 및 유저 집계 누적 |
| 데이터 준비 | DType 다운캐스팅 | Prep | category/int32/float32 적용으로 메모리 최적화 |
| 데이터 준비 | 최근활동 코호트 | Prep | recency ≤ 7 기준으로 비정상 라벨 보정 |
| 데이터 품질 | 결측치·음수값 처리 | Bayes | 음수 가격 보정 및 NaN 제거 |
| 머신러닝 | 스케일링 | ML | Standard / MinMax / Robust / None 비교 |
| 머신러닝 | Log1p 변환 | ML | Right-skew 분포 안정화 |
| 머신러닝 | IQR 클리핑 | ML | 이상치 영향 감소 |
| 머신러닝 | 클래스 불균형 처리 | ML | None / Class Weight / SMOTE 비교 |
| 딥러닝 | 시퀀스 정규화 | DL | Feature Std / Log1p / None 비교 |
| 딥러닝 | Pos Weight | DL | BCE Loss 클래스 가중치 적용 |
| 저장 최적화 | 압축 저장 | All | Parquet 및 Savez Compressed 사용 |

---

## 3. 모델

- **알고리즘 출처**: XGBoost — 2016년 워싱턴 대학교 **Tianqi Chen & Carlos Guestrin** 논문
- **선정 기준**: 모델 평가 지표값이 가장 좋은 **상위 3개** 선정 → **XGBoost ★ · LightGBM · CatBoost**
- **평가 기준**: ROC-AUC · Precision Acc · Feb AUC

### 모델 비교

| No | Model | 상태 | CV PR-AUC | Feb AUC | 임계값 | 최적 전처리 |
|:--:|-------|:----:|:---------:|:-------:|:------:|-------------|
| 1 | DecisionTree | 완료 | 0.9288 | 0.7773 | 0.42 | none · classweight |
| 2 | RandomForest | 완료 | 0.9365 | 0.7892 | 0.54 | robust · none |
| 3 | LogisticReg | 완료 | 0.9356 | 0.7860 | 0.59 | minmax · classweight |
| 4 | **XGBoost ★** | 완료 | **0.9370** | **0.7904** | 0.53 | robust · classweight |
| 5 | LightGBM | 완료 | 0.9370 | 0.7902 | 0.55 | standard · none |
| 6 | CatBoost | 완료 | 0.9370 | 0.7902 | 0.52 | none · none |
| 7 | Transformer | 완료 | - | 0.7855 | - | sequence norm = log |

### 모델 학습 방법 (XGBoost)
> 사용자별 집계 피처(접속일수, 플레이시간, 결제금액, 구매횟수 등)를 입력으로 첫 번째 의사결정나무를 학습한 뒤,
> 이전 트리가 오분류한 사용자들을 중심으로 새로운 트리를 순차적으로 추가 학습합니다.
> 각 트리는 이전 모델의 예측 오차를 보완하도록 생성되며, 최종적으로 여러 트리의 예측 결과를 결합해 이탈 여부를 판단합니다.

---

## 4. 스크린샷

> 스크린샷은 `assets/`(또는 `dashboard_streamlit/assets/`)에 추가하세요. (예시 경로)

```text
### 🔐 Face Login
![Face Login](assets/face_login.png)

### 📊 Dashboard — 모델 성능 진단
![Dashboard](assets/dashboard.png)

### 👤 개인 이탈 진단 + Action Plan
![진단](assets/diagnosis.png)

### 🛒 Simulation Site — 실시간 이탈률 분석
![Simulation](assets/simulation.png)
```

---

## 5. 프로젝트 구조

```text
SKN32-2nd_GAJIMA_Dev/
├─ backend/                        # FastAPI 백엔드 (포트 8090)
│  ├─ app/
│  │  ├─ interfaces/http/          # 라우터(auth·predictions·dashboard·sim·models …)
│  │  ├─ application/              # 유스케이스(diagnose·realtime·sim·charts·auth …)
│  │  ├─ domain/                   # 도메인 로직(risk_level·session_hazard …)
│  │  ├─ infrastructure/           # face(insightface)·files·mysql(옵션)·model_inference
│  │  ├─ schemas/ · validators/    # Pydantic 스키마·검증
│  │  └─ config.py · main.py
│  ├─ db/                          # DB 스키마/마이그레이션
│  └─ tests/                       # 백엔드 테스트
├─ dashboard_streamlit/            # Streamlit 대시보드 (포트 8501)
│  ├─ pages/                       # 01_face_login · 02_dashboard
│  ├─ services/                    # api_client·dashboard·prediction·chart·auth·face_utils·recommendation
│  └─ components/                  # layout·charts·kpi_cards·risk_table·error_state
├─ simulation_site/                # React + Vite + Express 시뮬 쇼핑몰 (포트 3000)
│  ├─ client/src/pages/            # Home · ProductList · ProductDetail · Cart · ComponentShowcase
│  ├─ client/src/components/       # FloatingChurnWidget · ChurnActionPanel · RecommendedProducts 등
│  ├─ server/                      # Express + tRPC
│  └─ shared/                      # 공용 타입/스키마
├─ src/models/                     # 모델 학습/추론 코드 (churn · session_bounce · next_category)
├─ models/                         # 학습 산출물 (prep 번들 · GRU/Transformer · buffalo_l)
├─ data/processed/                 # 전처리 데이터 · evaluation 산출물
├─ scripts/                        # 학습/평가 스크립트
├─ notebooks/ · reports/ · plans/  # 노트북 · 리포트 · 계획서
├─ run_all.ps1 / run_all.sh        # 서버 일괄 실행
├─ requirements.txt
└─ README.md
```

---

## 6. 개발 환경

### 소프트웨어 스펙

| 항목 | 버전 |
|------|------|
| Python | 3.12.7 |
| FastAPI / Uvicorn | 0.138.0 / 0.49.0 |
| Streamlit | 1.41.1 |
| XGBoost / LightGBM / CatBoost | 2.1.4 / 4.6.0 / 1.2.10 |
| scikit-learn | 1.5.2 |
| PyTorch | 2.5.1 (cpu) |
| insightface | 0.7.3 |
| pandas / pydantic | 2.2.3 / 2.13.4 |
| React / Vite / TypeScript | 19 / 7 / 5.9.3 |
| Express / tRPC | 4.21 / 11.6 |
| DB | MySQL 8.0 · Neon(PostgreSQL, 시뮬 로그) |

### 하드웨어 스펙

| 항목 | 사양 |
|------|------|
| OS | Microsoft Windows 11 Pro (64비트) |
| CPU / RAM / GPU | *(작성 필요)* |

---

## 7. 환경 설정

### 가상환경 및 패키지 설치

```bash
cd SKN32-2nd_GAJIMA_Dev
python -m venv .venv
.venv\Scripts\activate          # (Windows) / macOS·Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -r dashboard_streamlit/requirements.txt

cd simulation_site && pnpm install   # (또는 npm install)
```

### `.env` 설정 (시크릿은 커밋 금지 — `.env.example` 참고)

```env
# backend/.env
API_KEY=your-api-key
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=your_user
MYSQL_PASSWORD=your_password
MYSQL_DB=anchor

# dashboard_streamlit/.env
DASHBOARD_API_BASE_URL=http://127.0.0.1:8090
DASHBOARD_API_KEY=your-api-key
DASHBOARD_USE_MOCK=false
```

---

## 8. 실행 방법

> 프로젝트 루트(`SKN32-2nd_GAJIMA_Dev`)에서 실행. 아래는 실제 검증된 명령(Windows PowerShell 기준).
> 경로는 루트 기준 상대경로로 표기 — 절대경로로 쓰려면 앞에 본인 경로를 붙이세요.

### ① 백엔드 (이탈예측 API · FastAPI :8090)

```powershell
cd backend ; ..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8090
```

### ② 대시보드 (Streamlit :8501)

```powershell
.\.venv\Scripts\python.exe -m streamlit run dashboard_streamlit\app.py
```

### ③ 시뮬레이션 사이트 (프론트 · React :3000)

```bash
cd simulation_site && pnpm dev
```

### 🔗 접속 링크

| 서비스 | 주소 | 확인 |
|--------|------|------|
| 시뮬레이션 사이트(프론트) | <http://localhost:3000/> | HTTP 200, 정상 렌더(≈368KB) |
| 백엔드(이탈예측 API) | <http://127.0.0.1:8090> | `/health` → `ok:true` |
| 대시보드 | <http://127.0.0.1:8501> | Face Login 후 진입 |
| API 문서(참고) | <http://127.0.0.1:8090/docs> | Swagger UI |

> 대시보드는 **Face Login** 후 진입합니다. **백엔드를 먼저** 띄워야 추론/얼굴인식이 동작합니다.
