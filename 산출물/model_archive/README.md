# 📦 Model Archive — Anchor 프로젝트 모델 모음

이 폴더(`model_archive/`)는 프로젝트가 학습·생성한 **모든 모델 산출물을 한곳에 모은 사본**입니다.
원본은 프로젝트 루트의 [`models/`](../models/) 폴더이며, **실제 서비스(백엔드)는 원본 `models/` 경로를 읽습니다.**
이 아카이브는 보관·인수인계·문서화 용도이고, 내부 디렉터리 구조는 원본과 동일하게 보존했습니다.

> ⚠️ 코드/백엔드가 참조하는 경로는 원본 `models/...` 입니다. 이 폴더를 옮기거나 이름을 바꿔도 서비스 동작에는 영향이 없습니다(반대로, 이 폴더를 수정해도 서비스에 반영되지 않습니다).

---

## 🗂️ 용도별 분류 (4종)

| 분류 | 목적 | 모델 수 |
|------|------|:---:|
| **A. 이탈 예측 (Churn)** | 가입 고객의 7일 무활동 이탈 예측 — 프로젝트 핵심 | 7 |
| **B. 세션 바운스 (Session Bounce)** | 현재 세션 중 이탈(바운스) 실시간 예측 | 3 |
| **C. 다음 카테고리 (Next Category)** | 다음에 관심 가질 카테고리 예측 (추천 보조) | 1 |
| **D. 추천 (Recommendation)** | 장바구니/상품 추천 | 1 (+로직) |

---

## A. 이탈 예측 (Churn) — Y: 7일간 이벤트 로그 없는 유저 = 이탈

가입 고객의 집계 피처(recency·tenure·이벤트수·구매액 등 v2 스키마)로 이탈 확률을 예측합니다.
**실시간 추론(백엔드)은 `preprocessors/prep_{모델}_v2.joblib` 번들**(전처리 파이프라인 + 모델 + 보정 + threshold)을 로드합니다.
각 `churn/<모델>/` 폴더에는 **원시 모델 바이너리 + 설정(model_config.json) + 피처 스키마**가 들어 있습니다.

| 모델 | 타입 | 역할/특징 | 서빙 번들 (아카이브 경로) | 원시 모델 (아카이브 경로) |
|------|:---:|------|------|------|
| **XGBoost** ★ | tree | 최고 성능 후보, 얕은 트리(depth=3)+강정규화 | [`preprocessors/prep_XGBoost_v2.joblib`](preprocessors/prep_XGBoost_v2.joblib) | [`churn/xgboost/model.json`](churn/xgboost/model.json) + `preprocessor.joblib` |
| **LightGBM** | tree | 부스팅(gbdt), 빠른 학습/추론 | [`preprocessors/prep_LightGBM_v2.joblib`](preprocessors/prep_LightGBM_v2.joblib) | [`churn/lightgbm/model.pkl`](churn/lightgbm/model.pkl) · `model.txt` |
| **CatBoost** | tree | iter=450·depth=4, PRAUC 기준 튜닝 | [`preprocessors/prep_CatBoost_v2.joblib`](preprocessors/prep_CatBoost_v2.joblib) | [`churn/catboost/model.cbm`](churn/catboost/model.cbm) |
| **DecisionTree** | tree | 단일 트리(max_depth=6), class_weight=balanced | [`preprocessors/prep_DecisionTree_v2.joblib`](preprocessors/prep_DecisionTree_v2.joblib) | [`churn/decisiontree/model.joblib`](churn/decisiontree/model.joblib) |
| **RandomForest** | tree | 배깅 앙상블 | [`preprocessors/prep_RandomForest_v2.joblib`](preprocessors/prep_RandomForest_v2.joblib) | (번들에 포함) |
| **LogReg** | linear | 로지스틱 회귀 베이스라인 (log+scaler) | [`preprocessors/prep_LogReg_v2.joblib`](preprocessors/prep_LogReg_v2.joblib) | (번들에 포함) |
| **Transformer** | sequence | 주별 시퀀스[view·cart·purchase] 기반 DL, best_norm=log | — (시퀀스 npz 입력) | [`churn/transformer/model.pt`](churn/transformer/model.pt) |

- **전처리 리포트**: 각 모델 전처리 설명은 [`preprocessors/`](preprocessors/) 안의 `*_전처리리포트.md` 참고.
- **평가 지표·차트(권위 출처)**: 원본 `data/processed/evaluation/churn/<모델>/` (roc/pr curve, confusion, shap 등). RandomForest·LogReg는 평가 산출물이 아직 없습니다.
- **XGBoost 실험 변형**: [`churn/xgboost/runs/`](churn/xgboost/runs/) (cv_retune·venv_existing·venv_newcand) — 비교용 실험 런이며 서빙에는 쓰이지 않습니다.

---

## B. 세션 바운스 (Session Bounce) — 세션 중 이탈 실시간 예측

이커머스 세션 내 행동 시퀀스로 "이 세션에서 이탈할지"를 실시간 예측합니다.

| 모델 | 역할/특징 | 아카이브 경로 |
|------|------|------|
| **GRU** (SessionBounceGRU) | 주력 시퀀스 모델, val PR-AUC ≈ 0.603 | [`session_bounce/gru/model.pt`](session_bounce/gru/model.pt) (+ `model_config.json`, `category_index_map.json`) |
| **Transformer** | 시퀀스 비교 모델 (d=64·layers=2·heads=4) | [`session_bounce/transformer/model.pt`](session_bounce/transformer/model.pt) (+ `config.json`) |
| **LogReg baseline** | 세션바운스 베이스라인(경량) | [`sequence/session_bounce_model.joblib`](sequence/session_bounce_model.joblib) |

> 참고: [`sequence/transformer_meta.json`](sequence/transformer_meta.json)은 위 **Churn Transformer**의 메타데이터입니다(Y=churn). 폴더명이 `sequence`라 혼동 주의.

---

## C. 다음 카테고리 (Next Category)

유저의 최근 행동 시퀀스로 **다음에 관심 가질 카테고리**를 예측합니다(추천 보조 신호).

| 모델 | 역할/특징 | 아카이브 경로 |
|------|------|------|
| **CategoryGRU_v1** | seq_len=10·hidden=128 GRU, 선정지표 val_hit@4 | [`next_category/gru/model.pt`](next_category/gru/model.pt) (+ `model_config.json`, `category_index_map.json`) |

---

## D. 추천 (Recommendation)

장바구니/상품 추천을 담당합니다.

| 항목 | 역할/특징 | 아카이브 경로 |
|------|------|------|
| **CatBoost 추천 모델** | 추천 랭킹 모델 (≈23MB) | [`recommendation/catboost/CatBoost_rec_model.cbm`](recommendation/catboost/CatBoost_rec_model.cbm) (+ `CatBoost_rec_mapping.json`) |
| **cart_recommender** (로직) | 장바구니 기반 추천 로직 코드 | [`recommendation/cart_recommender.py`](recommendation/cart_recommender.py) |

---

## 📍 원본 ↔ 아카이브 경로 매핑

이 폴더의 모든 경로는 원본에서 `models/` → `model_archive/` 로 1:1 대응합니다.

| 아카이브 | 원본 (서비스가 읽는 경로) |
|------|------|
| `model_archive/churn/...` | `models/churn/...` |
| `model_archive/preprocessors/...` | `models/preprocessors/...` |
| `model_archive/session_bounce/...` | `models/session_bounce/...` |
| `model_archive/next_category/...` | `models/next_category/...` |
| `model_archive/recommendation/...` | `models/recommendation/...` |
| `model_archive/sequence/...` | `models/sequence/...` |

총 59개 파일 · 약 32MB.
