# 가지마 데이터 카드 (운영 정본) — 모델/대시보드 담당 인수용

운영 정본은 `data/processed/` 단일 루트. 대용량(parquet/npz)은 파일, DB엔 경로만. 한 장으로 "어디서 뭘 쓰는지" 정리.

## 폴더 지도
```
data/processed/
├── churn/             이탈 모델 입력 정본(향후 7일 이탈)
├── evaluation/        검증 산출물(대시보드 15시각화 데이터원)
├── recommendation/    추천(유사 카테고리·카탈로그)
├── session_bounce/    실시간 세션/바운스(churn30) 샘플·메타
└── (canonical_5m, next_category)  추가 예정
```

## churn/ — 이탈 모델 입력
| 파일 | 내용 | 쓰는 법 |
| --- | --- | --- |
| `train_tabular_v2.parquet` | 전체 1.27M 유저 × **27피처**(10정본 + category/brand/세션/remove/price 파생) + `churn`·`cohort_recency7` | 모델 학습 입력. 코호트만 쓰려면 `cohort_recency7==1` 필터 |
| `test_tabular_v2.parquet` | Feb 시간외삽 test | 평가용 |
- **X**: recency_days·tenure_days·ndays·n_events·n_view·n_cart·n_remove_from_cart·n_purchase·avg_price·purch_amt + (min/max/std_price·purchase_avg_price·remove_ratio·cart_purchase_ratio·n_categories·cat_entropy·n_brands·brand_loyalty·top_brand·top_category_id·n_sessions·events_per_session·last_*)
- **Y**: `churn`(예측 시점 이후 **7일 무이벤트**=1). 분할: train(2019-10~2020-01-25) / test(Feb).
- 모델별 전처리기: `../../models/preprocessors/prep_{Model}_v2.joblib` + `{Model}_전처리리포트.md`(기술선택·이유·컬럼).

## evaluation/ — 대시보드 15시각화 데이터원
| 파일 | 내용 |
| --- | --- |
| `eval_predictions.parquet` | user×model: y_true·y_score·y_pred·cohort·revenue·top_category·top_brand (8·10·11·14·15 계산) |
| `metrics_summary.json` | 모델별 ROC-AUC·PR-AUC·Brier·ECE·F1·threshold (개요 비교) |
| `curves.json` | 모델별 roc·pr·threshold·calibration |
| (shap_summary.json) | `pip install shap` 후 pp_eval_package 재실행 시 |
- 재현: `team_project_churn/preprocessing_project/v4_model_prep/src/pp_eval_package.py`

## recommendation/ — 유사 카테고리 추천
`category_similar.parquet`(category→유사 top10 cosine) · `category_catalog.parquet`(525 카테고리·top_brand·price) · `product_catalog.parquet`(54,571) · `category_code_map.csv`(이름 18개).
→ 유저 top_category → 유사 카테고리 → 인기상품 추천. (category_code 96.6% 결측이라 **id+행동유사도**로 추천)

## session_bounce/ — 실시간 세션/바운스(SB)
`sample_sessions.json`(바운스/장바구니이탈/구매 세션, 행동별 bounce_prob) · `meta.json`(churn30 라벨, AUC 0.8118).
→ 집계 churn(7일)이 못 잡는 **단발·즉시 이탈**을 세션 진행상태로 실시간 채점. 모델 `../../models/sequence/session_bounce_model.joblib`.

## 모델 담당 빠른 시작
1. `churn/train_tabular_v2.parquet` 로드 → `models/preprocessors/prep_{Model}_v2.joblib`로 transform → 학습.
2. 평가는 `pp_eval_package.py`로 eval_predictions/metrics/curves 생성 → 백엔드 `POST /models/submit`.
3. 대시보드는 evaluation/ 산출물을 읽어 15시각화 표시(현재) 또는 backend chart API 경유.
