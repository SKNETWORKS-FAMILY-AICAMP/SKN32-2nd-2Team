# v4 · CatBoost — 전처리 베이지안 결과 (인수인계용)

재현: `python preprocessing_project/v4_model_prep/src/models_bayes_one.py CatBoost` → `output/CatBoost/`.

## 1. 데이터 · X/Y
- 입력: `processed_5m/{train,test}_cohort_tabular.parquet` (코호트 recency≤7).
- X = 10피처(realtime-safe): recency_days, tenure_days, ndays, n_events, n_view, n_cart, n_remove_from_cart, n_purchase, avg_price, purch_amt
- Y = churn(7일 무활동). train 이탈률 82.2%.

## 2. 옵션별 best CV PR-AUC
- **scaler**: none 0.937 / robust 0.9356 / standard 0.9352
- **log_counts**: True 0.937 / False 0.9368
- **imbalance**: none 0.937 / classweight 0.9369 / smote 0.9356

## 3. 선택된 전처리 + HP
- 전처리: scaler=**none**, log1p=**True**, imbalance=**none**
- HP: {'iterations': 241, 'depth': 4, 'lr': 0.10769910202863792, 'l2_leaf_reg': 9.97731438146117}

## 4. 성능 (Feb 시간외삽)
- CV PR-AUC **0.937** | Feb PR-AUC 0.936(보정 0.9341) | **AUC 0.7902** | 임계값 0.52(F1 0.9181)
- base rate 0.8217 (PR-AUC는 양성다수라 높음 — AUC가 분별력 지표)

## 5. 산출물 · 서빙
- `prep_CatBoost.joblib`(전처리+모델+isotonic보정+임계값+feature_order), `CatBoost_bayes.json`, `CatBoost_{train,test}.parquet`(전처리본 백업).
- 서빙: 최근이벤트→10피처→(scaler/log)→calibrator.predict_proba→≥0.52. (`realtime_compat_test.py`)

## 6. 백엔드 제출(계약)
```json
{"model_name":"CatBoost_v4","model_type":"tree","artifact_path":"preprocessing_project/v4_model_prep/output/CatBoost/prep_CatBoost.joblib","preprocessing_config":{"scale":"none","log1p":true,"imbalance":"none","label":"churn","threshold":0.52,"calibrator":"isotonic"},"metrics":{"cv_pr_auc": 0.937, "oot_pr_auc": 0.936, "oot_pr_auc_cal": 0.9341, "oot_auc": 0.7902, "f1@thr": 0.9181, "base_rate": 0.8217}}
```