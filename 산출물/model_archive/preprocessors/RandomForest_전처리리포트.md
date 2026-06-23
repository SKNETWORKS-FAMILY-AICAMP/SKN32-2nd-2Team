# v4 · RandomForest — 전처리 베이지안 결과 (인수인계용)

재현: `python preprocessing_project/v4_model_prep/src/models_bayes_one.py RandomForest` → `output/RandomForest/`.

## 1. 데이터 · X/Y
- 입력: `processed_5m/{train,test}_cohort_tabular.parquet` (코호트 recency≤7).
- X = 10피처(realtime-safe): recency_days, tenure_days, ndays, n_events, n_view, n_cart, n_remove_from_cart, n_purchase, avg_price, purch_amt
- Y = churn(7일 무활동). train 이탈률 82.2%.

## 2. 옵션별 best CV PR-AUC
- **scaler**: robust 0.9365 / none 0.9363 / standard 0.9356
- **log_counts**: True 0.9365 / False 0.9364
- **imbalance**: none 0.9365 / classweight 0.9364 / smote 0.9348

## 3. 선택된 전처리 + HP
- 전처리: scaler=**robust**, log1p=**True**, imbalance=**none**
- HP: {'n_estimators': 243, 'max_depth': 17, 'min_samples_leaf': 50, 'max_features': 'sqrt'}

## 4. 성능 (Feb 시간외삽)
- CV PR-AUC **0.9365** | Feb PR-AUC 0.9356(보정 0.9329) | **AUC 0.7892** | 임계값 0.54(F1 0.9178)
- base rate 0.8217 (PR-AUC는 양성다수라 높음 — AUC가 분별력 지표)

## 5. 산출물 · 서빙
- `prep_RandomForest.joblib`(전처리+모델+isotonic보정+임계값+feature_order), `RandomForest_bayes.json`, `RandomForest_{train,test}.parquet`(전처리본 백업).
- 서빙: 최근이벤트→10피처→(scaler/log)→calibrator.predict_proba→≥0.54. (`realtime_compat_test.py`)

## 6. 백엔드 제출(계약)
```json
{"model_name":"RandomForest_v4","model_type":"tree","artifact_path":"preprocessing_project/v4_model_prep/output/RandomForest/prep_RandomForest.joblib","preprocessing_config":{"scale":"robust","log1p":true,"imbalance":"none","label":"churn","threshold":0.54,"calibrator":"isotonic"},"metrics":{"cv_pr_auc": 0.9365, "oot_pr_auc": 0.9356, "oot_pr_auc_cal": 0.9329, "oot_auc": 0.7892, "f1@thr": 0.9178, "base_rate": 0.8217}}
```