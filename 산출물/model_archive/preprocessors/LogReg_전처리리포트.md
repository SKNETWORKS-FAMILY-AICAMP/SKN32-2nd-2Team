# v4 · LogReg — 전처리 베이지안 결과 (인수인계용)

재현: `python preprocessing_project/v4_model_prep/src/models_bayes_one.py LogReg` → `output/LogReg/`.

## 1. 데이터 · X/Y
- 입력: `processed_5m/{train,test}_cohort_tabular.parquet` (코호트 recency≤7).
- X = 10피처(realtime-safe): recency_days, tenure_days, ndays, n_events, n_view, n_cart, n_remove_from_cart, n_purchase, avg_price, purch_amt
- Y = churn(7일 무활동). train 이탈률 82.2%.

## 2. 옵션별 best CV PR-AUC
- **scaler**: minmax 0.9356 / robust 0.9356 / standard 0.9356
- **log_counts**: True 0.9356 / False 0.9253
- **imbalance**: classweight 0.9356 / smote 0.9354 / none 0.9352

## 3. 선택된 전처리 + HP
- 전처리: scaler=**minmax**, log1p=**True**, imbalance=**classweight**
- HP: {'C': 13.921548533046511}

## 4. 성능 (Feb 시간외삽)
- CV PR-AUC **0.9356** | Feb PR-AUC 0.9344(보정 0.9324) | **AUC 0.786** | 임계값 0.59(F1 0.9176)
- base rate 0.8217 (PR-AUC는 양성다수라 높음 — AUC가 분별력 지표)

## 5. 산출물 · 서빙
- `prep_LogReg.joblib`(전처리+모델+isotonic보정+임계값+feature_order), `LogReg_bayes.json`, `LogReg_{train,test}.parquet`(전처리본 백업).
- 서빙: 최근이벤트→10피처→(scaler/log)→calibrator.predict_proba→≥0.59. (`realtime_compat_test.py`)

## 6. 백엔드 제출(계약)
```json
{"model_name":"LogReg_v4","model_type":"linear","artifact_path":"preprocessing_project/v4_model_prep/output/LogReg/prep_LogReg.joblib","preprocessing_config":{"scale":"minmax","log1p":true,"imbalance":"classweight","label":"churn","threshold":0.59,"calibrator":"isotonic"},"metrics":{"cv_pr_auc": 0.9356, "oot_pr_auc": 0.9344, "oot_pr_auc_cal": 0.9324, "oot_auc": 0.786, "f1@thr": 0.9176, "base_rate": 0.8217}}
```