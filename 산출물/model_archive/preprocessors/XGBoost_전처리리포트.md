# v4 · XGBoost — 전처리 베이지안 결과 (인수인계용)

재현: `python preprocessing_project/v4_model_prep/src/models_bayes_one.py XGBoost` → `output/XGBoost/`.

## 1. 데이터 · X/Y
- 입력: `processed_5m/{train,test}_cohort_tabular.parquet` (코호트 recency≤7).
- X = 10피처(realtime-safe): recency_days, tenure_days, ndays, n_events, n_view, n_cart, n_remove_from_cart, n_purchase, avg_price, purch_amt
- Y = churn(7일 무활동). train 이탈률 82.2%.

## 2. 옵션별 best CV PR-AUC
- **scaler**: robust 0.937 / none 0.937 / standard 0.9368
- **log_counts**: False 0.937 / True 0.9366
- **imbalance**: classweight 0.937 / none 0.937 / smote 0.928

## 3. 선택된 전처리 + HP
- 전처리: scaler=**robust**, log1p=**False**, imbalance=**classweight**
- HP: {'n_estimators': 312, 'max_depth': 5, 'lr': 0.035906165718883755, 'subsample': 0.7752520669964538, 'colsample': 0.694827223643275, 'min_child_weight': 5, 'reg_alpha': 0.7094774061671105, 'reg_lambda': 0.0699226965154636, 'scale_pos_weight': 4}

## 4. 성능 (Feb 시간외삽)
- CV PR-AUC **0.937** | Feb PR-AUC 0.9363(보정 0.9344) | **AUC 0.7904** | 임계값 0.53(F1 0.9178)
- base rate 0.8217 (PR-AUC는 양성다수라 높음 — AUC가 분별력 지표)

## 5. 산출물 · 서빙
- `prep_XGBoost.joblib`(전처리+모델+isotonic보정+임계값+feature_order), `XGBoost_bayes.json`, `XGBoost_{train,test}.parquet`(전처리본 백업).
- 서빙: 최근이벤트→10피처→(scaler/log)→calibrator.predict_proba→≥0.53. (`realtime_compat_test.py`)

## 6. 백엔드 제출(계약)
```json
{"model_name":"XGBoost_v4","model_type":"tree","artifact_path":"preprocessing_project/v4_model_prep/output/XGBoost/prep_XGBoost.joblib","preprocessing_config":{"scale":"robust","log1p":false,"imbalance":"classweight","label":"churn","threshold":0.53,"calibrator":"isotonic"},"metrics":{"cv_pr_auc": 0.937, "oot_pr_auc": 0.9363, "oot_pr_auc_cal": 0.9344, "oot_auc": 0.7904, "f1@thr": 0.9178, "base_rate": 0.8217}}
```