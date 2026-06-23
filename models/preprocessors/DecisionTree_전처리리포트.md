# v4 · DecisionTree — 전처리 베이지안 결과 (인수인계용)

재현: `python preprocessing_project/v4_model_prep/src/models_bayes_one.py DecisionTree` → `output/DecisionTree/`.

## 1. 데이터 · X/Y
- 입력: `processed_5m/{train,test}_cohort_tabular.parquet` (코호트 recency≤7).
- X = 10피처(realtime-safe): recency_days, tenure_days, ndays, n_events, n_view, n_cart, n_remove_from_cart, n_purchase, avg_price, purch_amt
- Y = churn(7일 무활동). train 이탈률 82.2%.

## 2. 옵션별 best CV PR-AUC
- **scaler**: none 0.9288 / standard 0.9283 / robust 0.9233
- **log_counts**: False 0.9288 / True 0.9283
- **imbalance**: classweight 0.9288 / smote 0.9283 / none 0.9161

## 3. 선택된 전처리 + HP
- 전처리: scaler=**none**, log1p=**False**, imbalance=**classweight**
- HP: {'max_depth': 12, 'min_samples_leaf': 35, 'ccp_alpha': 0.0005083825348819038}

## 4. 성능 (Feb 시간외삽)
- CV PR-AUC **0.9288** | Feb PR-AUC 0.9279(보정 0.9272) | **AUC 0.7773** | 임계값 0.42(F1 0.9137)
- base rate 0.8217 (PR-AUC는 양성다수라 높음 — AUC가 분별력 지표)

## 5. 산출물 · 서빙
- `prep_DecisionTree.joblib`(전처리+모델+isotonic보정+임계값+feature_order), `DecisionTree_bayes.json`, `DecisionTree_{train,test}.parquet`(전처리본 백업).
- 서빙: 최근이벤트→10피처→(scaler/log)→calibrator.predict_proba→≥0.42. (`realtime_compat_test.py`)

## 6. 백엔드 제출(계약)
```json
{"model_name":"DecisionTree_v4","model_type":"tree","artifact_path":"preprocessing_project/v4_model_prep/output/DecisionTree/prep_DecisionTree.joblib","preprocessing_config":{"scale":"none","log1p":false,"imbalance":"classweight","label":"churn","threshold":0.42,"calibrator":"isotonic"},"metrics":{"cv_pr_auc": 0.9288, "oot_pr_auc": 0.9279, "oot_pr_auc_cal": 0.9272, "oot_auc": 0.7773, "f1@thr": 0.9137, "base_rate": 0.8217}}
```