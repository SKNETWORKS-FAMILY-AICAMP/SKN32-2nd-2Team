# Churn Model Backend Contract

## 1. Active model

- Model: `CatBoost_Churn_v2`
- Model key: `catboost`
- Task: 7-day churn prediction
- Status: active candidate

## 2. Metrics source

Path:

```text
data/processed/evaluation/churn/catboost/metrics_summary.json
```

Fields:

| field | type | example | note |
| --- | --- | --- | --- |
| `model_name` | string | `CatBoost_Churn_v2` | display/model registry name |
| `model_key` | string | `catboost` | backend lookup key |
| `label_name` | string | `churn` | target label |
| `horizon_days` | int | `7` | prediction horizon |
| `n_train` | int | `109378` | train users |
| `n_test` | int | `109736` | evaluated users |
| `positive_rate` | float | `0.833` | churn rate |
| `roc_auc` | float | `0.7914` | dashboard metric |
| `pr_auc` | float | `0.9364` | dashboard metric |
| `best_threshold` | float | `0.52` | threshold used for `y_pred` |
| `precision` | float | `0.8614` | dashboard metric |
| `recall` | float | `0.9830` | dashboard metric |
| `f1` | float | `0.9182` | dashboard metric |
| `brier` | float | `0.1104` | calibration metric |
| `ece` | float | `0.0239` | calibration metric |
| `confusion_matrix` | object | `{tn, fp, fn, tp}` | dashboard chart |

## 3. Prediction source

Path:

```text
data/processed/evaluation/churn/catboost/eval_predictions.parquet
```

Shape:

```text
109736 rows x 11 columns
```

Columns:

| field | type | note |
| --- | --- | --- |
| `user_id` | int64 | user key |
| `model_name` | string | currently `CatBoost` |
| `split` | string | currently `test` |
| `y_true` | int32 | actual churn label |
| `y_score` | float64 | churn probability/score |
| `y_pred` | int32 | thresholded prediction |
| `cohort_flag` | int64 | cohort marker |
| `revenue` | float32 | user revenue/value |
| `top_category` | int64 | user top category for recommendation/dashboard |
| `top_brand` | string | user top brand |
| `threshold` | float64 | threshold used for prediction |

Recommended DB/API split:

- `model_metrics`: store one row per model run from `metrics_summary.json`
- `model_predictions`: store user-level rows from `eval_predictions.parquet`
