# Session Bounce GRU (Claude 담당)

계획서: `mdfile/GRU_분리모델_작업분담_및_통합계획.md` §3.

현재 행동 이후 **30분 내 추가 행동이 없을 확률**(`churn30` = 세션 바운스)을 이벤트 시퀀스로 예측한다. 장기 7일 이탈(XGBoost)과 별개 트랙.

## 결과 (test = 사용자 단위 group split, 179,163 시퀀스)

| 모델 | ROC-AUC | PR-AUC | Brier | F1 | 비고 |
| --- | ---: | ---: | ---: | ---: | --- |
| LogReg baseline (동일 split) | 0.7984 | 0.5288 | 0.1772 | 0.5719 | 시퀀스 평탄화 tabular |
| Session Bounce GRU v1 | 0.8282 | 0.5988 | 0.1233 | 0.5953 | hidden 64·1층·bce |
| **Session Bounce GRU (tuned)** | **0.8297** | **0.6013** | **0.1229** | **0.5964** | **채택** |

GRU가 전 지표에서 베이스라인을 능가 → 계획서 §3.1 기준 **운영 모델 채택**. 첫 이벤트 bounce recall 0.9997.

### 하이퍼파라미터 탐색 (Optuna/TPE 30 trials, GPU)

핵심 5축(hidden/layers/lr/dropout/loss) 탐색, 목적함수 = validation PR-AUC(test 미사용). LR 스케줄러(plateau)+grad clip(1.0) 기본 적용. 전 trial 로그: `data/processed/evaluation/session_bounce/gru/tuning_report.json`.

**best**: `hidden 64, layers 2, lr 1.08e-3, dropout 0.22, loss bce` (val PR-AUC 0.6033).

> v1 대비 개선폭은 소폭(전 지표 일관 ↑). recency 지배 구조상(보고서들과 정합) 큰 도약보단 안정적 개선. 추가 향상 여지: seq_len 20, 입력 채널 추가(step/누적), 확률 보정(isotonic), 시간 OOT.
> latency 1.3ms/1k는 이번 재학습이 **GPU**에서 측정된 값(metrics의 `cpu_latency_ms_per_1k` 필드명 주의). 실서빙 CPU 기준은 v1 측정치 ~6.5ms/1k 참고.

> 참고: 기존 `data/processed/session_bounce/meta.json`의 LogReg AUC 0.8118은 다른(소규모) 표본 기준이라 직접 비교 대상이 아니다. 공정 비교는 위 표(동일 dataset.npz·동일 split에서 재현한 LogReg 0.7984)로 한다.

## 파이프라인

```text
data2/clean/all_months_clean_core.csv  (5개월, 1,567만 이벤트)
  → preprocess.py  (유저 15만 표본, churn30 라벨 + censoring, 최근 10이벤트 시퀀스)
  → data/processed/session_bounce/gru/dataset.npz  (1,210,763 시퀀스)
  → baseline_logreg.py / gru_trainer.py  (동일 split)
  → models/session_bounce/gru/  +  data/processed/evaluation/session_bounce/{gru,logreg}/
```

### 재현

```bash
conda run -n venv python -m src.models.session_bounce.preprocess --n-users 150000 --cap-per-user 100
conda run -n venv python -m src.models.session_bounce.baseline_logreg
conda run -n venv python -m src.models.session_bounce.gru_trainer --epochs 30 --patience 4
```

### 실시간 추론

```python
from src.models.session_bounce.gru_inference import SessionBouncePredictor
p = SessionBouncePredictor()
p.predict([
    {"event_type": "view", "category_id": 1487580005134238553, "price": 5.24, "timestamp": 0},
    {"event_type": "cart", "category_id": 1487580005134238553, "price": 5.24, "timestamp": 40},
])  # -> {"session_bounce_probability": ...}
```

## 입력 스키마 (Codex/통합과 공유)

- `X_num (N,10,6)` = `[is_view, is_cart, is_remove, is_purchase, gap_log, price_log]` — next-category GRU와 동일 인코딩(§7)
- `X_cat (N,10)` = 카테고리 인덱스, **0 = padding** (`models/session_bounce/gru/category_index_map.json`)
- 라벨 `churn30`: 같은 사용자 다음 이벤트까지 간격 > 1800s(또는 없음). 관측 종료 30분 이내 미관측 구간은 censoring drop (§3.3).

## 산출물 (§3.6)

```text
src/models/session_bounce/{gru_model,gru_trainer,gru_inference}.py  (+ common.py, preprocess.py, baseline_logreg.py)
models/session_bounce/gru/{model.pt, model_config.json, feature_schema.json, category_index_map.json}
data/processed/evaluation/session_bounce/gru/{metrics_summary, eval_predictions.parquet, training_history, model_run_manifest}.json
```

## 통합 메모

- 서버 시작 시 `SessionBouncePredictor` 1회 로드 후 재사용(계획서 §8).
- 응답 키: `session_bounce_probability` (계획서 §6 통합 payload).
- 운영 threshold 0.3 (validation F1 기준). recall 우선(바운스 놓치지 않기) 성향이라 정책에서 조정 가능.
- 한계/후속: 데이터가 Oct 2019~Jan 2020 범위(clean_core에 Feb 부재). 시간 OOT 평가, 세션 한정 라벨 변형, 하이퍼파라미터 탐색은 후속 과제.
