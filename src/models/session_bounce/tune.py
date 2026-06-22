"""Session Bounce GRU 하이퍼파라미터 탐색 (Optuna/TPE, churn models_bayes.py 선례와 정합).

탐색 축(핵심 5종): hidden, layers, lr, dropout, loss(bce|bce_pos|focal[+gamma]).
고정: cat_emb=32, num_proj=16, weight_decay=1e-4, batch=512, scheduler=plateau, grad_clip=1.0, seq_len=10.
목적함수: validation PR-AUC 최대 (test 미사용, §3.4). trial은 속도 위해 epochs 단축.

종료 후 best 설정으로 gru_trainer.train()을 30에폭 재학습 → §3.6 산출물(model.pt/manifest 등) 저장.
탐색 로그: data/processed/evaluation/session_bounce/gru/tuning_report.json
"""
import argparse
import json

import numpy as np
import optuna

from src.models.session_bounce.common import EVAL_DIR, MODEL_DIR, load_dataset, split_masks
from src.models.session_bounce.gru_trainer import DEFAULTS, fit, train

SEED = 42


def make_objective(d, tr, va, n_cat, trial_epochs):
    def objective(trial):
        hp = dict(DEFAULTS)
        hp.update({
            "hidden": trial.suggest_categorical("hidden", [64, 128, 256]),
            "layers": trial.suggest_categorical("layers", [1, 2]),
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "dropout": trial.suggest_float("dropout", 0.1, 0.3),
            "loss": trial.suggest_categorical("loss", ["bce", "bce_pos", "focal"]),
            "patience": 3,
        })
        if hp["loss"] == "focal":
            hp["focal_gamma"] = trial.suggest_float("focal_gamma", 1.0, 3.0)
        _, _, best_pr, _ = fit(d, tr, va, n_cat, hp, max_epochs=trial_epochs, verbose=False)
        return best_pr
    return objective


def main(n_trials=30, trial_epochs=12, trial_subsample=1.0, final_epochs=30):
    d = load_dataset()
    tr, va, te = split_masks(d["split"])
    schema = json.loads((MODEL_DIR / "feature_schema.json").read_text(encoding="utf-8"))
    n_cat = schema["n_categories_incl_pad"]

    tr_use = tr
    if trial_subsample < 1.0:
        rng = np.random.default_rng(SEED)
        idx = np.flatnonzero(tr)
        keep = rng.choice(idx, size=int(len(idx) * trial_subsample), replace=False)
        tr_use = np.zeros_like(tr)
        tr_use[keep] = True
        print(f"[tune] trial train subsample {trial_subsample} -> {int(tr_use.sum()):,}")

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))

    def cb(study, trial):
        b = study.best_trial
        print(f"[tune] trial {trial.number:2d} PR-AUC={trial.value:.4f} "
              f"params={trial.params} | best={b.value:.4f}@{b.number}")

    study.optimize(make_objective(d, tr_use, va, n_cat, trial_epochs),
                   n_trials=n_trials, callbacks=[cb])

    best = dict(study.best_params)
    print(f"[tune] BEST val PR-AUC={study.best_value:.4f} params={best}")

    report = {
        "n_trials": n_trials, "trial_epochs": trial_epochs,
        "trial_subsample": trial_subsample, "metric": "val_pr_auc",
        "best_value": study.best_value, "best_params": best,
        "trials": [
            {"number": t.number, "value": t.value, "params": t.params,
             "state": str(t.state)}
            for t in study.trials
        ],
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "tuning_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # best 설정으로 전체(full train) 30에폭 재학습 → 최종 산출물 저장
    print("[tune] retraining best config on full train ...")
    summary = train(epochs=final_epochs, patience=4, **best)
    return study.best_value, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--trial-epochs", type=int, default=12)
    ap.add_argument("--trial-subsample", type=float, default=1.0)
    ap.add_argument("--final-epochs", type=int, default=30)
    a = ap.parse_args()
    main(a.n_trials, a.trial_epochs, a.trial_subsample, a.final_epochs)
