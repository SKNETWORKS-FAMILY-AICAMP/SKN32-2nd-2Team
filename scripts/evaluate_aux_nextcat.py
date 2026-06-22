from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate next-category baselines on aux_nextcat npz datasets."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("aux_nextcat/output/nextcat_dataset.npz"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("aux_nextcat/output/nextcat_baseline_metrics.json"),
    )
    return parser.parse_args()


def topk_metrics(y_true: np.ndarray, topk: np.ndarray) -> dict[str, float]:
    hits = topk == y_true[:, None]
    top1 = float(hits[:, 0].mean())
    hitk = float(hits.any(axis=1).mean())

    reciprocal_ranks = np.zeros(len(y_true), dtype=np.float32)
    hit_rows, hit_cols = np.where(hits)
    reciprocal_ranks[hit_rows] = 1.0 / (hit_cols + 1)
    mrr = float(reciprocal_ranks.mean())
    return {"top1": round(top1, 4), "hit@10": round(hitk, 4), "mrr@10": round(mrr, 4)}


def popularity_topk(y_train: np.ndarray, n_rows: int, top_k: int) -> np.ndarray:
    counts = Counter(int(label) for label in y_train if int(label) != 0)
    ranked = [label for label, _ in counts.most_common(top_k)]
    return np.tile(np.array(ranked, dtype=np.int64), (n_rows, 1))


def last_category_topk(
    x_test: np.ndarray, y_train: np.ndarray, top_k: int
) -> np.ndarray:
    global_popular = popularity_topk(y_train, 1, top_k).ravel().tolist()
    rows: list[list[int]] = []
    for sequence in x_test:
        ranked: list[int] = []
        for category_id in reversed(sequence.tolist()):
            category_id = int(category_id)
            if category_id == 0 or category_id in ranked:
                continue
            ranked.append(category_id)
            if len(ranked) >= top_k:
                break
        for category_id in global_popular:
            if category_id not in ranked:
                ranked.append(category_id)
            if len(ranked) >= top_k:
                break
        rows.append(ranked[:top_k])
    return np.array(rows, dtype=np.int64)


def main() -> None:
    args = parse_args()
    data = np.load(args.dataset)
    x_cat = data["X_cat"]
    y = data["y"]
    is_train = data["is_train"].astype(bool)

    x_test = x_cat[~is_train]
    y_train = y[is_train]
    y_test = y[~is_train]

    popularity = popularity_topk(y_train, len(y_test), args.top_k)
    last_category = last_category_topk(x_test, y_train, args.top_k)

    payload = {
        "dataset": str(args.dataset),
        "n_train": int(is_train.sum()),
        "n_test": int((~is_train).sum()),
        "top_k": args.top_k,
        "models": {
            "Popularity": topk_metrics(y_test, popularity),
            "Last-category": topk_metrics(y_test, last_category),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
