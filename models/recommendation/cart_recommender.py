from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "recommendation"


@dataclass(frozen=True)
class CartItem:
    product_id: int | None = None
    category_id: int | None = None
    quantity: int = 1


@dataclass(frozen=True)
class CategoryPrediction:
    category_id: int
    score: float
    base_category_id: int | None
    similarity: float | None
    reason: str


class CartRecommender:
    """Category-similarity based cart recommendation MVP.

    The source data has sparse category_code coverage, so this recommender uses
    category_id behavior similarity and falls back to popular products.
    """

    def __init__(self, data_dir: Path | str = DEFAULT_DATA_DIR) -> None:
        self.data_dir = Path(data_dir)
        self.product_catalog = pd.read_parquet(self.data_dir / "product_catalog.parquet")
        self.category_similar = pd.read_parquet(self.data_dir / "category_similar.parquet")
        self.category_catalog = pd.read_parquet(self.data_dir / "category_catalog.parquet")

        self.product_catalog["product_id"] = self.product_catalog["product_id"].astype("int64")
        self.product_catalog["category_id"] = self.product_catalog["category_id"].astype("int64")
        self.category_similar["category_id"] = self.category_similar["category_id"].astype("int64")
        self.category_similar["similar_category_id"] = self.category_similar[
            "similar_category_id"
        ].astype("int64")

        self._product_to_category = dict(
            zip(self.product_catalog["product_id"], self.product_catalog["category_id"])
        )
        self._category_meta = self.category_catalog.set_index("category_id").to_dict("index")

    def recommend(
        self,
        cart_items: Iterable[CartItem | dict],
        top_k: int = 10,
        similar_categories_per_base: int = 5,
        products_per_category: int = 30,
    ) -> list[dict]:
        items = [self._coerce_item(item) for item in cart_items]
        excluded_product_ids = {
            item.product_id for item in items if item.product_id is not None
        }
        category_weights = self._build_category_weights(items)

        if not category_weights:
            return self._popular_fallback(top_k, excluded_product_ids)

        candidate_categories = self._expand_categories(
            category_weights, similar_categories_per_base
        )
        candidates = self._rank_products(
            candidate_categories, excluded_product_ids, products_per_category
        )

        if len(candidates) < top_k:
            candidates.extend(
                self._popular_fallback(
                    top_k - len(candidates),
                    excluded_product_ids | {row["product_id"] for row in candidates},
                )
            )

        return candidates[:top_k]

    def recommend_from_category_sequence(
        self,
        category_sequence: Iterable[int],
        top_k: int = 10,
        predicted_categories: int = 5,
        products_per_category: int = 30,
    ) -> list[dict]:
        next_categories = self.predict_next_categories(
            category_sequence, top_k=predicted_categories
        )
        if not next_categories:
            return self._popular_fallback(top_k, set())

        candidate_categories = {
            prediction.category_id: {
                "score": prediction.score,
                "base_category_id": prediction.base_category_id,
                "similarity": prediction.similarity,
                "reason": prediction.reason,
            }
            for prediction in next_categories
        }
        candidates = self._rank_products(candidate_categories, set(), products_per_category)

        if len(candidates) < top_k:
            candidates.extend(
                self._popular_fallback(
                    top_k - len(candidates),
                    {row["product_id"] for row in candidates},
                )
            )

        return candidates[:top_k]

    def predict_next_categories(
        self,
        category_sequence: Iterable[int],
        top_k: int = 5,
        recency_decay: float = 0.7,
        similar_categories_per_base: int = 5,
    ) -> list[CategoryPrediction]:
        sequence = [int(category_id) for category_id in category_sequence]
        if not sequence:
            return []

        category_scores: dict[int, dict] = {}
        recent_sequence = sequence[-10:]
        for offset, base_category_id in enumerate(reversed(recent_sequence)):
            recency_weight = recency_decay**offset
            existing = category_scores.get(base_category_id)
            if existing is None or recency_weight > existing["score"]:
                category_scores[base_category_id] = {
                    "score": recency_weight,
                    "base_category_id": base_category_id,
                    "similarity": 1.0,
                    "reason": "sequence_last_category",
                }

            similar_rows = self.category_similar[
                self.category_similar["category_id"] == base_category_id
            ].nsmallest(similar_categories_per_base, "rank")
            for row in similar_rows.itertuples(index=False):
                category_id = int(row.similar_category_id)
                similarity = float(row.cosine)
                score = recency_weight * similarity
                existing = category_scores.get(category_id)
                if existing is None or score > existing["score"]:
                    category_scores[category_id] = {
                        "score": score,
                        "base_category_id": base_category_id,
                        "similarity": similarity,
                        "reason": "sequence_similar_category",
                    }

        predictions = [
            CategoryPrediction(
                category_id=category_id,
                score=round(float(info["score"]), 6),
                base_category_id=int(info["base_category_id"]),
                similarity=round(float(info["similarity"]), 6),
                reason=str(info["reason"]),
            )
            for category_id, info in category_scores.items()
        ]
        predictions.sort(key=lambda row: row.score, reverse=True)
        return predictions[:top_k]

    def _coerce_item(self, item: CartItem | dict) -> CartItem:
        if isinstance(item, CartItem):
            return item
        return CartItem(
            product_id=self._to_optional_int(item.get("product_id")),
            category_id=self._to_optional_int(item.get("category_id")),
            quantity=int(item.get("quantity", 1) or 1),
        )

    def _build_category_weights(self, items: list[CartItem]) -> dict[int, float]:
        weights: dict[int, float] = {}
        for item in items:
            category_id = item.category_id
            if category_id is None and item.product_id is not None:
                category_id = self._product_to_category.get(item.product_id)
            if category_id is None:
                continue
            weights[int(category_id)] = weights.get(int(category_id), 0.0) + max(
                item.quantity, 1
            )
        return weights

    def _expand_categories(
        self, category_weights: dict[int, float], similar_categories_per_base: int
    ) -> dict[int, dict]:
        candidates: dict[int, dict] = {}
        for base_category_id, base_weight in category_weights.items():
            candidates[base_category_id] = {
                "score": max(candidates.get(base_category_id, {}).get("score", 0.0), base_weight),
                "base_category_id": base_category_id,
                "similarity": 1.0,
                "reason": "same_category",
            }

            similar_rows = self.category_similar[
                self.category_similar["category_id"] == base_category_id
            ].nsmallest(similar_categories_per_base, "rank")

            for row in similar_rows.itertuples(index=False):
                category_id = int(row.similar_category_id)
                similarity = float(row.cosine)
                score = base_weight * similarity
                existing = candidates.get(category_id)
                if existing is None or score > existing["score"]:
                    candidates[category_id] = {
                        "score": score,
                        "base_category_id": base_category_id,
                        "similarity": similarity,
                        "reason": "similar_category",
                    }
        return candidates

    def _rank_products(
        self,
        candidate_categories: dict[int, dict],
        excluded_product_ids: set[int | None],
        products_per_category: int,
    ) -> list[dict]:
        rows: list[dict] = []
        for category_id, category_info in candidate_categories.items():
            products = (
                self.product_catalog[self.product_catalog["category_id"] == category_id]
                .sort_values("n_events", ascending=False)
                .head(products_per_category)
            )
            category_meta = self._category_meta.get(category_id, {})
            for product in products.itertuples(index=False):
                product_id = int(product.product_id)
                if product_id in excluded_product_ids:
                    continue
                popularity = float(product.n_events)
                score = float(category_info["score"]) * (1.0 + popularity / 1000.0)
                rows.append(
                    {
                        "product_id": product_id,
                        "category_id": category_id,
                        "base_category_id": self._to_optional_int(
                            category_info["base_category_id"]
                        ),
                        "rank_score": round(score, 6),
                        "similarity": self._to_optional_float(category_info["similarity"]),
                        "reason": category_info["reason"],
                        "price_median": self._to_optional_float(product.price_median),
                        "category_code": self._to_optional_str(
                            category_meta.get("category_code")
                        ),
                        "top_brand": self._to_optional_str(category_meta.get("top_brand")),
                        "n_events": int(product.n_events),
                    }
                )

        rows.sort(key=lambda row: (row["rank_score"], row["n_events"]), reverse=True)
        return self._dedupe_products(rows)

    def _popular_fallback(
        self, top_k: int, excluded_product_ids: set[int | None]
    ) -> list[dict]:
        products = self.product_catalog.sort_values("n_events", ascending=False)
        rows: list[dict] = []
        for product in products.itertuples(index=False):
            product_id = int(product.product_id)
            if product_id in excluded_product_ids:
                continue
            category_meta = self._category_meta.get(int(product.category_id), {})
            rows.append(
                {
                    "product_id": product_id,
                    "category_id": int(product.category_id),
                    "base_category_id": None,
                    "rank_score": round(1.0 + float(product.n_events) / 1000.0, 6),
                    "similarity": None,
                    "reason": "popular_fallback",
                    "price_median": self._to_optional_float(product.price_median),
                    "category_code": self._to_optional_str(
                        category_meta.get("category_code")
                    ),
                    "top_brand": self._to_optional_str(category_meta.get("top_brand")),
                    "n_events": int(product.n_events),
                }
            )
            if len(rows) >= top_k:
                break
        return rows

    @staticmethod
    def _dedupe_products(rows: list[dict]) -> list[dict]:
        seen: set[int] = set()
        deduped: list[dict] = []
        for row in rows:
            product_id = row["product_id"]
            if product_id in seen:
                continue
            seen.add(product_id)
            deduped.append(row)
        return deduped

    @staticmethod
    def _to_optional_int(value: object) -> int | None:
        if value is None or pd.isna(value):
            return None
        return int(value)

    @staticmethod
    def _to_optional_float(value: object) -> float | None:
        if value is None or pd.isna(value):
            return None
        return float(value)

    @staticmethod
    def _to_optional_str(value: object) -> str | None:
        if value is None or pd.isna(value):
            return None
        return str(value)
