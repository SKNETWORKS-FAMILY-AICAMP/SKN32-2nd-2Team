from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from models.recommendation import CartItem, CartRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Return top-N cart-based product recommendations."
    )
    parser.add_argument(
        "--cart-json",
        type=Path,
        help="JSON file with a list of cart items. Each item can include product_id, category_id, quantity.",
    )
    parser.add_argument(
        "--product-id",
        action="append",
        type=int,
        default=[],
        help="Product id in cart. Can be passed multiple times.",
    )
    parser.add_argument(
        "--category-id",
        action="append",
        type=int,
        default=[],
        help="Category id in cart. Can be passed multiple times.",
    )
    parser.add_argument(
        "--category-sequence",
        action="append",
        type=int,
        default=[],
        help="Historical category id sequence ordered oldest to newest. Can be passed multiple times.",
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--predicted-categories", type=int, default=5)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/recommendation"),
    )
    return parser.parse_args()


def load_cart_items(args: argparse.Namespace) -> list[CartItem | dict]:
    items: list[CartItem | dict] = []
    if args.cart_json:
        payload = json.loads(args.cart_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("items", [])
        items.extend(payload)
    items.extend(CartItem(product_id=product_id) for product_id in args.product_id)
    items.extend(CartItem(category_id=category_id) for category_id in args.category_id)
    return items


def main() -> None:
    args = parse_args()
    recommender = CartRecommender(args.data_dir)
    if args.category_sequence:
        next_categories = recommender.predict_next_categories(
            args.category_sequence, top_k=args.predicted_categories
        )
        recommendations = recommender.recommend_from_category_sequence(
            args.category_sequence,
            top_k=args.top_k,
            predicted_categories=args.predicted_categories,
        )
        payload = {
            "next_categories": [category.__dict__ for category in next_categories],
            "recommendations": recommendations,
        }
    else:
        payload = {
            "recommendations": recommender.recommend(
                load_cart_items(args), top_k=args.top_k
            )
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
