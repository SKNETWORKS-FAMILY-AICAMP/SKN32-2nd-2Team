# Cart Recommendation Contract

## Goal
Return top-N product recommendations from current cart context.

## Data Source
- `data/processed/recommendation/product_catalog.parquet`
- `data/processed/recommendation/category_similar.parquet`
- `data/processed/recommendation/category_catalog.parquet`
- `data/processed/recommendation/category_code_map.csv`

## Request
```json
{
  "items": [
    {
      "product_id": 3762,
      "category_id": 1487580005411062629,
      "quantity": 1
    }
  ],
  "top_k": 10
}
```

`product_id` or `category_id` can be omitted when the other value exists.

Sequence-aware recommendation can pass historical category IDs ordered oldest to newest.

```bash
python scripts/recommend_cart_products.py \
  --category-sequence 1487580005411062629 \
  --category-sequence 1487580005595612013 \
  --top-k 10
```

## Response
```json
{
  "recommendations": [
    {
      "product_id": 4185,
      "category_id": 1487580005411062629,
      "base_category_id": 1487580005411062629,
      "rank_score": 8.068,
      "similarity": 1.0,
      "reason": "same_category",
      "price_median": 19.33,
      "category_code": null,
      "top_brand": "cnd",
      "n_events": 7068
    }
  ]
}
```

## Ranking
1. Resolve cart product IDs to category IDs.
2. Weight categories by cart quantity.
3. Expand to similar categories using category cosine similarity.
4. Rank popular products in candidate categories.
5. Exclude products already present in the cart.
6. Fall back to globally popular products when cart context is empty or sparse.

## Sequence-aware Ranking
1. Read historical `category_id` events in order.
2. Predict next-interest categories with a recency baseline:
   - most recent category gets the highest score,
   - older categories decay by recency,
   - similar categories are added through cosine similarity.
3. Rank popular products inside predicted categories.
4. Return both `next_categories` and product `recommendations`.

This keeps the interface ready for a GRU/SASRec next-category model: the model can
replace the recency baseline while the downstream product ranking stays the same.
