-- Run schema_neon.sql first.
-- psql example from repository root:
--   psql "$DATABASE_URL" -f simulation_site/neon/schema_neon.sql
--   psql "$DATABASE_URL" -f simulation_site/neon/catalog_import.sql
--
-- The \copy command reads local CSV files and uploads them to Neon.

\copy sim_category_catalog(category_id, category_code, display_name, top_brand, price_median, n_products, n_events, price_sum, source_dataset) FROM 'simulation_site/neon/seed/categories.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy sim_brand_catalog(brand, n_products, n_categories, n_events, price_median, source_dataset) FROM 'simulation_site/neon/seed/brands.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy sim_product_catalog(product_id, category_id, brand, price_median, n_events, display_name, is_active, source_dataset) FROM 'simulation_site/neon/seed/products.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy sim_category_similarity(category_id, rank, similar_category_id, cosine, source_dataset) FROM 'simulation_site/neon/seed/category_similarity.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

