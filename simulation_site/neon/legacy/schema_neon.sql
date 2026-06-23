-- Neon Postgres schema for the ecommerce simulation site.
-- Operational MySQL remains separate. Neon keeps simulation events and
-- lightweight REES46-derived product/category catalog tables.

CREATE TABLE IF NOT EXISTS sim_category_catalog (
  category_id TEXT PRIMARY KEY,
  category_code TEXT,
  display_name TEXT NOT NULL,
  top_brand TEXT,
  price_median NUMERIC(12, 2),
  n_products INTEGER DEFAULT 0,
  n_events BIGINT DEFAULT 0,
  price_sum NUMERIC(16, 2),
  source_dataset TEXT DEFAULT 'REES46 cosmetics 2019-Oct..2020-Feb',
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sim_brand_catalog (
  brand TEXT PRIMARY KEY,
  n_products INTEGER DEFAULT 0,
  n_categories INTEGER DEFAULT 0,
  n_events BIGINT DEFAULT 0,
  price_median NUMERIC(12, 2),
  source_dataset TEXT DEFAULT 'REES46 cosmetics 2019-Oct..2020-Feb',
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sim_product_catalog (
  product_id TEXT PRIMARY KEY,
  category_id TEXT REFERENCES sim_category_catalog(category_id),
  brand TEXT REFERENCES sim_brand_catalog(brand),
  price_median NUMERIC(12, 2),
  n_events BIGINT DEFAULT 0,
  display_name TEXT NOT NULL,
  is_active BOOLEAN DEFAULT true,
  source_dataset TEXT DEFAULT 'REES46 cosmetics 2019-Oct..2020-Feb',
  updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS sim_category_similarity (
  category_id TEXT REFERENCES sim_category_catalog(category_id),
  rank INTEGER NOT NULL,
  similar_category_id TEXT REFERENCES sim_category_catalog(category_id),
  cosine DOUBLE PRECISION,
  source_dataset TEXT DEFAULT 'REES46 cosmetics 2019-Oct..2020-Feb',
  updated_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (category_id, rank)
);

CREATE TABLE IF NOT EXISTS sim_event_log (
  event_id TEXT PRIMARY KEY,
  user_id TEXT NOT NULL,
  session_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  event_time TIMESTAMPTZ NOT NULL DEFAULT now(),
  product_id TEXT REFERENCES sim_product_catalog(product_id),
  category_id TEXT REFERENCES sim_category_catalog(category_id),
  brand TEXT REFERENCES sim_brand_catalog(brand),
  price NUMERIC(12, 2),
  quantity INTEGER DEFAULT 1,
  page_url TEXT,
  referrer TEXT,
  device_type TEXT,
  payload_json JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_sim_event_user_time
  ON sim_event_log(user_id, event_time DESC);

CREATE INDEX IF NOT EXISTS idx_sim_event_session_time
  ON sim_event_log(session_id, event_time);

CREATE INDEX IF NOT EXISTS idx_sim_product_category
  ON sim_product_catalog(category_id);

CREATE INDEX IF NOT EXISTS idx_sim_product_brand
  ON sim_product_catalog(brand);

