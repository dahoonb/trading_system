# feature_repo/fundamental_features.py

import os
from datetime import timedelta
from feast import FeatureView, Field, FileSource, ValueType
from feast.data_format import ParquetFormat
from feast.types import from_value_type

# --- Project Root Setup ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from feature_repo.entity import ticker

# --- Data Source Definition ---
fundamental_features_source = FileSource(
    path=os.path.join(PROJECT_ROOT, "data", "fundamental_features.parquet"),
    # --- CHANGE 2: Add the file_format parameter ---
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="A file source for daily-refreshed fundamental stock features.",
)

# --- Feature View Definition ---
fundamental_features_fv = FeatureView(
    name="fundamental_features",
    entities=[ticker],
    ttl=timedelta(days=2), # Data is refreshed daily, so a short TTL is appropriate
    online=True,
    source=fundamental_features_source,
    tags={"source": "fundamental_etl"},
    schema=[
        Field(name="price_to_book", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="price_to_earnings", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="dividend_yield", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="buyback_yield", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="shareholder_yield", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="piotroski_f_score", dtype=from_value_type(ValueType.INT64)),
        Field(name="roa", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="cfo", dtype=from_value_type(ValueType.DOUBLE)),
    ],
)