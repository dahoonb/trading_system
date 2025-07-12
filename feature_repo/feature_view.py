# feature_repo/feature_view.py

import sys
import os
from datetime import timedelta

from feast import FeatureView, Field, FileSource
from feast.data_format import ParquetFormat
from feast.value_type import ValueType
from feast.types import from_value_type

# --- START: ADD THIS BLOCK TO FIX THE IMPORT PATH ---
# This block makes the script self-aware of its location and adds the project root to the path.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# --- END: ADD THIS BLOCK ---

from feature_repo.entity import ticker

# --- DEFINITION 1: Define DataSources directly in this file ---
features_source = FileSource(
    path=os.path.join(PROJECT_ROOT, "data", "features.parquet"),
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="A file source for precomputed daily OHLCV and technical indicator features.",
)

tca_features_source = FileSource(
    path=os.path.join(PROJECT_ROOT, "data", "tca_features.parquet"),
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="A file source for aggregated TCA metrics like average slippage.",
)

# --- DEFINITION 2: Define FeatureViews that use the sources above ---
all_ticker_features_fv = FeatureView(
    name="all_ticker_features",
    entities=[ticker],
    ttl=timedelta(days=5),
    online=True,
    source=features_source, # Use the source defined above
    tags={"source": "precomputed_etl"},
    schema=[
        Field(name="open", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="high", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="low", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="close", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="volume", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="sma_20", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="sma_200", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="rsi_14", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="atr_20", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="realized_vol_20d", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="oer_5d", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="vol_regime", dtype=from_value_type(ValueType.INT64)),
        # FIX: Add new cross-sectional rank features to the view
        Field(name="cs_rank_21d", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="cs_rank_63d", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="cs_rank_126d", dtype=from_value_type(ValueType.DOUBLE)),
        Field(name="cs_rank_252d", dtype=from_value_type(ValueType.DOUBLE)),
    ],
)

tca_features_fv = FeatureView(
    name="tca_features",
    entities=[ticker],
    ttl=timedelta(days=10),
    online=True,
    source=tca_features_source, # Use the source defined above
    tags={"source": "tca_etl"},
    schema=[
        Field(name="avg_slippage_5d", dtype=from_value_type(ValueType.DOUBLE)),
    ],
)