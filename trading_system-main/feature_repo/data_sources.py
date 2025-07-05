# feature_repo/data_sources.py
import os
from feast import FileSource
from feast.data_format import ParquetFormat

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(REPO_DIR)

# Existing source for primary features
features_source = FileSource(
    name="features_source",
    path=os.path.join(PROJECT_ROOT, "data", "features.parquet"),
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="A file source for precomputed daily OHLCV and technical indicator features.",
)

# --- MODIFICATION: Add new source for TCA features ---
tca_features_source = FileSource(
    name="tca_features_source",
    path=os.path.join(PROJECT_ROOT, "data", "tca_features.parquet"),
    file_format=ParquetFormat(),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
    description="A file source for aggregated TCA metrics like average slippage.",
)