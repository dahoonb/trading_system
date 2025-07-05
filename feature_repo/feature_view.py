# feature_repo/feature_view.py
from datetime import timedelta
from feast import FeatureView, Field
from feast.types import Float64, Int64

from entity import ticker
from data_sources import features_source, tca_features_source  # Import new source

# Existing feature view for primary features
all_ticker_features_fv = FeatureView(
    name="all_ticker_features",
    entities=[ticker],
    ttl=timedelta(days=5),
    schema=[
        Field(name="open", dtype=Float64),
        Field(name="high", dtype=Float64),
        Field(name="low", dtype=Float64),
        Field(name="close", dtype=Float64),
        Field(name="volume", dtype=Float64),
        Field(name="sma_20", dtype=Float64),
        Field(name="sma_200", dtype=Float64),
        Field(name="rsi_14", dtype=Float64),
        Field(name="atr_20", dtype=Float64),
        Field(name="realized_vol_20d", dtype=Float64),
        Field(name="oer_5d", dtype=Float64),
        Field(name="vol_regime", dtype=Int64),
        Field(name="vix_regime_5_state", dtype=Int64),
    ],
    online=True,
    source=features_source,
    tags={"source": "precomputed_etl"},
)

# --- MODIFICATION: Add new feature view for TCA features ---
tca_features_fv = FeatureView(
    name="tca_features",
    entities=[ticker],
    ttl=timedelta(days=10), # Longer TTL as these features might be less frequent
    schema=[
        Field(name="avg_slippage_5d", dtype=Float64),
    ],
    online=True,
    source=tca_features_source,
    tags={"source": "tca_etl"},
)