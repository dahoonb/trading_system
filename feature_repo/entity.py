# feature_repo/entity.py
from feast import Entity, ValueType

# Define an entity for the ticker symbol. This is the primary entity for the system.
ticker = Entity(
    name="ticker",
    join_keys=["ticker_id"],
    value_type=ValueType.INT64,
    description="Stock ticker symbol (as an integer hash)",
)

# The 'method_entity' has been removed as it is not used in the final
# feature views or ML models. This keeps the repository definition clean.