# feature_repo/entity.py
from feast import Entity, ValueType

# Define an entity for the ticker symbol. This is the primary entity for the system.
ticker = Entity(
    name="ticker",
    join_keys=["ticker_id"], # This must match the column name in your data sources
    value_type=ValueType.STRING,
    description="Stock ticker symbol",
)

# The 'method_entity' has been removed as it is not used in the final
# feature views or ML models. This keeps the repository definition clean.