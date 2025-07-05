# core/events.py

from datetime import datetime
from typing import Optional

class Event:
    """Base class for all event objects."""
    pass

class MarketEvent(Event):
    """Handles the event of receiving a new market update."""
    def __init__(self, symbol: str, timestamp: datetime, data: dict):
        self.type = 'MARKET'
        self.symbol = symbol
        self.timestamp = timestamp
        self.data = data

class SignalEvent(Event):
    """Handles the event of a Strategy object generating a trading signal."""
    def __init__(self, strategy_id: str, symbol: str, direction: str, strength: float = 1.0):
        self.type = 'SIGNAL'
        # --- MODIFICATION: Added strategy_id ---
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.direction = direction
        self.strength = strength

class OrderEvent(Event):
    """Handles the event of sending an Order to an execution system."""
    def __init__(self, strategy_id: str, symbol: str, order_type: str, quantity: int, direction: str, arrival_price: float = 0.0):
        self.type = 'ORDER'
        # --- MODIFICATION: Added strategy_id ---
        self.strategy_id = strategy_id
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.arrival_price = arrival_price

class FillEvent(Event):
    """Encapsulates the information of a filled order from the brokerage."""
    def __init__(self, timestamp: datetime, symbol: str, exchange: str,
                 quantity: int, direction: str, fill_cost: float, commission: float = 0.0,
                 strategy_id: Optional[str] = None):
        self.type = 'FILL'
        self.timestamp = timestamp
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = commission
        # --- MODIFICATION: Added strategy_id ---
        self.strategy_id = strategy_id

    @property
    def average_price(self) -> float:
        if self.quantity == 0: return 0.0
        return self.fill_cost / self.quantity