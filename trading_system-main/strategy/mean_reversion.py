# strategy/mean_reversion.py

import pandas as pd
from core.events import SignalEvent
from strategy.base import BaseStrategy

class RsiMeanReversionStrategy(BaseStrategy):
    """An RSI-based mean reversion strategy."""
    def __init__(self, event_queue, symbols, strategy_id: str, rsi_period=14, overbought=70, oversold=30, state_file="state/mean_reversion_rsi_state.pkl"):
        super().__init__(event_queue, symbols, state_file)
        # --- MODIFICATION: Store strategy_id ---
        self.strategy_id = strategy_id
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    def calculate_signals(self, event):
        if event.type != 'MARKET' or event.symbol not in self.symbols:
            return

        symbol = event.symbol
        new_row = pd.DataFrame([event.data], index=[event.timestamp])
        self.data[symbol] = pd.concat([self.data[symbol], new_row])

        if len(self.data[symbol]) > self.rsi_period:
            close_prices = self.data[symbol]['close']
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            
            rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else float('inf')
            rsi = 100 - (100 / (1 + rs))

            if rsi > self.overbought and self.signals[symbol] != 'SHORT':
                # --- MODIFICATION: Pass strategy_id to SignalEvent ---
                signal = SignalEvent(self.strategy_id, symbol, 'SHORT')
                self.event_queue.put(signal)
                self.signals[symbol] = 'SHORT'
            elif rsi < self.oversold and self.signals[symbol] != 'LONG':
                # --- MODIFICATION: Pass strategy_id to SignalEvent ---
                signal = SignalEvent(self.strategy_id, symbol, 'LONG')
                self.event_queue.put(signal)
                self.signals[symbol] = 'LONG'