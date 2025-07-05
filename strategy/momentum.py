# strategy/momentum.py

import pandas as pd
from core.events import SignalEvent
from strategy.base import BaseStrategy

class MovingAverageCrossoverStrategy(BaseStrategy):
    """A moving average crossover strategy."""
    def __init__(self, event_queue, symbols, strategy_id: str, short_window=50, long_window=200, state_file="state/momentum_ma_state.pkl"):
        super().__init__(event_queue, symbols, state_file)
        # --- MODIFICATION: Store strategy_id ---
        self.strategy_id = strategy_id
        self.short_window = short_window
        self.long_window = long_window

    def calculate_signals(self, event):
        if event.type != 'MARKET' or event.symbol not in self.symbols:
            return
            
        symbol = event.symbol
        new_row = pd.DataFrame([event.data], index=[event.timestamp])
        self.data[symbol] = pd.concat([self.data[symbol], new_row])

        if len(self.data[symbol]) > self.long_window:
            close_prices = self.data[symbol]['close']
            short_ma = close_prices.rolling(window=self.short_window).mean().iloc[-1]
            long_ma = close_prices.rolling(window=self.long_window).mean().iloc[-1]

            if short_ma > long_ma and self.signals[symbol] != 'LONG':
                # --- MODIFICATION: Pass strategy_id to SignalEvent ---
                signal = SignalEvent(self.strategy_id, symbol, 'LONG')
                self.event_queue.put(signal)
                self.signals[symbol] = 'LONG'
            elif short_ma < long_ma and self.signals[symbol] != 'SHORT':
                # --- MODIFICATION: Pass strategy_id to SignalEvent ---
                signal = SignalEvent(self.strategy_id, symbol, 'SHORT')
                self.event_queue.put(signal)
                self.signals[symbol] = 'SHORT'