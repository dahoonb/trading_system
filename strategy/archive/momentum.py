# strategy/momentum.py

import pandas as pd
import numpy as np
import talib
from core.events import SignalEvent
from strategy.base import BaseStrategy

class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    An adaptive moving average crossover strategy with a multi-layered defense system.
    The core logic is now adaptive: it uses faster moving averages in low-volatility
    environments to be more responsive, and slower moving averages in high-volatility
    environments to reduce whipsaws.

    Defense System Layers:
    1.  **Market Regime Filter:** Only trades when the broad market (SPY) is in an uptrend,
        using both a slow MA and a fast Rate-of-Change (ROC) filter for crash protection.
    2.  **Volatility Filter:** Only trades when asset volatility is within a specified range.
    3.  **ADX Trend Strength Filter:** Only enters trades when the trend is strong.
    4.  **Trailing Stop-Loss:** A tactical exit rule to protect profits and limit drawdowns.
    5.  **Structural Exit:** The classic moving average "death cross" as a final exit signal,
        using the dynamically selected moving averages.
    """
    def __init__(self, event_queue, symbols, strategy_id: str,
                 # --- NEW: Adaptive MA Parameters ---
                 fast_ma_windows=(20, 50),
                 slow_ma_windows=(50, 200),
                 volatility_regime_period=100,
                 # --- Existing Filter Parameters ---
                 atr_period=20,
                 volatility_threshold_min=0.015,
                 volatility_threshold_max=0.05,
                 adx_period=14,
                 adx_threshold=25,
                 trailing_stop_atr_multiplier=2.5,
                 market_roc_period=21,
                 market_roc_threshold=-0.08,
                 state_file="state/momentum_ma_state.pkl"):
        """
        Initializes the adaptive momentum strategy.
        """
        super().__init__(event_queue, symbols, state_file)
        self.strategy_id = strategy_id

        # Store adaptive parameters
        self.fast_ma_windows = fast_ma_windows
        self.slow_ma_windows = slow_ma_windows
        self.volatility_regime_period = int(volatility_regime_period)

        # Store filter parameters
        self.atr_period = int(atr_period)
        self.volatility_threshold_min = float(volatility_threshold_min)
        self.volatility_threshold_max = float(volatility_threshold_max)
        self.adx_period = int(adx_period)
        self.adx_threshold = float(adx_threshold)
        self.trailing_stop_atr_multiplier = float(trailing_stop_atr_multiplier)
        self.market_roc_period = int(market_roc_period)
        self.market_roc_threshold = float(market_roc_threshold)

        # This dictionary will store the highest price reached since entering a position.
        self.position_high_water_mark = {s: 0.0 for s in self.symbols}

    def calculate_signals(self, event):
        """
        Processes a market event to generate trading signals based on a hierarchy of rules,
        including the new adaptive MA logic.
        """
        if event.type != 'MARKET' or event.symbol not in self.symbols:
            return

        symbol = event.symbol
        new_row = pd.DataFrame([event.data], index=[event.timestamp])
        self.data[symbol] = pd.concat([self.data[symbol], new_row])

        # --- DEFENSE LAYER 1: Ensure enough data for all calculations ---
        min_data_length = max(self.slow_ma_windows[1], self.volatility_regime_period * 2, self.adx_period * 2)
        if len(self.data[symbol]) <= min_data_length or ('SPY' in self.symbols and len(self.data['SPY']) <= 200):
            return

        # --- DEFENSE LAYER 2: Market Regime Filters (SPY) ---
        if 'SPY' in self.symbols:
            spy_close_prices = self.data['SPY']['close']
            spy_price = spy_close_prices.iloc[-1]

            # Fast ROC Crash Filter
            if len(spy_close_prices) > self.market_roc_period:
                spy_roc = spy_close_prices.pct_change(periods=self.market_roc_period).iloc[-1]
                if spy_roc < self.market_roc_threshold:
                    if self.signals.get(symbol) == 'LONG':
                        signal = SignalEvent(self.strategy_id, symbol, 'EXIT')
                        self.event_queue.put(signal)
                        self.signals[symbol] = 'EXIT'
                        self.position_high_water_mark[symbol] = 0.0
                    return # Exit immediately

            # Slow MA Trend Filter
            spy_long_ma = spy_close_prices.rolling(200).mean().iloc[-1]
            if spy_price < spy_long_ma:
                if self.signals.get(symbol) == 'LONG':
                    signal = SignalEvent(self.strategy_id, symbol, 'EXIT')
                    self.event_queue.put(signal)
                    self.signals[symbol] = 'EXIT'
                    self.position_high_water_mark[symbol] = 0.0
                return

        close_prices = self.data[symbol]['close']
        high_prices = self.data[symbol]['high']
        low_prices = self.data[symbol]['low']
        price = close_prices.iloc[-1]

        # --- DEFENSE LAYER 3: Volatility Filter ---
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.atr_period).iloc[-1]
        volatility_pct = atr / price if price > 0 else 0

        if not (self.volatility_threshold_min < volatility_pct < self.volatility_threshold_max):
            if self.signals.get(symbol) == 'LONG':
                signal = SignalEvent(self.strategy_id, symbol, 'EXIT')
                self.event_queue.put(signal)
                self.signals[symbol] = 'EXIT'
                self.position_high_water_mark[symbol] = 0.0
            return

        # --- DEFENSE LAYER 4: Tactical Trailing Stop-Loss (Highest Priority Exit) ---
        if self.signals.get(symbol) == 'LONG':
            self.position_high_water_mark[symbol] = max(self.position_high_water_mark[symbol], price)
            stop_price = self.position_high_water_mark[symbol] - (atr * self.trailing_stop_atr_multiplier)

            if price < stop_price:
                signal = SignalEvent(self.strategy_id, symbol, 'EXIT')
                self.event_queue.put(signal)
                self.signals[symbol] = 'EXIT'
                self.position_high_water_mark[symbol] = 0.0
                return

        # --- NEW: ADAPTIVE MA LOGIC ---
        log_returns = np.log(close_prices / close_prices.shift(1))
        current_vol = log_returns.rolling(window=self.volatility_regime_period).std().iloc[-1]
        long_term_avg_vol = log_returns.rolling(window=self.volatility_regime_period * 2).std().mean()

        if pd.isna(current_vol) or pd.isna(long_term_avg_vol):
            return # Not enough data for volatility regime calculation

        if current_vol < long_term_avg_vol:
            short_window, long_window = self.fast_ma_windows
        else:
            short_window, long_window = self.slow_ma_windows
        # --- END OF ADAPTIVE LOGIC ---

        # --- CORE SIGNAL LOGIC (using dynamic windows) ---
        short_ma = close_prices.rolling(window=short_window).mean().iloc[-1]
        long_ma = close_prices.rolling(window=long_window).mean().iloc[-1]

        # --- DEFENSE LAYER 5: ADX Trend Strength Filter (for ENTRY only) ---
        adx = talib.ADX(
            high_prices.to_numpy(dtype=float),
            low_prices.to_numpy(dtype=float),
            close_prices.to_numpy(dtype=float),
            timeperiod=self.adx_period
        )
        current_adx = adx[-1]

        # Entry Signal: Golden Cross AND Strong Trend (ADX > threshold)
        if short_ma > long_ma and current_adx > self.adx_threshold and self.signals.get(symbol) != 'LONG':
            signal = SignalEvent(self.strategy_id, symbol, 'LONG')
            self.event_queue.put(signal)
            self.signals[symbol] = 'LONG'
            self.position_high_water_mark[symbol] = price

        # Structural Exit Signal: Death Cross
        elif short_ma < long_ma and self.signals.get(symbol) == 'LONG':
            signal = SignalEvent(self.strategy_id, symbol, 'EXIT')
            self.event_queue.put(signal)
            self.signals[symbol] = 'EXIT'
            self.position_high_water_mark[symbol] = 0.0