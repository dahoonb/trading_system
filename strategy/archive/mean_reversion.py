# strategy/mean_reversion.py

import pandas as pd
import talib
from core.events import SignalEvent
from strategy.base import BaseStrategy

class RsiMeanReversionStrategy(BaseStrategy):
    """
    An RSI-based mean reversion strategy.
    
    MODIFIED to be a PURE COUNTER-TREND strategy by removing the long-term
    trend filter. This is intended to provide true diversification to the
    portfolio, especially during market downturns where the momentum strategy
    is expected to be inactive or losing.

    It still maintains its own defensive layers:
    1.  **Volatility Filter:** Avoids trading in excessively calm or chaotic markets.
    2.  **ATR Stop-Loss:** A hard, volatility-adjusted stop to cut losses on failed reversions.
    3.  **RSI-based Exits:** Sells into strength when the instrument becomes overbought.
    """
    def __init__(self, event_queue, symbols, strategy_id: str,
                 rsi_period=14, overbought_threshold=70, oversold_threshold=30,
                 long_ma_period=200, # This parameter is no longer used but kept for API consistency
                 atr_period=20,
                 volatility_threshold_min=0.01, volatility_threshold_max=0.05,
                 stop_loss_atr_multiplier=2.0,
                 state_file="state/mean_reversion_rsi_state.pkl"):
        """
        Initializes the mean reversion strategy.

        Args:
            event_queue: The system's main event queue.
            symbols (list): A list of symbol strings.
            strategy_id (str): A unique identifier for this strategy instance.
            rsi_period (int): The lookback period for the Relative Strength Index (RSI).
            overbought_threshold (int): The RSI level above which an asset is considered overbought.
            oversold_threshold (int): The RSI level below which an asset is considered oversold.
            long_ma_period (int): No longer used. Kept for config compatibility.
            atr_period (int): The lookback period for the Average True Range (ATR) calculation.
            volatility_threshold_min (float): The minimum ATR/price volatility to allow trades.
            volatility_threshold_max (float): The maximum ATR/price volatility to allow trades.
            stop_loss_atr_multiplier (float): The multiple of ATR to set the stop-loss below entry.
            state_file (str): Path to save/load strategy state.
        """
        super().__init__(event_queue, symbols, state_file)
        self.strategy_id = strategy_id
        self.rsi_period = rsi_period
        self.overbought = overbought_threshold
        self.oversold = oversold_threshold
        # self.long_ma_period = long_ma_period # No longer used
        self.atr_period = atr_period
        self.volatility_threshold_min = volatility_threshold_min
        self.volatility_threshold_max = volatility_threshold_max
        self.stop_loss_atr_multiplier = stop_loss_atr_multiplier

        # State tracking dictionaries
        self.previous_rsi = {s: None for s in self.symbols}
        self.entry_price = {s: 0.0 for s in self.symbols}

    def calculate_signals(self, event):
        """
        Processes a market event to generate trading signals based on a hierarchy of rules.
        """
        if event.type != 'MARKET' or event.symbol not in self.symbols:
            return

        symbol = event.symbol
        new_row = pd.DataFrame([event.data], index=[event.timestamp])
        self.data[symbol] = pd.concat([self.data[symbol], new_row])

        # Ensure we have enough data for the longest calculation
        min_data_length = max(self.atr_period, self.rsi_period) + 2
        if len(self.data[symbol]) < min_data_length:
            return

        # Use .to_numpy() for performance with TA-Lib
        close_prices = self.data[symbol]['close'].to_numpy(dtype=float)
        high_prices = self.data[symbol]['high'].to_numpy(dtype=float)
        low_prices = self.data[symbol]['low'].to_numpy(dtype=float)
        price = close_prices[-1]

        # --- Calculate Indicators ---
        atr = talib.ATR(high_prices, low_prices, close_prices, timeperiod=self.atr_period)[-1]
        current_rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)[-1]
        # The long-term moving average is intentionally NOT calculated anymore.
        # long_ma = talib.SMA(close_prices, timeperiod=self.long_ma_period)[-1]
        prev_rsi = self.previous_rsi.get(symbol)
        self.previous_rsi[symbol] = current_rsi # Update state for the next event

        # --- DEFENSE LAYER 1: Volatility Filter ---
        volatility_pct = atr / price if price > 0 else 0
        if not (self.volatility_threshold_min < volatility_pct < self.volatility_threshold_max):
            # If volatility is outside our desired range, exit any open position and do not enter new ones.
            if self.signals.get(symbol) == 'LONG':
                signal = SignalEvent(self.strategy_id, symbol, 'EXIT')
                self.event_queue.put(signal)
                self.signals[symbol] = 'EXIT'
                self.entry_price[symbol] = 0.0 # Reset on exit
            return # Stop processing this symbol

        # --- DEFENSE LAYER 2: ATR Stop-Loss (Highest Priority Exit) ---
        if self.signals.get(symbol) == 'LONG' and self.entry_price[symbol] > 0:
            stop_price = self.entry_price[symbol] - (atr * self.stop_loss_atr_multiplier)
            if price < stop_price:
                print(f"STOP-LOSS HIT for {symbol}: Price {price:.2f} < Stop {stop_price:.2f}")
                signal = SignalEvent(self.strategy_id, symbol, 'EXIT')
                self.event_queue.put(signal)
                self.signals[symbol] = 'EXIT'
                self.entry_price[symbol] = 0.0 # Reset on exit
                return # Exit signal sent, no further action needed for this event.

        # --- CORE SIGNAL LOGIC ---
        # Entry Signal: RSI crosses UP through oversold. The trend filter has been removed
        # to allow this strategy to be a true counter-trend diversifier.
        if prev_rsi is not None and current_rsi > self.oversold and prev_rsi <= self.oversold and self.signals.get(symbol) != 'LONG':
            signal = SignalEvent(self.strategy_id, symbol, 'LONG')
            self.event_queue.put(signal)
            self.signals[symbol] = 'LONG'
            self.entry_price[symbol] = price # Record entry price for stop-loss calculation

        # Exit Signal (Profit Taking): RSI crosses DOWN through overbought.
        elif prev_rsi is not None and current_rsi < self.overbought and prev_rsi >= self.overbought and self.signals.get(symbol) == 'LONG':
            signal = SignalEvent(self.strategy_id, symbol, 'EXIT')
            self.event_queue.put(signal)
            self.signals[symbol] = 'EXIT'
            self.entry_price[symbol] = 0.0 # Reset on exit