# strategy/low_volatility.py

import pandas as pd
from core.events import SignalEvent
from strategy.base import BaseStrategy
import feast
import os

class LowVolatilityStrategy(BaseStrategy):
    """
    A long-only strategy that invests in a portfolio of the least volatile
    stocks, with a momentum overlay to avoid stocks in a persistent downtrend.
    Rebalances monthly.
    """
    def __init__(self, event_queue, symbols, strategy_id: str,
                 rebalance_freq_days: int = 30,
                 volatility_lookback_days: int = 252,
                 momentum_lookback_days: int = 126, # 6-month momentum
                 volatility_percentile: float = 0.2, # Bottom 20% by volatility
                 max_portfolio_size: int = 30,
                 feature_repo_path: str = "feature_repo/",
                 state_file="state/low_vol_state.pkl"):
        """
        Initializes the Low-Volatility strategy.
        """
        super().__init__(event_queue, symbols, state_file)
        self.strategy_id = strategy_id
        self.rebalance_freq_days = rebalance_freq_days
        self.volatility_lookback_days = volatility_lookback_days
        self.momentum_lookback_days = momentum_lookback_days
        self.volatility_percentile = volatility_percentile
        self.max_portfolio_size = max_portfolio_size
        
        self.last_rebalance_date = None
        self.target_portfolio = set()

        self.feature_store = feast.FeatureStore(repo_path=os.path.join(os.getcwd(), feature_repo_path))

    def calculate_signals(self, event):
        """
        Checks for rebalance date, then constructs and aligns to the new
        low-volatility, positive-momentum portfolio.
        """
        if event.type != 'MARKET':
            return

        current_date = event.timestamp.date()
        if self.last_rebalance_date and (current_date - self.last_rebalance_date).days < self.rebalance_freq_days:
            return

        print(f"{self.strategy_id}: Rebalancing portfolio on {current_date}...")
        self.last_rebalance_date = current_date

        # --- Construct Target Portfolio ---
        try:
            # 1. Calculate volatility and momentum for all symbols
            all_metrics = []
            for symbol in self.symbols:
                hist_data = self.data[symbol]
                if len(hist_data) < self.volatility_lookback_days:
                    continue
                
                # Volatility: Standard deviation of daily log returns
                log_returns = pd.np.log(hist_data['close'] / hist_data['close'].shift(1))
                volatility = log_returns.tail(self.volatility_lookback_days).std() * pd.np.sqrt(252)
                
                # Momentum: 6-month total return
                momentum = (hist_data['close'].iloc[-1] / hist_data['close'].iloc[-self.momentum_lookback_days]) - 1
                
                all_metrics.append({'ticker_id': symbol, 'volatility': volatility, 'momentum': momentum})

            if not all_metrics: return
            metrics_df = pd.DataFrame(all_metrics)

            # 2. Volatility Screen: Filter for the least volatile stocks
            vol_threshold = metrics_df['volatility'].quantile(self.volatility_percentile)
            low_vol_stocks = metrics_df[metrics_df['volatility'] <= vol_threshold]

            # 3. Momentum Filter: From low-vol stocks, keep only those with positive momentum
            low_vol_positive_momentum = low_vol_stocks[low_vol_stocks['momentum'] > 0]
            
            # 4. Final Selection: Sort by volatility to get the most stable candidates
            low_vol_positive_momentum = low_vol_positive_momentum.sort_values(by='volatility', ascending=True)
            
            self.target_portfolio = set(low_vol_positive_momentum.head(self.max_portfolio_size)['ticker_id'])
            print(f"{self.strategy_id}: New target portfolio has {len(self.target_portfolio)} stocks.")

        except Exception as e:
            print(f"ERROR in {self.strategy_id}: Could not construct target portfolio. Error: {e}")
            return

        # --- Generate Orders to Align with Target Portfolio ---
        current_holdings = {s for s, sig in self.signals.items() if sig == 'LONG'}
        
        for symbol in current_holdings - self.target_portfolio:
            self.event_queue.put(SignalEvent(self.strategy_id, symbol, 'EXIT'))
            self.signals[symbol] = 'EXIT'
            print(f"{self.strategy_id}: EXIT signal for {symbol}")

        for symbol in self.target_portfolio - current_holdings:
            self.event_queue.put(SignalEvent(self.strategy_id, symbol, 'LONG'))
            self.signals[symbol] = 'LONG'
            print(f"{self.strategy_id}: LONG signal for {symbol}")