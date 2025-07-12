# strategy/cross_sectional_momentum.py

import pandas as pd
from core.events import SignalEvent
from strategy.base import BaseStrategy

class CrossSectionalMomentumStrategy(BaseStrategy):
    """
    A long-only strategy that invests in a portfolio of stocks with the
    strongest relative price momentum over the past 12 months, skipping
    the most recent month. Rebalances monthly.
    """
    def __init__(self, event_queue, symbols, strategy_id: str,
                 rebalance_freq_days: int = 30,
                 momentum_lookback_days: int = 252, # 12 months
                 momentum_skip_days: int = 21, # 1 month
                 momentum_percentile: float = 0.9, # Top 10% by momentum
                 max_portfolio_size: int = 30,
                 state_file="state/cs_momentum_state.pkl"):
        """
        Initializes the Cross-Sectional Momentum strategy.
        """
        super().__init__(event_queue, symbols, state_file)
        self.strategy_id = strategy_id
        self.rebalance_freq_days = rebalance_freq_days
        self.momentum_lookback_days = momentum_lookback_days
        self.momentum_skip_days = momentum_skip_days
        self.momentum_percentile = momentum_percentile
        self.max_portfolio_size = max_portfolio_size
        
        self.last_rebalance_date = None
        self.target_portfolio = set()

    def calculate_signals(self, event):
        """
        Checks for rebalance date, then constructs and aligns to the new
        top-momentum portfolio.
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
            all_metrics = []
            for symbol in self.symbols:
                hist_data = self.data[symbol]
                if len(hist_data) < self.momentum_lookback_days + self.momentum_skip_days:
                    continue
                
                # Calculate 12-1 momentum
                # Price from ~12 months ago (start of lookback)
                price_start = hist_data['close'].iloc[-(self.momentum_lookback_days + self.momentum_skip_days)]
                # Price from ~1 month ago (end of lookback)
                price_end = hist_data['close'].iloc[-self.momentum_skip_days]
                
                momentum = (price_end / price_start) - 1
                all_metrics.append({'ticker_id': symbol, 'momentum': momentum})

            if not all_metrics: return
            metrics_df = pd.DataFrame(all_metrics)

            # 2. Momentum Screen: Filter for the top percentile of performers
            momentum_threshold = metrics_df['momentum'].quantile(self.momentum_percentile)
            top_performers = metrics_df[metrics_df['momentum'] >= momentum_threshold]
            
            # 3. Final Selection: Sort by momentum to get the strongest candidates
            top_performers = top_performers.sort_values(by='momentum', ascending=False)
            
            self.target_portfolio = set(top_performers.head(self.max_portfolio_size)['ticker_id'])
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