# strategy/quality_value.py

import pandas as pd
from core.events import SignalEvent
from strategy.base import BaseStrategy
import feast
import os

class QualityValueStrategy(BaseStrategy):
    """
    A long-only strategy that invests in a portfolio of stocks that are
    both cheap (low Price-to-Book ratio) and fundamentally strong
    (high Piotroski F-Score). Rebalances quarterly.
    """
    def __init__(self, event_queue, symbols, strategy_id: str,
                 rebalance_freq_days: int = 90,
                 value_percentile: float = 0.2, # Bottom 20% by P/B
                 quality_f_score_threshold: int = 7, # F-Score of 7 or higher
                 max_portfolio_size: int = 30,
                 feature_repo_path: str = "feature_repo/",
                 state_file="state/quality_value_state.pkl"):
        """
        Initializes the Quality-Value strategy.
        """
        super().__init__(event_queue, symbols, state_file)
        self.strategy_id = strategy_id
        self.rebalance_freq_days = rebalance_freq_days
        self.value_percentile = value_percentile
        self.quality_f_score_threshold = quality_f_score_threshold
        self.max_portfolio_size = max_portfolio_size
        
        self.last_rebalance_date = None
        self.target_portfolio = set() # The set of symbols we want to hold

        # In a real system, this would be injected. For simplicity, we instantiate it here.
        self.feature_store = feast.FeatureStore(repo_path=os.path.join(os.getcwd(), feature_repo_path))

    def calculate_signals(self, event):
        """
        Checks if it's time to rebalance. If so, it constructs the new target
        portfolio and generates signals to align the current portfolio with it.
        """
        if event.type != 'MARKET':
            return

        current_date = event.timestamp.date()

        # --- Rebalance Check ---
        if self.last_rebalance_date and (current_date - self.last_rebalance_date).days < self.rebalance_freq_days:
            return # Not time to rebalance yet

        print(f"{self.strategy_id}: Rebalancing portfolio on {current_date}...")
        self.last_rebalance_date = current_date

        # --- Construct Target Portfolio ---
        try:
            # 1. Fetch features for all symbols in the universe
            entity_rows = [{"ticker_id": s} for s in self.symbols]
            features_df = self.feature_store.get_online_features(
                features=[
                    "fundamental_features:price_to_book",
                    "fundamental_features:piotroski_f_score"
                ],
                entity_rows=entity_rows
            ).to_df()

            # 2. Value Screen: Filter for stocks in the bottom percentile for P/B
            # A lower P/B is better (cheaper)
            value_threshold = features_df['price_to_book'].quantile(self.value_percentile)
            value_stocks = features_df[features_df['price_to_book'] <= value_threshold]

            # 3. Quality Screen: From the value stocks, filter for high F-Score
            quality_value_stocks = value_stocks[value_stocks['piotroski_f_score'] >= self.quality_f_score_threshold]
            
            # 4. Final Selection: Sort by F-score and P/B to get the best candidates
            quality_value_stocks = quality_value_stocks.sort_values(
                by=['piotroski_f_score', 'price_to_book'], 
                ascending=[False, True] # Highest F-score, lowest P/B
            )
            
            self.target_portfolio = set(quality_value_stocks.head(self.max_portfolio_size)['ticker_id'])
            print(f"{self.strategy_id}: New target portfolio has {len(self.target_portfolio)} stocks.")

        except Exception as e:
            print(f"ERROR in {self.strategy_id}: Could not construct target portfolio. Error: {e}")
            return # Abort rebalance if feature fetching fails

        # --- Generate Orders to Align with Target Portfolio ---
        current_holdings = {s for s, sig in self.signals.items() if sig == 'LONG'}

        # Generate EXIT signals for stocks no longer in the target portfolio
        for symbol in current_holdings - self.target_portfolio:
            self.event_queue.put(SignalEvent(self.strategy_id, symbol, 'EXIT'))
            self.signals[symbol] = 'EXIT'
            print(f"{self.strategy_id}: EXIT signal for {symbol}")

        # Generate LONG signals for new stocks in the target portfolio
        for symbol in self.target_portfolio - current_holdings:
            self.event_queue.put(SignalEvent(self.strategy_id, symbol, 'LONG'))
            self.signals[symbol] = 'LONG'
            print(f"{self.strategy_id}: LONG signal for {symbol}")