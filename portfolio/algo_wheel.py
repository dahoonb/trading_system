# portfolio/algo_wheel.py

import random
import duckdb
import os
import pandas as pd
import joblib
from typing import List, Dict, Any

from core.config_loader import config

class AlgoWheel:
    """
    An adaptive, contextual Algo Wheel that selects execution algorithms
    using an ML policy model, with a fallback to a simple bandit algorithm.
    """
    def __init__(self, tca_db_path: str = "data/tca_log.duckdb"):
        """
        Initializes the AlgoWheel, loading the ML policy model and its
        feature encoder if they exist.

        Args:
            tca_db_path (str): Path to the DuckDB database for TCA logs.
        """
        self.tca_db_path = tca_db_path
        self.config = config.get('algo_wheel', {})
        self.algos: List[str] = self.config.get('available_methods', ['ADAPTIVE', 'MOC', 'LOC', 'TWAP'])
        self.epsilon: float = self.config.get('epsilon', 0.10)
        self.min_fills_for_stats: int = self.config.get('min_fills_per_method', 20)

        # Ensure the database directory exists
        db_dir = os.path.dirname(self.tca_db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        # Load the ML policy model and its feature encoder
        self.policy_model = None
        self.feature_encoder = None
        
        # Use a models directory relative to the project root for robustness
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_path = os.path.join(model_dir, "algowheel_policy_model.pkl")
        encoder_path = os.path.join(model_dir, "algowheel_feature_encoder.pkl")
        
        try:
            if os.path.exists(model_path) and os.path.exists(encoder_path):
                self.policy_model = joblib.load(model_path)
                self.feature_encoder = joblib.load(encoder_path)
                print(f"Successfully loaded AlgoWheel policy model and encoder.")
            else:
                print(f"Warning: AlgoWheel policy model or encoder not found. Will use fallback bandit logic.")
        except Exception as e:
            print(f"Error loading AlgoWheel policy model/encoder: {e}. Will use fallback bandit logic.")

    def get_execution_decision(self, symbol: str, order_context: Dict[str, Any]) -> str:
        """
        Selects an execution algorithm using the ML policy model if available.
        If the model is not present or fails, it falls back to an epsilon-greedy bandit.

        Args:
            symbol (str): The ticker symbol of the instrument.
            order_context (Dict[str, Any]): A dictionary with context like
                                            {'order_size': int, 'volatility': float}.
        """
        # --- Primary Path: Use the ML Policy Model ---
        if self.policy_model and self.feature_encoder:
            try:
                # 1. Engineer categorical features from the live order context.
                # These bins MUST match the logic in the bootstrap trainer script.
                order_size = order_context.get('order_size', 100)
                volatility = order_context.get('volatility', 0.015) # Normalized volatility (e.g., ATR/price)

                size_cat = 'large' if order_size > 500 else 'medium' if order_size > 100 else 'small'
                vol_cat = 'high_vol' if volatility > 0.03 else 'mid_vol' if volatility > 0.01 else 'low_vol'
                
                # 2. Transform features into numerical format using the loaded encoder.
                context_df = pd.DataFrame([[size_cat, vol_cat]], columns=['order_size_cat', 'volatility_cat'])
                context_encoded = self.feature_encoder.transform(context_df)
                
                # 3. Make a prediction to get the best algorithm for this context.
                prediction = self.policy_model.predict(context_encoded)[0]
                
                print(f"AlgoWheel: ML Policy selected '{prediction}' for context: size={size_cat}, vol={vol_cat}.")
                return prediction
            except Exception as e:
                print(f"Error during AlgoWheel ML prediction: {e}. Falling back to bandit logic.")

        # --- Fallback Path: Use the Epsilon-Greedy Bandit ---
        return self._get_decision_from_bandit()

    def _get_decision_from_bandit(self) -> str:
        """
        The fallback logic that selects an algorithm based on historical performance
        of both filled and missed trades, using an epsilon-greedy approach.
        """
        try:
            with duckdb.connect(self.tca_db_path, read_only=True) as con:
                fills_query = "SELECT algo_used, AVG(slippage) as avg_slippage, COUNT(*) as fill_count FROM execution_log GROUP BY algo_used"
                misses_query = "SELECT algo_used, SUM(opportunity_cost) as total_opp_cost, COUNT(*) as miss_count FROM missed_trades GROUP BY algo_used"
                
                fills_df = con.execute(fills_query).fetchdf()
                misses_df = con.execute(misses_query).fetchdf()
        except duckdb.Error:
            # If the database can't be read, return a safe default
            return 'ADAPTIVE'

        if fills_df.empty:
            # If there's no data at all, explore randomly
            return random.choice(self.algos)

        # Merge performance data and calculate a total cost score
        performance_df = pd.merge(fills_df, misses_df, on='algo_used', how='left').fillna(0)
        
        performance_df['total_trades'] = performance_df['fill_count'] + performance_df['miss_count']
        
        # Calculate total cost per trade, avoiding division by zero
        performance_df['total_cost_per_trade'] = performance_df.apply(
            lambda row: row['avg_slippage'] + (row['total_opp_cost'] / row['total_trades']) if row['total_trades'] > 0 else row['avg_slippage'],
            axis=1
        )
        
        # Filter for algorithms with enough data to be statistically significant
        qualified_algos = performance_df[performance_df['total_trades'] >= self.min_fills_for_stats]

        if qualified_algos.empty:
            # Not enough data on any single algo, so explore randomly
            return random.choice(self.algos)

        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            print("AlgoWheel (Bandit): Exploring with random selection.")
            return random.choice(self.algos)
        else:
            # Exploit: choose the algorithm with the lowest total cost per trade
            best_algo_row = qualified_algos.loc[qualified_algos['total_cost_per_trade'].idxmin()]
            best_algo = best_algo_row['algo_used']
            print(f"AlgoWheel (Bandit): Exploiting with best algo: {best_algo} (Total Cost/Trade: {best_algo_row['total_cost_per_trade']:.4f})")
            return best_algo