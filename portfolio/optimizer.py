# portfolio/optimizer.py

import pandas as pd
import numpy as np
import logging
from scipy.optimize import minimize

from core.config_loader import config

logger = logging.getLogger("TradingSystem")

class PortfolioOptimizer:
    """
    Calculates optimal capital allocation weights and a volatility-based scaling
    factor to proactively manage portfolio risk and return.
    
    MODIFIED to use Minimum Variance Optimization instead of Sharpe Ratio
    maximization. This creates a more stable, risk-averse portfolio that is
    less likely to be over-concentrated in a single strategy before a
    market regime change.
    """
    def __init__(self, strategy_ids: list):
        """
        Initializes the PortfolioOptimizer.

        Args:
            strategy_ids (list): A list of unique string identifiers for each strategy.
        """
        self.strategy_ids = strategy_ids
        optimizer_config = config.get('portfolio_optimizer', {})
        self.lookback_days = optimizer_config.get('performance_lookback_days', 60)
        
        # Volatility Targeting Parameters remain a key part of risk management
        self.target_volatility = optimizer_config.get('target_volatility_pct', 15.0) / 100.0
        self.max_leverage_factor = optimizer_config.get('max_leverage_factor', 1.5)
        
        self.returns_df = pd.DataFrame(columns=self.strategy_ids)
        self.optimal_weights = self._get_equal_weights()
        
        logger.info(f"PortfolioOptimizer initialized for strategies: {strategy_ids} with a {self.lookback_days}-day lookback.")
        logger.info(f"Optimization Method: Minimum Variance")
        logger.info(f"Volatility Target: {self.target_volatility:.1%}, Max Leverage Factor: {self.max_leverage_factor}x")

    def _get_equal_weights(self) -> dict:
        """Returns a dictionary of equal weights for all strategies."""
        num_strategies = len(self.strategy_ids)
        return {sid: 1.0 / num_strategies for sid in self.strategy_ids} if num_strategies > 0 else {}

    def track_daily_returns(self, daily_returns: dict):
        """Appends the latest daily returns for each strategy."""
        # Using pd.Timestamp.now().normalize() ensures we have one entry per day
        today = pd.Timestamp.now().normalize()
        new_row = pd.Series(daily_returns, name=today)
        
        # Avoid duplicating rows if called multiple times on the same day
        if today in self.returns_df.index:
            self.returns_df.loc[today] = new_row
        else:
            self.returns_df = pd.concat([self.returns_df, new_row.to_frame().T])

        # Trim the DataFrame to maintain the lookback window
        if len(self.returns_df) > self.lookback_days:
            self.returns_df = self.returns_df.iloc[-self.lookback_days:]

    def calculate_portfolio_volatility(self) -> float:
        """
        Calculates the annualized volatility of the entire portfolio based on
        the historical returns of the strategies and their current optimal weights.
        """
        if len(self.returns_df) < 10: # Need a minimum number of days for a meaningful calculation
            return 0.0
        
        # Weight the returns of each strategy by the current optimal weights
        # This creates a single series of historical portfolio returns
        portfolio_returns = self.returns_df.dot(pd.Series(self.optimal_weights))
        
        # Calculate the standard deviation of daily returns and annualize it
        annualized_vol = portfolio_returns.std() * np.sqrt(252)
        
        return annualized_vol if pd.notna(annualized_vol) else 0.0

    def get_volatility_scaling_factor(self) -> float:
        """
        Returns a scaling factor based on the ratio of target to realized volatility.
        This is a proactive risk management layer that adjusts overall portfolio leverage.
        - Factor > 1.0: Increase risk (realized vol is below target).
        - Factor < 1.0: Decrease risk (realized vol is above target).
        """
        realized_vol = self.calculate_portfolio_volatility()
        
        if realized_vol < 0.001: # Avoid division by zero or extreme scaling if volatility is near zero
            logger.warning("Realized volatility is near zero. Capping leverage factor to avoid excessive risk.")
            return self.max_leverage_factor

        scaling_factor = self.target_volatility / realized_vol
        
        # Cap the scaling factor to prevent excessive leverage
        final_factor = min(scaling_factor, self.max_leverage_factor)
        
        logger.info(f"Volatility Targeting: Realized Vol={realized_vol:.2%}, Target Vol={self.target_volatility:.2%}, Scaling Factor={final_factor:.2f}")
        
        return final_factor

    def calculate_optimal_weights(self):
        """
        Performs Minimum Variance Optimization to find the set of weights that
        results in the lowest possible portfolio volatility, given the historical
        covariance of the strategies.
        """
        if len(self.returns_df) < self.lookback_days / 2:
            logger.warning("Not enough historical return data for optimization. Using equal weights.")
            self.optimal_weights = self._get_equal_weights()
            return

        # --- OBJECTIVE FUNCTION: Minimize portfolio volatility ---
        # The function to be minimized by the optimizer. It takes a numpy array
        # of weights and returns the portfolio's annualized volatility.
        def objective_func_min_volatility(weights):
            # We only need the covariance matrix to calculate portfolio variance
            cov_matrix = self.returns_df.cov() * 252 # Annualized covariance
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            return np.sqrt(portfolio_variance) # Return standard deviation (volatility)

        num_strategies = len(self.strategy_ids)
        initial_guess = np.array(num_strategies * [1. / num_strategies])
        
        # Constraint: All weights must sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: Each weight must be between 0 and 1 (no shorting strategies)
        bounds = tuple((0, 1) for _ in range(num_strategies))

        try:
            # --- Call the SciPy minimizer ---
            result = minimize(
                fun=objective_func_min_volatility, 
                x0=initial_guess, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=constraints
            )
            
            if result.success:
                self.optimal_weights = dict(zip(self.strategy_ids, result.x))
                logger.info(f"Successfully calculated Minimum Variance weights: {self.optimal_weights}")
            else:
                logger.error(f"Minimum Variance optimization failed: {result.message}. Reverting to equal weights.")
                self.optimal_weights = self._get_equal_weights()
        except Exception as e:
            logger.error(f"Exception during Minimum Variance optimization: {e}. Reverting to equal weights.", exc_info=True)
            self.optimal_weights = self._get_equal_weights()

    def get_weights(self) -> dict:
        """Returns the last calculated optimal weights."""
        return self.optimal_weights