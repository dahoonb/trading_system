# portfolio/risk_manager.py

import logging
from core.config_loader import config

logger = logging.getLogger("TradingSystem")

class RiskManager:
    """
    Manages portfolio-level risk by monitoring equity and adjusting
    risk exposure based on performance (e.g., drawdowns). This acts as
    the "first line of defense" before the hard kill-switch.
    """
    def __init__(self, initial_capital: float):
        """
        Initializes the RiskManager.

        Args:
            initial_capital (float): The starting capital of the portfolio.
        """
        self.initial_capital = initial_capital
        self.current_equity = initial_capital
        self.peak_equity = initial_capital
        
        # Load drawdown thresholds from the global configuration
        risk_config = config.get('portfolio_risk', {})
        self.drawdown_thresholds = risk_config.get('drawdown_throttles', [
            {'level': 0.05, 'scale': 0.5},
            {'level': 0.10, 'scale': 0.25},
            {'level': 0.15, 'scale': 0.0}
        ])
        # Sort thresholds by level, descending, to ensure the most severe throttle is applied first
        self.drawdown_thresholds.sort(key=lambda x: x['level'], reverse=True)
        
        logger.info(f"RiskManager initialized with {len(self.drawdown_thresholds)} drawdown throttles.")

    def update_equity(self, total_portfolio_value: float):
        """
        Updates the current and peak equity of the portfolio. This should be called
        after every fill or at regular intervals.

        Args:
            total_portfolio_value (float): The current total market value of the portfolio (cash + positions).
        """
        self.current_equity = total_portfolio_value
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity
            logger.info(f"New peak equity reached: ${self.peak_equity:,.2f}")

    def get_risk_scaling_factor(self) -> float:
        """
        Calculates a scaling factor for position sizing based on the current drawdown.
        A factor of 1.0 means full risk, while a smaller factor reduces exposure.

        Returns:
            float: A scaling factor between 0.0 and 1.0.
        """
        if self.peak_equity <= 0:
            return 1.0 # Avoid division by zero if equity is wiped out

        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity

        # Check thresholds to see if a risk reduction is needed
        for throttle in self.drawdown_thresholds:
            if drawdown >= throttle['level']:
                logger.warning(
                    f"Drawdown of {drawdown:.2%} has breached the {throttle['level']:.0%} threshold. "
                    f"Scaling new positions by a factor of {throttle['scale']:.2f}."
                )
                return throttle['scale']
        
        # If no threshold is breached, no scaling is applied
        return 1.0