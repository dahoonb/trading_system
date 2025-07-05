# backtester/performance.py

import pandas as pd
import numpy as np
from typing import Dict, Any

def calculate_performance_metrics(equity_curve: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """
    Calculates a dictionary of performance metrics from a pandas Series of equity values.

    Args:
        equity_curve (pd.Series): A pandas Series with datetime index and portfolio equity values.
        risk_free_rate (float): The annualized risk-free rate for Sharpe ratio calculation.

    Returns:
        Dict[str, Any]: A dictionary containing key performance metrics.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        return {
            "Total Return (%)": 0.0, "CAGR (%)": 0.0, "Sharpe Ratio": 0.0,
            "Max Drawdown (%)": 0.0, "Volatility (%)": 0.0, "Error": "Equity curve too short."
        }

    # Total Return
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100

    # Compound Annual Growth Rate (CAGR)
    duration_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    if duration_years == 0:
        cagr = 0.0
    else:
        cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / duration_years) - 1) * 100

    # Annualized Volatility
    daily_returns = equity_curve.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) * 100

    # Sharpe Ratio
    if volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = ((cagr / 100) - risk_free_rate) / (volatility / 100)

    # Maximum Drawdown
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min() * 100

    return {
        "Total Return (%)": round(total_return, 2),
        "CAGR (%)": round(cagr, 2),
        "Sharpe Ratio": round(sharpe_ratio, 3),
        "Max Drawdown (%)": round(max_drawdown, 2),
        "Volatility (%)": round(volatility, 2)
    }