# portfolio/shared_sizing_logic.py

def get_atr_volatility_adjusted_size(capital: float, risk_per_trade_pct: float, price: float, atr: float) -> int:
    """
    Calculates a position size based on volatility (ATR) to target a
    fixed percentage of capital risk per trade.

    Args:
        capital (float): Total capital of the portfolio.
        risk_per_trade_pct (float): The percentage of capital to risk (e.g., 0.01 for 1%).
        price (float): The current price of the asset.
        atr (float): The Average True Range of the asset.

    Returns:
        int: The number of shares to trade.
    """
    if atr <= 0 or price <= 0:
        return 0

    # Calculate the dollar amount to risk
    dollar_risk_per_trade = capital * risk_per_trade_pct

    # Determine the dollar amount of risk per share using ATR (e.g., a 2x ATR stop)
    # This is a common convention for setting a volatility-adjusted stop-loss.
    stop_loss_per_share = 2 * atr

    if stop_loss_per_share <= 0:
        return 0

    # Calculate the number of shares to achieve the desired dollar risk
    # This formula determines how many shares you can buy so that if your
    # 2*ATR stop-loss is hit, you only lose your target dollar risk amount.
    position_size = int(dollar_risk_per_trade / stop_loss_per_share)

    return position_size