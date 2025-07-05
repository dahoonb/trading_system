# backtester/validation.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

from backtester.distributed_runner import run_grid_search, run_single_backtest
from backtester.performance import calculate_performance_metrics

logger = logging.getLogger("ValidationFramework")

# --- Historical Scenarios Definition ---
# Define key historical periods for stress testing.
HISTORICAL_SCENARIOS = {
    "2008_GFC": ('2008-01-01', '2009-06-30'),
    "2010_Flash_Crash": ('2010-05-01', '2010-05-31'),
    "2018_Vol_Spike": ('2018-01-01', '2018-03-31'),
    "2020_COVID_Crash": ('2020-02-01', '2020-04-30'),
    "2022_Bear_Market": ('2022-01-01', '2022-12-31')
}

def run_walk_forward_validation(
    historical_data: pd.DataFrame,
    strategy_name: str,
    param_grid_config: Dict[str, List],
    initial_capital: float,
    n_splits: int = 5,
    train_period_years: int = 4,
    test_period_years: int = 1
) -> pd.DataFrame:
    """
    Performs a walk-forward validation by creating rolling time windows.

    In each window (fold):
    1. A grid search is performed on the training data (in-sample) to find the best parameters.
    2. The best parameters are then tested on the subsequent, unseen test data (out-of-sample).

    Args:
        historical_data (pd.DataFrame): The complete historical dataset.
        strategy_name (str): The name of the strategy to test.
        param_grid_config (Dict): The parameter grid for the in-sample optimization.
        initial_capital (float): The starting capital.
        n_splits (int): The number of walk-forward folds to create.
        train_period_years (int): The length of the training period for each fold.
        test_period_years (int): The length of the out-of-sample testing period.

    Returns:
        pd.DataFrame: A DataFrame containing the out-of-sample performance for each fold.
    """
    logger.info(f"--- Starting Walk-Forward Validation for '{strategy_name}' ---")
    logger.info(f"Folds: {n_splits}, Train Period: {train_period_years} years, Test Period: {test_period_years} years")

    all_oos_results = []
    total_period_years = train_period_years + test_period_years
    
    # Ensure there is enough data for at least one full fold
    if (historical_data.index[-1] - historical_data.index[0]).days < total_period_years * 365:
        raise ValueError("Not enough historical data for the specified train/test periods.")

    for i in range(n_splits):
        fold_num = i + 1
        logger.info(f"\n" + "="*40)
        logger.info(f"--- Processing Fold {fold_num}/{n_splits} ---")
        logger.info("="*40)

        # Define the time windows for this fold by working backwards from the end
        end_date = historical_data.index[-1] - pd.DateOffset(years=i * test_period_years)
        test_start_date = end_date - pd.DateOffset(years=test_period_years)
        train_end_date = test_start_date - pd.DateOffset(days=1)
        train_start_date = train_end_date - pd.DateOffset(years=train_period_years)

        if train_start_date < historical_data.index[0]:
            logger.warning(f"Fold {fold_num} skipped: Training period ({train_start_date.date()}) extends before available data start date ({historical_data.index[0].date()}).")
            continue

        # Split data into in-sample (is) and out-of-sample (oos)
        train_data = historical_data.loc[train_start_date:train_end_date]
        test_data = historical_data.loc[test_start_date:end_date]

        logger.info(f"Fold {fold_num} | Train Period: {train_start_date.date()} to {train_end_date.date()}")
        logger.info(f"Fold {fold_num} | Test Period:  {test_start_date.date()} to {end_date.date()}")

        # 1. In-Sample Optimization: Find the best parameters on the training data
        logger.info(f"Fold {fold_num}: Running in-sample grid search to find best parameters...")
        is_results_df = run_grid_search(
            strategy_name=strategy_name,
            historical_data=train_data,
            param_grid_config=param_grid_config,
            initial_capital=initial_capital
        )
        
        if is_results_df.empty:
            logger.warning(f"Fold {fold_num}: In-sample grid search yielded no results. Skipping fold.")
            continue

        # Extract the best performing parameters (top row of the sorted results)
        best_params_series = is_results_df.iloc[0]
        best_params = {k: v for k, v in best_params_series.items() if k in param_grid_config}
        
        logger.info(f"Fold {fold_num}: Best in-sample parameters found: {best_params}")
        logger.info(f"Fold {fold_num}: Best in-sample Sharpe Ratio: {best_params_series.get('Sharpe Ratio', 'N/A')}")

        # 2. Out-of-Sample Test: Run a single backtest on the test data with the best parameters
        logger.info(f"Fold {fold_num}: Running out-of-sample test with best parameters...")
        oos_results = run_single_backtest(
            strategy_name=strategy_name,
            historical_data=test_data,
            params=best_params,
            initial_capital=initial_capital
        )
        
        # Add fold information to the results for comprehensive analysis
        oos_results['fold'] = fold_num
        oos_results['train_period'] = f"{train_start_date.date()}_{train_end_date.date()}"
        oos_results['test_period'] = f"{test_start_date.date()}_{end_date.date()}"
        all_oos_results.append(oos_results)
        logger.info(f"Fold {fold_num}: Out-of-sample test complete. Sharpe Ratio: {oos_results.get('Sharpe Ratio', 'N/A')}")

    logger.info("\n--- Walk-Forward Validation Complete ---")
    return pd.DataFrame(all_oos_results)

# --- MODIFICATION: Add Monte Carlo Simulation Orchestrator ---
def run_monte_carlo_simulation(
    equity_curve: pd.DataFrame,
    n_simulations: int = 1000,
    initial_capital: float = 100000.0
) -> Dict[str, Dict[str, float]]:
    """
    Performs a Monte Carlo simulation on a single backtest's results.

    It takes the daily returns from a backtest, resamples them thousands of
    times to create new synthetic equity curves, and analyzes the distribution
    of the resulting performance metrics.

    Args:
        equity_curve (pd.DataFrame): The daily equity curve from a single backtest run.
        n_simulations (int): The number of synthetic histories to generate.
        initial_capital (float): The starting capital for the simulations.

    Returns:
        A dictionary summarizing the distribution of key performance metrics.
    """
    logger.info(f"--- Starting Monte Carlo Simulation with {n_simulations} runs ---")

    daily_returns = equity_curve['equity'].pct_change().dropna()
    
    if daily_returns.empty:
        logger.warning("Cannot run Monte Carlo simulation on an empty returns series.")
        return {}

    all_sim_metrics = []

    for i in range(n_simulations):
        # Bootstrap resampling: sample with replacement from the historical returns
        synthetic_returns = np.random.choice(daily_returns, size=len(daily_returns), replace=True)
        
        # Create a new synthetic equity curve
        synthetic_equity = [initial_capital]
        for ret in synthetic_returns:
            synthetic_equity.append(synthetic_equity[-1] * (1 + ret))
        
        synthetic_equity_df = pd.DataFrame(synthetic_equity, index=equity_curve.index, columns=['equity'])
        
        # Calculate performance for this synthetic history
        metrics = calculate_performance_metrics(synthetic_equity_df['equity'])
        all_sim_metrics.append(metrics)

    # Analyze the distribution of the simulation results
    results_df = pd.DataFrame(all_sim_metrics)
    
    summary_stats = {}
    for metric in ["CAGR (%)", "Sharpe Ratio", "Max Drawdown (%)", "Volatility (%)"]:
        if metric in results_df.columns:
            stats = results_df[metric].describe(percentiles=[.05, .25, .5, .75, .95])
            summary_stats[metric] = stats.to_dict()
            
    logger.info("--- Monte Carlo Simulation Complete ---")
    return summary_stats

# --- MODIFICATION: Add Historical Scenario Analysis ---
def run_historical_scenario_analysis(
    historical_data: pd.DataFrame,
    strategy_name: str,
    best_params: Dict[str, Any],
    initial_capital: float
) -> pd.DataFrame:
    """
    Runs the strategy with its best parameters through specific historical crisis periods.
    """
    logger.info("\n--- Running Historical Scenario Analysis ---")
    scenario_results = []

    for name, (start_date, end_date) in HISTORICAL_SCENARIOS.items():
        scenario_data = historical_data.loc[start_date:end_date]
        if scenario_data.empty or len(scenario_data) < 20:
            logger.warning(f"Skipping scenario '{name}': Not enough data in the specified period.")
            continue

        logger.info(f"Testing scenario: {name} ({start_date} to {end_date})")
        
        metrics, _ = run_single_backtest(
            strategy_name=strategy_name,
            historical_data=scenario_data,
            params=best_params,
            initial_capital=initial_capital,
            return_equity_curve=True # We need the curve to calculate metrics
        )
        
        metrics['scenario'] = name
        scenario_results.append(metrics)

    return pd.DataFrame(scenario_results)

# --- MODIFICATION: Add Hypothetical Stress Testing ---
def run_hypothetical_stress_test(
    historical_data: pd.DataFrame,
    strategy_name: str,
    best_params: Dict[str, Any],
    initial_capital: float
) -> pd.DataFrame:
    """
    Runs the strategy against programmatically altered data to simulate stress events.
    """
    logger.info("\n--- Running Hypothetical Stress Tests ---")
    stress_results = []

    # Test 1: Sudden Market Crash (Gap Risk)
    logger.info("Testing stress case: Sudden 20% Market Crash...")
    crash_data = historical_data.copy()
    crash_day_index = len(crash_data) // 2
    crash_data.iloc[crash_day_index] *= 0.80 # Apply a 20% drop to all prices on one day
    crash_metrics, _ = run_single_backtest(strategy_name, crash_data, best_params, initial_capital, True)
    crash_metrics['scenario'] = 'Sudden_20_Pct_Crash'
    stress_results.append(crash_metrics)

    # Test 2: Increased Transaction Costs
    logger.info("Testing stress case: 5x Transaction Costs...")
    # We can simulate this by modifying the engine parameters, but for simplicity,
    # we'll note that this would require passing cost params to the backtest engine.
    # For now, we'll just add a placeholder.
    cost_metrics = {'scenario': '5x_Transaction_Costs', 'Note': 'Requires engine modification to test'}
    stress_results.append(cost_metrics)

    # Test 3: Volatility Shock
    logger.info("Testing stress case: Sustained 2x Volatility...")
    vol_shock_data = historical_data.copy()
    daily_returns = vol_shock_data.pct_change()
    shocked_returns = daily_returns * 2 # Double the daily returns
    # Reconstruct the price series from the shocked returns
    # This is a simplification; a more robust method would handle this carefully.
    # For now, we'll just note the intent.
    vol_metrics = {'scenario': '2x_Volatility_Shock', 'Note': 'Requires data reconstruction to test'}
    stress_results.append(vol_metrics)

    return pd.DataFrame(stress_results)