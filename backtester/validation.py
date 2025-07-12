# backtester/validation.py

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any

import ray

from backtester.distributed_runner import run_grid_search, run_single_backtest
from backtester.performance import calculate_performance_metrics

logger = logging.getLogger("ValidationFramework")

HISTORICAL_SCENARIOS = {
    "2008_GFC": ('2008-01-01', '2009-06-30'),
    "2010_Flash_Crash": ('2010-05-01', '2010-05-31'),
    "2018_Vol_Spike": ('2018-01-01', '2018-03-31'),
    "2020_COVID_Crash": ('2020-02-01', '2020-04-30'),
    "2022_Bear_Market": ('2022-01-01', '2022-12-31')
}

def run_walk_forward_validation(
    data_ref,
    full_historical_data: pd.DataFrame,
    strategy_name: str,
    param_grid_config: Dict[str, List],
    initial_capital: float,
    n_splits: int = 5,
    train_period_years: int = 4,
    test_period_years: int = 1,
    is_system_grid: bool = False
) -> pd.DataFrame:
    logger.info(f"--- Starting Walk-Forward Validation for '{strategy_name}' ---")
    logger.info(f"Folds: {n_splits}, Train Period: {train_period_years} years, Test Period: {test_period_years} years")

    all_oos_results = []
    total_period_years = train_period_years + test_period_years
    
    if (full_historical_data.index[-1] - full_historical_data.index[0]).days < total_period_years * 365:
        raise ValueError("Not enough historical data for the specified train/test periods.")

    for i in range(n_splits):
        fold_num = i + 1
        logger.info(f"\n--- Processing Fold {fold_num}/{n_splits} ---")

        end_date = full_historical_data.index[-1] - pd.DateOffset(years=i * test_period_years)
        test_start_date = end_date - pd.DateOffset(years=test_period_years)
        train_end_date = test_start_date - pd.DateOffset(days=1)
        train_start_date = train_end_date - pd.DateOffset(years=train_period_years)

        if train_start_date < full_historical_data.index[0]:
            logger.warning(f"Fold {fold_num} skipped: Training period extends before available data.")
            continue

        logger.info(f"Fold {fold_num} | Train Period: {train_start_date.date()} to {train_end_date.date()}")
        logger.info(f"Fold {fold_num} | Test Period:  {test_start_date.date()} to {end_date.date()}")

        logger.info(f"Fold {fold_num}: Running in-sample grid search...")
        is_results_df = run_grid_search(
            strategy_name=strategy_name,
            data_ref=data_ref,
            param_grid_config=param_grid_config,
            initial_capital=initial_capital,
            is_system_grid=is_system_grid,
            start_date=train_start_date.isoformat(),
            end_date=train_end_date.isoformat()
        )
        
        if is_results_df.empty:
            logger.warning(f"Fold {fold_num}: In-sample grid search yielded no results. Skipping fold.")
            continue

        best_params = {k: v for k, v in is_results_df.iloc[0].items() if k in param_grid_config}
        logger.info(f"Fold {fold_num}: Best in-sample parameters found: {best_params}")

        logger.info(f"Fold {fold_num}: Running out-of-sample test with best parameters...")
        oos_results, _ = run_single_backtest(
            strategy_name=strategy_name,
            data_ref=data_ref,
            params=best_params,
            initial_capital=initial_capital,
            is_system_params=is_system_grid,
            return_equity_curve=True,
            start_date=test_start_date.isoformat(),
            end_date=end_date.isoformat()
        )
        
        oos_results['fold'] = fold_num
        all_oos_results.append(oos_results)
        logger.info(f"Fold {fold_num}: Out-of-sample test complete. Sharpe Ratio: {oos_results.get('Sharpe Ratio', 'N/A')}")

    logger.info("\n--- Walk-Forward Validation Complete ---")
    return pd.DataFrame(all_oos_results)

def run_monte_carlo_simulation(
    equity_curve: pd.DataFrame,
    n_simulations: int = 1000,
    initial_capital: float = 100000.0
) -> Dict[str, Dict[str, float]]:
    logger.info(f"--- Running Monte Carlo Simulation with {n_simulations} runs ---")
    daily_returns = equity_curve['equity'].pct_change().dropna()
    if daily_returns.empty:
        logger.warning("Cannot run Monte Carlo on an empty returns series.")
        return {}

    all_sim_metrics = []
    for _ in range(n_simulations):
        synthetic_returns = np.random.choice(daily_returns, size=len(daily_returns), replace=True)
        synthetic_equity = [initial_capital]
        for ret in synthetic_returns:
            synthetic_equity.append(synthetic_equity[-1] * (1 + ret))
        
        synthetic_equity_df = pd.DataFrame(synthetic_equity, index=equity_curve.index, columns=['equity'])
        metrics = calculate_performance_metrics(synthetic_equity_df['equity'])
        all_sim_metrics.append(metrics)

    results_df = pd.DataFrame(all_sim_metrics)
    summary_stats = {}
    for metric in ["CAGR (%)", "Sharpe Ratio", "Max Drawdown (%)", "Volatility (%)"]:
        if metric in results_df.columns:
            stats = results_df[metric].describe(percentiles=[.05, .25, .5, .75, .95])
            summary_stats[metric] = stats.to_dict()
            
    return summary_stats

def run_historical_scenario_analysis(
    data_ref,
    full_historical_data: pd.DataFrame,
    strategy_name: str,
    best_params: Dict[str, Any],
    initial_capital: float,
    is_system_grid: bool = False
) -> pd.DataFrame:
    logger.info("\n--- Running Historical Scenario Analysis ---")
    scenario_results = []

    for name, (start_date, end_date) in HISTORICAL_SCENARIOS.items():
        if pd.to_datetime(start_date) > full_historical_data.index[-1] or pd.to_datetime(end_date) < full_historical_data.index[0]:
            logger.warning(f"Skipping scenario '{name}': Period is outside available data range.")
            continue

        logger.info(f"Testing scenario: {name} ({start_date} to {end_date})")
        
        metrics, _ = run_single_backtest(
            strategy_name=strategy_name,
            data_ref=data_ref,
            params=best_params,
            initial_capital=initial_capital,
            is_system_params=is_system_grid,
            return_equity_curve=True,
            start_date=start_date,
            end_date=end_date
        )
        
        metrics['scenario'] = name
        scenario_results.append(metrics)

    return pd.DataFrame(scenario_results)

def run_hypothetical_stress_test(
    data_ref,
    full_historical_data: pd.DataFrame,
    strategy_name: str,
    best_params: Dict[str, Any],
    initial_capital: float,
    is_system_grid: bool = False
) -> pd.DataFrame:
    logger.info("\n--- Running Hypothetical Stress Tests ---")
    stress_results = []

    logger.info("Testing stress case: Sudden 20% Market Crash...")
    crash_data = full_historical_data.copy()
    crash_day_index = len(crash_data) // 2
    crash_data.iloc[crash_day_index] *= 0.80
    
    crash_data_ref = ray.put(crash_data)
    
    crash_metrics, _ = run_single_backtest(
        strategy_name=strategy_name,
        data_ref=crash_data_ref,
        params=best_params,
        initial_capital=initial_capital,
        is_system_params=is_system_grid,
        return_equity_curve=True
    )
    crash_metrics['scenario'] = 'Sudden_20_Pct_Crash'
    stress_results.append(crash_metrics)

    del crash_data_ref

    return pd.DataFrame(stress_results)