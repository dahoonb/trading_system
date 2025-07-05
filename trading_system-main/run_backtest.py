# run_backtest.py

import pandas as pd
import os
import argparse
import logging
import ray
import json
from typing import Dict, List

from backtester.distributed_runner import run_grid_search, run_single_backtest
from backtester.validation import run_walk_forward_validation, run_monte_carlo_simulation, run_historical_scenario_analysis
from core.config_loader import config
from utils.logger import setup_logger

# Setup loggers for all relevant modules to ensure all output is captured
setup_logger(logger_name="BacktestRun", log_level=logging.INFO)
setup_logger(logger_name="ValidationFramework", log_level=logging.INFO)
setup_logger(logger_name="DistributedRunner", log_level=logging.INFO)
setup_logger(logger_name="BacktestEngine", log_level=logging.INFO)

# Grid for optimizing individual strategy logic (e.g., moving average periods)
STRATEGY_PARAM_GRIDS = {
    'momentum': {
        'short_window': [20, 30, 50],
        'long_window': [100, 150, 200]
    },
    'mean_reversion': {
        'rsi_period': [10, 14, 21],
        'overbought': [70, 75],
        'oversold': [25, 30]
    }
}

# Grid for optimizing system-level risk and portfolio management parameters
SYSTEM_PARAM_GRIDS = {
    'risk_system': {
        'portfolio_optimizer': [
            {'target_volatility_pct': 10.0, 'max_leverage_factor': 1.2},
            {'target_volatility_pct': 15.0, 'max_leverage_factor': 1.5},
        ],
        'portfolio_risk': [
            {'drawdown_throttles': [{'level': 0.05, 'scale': 0.5}, {'level': 0.10, 'scale': 0.25}]},
            {'drawdown_throttles': [{'level': 0.08, 'scale': 0.5}, {'level': 0.15, 'scale': 0.10}]},
        ]
    }
}

def load_all_historical_data(symbols: list, data_dir: str) -> pd.DataFrame:
    """
    Loads and combines historical CSV data for all symbols into a single
    DataFrame with multi-level columns.
    """
    all_dfs = []
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col='date', parse_dates=True)
            # Create multi-level columns: ('AAPL', 'open'), ('AAPL', 'close'), etc.
            df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
            all_dfs.append(df)
        else:
            print(f"Warning: Data file not found for symbol {symbol} at {file_path}")
    
    if not all_dfs:
        raise FileNotFoundError("No historical data files were found.")
        
    # Combine all dataframes, aligning by date index and dropping rows with any missing data
    combined_df = pd.concat(all_dfs, axis=1)
    combined_df.dropna(inplace=True)
    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run advanced backtests for trading strategies and systems.")
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=['momentum', 'mean_reversion'],
        help="The name of the strategy to backtest."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="grid_search",
        choices=['grid_search', 'walk_forward', 'scenario_analysis', 'single_run', 'monte_carlo'],
        help="The type of backtest to run."
    )
    parser.add_argument(
        "--config_set",
        type=str,
        default="strategy",
        choices=['strategy', 'risk_system'],
        help="Which set of parameters to test: 'strategy' or 'risk_system'."
    )
    parser.add_argument(
        "--log_fills_to",
        type=str,
        default=None,
        help="Optional path to a DuckDB file to log all simulated fills for TCA bootstrapping."
    )
    args = parser.parse_args()

    print(f"--- Starting Backtest Run for '{args.strategy}' in '{args.mode}' mode ---")
    if args.config_set == 'risk_system':
        print(f"--- Testing Parameter Set: '{args.config_set}' ---")

    initial_capital = config['portfolio']['initial_capital']
    symbols_to_trade = config['symbols']
    data_directory = "data/historical_csv"

    try:
        historical_data = load_all_historical_data(symbols_to_trade, data_directory)
        print(f"Data loaded for {len(symbols_to_trade)} symbols, with {len(historical_data)} trading days.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure data exists in '{data_directory}'.")
        exit(1)

    # Select the correct parameter grid based on user input
    if args.config_set == 'strategy':
        param_grid = STRATEGY_PARAM_GRIDS.get(args.strategy)
    else: # risk_system
        param_grid = SYSTEM_PARAM_GRIDS.get('risk_system')

    if not param_grid:
        print(f"Error: No parameter grid defined for config set '{args.config_set}'.")
        exit(1)
    
    # Execute the selected mode
    if args.mode == 'single_run':
        print("\nRunning single definitive backtest with production parameters...")
        # Use the strategy's default parameters from the main config
        prod_params = config['strategies'][args.strategy]
        
        results, equity_curve = run_single_backtest(
            strategy_name=args.strategy, historical_data=historical_data,
            params=prod_params, initial_capital=initial_capital,
            return_equity_curve=True, tca_log_path=args.log_fills_to
        )
        print("\n--- Single Run Performance ---")
        print(pd.Series(results).to_string())
        if args.log_fills_to:
            print(f"\nSimulated fills have been logged to '{args.log_fills_to}'")

    elif args.mode == 'grid_search':
        print("\nRunning Grid Search on full dataset...")
        results_df = run_grid_search(
            strategy_name=args.strategy, historical_data=historical_data,
            param_grid_config=param_grid, initial_capital=initial_capital,
            is_system_grid=(args.config_set == 'risk_system')
        )
        print("\n--- Grid Search Results ---")
        print(results_df.to_string())

    elif args.mode == 'walk_forward':
        print("\nRunning Walk-Forward Validation...")
        results_df = run_walk_forward_validation(
            historical_data=historical_data, strategy_name=args.strategy,
            param_grid_config=param_grid, initial_capital=initial_capital,
            is_system_grid=(args.config_set == 'risk_system')
        )
        print("\n--- Walk-Forward OOS Results ---")
        print(results_df.to_string())

    elif args.mode == 'scenario_analysis':
        print("\nStep 1: Finding best parameters via Grid Search...")
        grid_results = run_grid_search(
            strategy_name=args.strategy, historical_data=historical_data,
            param_grid_config=param_grid, initial_capital=initial_capital,
            is_system_grid=(args.config_set == 'risk_system')
        )
        if not grid_results.empty:
            best_params = {k: v for k, v in grid_results.iloc[0].items() if k in param_grid}
            print(f"\nStep 2: Testing best parameters against scenarios: {best_params}")
            
            historical_results = run_historical_scenario_analysis(
                historical_data, args.strategy, best_params, initial_capital,
                is_system_grid=(args.config_set == 'risk_system')
            )
            print("\n--- Historical Scenario Analysis Results ---")
            print(historical_results.to_string())
        else:
            print("Grid search found no viable parameters. Cannot run scenario analysis.")
            
    elif args.mode == 'monte_carlo':
        print("\nStep 1: Finding best parameters via Grid Search...")
        grid_results = run_grid_search(
            strategy_name=args.strategy, historical_data=historical_data,
            param_grid_config=param_grid, initial_capital=initial_capital
        )
        if not grid_results.empty:
            best_params = {k: v for k, v in grid_results.iloc[0].items() if k in param_grid}
            print(f"\nStep 2: Running definitive backtest with best parameters: {best_params}")
            
            _, equity_curve_df = run_single_backtest(
                strategy_name=args.strategy, historical_data=historical_data,
                params=best_params, initial_capital=initial_capital, return_equity_curve=True
            )
            
            print("\nStep 3: Running Monte Carlo simulation...")
            mc_results = run_monte_carlo_simulation(equity_curve_df, n_simulations=1000, initial_capital=initial_capital)
            
            print("\n--- Monte Carlo Simulation Results (Distribution of Outcomes) ---")
            print(json.dumps(mc_results, indent=4))

    print("\n--- Backtest Run Complete ---")
    
    if ray.is_initialized():
        ray.shutdown()
        print("\nRay has been shut down.")