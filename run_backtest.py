# run_backtest.py

import pandas as pd
import os
import argparse
import logging
import ray
import json
from typing import Dict, List, Optional

from backtester.distributed_runner import run_grid_search, run_single_backtest
from backtester.validation import run_walk_forward_validation, run_monte_carlo_simulation, run_historical_scenario_analysis, run_hypothetical_stress_test
from core.config_loader import config
from utils.logger import setup_logger
from etl.fundamental_feature_etl import get_sp500_tickers # Import the function

# Setup loggers for all relevant modules to ensure comprehensive output
setup_logger(logger_name="BacktestRun", log_level=logging.INFO)
setup_logger(logger_name="ValidationFramework", log_level=logging.INFO)
setup_logger(logger_name="DistributedRunner", log_level=logging.INFO)
setup_logger(logger_name="BacktestEngine", log_level=logging.INFO)

# ==============================================================================
# --- CENTRALIZED PARAMETER GRIDS FOR RESEARCH & RECALIBRATION ---
# ==============================================================================

# These grids define the universe of parameters to be tested for individual strategies.
# They are used for strategy-specific optimization and validation.
STRATEGY_PARAM_GRIDS: Dict[str, Dict[str, List]] = {
    'momentum_ma': {
        # --- Adaptive MA Parameters ---
        'fast_ma_windows': [(20, 50), (30, 60)],
        'slow_ma_windows': [(50, 150), (60, 200)],
        'volatility_regime_period': [100, 120],
        
        # --- Defensive Filter Parameters ---
        'volatility_threshold_min': [0.01, 0.015],
        'volatility_threshold_max': [0.04, 0.06],
        'adx_period': [14, 21],
        'adx_threshold': [20, 25],
        'trailing_stop_atr_multiplier': [2.5, 3.0],
        'market_roc_period': [21],
        'market_roc_threshold': [-0.08, -0.10]
    },
    'mean_reversion_rsi': {
        'rsi_period': [10, 14, 21],
        'overbought_threshold': [70, 75, 80],
        'oversold_threshold': [20, 25, 30],
        'atr_period': [20],
        'volatility_threshold_min': [0.01, 0.015],
        'volatility_threshold_max': [0.04, 0.05, 0.06],
        'stop_loss_atr_multiplier': [1.5, 2.0, 2.5]
    }
}

# This grid defines parameters for portfolio-level systems.
# It's used to test how different risk management settings affect the
# combined performance of all enabled strategies.
SYSTEM_PARAM_GRIDS: Dict[str, Dict[str, List]] = {
    'risk_system': {
        # --- Portfolio Optimizer Settings ---
        'portfolio_optimizer': [
            # Test different risk appetites
            {'target_volatility_pct': 10.0, 'max_leverage_factor': 1.2},
            {'target_volatility_pct': 15.0, 'max_leverage_factor': 1.5},
            {'target_volatility_pct': 20.0, 'max_leverage_factor': 1.8}
        ],
        # --- Dynamic Risk Manager Settings ---
        'portfolio_risk': [
            # Test different drawdown sensitivities
            {'drawdown_throttles': [{'level': 0.05, 'scale': 0.5}, {'level': 0.10, 'scale': 0.25}]},
            {'drawdown_throttles': [{'level': 0.08, 'scale': 0.4}, {'level': 0.15, 'scale': 0.1}]}
        ]
    }
}

def load_all_historical_data(symbols: list, data_dir: str) -> pd.DataFrame:
    """
    Loads and combines historical CSV data for all specified symbols into a
    single multi-indexed DataFrame.
    """
    all_dfs = []
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col='date', parse_dates=True)
            # Create a multi-index column structure: (Symbol, Feature)
            df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
            all_dfs.append(df)
        else:
            logging.warning(f"Data file not found for symbol {symbol} at {file_path}")
    
    if not all_dfs:
        raise FileNotFoundError(f"No historical data files were found in '{data_dir}'.")
        
    # Combine all individual DataFrames and drop rows with any missing data
    combined_df = pd.concat(all_dfs, axis=1)
    combined_df.dropna(inplace=True)
    return combined_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run advanced backtests for trading strategies and systems.")
    parser.add_argument(
        "--strategy", type=str,
        choices=['momentum_ma', 'mean_reversion_rsi', 'full_system'],
        help="The specific strategy to test, or 'full_system' to test the entire portfolio."
    )
    parser.add_argument(
        "--mode", type=str, default="grid_search",
        choices=['grid_search', 'walk_forward', 'monte_carlo', 'scenario_analysis'],
        help="The type of backtest to run."
    )
    
    args = parser.parse_args()

    # Determine if this is a full system run or a single strategy run
    is_system_run = (args.strategy == 'full_system')
    
    if is_system_run:
        print(f"--- Starting Backtest Run for FULL SYSTEM in '{args.mode}' mode ---")
        param_grid = SYSTEM_PARAM_GRIDS['risk_system']
        strategy_name_for_runner = None # Signal to runner to use all enabled strategies
    else:
        print(f"--- Starting Backtest Run for '{args.strategy}' in '{args.mode}' mode ---")
        if args.strategy not in STRATEGY_PARAM_GRIDS:
            print(f"Error: No parameter grid defined for strategy '{args.strategy}'."); exit(1)
        param_grid = STRATEGY_PARAM_GRIDS[args.strategy]
        strategy_name_for_runner = args.strategy

    initial_capital = config['portfolio']['initial_capital']
    data_directory = "data/historical_csv"
    
    # --- NEW: Define the two universes ---
    # Alpha Universe: The broad set for ranking and context
    alpha_universe_symbols = get_sp500_tickers() 
    if not alpha_universe_symbols:
        print("Could not fetch alpha universe. Aborting.")
        exit(1)

    # Trading Universe: The smaller, executable set from the config
    trading_universe_symbols = config['symbols']

    # --- CHANGE: Load data for the FULL alpha universe ---
    try:
        # This will now load data for all 500+ stocks
        historical_data = load_all_historical_data(alpha_universe_symbols, data_directory)
        print(f"Data loaded for {len(alpha_universe_symbols)} symbols in the alpha universe.")
    except FileNotFoundError as e:
        print(f"Error: {e}"); exit(1)


    # Initialize Ray for distributed processing
    if not ray.is_initialized():
        ray.init(logging_level=logging.ERROR, ignore_reinit_error=True)
    data_ref = ray.put(historical_data)

    # --- Execute the chosen backtesting mode ---

    if args.mode == 'grid_search':
        results_df = run_grid_search(
            strategy_name=strategy_name_for_runner, 
            data_ref=data_ref,
            param_grid_config=param_grid, 
            initial_capital=initial_capital,
            is_system_grid=is_system_run,
            trading_universe=trading_universe_symbols # Pass the smaller list
        )
        print("\n--- Grid Search Results ---")
        print(results_df.to_string())

    elif args.mode == 'walk_forward':
        results_df = run_walk_forward_validation(
            data_ref=data_ref,
            full_historical_data=historical_data,
            strategy_name=strategy_name_for_runner,
            param_grid_config=param_grid,
            initial_capital=initial_capital,
            is_system_grid=is_system_run,
            trading_universe=trading_universe_symbols # Pass the smaller list
        )
        print("\n--- Walk-Forward Out-of-Sample Results ---")
        print(results_df.to_string())

    elif args.mode in ['monte_carlo', 'scenario_analysis']:
        print("\nStep 1: Finding best parameters via Grid Search to use for analysis...")
        grid_results = run_grid_search(
            strategy_name=strategy_name_for_runner, 
            data_ref=data_ref,
            param_grid_config=param_grid, 
            initial_capital=initial_capital,
            is_system_grid=is_system_run,
            trading_universe=trading_universe_symbols # Pass the smaller list
        )
        if grid_results.empty:
            print("Grid search found no viable parameters. Cannot proceed.")
        else:
            # Select the best parameters (top row after sorting by Sharpe)
            best_params = {k: v for k, v in grid_results.iloc[0].items() if k in param_grid}
            print(f"\nStep 2: Using best parameters for further analysis: {best_params}")
            
            if args.mode == 'monte_carlo':
                # Run a single backtest with the best params to get the equity curve
                _, equity_curve_df = run_single_backtest(
                    strategy_name=strategy_name_for_runner, data_ref=data_ref,
                    params=best_params, initial_capital=initial_capital,
                    is_system_params=is_system_run, return_equity_curve=True,
                    trading_universe=trading_universe_symbols # Pass the smaller list
                )
                print("\nStep 3: Running Monte Carlo simulation...")
                mc_results = run_monte_carlo_simulation(equity_curve_df, n_simulations=1000, initial_capital=initial_capital)
                print("\n--- Monte Carlo Simulation Results ---")
                print(json.dumps(mc_results, indent=4))

            elif args.mode == 'scenario_analysis':
                print("\nStep 3: Running Scenario and Stress tests...")
                historical_results = run_historical_scenario_analysis(
                    data_ref=data_ref, full_historical_data=historical_data,
                    strategy_name=strategy_name_for_runner, best_params=best_params, 
                    initial_capital=initial_capital, is_system_grid=is_system_run,
                    trading_universe=trading_universe_symbols # Pass the smaller list
                )
                hypothetical_results = run_hypothetical_stress_test(
                    data_ref=data_ref, full_historical_data=historical_data,
                    strategy_name=strategy_name_for_runner, best_params=best_params, 
                    initial_capital=initial_capital, is_system_grid=is_system_run,
                    trading_universe=trading_universe_symbols # Pass the smaller list
                )
                
                print("\n--- Historical Scenario Analysis Results ---")
                print(historical_results.to_string())
                print("\n--- Hypothetical Stress Test Results ---")
                print(hypothetical_results.to_string())

    print("\n--- Backtest Run Complete ---")
    
    if ray.is_initialized():
        ray.shutdown()