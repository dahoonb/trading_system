# run_recalibration.py

import pandas as pd
import os
import argparse
import logging
import yaml
import ray
from ruamel.yaml import YAML
from typing import List, Dict

from backtester.distributed_runner import run_grid_search, run_single_backtest
from backtester.validation import run_walk_forward_validation, run_monte_carlo_simulation, run_historical_scenario_analysis
from core.config_loader import config
from utils.logger import setup_logger
from run_backtest import STRATEGY_PARAM_GRIDS as PARAM_GRIDS

# Setup a dedicated logger for this critical process
setup_logger(logger_name="RecalibrationRun", log_level=logging.INFO)
logger = logging.getLogger("RecalibrationRun")

def load_all_historical_data(symbols: list, data_dir: str) -> pd.DataFrame:
    """Loads and combines historical CSV data for all symbols."""
    all_dfs = []
    for symbol in symbols:
        file_path = os.path.join(data_dir, f"{symbol}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col='date', parse_dates=True)
            df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
            all_dfs.append(df)
    if not all_dfs:
        raise FileNotFoundError("No historical data files were found.")
    combined_df = pd.concat(all_dfs, axis=1).dropna()
    return combined_df

def check_mandate(validation_results: dict, mandate: dict) -> tuple[bool, List[str]]:
    """Checks if the validation results pass the performance mandate."""
    failures = []
    logger.info("--- Checking Candidate Against Validation Mandate ---")

    # Check Walk-Forward Results
    wf_results = validation_results.get('walk_forward')
    wf_mandate = mandate.get('walk_forward_mandate', {})
    if wf_results is not None and not wf_results.empty:
        avg_sharpe = wf_results['Sharpe Ratio'].mean()
        min_sharpe = wf_results['Sharpe Ratio'].min()
        avg_dd = abs(wf_results['Max Drawdown (%)'].mean())

        if avg_sharpe < wf_mandate.get('min_avg_sharpe_ratio', 1.0):
            failures.append(f"Walk-Forward FAIL: Avg Sharpe {avg_sharpe:.2f} < Mandate {wf_mandate.get('min_avg_sharpe_ratio', 1.0):.2f}")
        if min_sharpe < wf_mandate.get('min_sharpe_ratio_in_any_fold', 0.5):
            failures.append(f"Walk-Forward FAIL: Min Sharpe {min_sharpe:.2f} < Mandate {wf_mandate.get('min_sharpe_ratio_in_any_fold', 0.5):.2f}")
        if avg_dd > wf_mandate.get('max_avg_drawdown_pct', 15.0):
            failures.append(f"Walk-Forward FAIL: Avg Drawdown {avg_dd:.2f}% > Mandate {wf_mandate.get('max_avg_drawdown_pct', 15.0):.2f}%")
    else:
        failures.append("Walk-Forward FAIL: No results were generated.")

    # Check Monte Carlo Results
    mc_results = validation_results.get('monte_carlo')
    mc_mandate = mandate.get('monte_carlo_mandate', {})
    if mc_results:
        cagr_stats = mc_results.get("CAGR (%)", {})
        dd_stats = mc_results.get("Max Drawdown (%)", {})
        cagr_5p = cagr_stats.get('5%', -999)
        dd_5p = abs(dd_stats.get('5%', 999))

        if cagr_5p < mc_mandate.get('min_cagr_at_5th_percentile', 0.0):
            failures.append(f"Monte Carlo FAIL: 5th Pctile CAGR {cagr_5p:.2f}% < Mandate {mc_mandate.get('min_cagr_at_5th_percentile', 0.0):.2f}%")
        if dd_5p > mc_mandate.get('max_drawdown_at_5th_percentile', 20.0):
            failures.append(f"Monte Carlo FAIL: 5th Pctile Drawdown {dd_5p:.2f}% > Mandate {mc_mandate.get('max_drawdown_at_5th_percentile', 20.0):.2f}%")
    else:
        failures.append("Monte Carlo FAIL: No results were generated.")

    # Check Scenario Results
    sc_results = validation_results.get('scenarios')
    sc_mandate = mandate.get('scenario_mandate', {})
    if sc_results is not None and not sc_results.empty:
        gfc_row = sc_results[sc_results['scenario'] == '2008_GFC']
        covid_row = sc_results[sc_results['scenario'] == '2020_COVID_Crash']
        if not gfc_row.empty and abs(gfc_row['Max Drawdown (%)'].iloc[0]) > sc_mandate.get('max_drawdown_in_2008_gfc_pct', 25.0):
            failures.append(f"Scenario FAIL: GFC Drawdown {abs(gfc_row['Max Drawdown (%)'].iloc[0]):.2f}% > Mandate {sc_mandate.get('max_drawdown_in_2008_gfc_pct', 25.0):.2f}%")
        if not covid_row.empty and abs(covid_row['Max Drawdown (%)'].iloc[0]) > sc_mandate.get('max_drawdown_in_2020_covid_pct', 20.0):
            failures.append(f"Scenario FAIL: COVID Drawdown {abs(covid_row['Max Drawdown (%)'].iloc[0]):.2f}% > Mandate {sc_mandate.get('max_drawdown_in_2020_covid_pct', 20.0):.2f}%")
    else:
        failures.append("Scenario FAIL: No results were generated.")

    return len(failures) == 0, failures

def update_production_config(strategy_name: str, new_params: dict):
    """Programmatically updates the main config.yaml with new parameters, preserving comments."""
    logger.info(f"Promoting new parameters to production config: {new_params}")
    yaml_processor = YAML()
    yaml_processor.preserve_quotes = True
    config_path = 'config.yaml'

    with open(config_path, 'r') as f:
        prod_config = yaml_processor.load(f)

    # Update the specific strategy's parameters
    for key, value in new_params.items():
        if key in prod_config['strategies'][strategy_name]:
            # Convert numpy types to native Python types for YAML serialization
            if isinstance(value, (pd.Int64Dtype, pd.Float64Dtype, int, float)):
                 prod_config['strategies'][strategy_name][key] = float(value) if '.' in str(value) else int(value)
            else:
                 prod_config['strategies'][strategy_name][key] = value

    with open(config_path, 'w') as f:
        yaml_processor.dump(prod_config, f)

    logger.info("Production config.yaml has been successfully updated. The live system will use these new parameters on its next run.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full recalibration and promotion pipeline.")
    parser.add_argument("--strategy", type=str, required=True, choices=['momentum_ma', 'mean_reversion_rsi'])
    args = parser.parse_args()

    logger.info(f"--- Starting Full Recalibration Pipeline for '{args.strategy}' ---")
    logger.info("This process validates the *method* of periodic re-optimization, not a single static parameter set.")

    try:
        historical_data = load_all_historical_data(config['symbols'], "data/historical_csv")
        with open('validation_mandate.yaml', 'r') as f:
            mandate = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.critical(f"Failed to load necessary files: {e}. Aborting."); exit(1)

    if not ray.is_initialized():
        ray.init(logging_level=logging.ERROR, ignore_reinit_error=True)
    data_ref = ray.put(historical_data)

    param_grid = PARAM_GRIDS[args.strategy]
    validation_results = {}

    # --- STEP 1: RUN WALK-FORWARD OPTIMIZATION ---
    # This is now the primary test. It simulates re-optimizing the strategy each year.
    logger.info("\nStep 1: Running Walk-Forward Optimization...")
    oos_results_df = run_walk_forward_validation(
        data_ref=data_ref,
        full_historical_data=historical_data,
        strategy_name=args.strategy,
        param_grid_config=param_grid,
        initial_capital=config['portfolio']['initial_capital'],
        n_splits=10,
        train_period_years=4,
        test_period_years=1
    )
    validation_results['walk_forward'] = oos_results_df
    print("\n--- Walk-Forward Optimization Out-of-Sample Results ---")
    print(oos_results_df.to_string())

    # --- STEP 2: FIND BEST *CURRENT* PARAMETERS FOR FINAL CHECKS ---
    # To run MC and Scenario tests, we need a single parameter set.
    # The most logical choice is the one we would deploy *today*, which is found
    # by optimizing on the most recent training data.
    logger.info("\nStep 2: Finding best parameters on most recent data for MC/Scenario tests...")
    train_years = 4
    recent_train_start = (historical_data.index[-1] - pd.DateOffset(years=train_years)).strftime('%Y-%m-%d')
    grid_results = run_grid_search(
        strategy_name=args.strategy,
        data_ref=data_ref,
        param_grid_config=param_grid,
        initial_capital=config['portfolio']['initial_capital'],
        start_date=recent_train_start,
        end_date=historical_data.index[-1].strftime('%Y-%m-%d')
    )
    if grid_results.empty:
        logger.error("Grid search on recent data failed. Aborting."); exit(1)

    best_current_params = {k: v for k, v in grid_results.iloc[0].items() if k in param_grid}
    logger.info(f"Best parameters for the current regime found: {best_current_params}")

    # --- STEP 3: RUN CONCLUDING VALIDATION ON CURRENT BEST PARAMETERS ---
    logger.info("\nStep 3: Running concluding validation on best current parameters...")
    _, equity_curve_df = run_single_backtest(
        strategy_name=args.strategy,
        data_ref=data_ref,
        params=best_current_params,
        initial_capital=config['portfolio']['initial_capital'],
        return_equity_curve=True
    )
    equity_curve_df.to_csv(f"results/final_candidate_equity_{args.strategy}.csv")
    logger.info(f"Saved final candidate equity curve to 'results/final_candidate_equity_{args.strategy}.csv'")

    validation_results['monte_carlo'] = run_monte_carlo_simulation(equity_curve_df, initial_capital=config['portfolio']['initial_capital'])
    validation_results['scenarios'] = run_historical_scenario_analysis(
        data_ref=data_ref,
        full_historical_data=historical_data,
        strategy_name=args.strategy,
        best_params=best_current_params,
        initial_capital=config['portfolio']['initial_capital']
    )

    # --- STEP 4: FINAL DECISION ---
    logger.info("\nStep 4: Checking full validation suite against performance mandate...")
    passed, failures = check_mandate(validation_results, mandate)

    if passed:
        logger.info("\n--- DECISION: MANDATE PASSED ---")
        logger.info("The process of periodic re-optimization is robust. Promoting most recent best parameters to live configuration.")
        update_production_config(args.strategy, best_current_params)
    else:
        logger.warning("\n--- DECISION: MANDATE FAILED ---")
        logger.warning("The strategy's re-optimization process did not meet the performance mandate. Production config will not be changed.")
        logger.warning("Failure reasons:")
        for reason in failures:
            logger.warning(f"  - {reason}")

    if ray.is_initialized():
        ray.shutdown()

    logger.info("\n--- Recalibration Pipeline Finished ---")