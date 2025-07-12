# run_portfolio_recalibration.py

import pandas as pd
import os
import argparse
import logging
import yaml
import ray
from ruamel.yaml import YAML
from typing import List, Dict, Any

# --- Use the system-level grid and the main data loader ---
from run_backtest import SYSTEM_PARAM_GRIDS, load_all_historical_data
from backtester.validation import run_walk_forward_validation, run_monte_carlo_simulation, run_historical_scenario_analysis, run_hypothetical_stress_test
from backtester.distributed_runner import run_single_backtest, run_grid_search
from core.config_loader import config
from utils.logger import setup_logger

# Setup a dedicated logger for this critical process
setup_logger(logger_name="PortfolioRecalibration", log_level=logging.INFO)
logger = logging.getLogger("PortfolioRecalibration")

def check_portfolio_mandate(validation_results: dict, mandate: dict) -> tuple[bool, List[str]]:
    """
    Checks if the combined portfolio's validation results pass the performance mandate.
    This is identical to the previous check_mandate function.
    """
    failures = []
    logger.info("--- Checking Portfolio System Against Validation Mandate ---")

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

    # Check Monte Carlo Results
    mc_results = validation_results.get('monte_carlo')
    mc_mandate = mandate.get('monte_carlo_mandate', {})
    if mc_results:
        cagr_5p = mc_results.get('CAGR (%)', {}).get('5%', -99)
        dd_5p = abs(mc_results.get('Max Drawdown (%)', {}).get('5%', 99))

        if cagr_5p < mc_mandate.get('min_cagr_at_5th_percentile', 0.0):
            failures.append(f"Monte Carlo FAIL: 5th Pctile CAGR {cagr_5p:.2f}% < Mandate {mc_mandate.get('min_cagr_at_5th_percentile', 0.0):.2f}%")
        if dd_5p > mc_mandate.get('max_drawdown_at_5th_percentile', 20.0):
            failures.append(f"Monte Carlo FAIL: 5th Pctile Drawdown {dd_5p:.2f}% > Mandate {mc_mandate.get('max_drawdown_at_5th_percentile', 20.0):.2f}%")

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

    return len(failures) == 0, failures

def update_production_config(new_params: Dict[str, Any]):
    """
    Programmatically updates the portfolio-level sections of config.yaml.
    """
    logger.info(f"Promoting new PORTFOLIO parameters to production config: {new_params}")
    yaml_processor = YAML()
    yaml_processor.preserve_quotes = True
    config_path = 'config.yaml'

    with open(config_path, 'r') as f:
        prod_config = yaml_processor.load(f)

    # Update the specific portfolio sections
    if 'portfolio_optimizer' in new_params:
        prod_config['portfolio_optimizer'].update(new_params['portfolio_optimizer'])
    if 'portfolio_risk' in new_params:
        prod_config['portfolio_risk'].update(new_params['portfolio_risk'])

    with open(config_path, 'w') as f:
        yaml_processor.dump(prod_config, f)

    logger.info("Production config.yaml has been successfully updated with new portfolio parameters.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full PORTFOLIO recalibration and promotion pipeline.")
    args = parser.parse_args()

    logger.info("--- Starting Full PORTFOLIO Recalibration Pipeline ---")
    logger.info("This process validates the entire system, including all enabled strategies and portfolio-level risk management.")

    try:
        historical_data = load_all_historical_data(config['symbols'], "data/historical_csv")
        with open('validation_mandate.yaml', 'r') as f:
            mandate = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError) as e:
        logger.critical(f"Failed to load necessary files: {e}. Aborting."); exit(1)

    if not ray.is_initialized():
        ray.init(logging_level=logging.ERROR, ignore_reinit_error=True)
    data_ref = ray.put(historical_data)

    # Use the system-level parameter grid for optimization
    param_grid = SYSTEM_PARAM_GRIDS['risk_system']

    logger.info("\nStep 1: Running Walk-Forward Optimization on the full portfolio system...")
    validation_results = {}
    oos_results_df = run_walk_forward_validation(
        data_ref=data_ref,
        full_historical_data=historical_data,
        strategy_name=None,  # Set to None to signal a full system run
        param_grid_config=param_grid,
        initial_capital=config['portfolio']['initial_capital'],
        is_system_grid=True, # This is the key flag
        n_splits=10,
        train_period_years=4,
        test_period_years=1
    )
    validation_results['walk_forward'] = oos_results_df
    print("\n--- Portfolio Walk-Forward Optimization Out-of-Sample Results ---")
    print(oos_results_df.to_string())

    logger.info("\nStep 2: Finding best system parameters on most recent data for MC/Scenario tests...")
    recent_train_start = (historical_data.index[-1] - pd.DateOffset(years=4)).strftime('%Y-%m-%d')
    grid_results = run_grid_search(
        strategy_name=None, # Full system run
        data_ref=data_ref,
        param_grid_config=param_grid,
        initial_capital=config['portfolio']['initial_capital'],
        is_system_grid=True,
        start_date=recent_train_start,
        end_date=historical_data.index[-1].strftime('%Y-%m-%d')
    )

    if grid_results.empty:
        logger.error("Grid search on recent data failed. Aborting final validation.")
        exit(1)

    best_current_params = {k: v for k, v in grid_results.iloc[0].items() if k in param_grid}
    logger.info(f"Best system parameters for current regime: {best_current_params}")

    logger.info("\nStep 3: Running concluding validation on best current system parameters...")
    _, equity_curve_df = run_single_backtest(
        strategy_name=None, # Full system run
        data_ref=data_ref,
        params=best_current_params,
        initial_capital=config['portfolio']['initial_capital'],
        is_system_params=True,
        return_equity_curve=True
    )
    equity_curve_df.to_csv(f"results/final_portfolio_equity.csv")
    logger.info("Saved final portfolio equity curve to 'results/final_portfolio_equity.csv'")

    validation_results['monte_carlo'] = run_monte_carlo_simulation(equity_curve_df, initial_capital=config['portfolio']['initial_capital'])
    validation_results['scenarios'] = run_historical_scenario_analysis(
        data_ref=data_ref,
        full_historical_data=historical_data,
        strategy_name=None, # Full system run
        best_params=best_current_params,
        initial_capital=config['portfolio']['initial_capital'],
        is_system_grid=True
    )

    logger.info("\nStep 4: Checking full validation suite against performance mandate...")
    passed, failures = check_portfolio_mandate(validation_results, mandate)

    if passed:
        logger.info("\n--- DECISION: PORTFOLIO MANDATE PASSED ---")
        logger.info("Promoting new portfolio-level parameters to live production configuration.")
        update_production_config(best_current_params)
    else:
        logger.warning("\n--- DECISION: PORTFOLIO MANDATE FAILED ---")
        logger.warning("The portfolio system's re-optimization process did not meet the performance mandate. Production config will not be changed.")
        logger.warning("Failure reasons:")
        for reason in failures:
            logger.warning(f"  - {reason}")

    if ray.is_initialized():
        ray.shutdown()

    logger.info("\n--- Portfolio Recalibration Pipeline Finished ---")