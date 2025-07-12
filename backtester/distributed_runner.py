# backtester/distributed_runner.py

import ray
import pandas as pd
import itertools
import logging
from typing import List, Dict, Any, Tuple, Optional
import copy

from backtester.engine import BacktestEngine
from core.config_loader import config as global_config

# --- NOTE: The STRATEGY_MAP is no longer needed here ---
# The LivePortfolioManager now handles the creation of strategies based on the config.
# This makes the runner more generic and decoupled from specific strategy implementations.

logger = logging.getLogger("DistributedRunner")

@ray.remote
def run_backtest_remotely(
    strategy_name: Optional[str],
    data_ref,
    params: Dict[str, Any],
    initial_capital: float,
    is_system_params: bool,
    trading_universe: List[str], # Add this parameter
    return_equity_curve: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Any:
    """
    A remote Ray task to run a single backtest instance.
    MODIFIED to work with the new BacktestEngine that internally manages strategies.
    The engine is now responsible for creating the LivePortfolioManager, which in
    turn creates the necessary strategies based on the provided configuration.
    """
    if start_date and end_date:
        historical_data = data_ref.loc[start_date:end_date]
    else:
        historical_data = data_ref

    # Deepcopy the global config to ensure this run is isolated.
    local_config = copy.deepcopy(global_config)

    # The primary role of this function is now to correctly override the
    # configuration for this specific backtest run based on the `params`
    # from the grid search.

    if is_system_params:
        # The `params` dict can contain both system-level and strategy-level overrides.
        # This logic updates the local_config with the parameters for this run.
        for key, value in params.items():
            if key in local_config: # e.g., 'portfolio_optimizer', 'portfolio_risk'
                local_config[key].update(value)
            elif key in local_config.get('strategies', {}):
                # This allows overriding specific strategy params during a system run,
                # for example, testing different window sizes for a strategy.
                local_config['strategies'][key].update(value)
    else:
        # This block handles the case where we are optimizing parameters for just
        # a single strategy, not the whole system.
        if strategy_name and strategy_name in local_config.get('strategies', {}):
            local_config['strategies'][strategy_name].update(params)
        else:
            logger.warning(f"Single-strategy run specified for '{strategy_name}', but it was not found in the config. Running with default parameters.")


    # --- SIMPLIFIED ENGINE INITIALIZATION ---
    # The BacktestEngine no longer takes a 'strategies' list. It uses the
    # config_override to create the LivePortfolioManager, which then creates
    # the strategies based on that same config.
    engine = BacktestEngine(
        historical_data=historical_data,
        initial_capital=initial_capital,
        config_override=local_config,
        trading_universe=trading_universe # Pass it down
    )

    performance_metrics, equity_curve_df = engine.run()

    # Attach the input parameters to the results for easy analysis.
    result = params.copy()
    result.update(performance_metrics)

    if return_equity_curve:
        return result, equity_curve_df
    else:
        return result

def _generate_parameter_grid(param_config: Dict[str, List]) -> List[Dict[str, Any]]:
    """Helper function to create all combinations from a parameter grid config."""
    keys = param_config.keys()
    values = param_config.values()
    value_combinations = list(itertools.product(*values))
    grid = [dict(zip(keys, v)) for v in value_combinations]
    return grid

def run_grid_search(
    strategy_name: Optional[str],
    data_ref,
    param_grid_config: Dict[str, List],
    initial_capital: float,
    is_system_grid: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Runs a grid search by fanning out remote backtest tasks to Ray.
    """
    param_combinations = _generate_parameter_grid(param_grid_config)

    futures = [
        run_backtest_remotely.remote(
            strategy_name=strategy_name,
            data_ref=data_ref,
            params=params,
            initial_capital=initial_capital,
            is_system_params=is_system_grid,
            return_equity_curve=False,
            start_date=start_date,
            end_date=end_date
        ) for params in param_combinations
    ]
    results = ray.get(futures)

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    if "Sharpe Ratio" in results_df.columns:
        results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)

    return results_df

def run_single_backtest(
    strategy_name: Optional[str],
    data_ref,
    params: Dict[str, Any],
    initial_capital: float,
    is_system_params: bool = False,
    return_equity_curve: bool = False,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Any:
    """
    Runs a single backtest instance remotely on Ray.
    """
    future = run_backtest_remotely.remote(
        strategy_name=strategy_name,
        data_ref=data_ref,
        params=params,
        initial_capital=initial_capital,
        is_system_params=is_system_params,
        return_equity_curve=return_equity_curve,
        start_date=start_date,
        end_date=end_date
    )
    return ray.get(future)