# backtester/distributed_runner.py

import ray
import pandas as pd
import itertools
import logging
from typing import List, Dict, Any, Tuple

from backtester.engine import BacktestEngine
from strategy.momentum import MovingAverageCrossoverStrategy
from strategy.mean_reversion import RsiMeanReversionStrategy
from core.config_loader import config as global_config

logger = logging.getLogger("DistributedRunner")

# A mapping from strategy names to their classes for easy instantiation
STRATEGY_MAP = {
    'momentum': MovingAverageCrossoverStrategy,
    'mean_reversion': RsiMeanReversionStrategy
}

@ray.remote
def run_backtest_remotely(
    strategy_name: str, 
    data_ref, 
    params: Dict[str, Any],
    initial_capital: float, 
    is_system_params: bool,
    return_equity_curve: bool = False,
    tca_log_path: str = None
) -> Dict[str, Any] | Tuple[Dict[str, Any], pd.DataFrame]:
    """
    A Ray remote function to run a single backtest instance. This is the
    core worker task for all backtesting operations. It can handle both
    strategy parameter overrides and system config overrides, and can
    optionally log simulated fills for TCA.
    """
    # Apply config overrides if testing system parameters
    if is_system_params:
        import copy
        from core.config_loader import config as worker_config
        
        # Create a deep copy for this worker to avoid race conditions
        local_config = copy.deepcopy(worker_config)
        
        for key, value in params.items():
            if key in local_config:
                local_config[key].update(value)
        
        # The strategy will use its default parameters from the now-modified config
        strategy_params = local_config['strategies'][strategy_name]
    else:
        # If testing strategy params, the 'params' dict is passed directly to the strategy
        strategy_params = params
        local_config = global_config

    historical_data = ray.get(data_ref)
    strategy_class = STRATEGY_MAP[strategy_name]
    symbols = list(set(col[0] for col in historical_data.columns))
    
    # The strategy is instantiated inside the remote worker
    strategy = strategy_class(event_queue=None, symbols=symbols, state_file=None, **strategy_params)
    
    engine = BacktestEngine(
        historical_data=historical_data,
        strategy=strategy,
        initial_capital=initial_capital,
        config_override=local_config,
        tca_log_path=tca_log_path
    )
    
    performance_metrics, equity_curve_df = engine.run()
    
    # Combine the input parameters with the output metrics for a complete result record
    result = params.copy()
    result.update(performance_metrics)
    
    if return_equity_curve:
        return result, equity_curve_df
    else:
        return result

def _generate_parameter_grid(param_config: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generates all combinations of parameters from a config dictionary."""
    keys = param_config.keys()
    values = param_config.values()
    
    value_combinations = list(itertools.product(*values))
    
    grid = [dict(zip(keys, v)) for v in value_combinations]
    return grid

def run_grid_search(strategy_name: str, historical_data: pd.DataFrame,
                    param_grid_config: Dict[str, List], initial_capital: float,
                    is_system_grid: bool = False) -> pd.DataFrame:
    """
    Orchestrates a distributed grid search of backtests using Ray.
    """
    if not ray.is_initialized():
        ray.init(logging_level=logging.ERROR)

    data_ref = ray.put(historical_data)
    param_combinations = _generate_parameter_grid(param_grid_config)
    
    futures = [
        run_backtest_remotely.remote(strategy_name, data_ref, params, initial_capital, is_system_grid)
        for params in param_combinations
    ]

    results = ray.get(futures)
    
    if not results: 
        return pd.DataFrame()
        
    results_df = pd.DataFrame(results)
    if "Sharpe Ratio" in results_df.columns:
        results_df = results_df.sort_values(by="Sharpe Ratio", ascending=False).reset_index(drop=True)
        
    return results_df

def run_single_backtest(
    strategy_name: str, 
    historical_data: pd.DataFrame,
    params: Dict[str, Any], 
    initial_capital: float,
    is_system_params: bool = False, 
    return_equity_curve: bool = False,
    tca_log_path: str = None
) -> Any:
    """
    Orchestrates a single backtest run, optionally returning the equity curve
    and logging simulated fills to a specified path.
    """
    if not ray.is_initialized():
        ray.init(logging_level=logging.ERROR)

    data_ref = ray.put(historical_data)
    
    future = run_backtest_remotely.remote(
        strategy_name, 
        data_ref, 
        params, 
        initial_capital, 
        is_system_params, 
        return_equity_curve, 
        tca_log_path
    )
    result = ray.get(future)
    
    return result