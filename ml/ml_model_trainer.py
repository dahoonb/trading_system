# ml/ml_model_trainer.py
import argparse
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score
import feast
import mlflow
import mlflow.sklearn
import joblib
import yaml
import os
import datetime
import logging
import sys
import json
from typing import Dict, Any, Optional, List, Tuple
import yfinance as yf
import hashlib
from tqdm import tqdm

# --- Project Root Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Logger Setup ---
try:
    from utils.logger import setup_logger
    logger = setup_logger(logger_name="MLModelTrainer", also_log_to_console=True)
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)-8s - %(message)s')
    logger = logging.getLogger("MLModelTrainer_Fallback")

def stable_id(symbol: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(symbol.encode("utf-8"), digest_size=8).digest(),
        "big",
        signed=True
    )

def get_app_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    abs_config_path = os.path.join(PROJECT_ROOT, config_path)
    try:
        with open(abs_config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"Config file not found at '{abs_config_path}'.")
        raise

def get_sp500_tickers() -> list[str]:
    try:
        payload = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = payload[0]['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        logger.info(f"Successfully fetched {len(tickers)} S&P 500 tickers for ML training universe.")
        return tickers
    except Exception as e:
        logger.error(f"Could not fetch S&P 500 tickers: {e}. Aborting.")
        return []

def fetch_training_data(
    config: Dict[str, Any],
    start_date_str: str,
    end_date_str: str,
    symbol_universe: List[str]
) -> Optional[pd.DataFrame]:
    logger.info(f"Fetching training data for {len(symbol_universe)} symbols from {start_date_str} to {end_date_str}...")
    feature_repo_path = os.path.join(PROJECT_ROOT, config.get('feature_repo_path', 'feature_repo'))
    
    try:
        store = feast.FeatureStore(repo_path=feature_repo_path)
        
        batch_size = 50
        all_batch_dfs = []
        
        technical_feature_refs = [
            "all_ticker_features:close", "all_ticker_features:sma_20", "all_ticker_features:sma_200",
            "all_ticker_features:rsi_14", "all_ticker_features:atr_20", "all_ticker_features:vol_regime",
            "all_ticker_features:oer_5d", "all_ticker_features:realized_vol_20d",
            "all_ticker_features:cs_rank_21d", "all_ticker_features:cs_rank_63d",
            "all_ticker_features:cs_rank_126d", "all_ticker_features:cs_rank_252d",
        ]
        fundamental_feature_refs = [
            "fundamental_features:price_to_book", "fundamental_features:piotroski_f_score",
            "fundamental_features:shareholder_yield",
        ]
        
        num_batches = (len(symbol_universe) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(symbol_universe), batch_size), total=num_batches, desc="Fetching feature data in batches"):
            batch_symbols = symbol_universe[i:i + batch_size]
            
            date_range = pd.to_datetime(pd.date_range(start=start_date_str, end=end_date_str, freq='B'))
            entity_df_list = [
                pd.DataFrame({"event_timestamp": date_range, "ticker_id": stable_id(symbol)})
                for symbol in batch_symbols
            ]
            entity_df = pd.concat(entity_df_list, ignore_index=True)
            entity_df["ticker_id"] = entity_df["ticker_id"].astype("int64")
            
            historical_job = store.get_historical_features(entity_df=entity_df, features=technical_feature_refs)
            historical_df = historical_job.to_df()

            if historical_df.empty:
                logger.warning(f"No historical data for batch starting with {batch_symbols[0]}. Skipping.")
                continue

            online_entity_rows = [{"ticker_id": stable_id(s)} for s in batch_symbols]
            online_features_df = store.get_online_features(
                features=fundamental_feature_refs,
                entity_rows=online_entity_rows
            ).to_df()

            batch_training_df = pd.merge(historical_df, online_features_df, on="ticker_id", how="left")
            all_batch_dfs.append(batch_training_df)

        if not all_batch_dfs:
            logger.error("No data could be fetched for any batch. Aborting.")
            return None

        logger.info("All batches fetched. Concatenating into final training DataFrame...")
        training_df = pd.concat(all_batch_dfs, ignore_index=True)

        logger.info("Engineering cross-sectional target variable based on 5-day forward returns...")
        forward_days = 5
        training_df.sort_values(by=['ticker_id', 'event_timestamp'], inplace=True)
        training_df['fwd_return'] = training_df.groupby('ticker_id')['close'].shift(-forward_days) / training_df['close'] - 1
        training_df.dropna(subset=['fwd_return', 'close'], inplace=True)

        training_df['fwd_return_rank'] = training_df.groupby('event_timestamp')['fwd_return'].rank(pct=True)

        top_quintile_threshold = 0.8
        training_df['target'] = np.where(training_df['fwd_return_rank'] >= top_quintile_threshold, 1, 0)
        training_df['target'] = training_df['target'].astype(int)
        
        target_dist = training_df['target'].value_counts(normalize=True)
        logger.info(f"Target variable distribution: Class 0: {target_dist.get(0, 0):.2%}, Class 1: {target_dist.get(1, 0):.2%}")
        
        logger.info(f"Successfully fetched and prepared {len(training_df)} data points for training.")
        return training_df

    except Exception as e:
        logger.error(f"Failed to fetch or process training data: {e}", exc_info=True)
        return None

def train_model(ml_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[Optional[lgb.LGBMClassifier], Optional[StandardScaler], pd.DataFrame, pd.Series]:
    """Trains the ML model using LightGBM with a robust time-based split."""
    logger.info("Starting model training with LightGBM...")
    
    logger.info("Splitting data into train, validation, and test sets based on time...")
    ml_df = ml_df.sort_values('event_timestamp').reset_index(drop=True)
    X = ml_df[feature_cols]
    y = ml_df['target']

    n = len(ml_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    logger.info(f"Train set size: {len(X_train)}, Val set size: {len(X_val)}, Test set size: {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
        'n_estimators': 2000, 'learning_rate': 0.01, 'num_leaves': 20,
        'max_depth': 5, 'seed': 42, 'n_jobs': -1, 'verbose': -1,
        'colsample_bytree': 0.7, 'subsample': 0.7, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
    }

    model = lgb.LGBMClassifier(**lgb_params)
    
    # FIX: Pass feature names to fit method to avoid warnings later
    model.fit(X_train_scaled, y_train,
              eval_set=[(X_val_scaled, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)],
              feature_name=feature_cols)
    
    logger.info("Model training completed.")
    return model, scaler, X_test, y_test

def evaluate_model(model, scaler, X_test, y_test) -> Dict[str, Any]:
    logger.info("Evaluating model performance...")
    # FIX: Convert scaled numpy array back to DataFrame to preserve feature names
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    metrics = {
        "test_roc_auc": roc_auc_score(y_test, y_pred_proba),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1_score": f1_score(y_test, y_pred, zero_division=0)
    }
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

def load_production_metrics(metrics_path: str) -> Dict[str, Any]:
    if not os.path.exists(metrics_path):
        logger.warning("No production metrics file found. Assuming baseline performance.")
        return {"test_roc_auc": 0.5}
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load production metrics from {metrics_path}: {e}")
        return {"test_roc_auc": 0.5}

def save_to_production(model, scaler, metrics, model_dir, metrics_path):
    logger.info(f"Saving new model and metrics to production at {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "ml_vetoing_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info("Promotion successful.")

def main(cli_args):
    config = get_app_config(cli_args.config_file)
    ml_config = config.get('ml', {})
    model_dir = os.path.join(PROJECT_ROOT, ml_config.get('model_directory', 'models'))
    prod_metrics_path = os.path.join(model_dir, "production_metrics.json")
    
    training_universe = get_sp500_tickers()
    if not training_universe:
        logger.critical("Could not get training universe, aborting.")
        return

    train_end_date = cli_args.end_date or datetime.date.today().strftime('%Y-%m-%d')
    train_start_date = cli_args.start_date or (pd.to_datetime(train_end_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')

    mlflow.set_experiment(ml_config.get('mlflow_experiment_name', 'AlphaExcessReturnPrediction'))
    
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params({"start_date": train_start_date, "end_date": train_end_date, "universe_size": len(training_universe)})

        champion_metrics = load_production_metrics(prod_metrics_path)
        champion_auc = champion_metrics.get("test_roc_auc", 0.5)
        logger.info(f"Current Champion Model AUC: {champion_auc:.4f}")

        ml_df = fetch_training_data(config, train_start_date, train_end_date, symbol_universe=training_universe)
        
        if ml_df is None or ml_df.empty:
            logger.critical("Data fetching failed. Aborting training run.")
            return

        feature_cols = [
            "sma_20", "sma_200", "rsi_14", "atr_20", "vol_regime", 
            "oer_5d", "realized_vol_20d", 
            "price_to_book", "piotroski_f_score", "shareholder_yield",
            "cs_rank_21d", "cs_rank_63d", "cs_rank_126d", "cs_rank_252d",
        ]
        
        for col in feature_cols:
            if col not in ml_df.columns:
                ml_df[col] = 0
        ml_df[feature_cols] = ml_df[feature_cols].fillna(0)

        challenger_model, challenger_scaler, X_test, y_test = train_model(ml_df, feature_cols)
        challenger_metrics = evaluate_model(challenger_model, challenger_scaler, X_test, y_test)
        challenger_auc = challenger_metrics.get("test_roc_auc", 0.0)
        mlflow.log_metrics(challenger_metrics)
        logger.info(f"Challenger Model AUC: {challenger_auc:.4f}")

        promotion_threshold = config.get('ml_pipeline', {}).get('promotion_threshold', 0.02)
        
        if challenger_auc > champion_auc + promotion_threshold:
            logger.info("DECISION: Challenger model is superior. Promoting to production.")
            mlflow.set_tag("promotion_decision", "PROMOTED")
            save_to_production(challenger_model, challenger_scaler, challenger_metrics, model_dir, prod_metrics_path)
            print("ML Trainer Decision: PROMOTED")
        else:
            logger.info("DECISION: Challenger did not meet promotion criteria. Keeping champion model.")
            mlflow.set_tag("promotion_decision", "KEPT_CHAMPION")
            print("ML Trainer Decision: KEPT_CHAMPION")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and promote a cost-aware ML model.")
    parser.add_argument("--config_file", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--start_date", type=str, default=None, help="Start date for training data (YYYY-MM-DD).")
    parser.add_argument("--end_date", type=str, default=None, help="End date for training data (YYYY-MM-DD).")
    cli_args = parser.parse_args()
    main(cli_args)