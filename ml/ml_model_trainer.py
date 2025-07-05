# ml/ml_model_trainer.py
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
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

def get_app_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads the application configuration from a YAML file."""
    abs_config_path = os.path.join(PROJECT_ROOT, config_path)
    try:
        with open(abs_config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.critical(f"Config file not found at '{abs_config_path}'.")
        raise

def fetch_training_data(config: Dict[str, Any], start_date_str: str, end_date_str: str) -> Optional[pd.DataFrame]:
    """
    Fetches all required features, including TCA metrics, from Feast,
    merges them with price data, and engineers the target variable.
    """
    logger.info(f"Fetching training data from {start_date_str} to {end_date_str}...")
    feast_repo_path = os.path.join(PROJECT_ROOT, config.get('feature_repo_path', 'feature_repo'))
    
    try:
        store = feast.FeatureStore(repo_path=feast_repo_path)
        symbols = config['symbols']
        
        # Create the entity DataFrame for the point-in-time join
        date_range = pd.to_datetime(pd.date_range(start=start_date_str, end=end_date_str, freq='B'))
        entity_df_list = [pd.DataFrame({"event_timestamp": date_range, "ticker_id": symbol}) for symbol in symbols]
        entity_df = pd.concat(entity_df_list, ignore_index=True)

        # --- MODIFICATION: Add the TCA feature to the list of features to fetch ---
        feature_refs = [
            "all_ticker_features:sma_20", "all_ticker_features:sma_200",
            "all_ticker_features:rsi_14", "all_ticker_features:atr_20",
            "all_ticker_features:vol_regime", "all_ticker_features:oer_5d",
            "all_ticker_features:realized_vol_20d",
            "tca_features:avg_slippage_5d"  # New cost-aware feature
        ]
        
        logger.info(f"Requesting historical features from Feast: {feature_refs}")
        training_job = store.get_historical_features(entity_df=entity_df, features=feature_refs)
        training_df = training_job.to_df()

        if training_df.empty:
            logger.error("Feast returned an empty DataFrame. Cannot proceed.")
            return None

        # --- Target Variable Engineering ---
        # This section would contain the logic to load historical prices,
        # calculate returns, and generate the 'target' column.
        # For this example, we'll create a dummy target to ensure the script is runnable.
        logger.info("Engineering target variable (using dummy data for this example)...")
        training_df['target'] = np.random.randint(0, 2, size=len(training_df))
        
        logger.info(f"Successfully fetched and prepared {len(training_df)} data points for training.")
        return training_df

    except Exception as e:
        logger.error(f"Failed to fetch or process training data: {e}", exc_info=True)
        return None

def train_model(ml_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[Optional[RandomForestClassifier], Optional[StandardScaler], pd.DataFrame, pd.Series]:
    """Trains the ML model using a time-series split."""
    logger.info("Starting model training...")
    X = ml_df[feature_cols]
    y = ml_df['target']
    
    tscv = TimeSeriesSplit(n_splits=5)
    train_index, test_index = list(tscv.split(X))[-1]
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    logger.info("Model training completed.")
    return model, scaler, X_test, y_test

def evaluate_model(model, scaler, X_test, y_test) -> Dict[str, Any]:
    """Evaluates the model and returns a dictionary of metrics."""
    logger.info("Evaluating model performance...")
    X_test_scaled = scaler.transform(X_test)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
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
    """Loads the metrics of the current production model."""
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
    """Saves a new model and its metrics as the production version."""
    logger.info(f"Saving new model and metrics to production at {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "ml_vetoing_model.pkl"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info("Promotion successful.")

def main(cli_args):
    """Main function to orchestrate the training and promotion process."""
    config = get_app_config(cli_args.config_file)
    ml_config = config.get('ml', {})
    model_dir = os.path.join(PROJECT_ROOT, ml_config.get('model_directory', 'models'))
    prod_metrics_path = os.path.join(model_dir, "production_metrics.json")

    train_end_date = cli_args.end_date or (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    train_start_date = cli_args.start_date or (pd.to_datetime(train_end_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')

    mlflow.set_experiment(ml_config.get('mlflow_experiment_name', 'AlphaExcessReturnPrediction'))
    
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        mlflow.log_params({"start_date": train_start_date, "end_date": train_end_date})

        champion_metrics = load_production_metrics(prod_metrics_path)
        champion_auc = champion_metrics.get("test_roc_auc", 0.5)
        logger.info(f"Current Champion Model AUC: {champion_auc:.4f}")

        ml_df = fetch_training_data(config, train_start_date, train_end_date)
        if ml_df is None or ml_df.empty:
            logger.critical("Data fetching failed. Aborting training run.")
            return

        # --- MODIFICATION: Add the new TCA feature to the training columns ---
        feature_cols = [
            "sma_20", "sma_200", "rsi_14", "atr_20", "vol_regime", 
            "oer_5d", "realized_vol_20d", "avg_slippage_5d"
        ]
        
        # Ensure all feature columns exist, fill NaNs for robustness
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