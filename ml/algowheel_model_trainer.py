# ml/algowheel_model_trainer.py

import duckdb
import pandas as pd
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# --- Project Root Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Basic logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AlgoWheelTrainer")

def create_training_set(tca_db_path: str) -> tuple[pd.DataFrame, pd.Series] | tuple[None, None]:
    """
    Creates a feature set and labels for training the AlgoWheel policy model.

    Returns:
        A tuple of (features_df, labels_series) or (None, None) if not possible.
    """
    if not os.path.exists(tca_db_path):
        logger.warning(f"TCA database not found at '{tca_db_path}'. Cannot train AlgoWheel model.")
        return None, None

    try:
        with duckdb.connect(tca_db_path, read_only=True) as con:
            # For a real implementation, you would join this with market data
            # to get real context like volatility, spread, etc.
            # Here, we simulate context from the available data.
            query = "SELECT algo_used, slippage, quantity FROM execution_log WHERE slippage IS NOT NULL"
            df = con.execute(query).fetchdf()
    except Exception as e:
        logger.error(f"Failed to read from TCA database: {e}")
        return None, None

    if df.empty or len(df) < 50: # Need a reasonable amount of data
        logger.warning("Not enough data in TCA log to train a meaningful model.")
        return None, None

    # 1. Engineer Contextual Features (simulated for this example)
    # Categorize order size and market condition (proxied by slippage magnitude)
    df['order_size_cat'] = pd.qcut(df['quantity'].abs(), q=3, labels=['small', 'medium', 'large'], duplicates='drop').cat.codes
    df['market_condition_cat'] = pd.qcut(df['slippage'].abs(), q=3, labels=['calm', 'normal', 'volatile'], duplicates='drop').cat.codes
    
    # 2. Engineer Labels: The "best" algorithm for each context
    # Find the algorithm with the lowest mean slippage for each context group.
    df['avg_slippage_in_context'] = df.groupby(['order_size_cat', 'market_condition_cat', 'algo_used'])['slippage'].transform('mean')
    
    # For each context, find the best performing algo
    best_algo_map = df.loc[df.groupby(['order_size_cat', 'market_condition_cat'])['avg_slippage_in_context'].idxmin()]
    best_algo_map = best_algo_map.set_index(['order_size_cat', 'market_condition_cat'])['algo_used'].to_dict()

    # Assign the best algo as the target label for each row
    df['target'] = df.set_index(['order_size_cat', 'market_condition_cat']).index.map(best_algo_map.get)
    
    df.dropna(subset=['target'], inplace=True)

    if df.empty:
        logger.warning("DataFrame is empty after label engineering. Cannot train.")
        return None, None

    features = df[['order_size_cat', 'market_condition_cat']]
    labels = df['target']
    
    logger.info(f"Created training set with {len(features)} samples.")
    return features, labels

def train_and_save_model(features: pd.DataFrame, labels: pd.Series, model_path: str):
    """Trains a Decision Tree classifier and saves it to disk."""
    if features.empty:
        logger.warning("Feature set is empty. Skipping model training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=42, stratify=labels)
    
    model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Trained AlgoWheel policy model with test accuracy: {accuracy:.2f}")
    
    # Save the trained model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"AlgoWheel policy model saved to '{model_path}'")

if __name__ == "__main__":
    logger.info("--- Starting AlgoWheel Policy Model Trainer ---")
    TCA_DB_PATH = os.path.join(PROJECT_ROOT, "data", "tca_log.duckdb")
    MODEL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "algowheel_policy_model.pkl")
    
    features, labels = create_training_set(TCA_DB_PATH)
    if features is not None and labels is not None:
        train_and_save_model(features, labels, MODEL_OUTPUT_PATH)
    logger.info("--- AlgoWheel Policy Model Trainer Finished ---")