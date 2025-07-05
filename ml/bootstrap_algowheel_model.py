# ml/bootstrap_algowheel_model.py

import pandas as pd
import os
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
import logging

# --- Project Root Setup ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Basic logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AlgoWheelBootstrapper")

def bootstrap_policy_model():
    """
    Creates and saves an initial policy model for the AlgoWheel based on
    a set of heuristic rules. This solves the "cold start" problem.
    """
    logger.info("--- Starting AlgoWheel Policy Model Bootstrap Process ---")

    # 1. Define Heuristic Rules
    # Context: (order_size_category, market_volatility_category) -> best_algorithm
    # This captures expert domain knowledge about execution.
    heuristic_rules = {
        # Small Orders
        ('small', 'low_vol'): 'LOC',      # Small, calm market: Be passive, save the spread.
        ('small', 'mid_vol'): 'ADAPTIVE', # Small, normal market: Balance speed and cost.
        ('small', 'high_vol'): 'MOC',      # Small, volatile market: Guarantee execution.
        
        # Medium Orders
        ('medium', 'low_vol'): 'LOC',      # Medium, calm market: Try to be passive.
        ('medium', 'mid_vol'): 'ADAPTIVE', # Medium, normal market: Balance speed and cost.
        ('medium', 'high_vol'): 'ADAPTIVE', # Medium, volatile market: Let IB's algo manage it.
        
        # Large Orders
        ('large', 'low_vol'): 'TWAP',     # Large, calm market: Work the order to minimize impact.
        ('large', 'mid_vol'): 'TWAP',     # Large, normal market: Work the order.
        ('large', 'high_vol'): 'TWAP',     # Large, volatile market: Definitely work the order.
    }
    logger.info(f"Defined {len(heuristic_rules)} heuristic rules for the policy model.")

    # 2. Generate Synthetic Training Data
    # We create multiple examples for each rule to ensure the model learns them.
    X_data = []
    y_data = []
    for (order_size, volatility), algo in heuristic_rules.items():
        for _ in range(50): # Create 50 samples per rule
            X_data.append([order_size, volatility])
            y_data.append(algo)

    X_df = pd.DataFrame(X_data, columns=['order_size_cat', 'volatility_cat'])
    y_series = pd.Series(y_data, name='target_algo')
    logger.info(f"Generated a synthetic training dataset with {len(X_df)} samples.")

    # 3. Encode Features and Labels
    # The model needs numerical inputs, so we encode the categorical features.
    feature_encoder = OrdinalEncoder(categories=[
        ['small', 'medium', 'large'],
        ['low_vol', 'mid_vol', 'high_vol']
    ])
    
    X_encoded = feature_encoder.fit_transform(X_df)
    
    # 4. Train the Decision Tree Model
    logger.info("Training the Decision Tree policy model...")
    policy_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    policy_model.fit(X_encoded, y_series)
    logger.info("Model training complete.")

    # 5. Save the Model and the Encoder
    # We must save both the model and the encoder to ensure that the live system
    # can process new data in exactly the same way.
    model_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "algowheel_policy_model.pkl")
    encoder_path = os.path.join(model_dir, "algowheel_feature_encoder.pkl")
    
    joblib.dump(policy_model, model_path)
    joblib.dump(feature_encoder, encoder_path)
    
    logger.info(f"Successfully saved policy model to: '{model_path}'")
    logger.info(f"Successfully saved feature encoder to: '{encoder_path}'")
    logger.info("--- Bootstrap Process Finished ---")

if __name__ == "__main__":
    bootstrap_policy_model()