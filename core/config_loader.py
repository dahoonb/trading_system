# core/config_loader.py

import yaml
from typing import Dict, Any

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the main YAML configuration file for the trading system.

    This function reads the specified YAML file and returns its contents
    as a Python dictionary, making all parameters easily accessible
    throughout the application.

    Args:
        path (str): The file path to the configuration file.

    Returns:
        Dict[str, Any]: A dictionary containing all configuration parameters.
    
    Raises:
        FileNotFoundError: If the configuration file cannot be found at the
                         specified path.
        yaml.YAMLError: If the file is not a valid YAML document.
    """
    try:
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration file 'config.yaml' loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at '{path}'.")
        raise
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing configuration file '{path}': {e}")
        raise

# Load the configuration once at startup to be imported by other modules
config = load_config()