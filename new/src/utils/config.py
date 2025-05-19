# src/utils/config.py

import yaml
import os
import sys
import logging

# Configure a basic logger for the config loading process itself
# This logger will be replaced by the main logging setup later,
# but is useful for errors during initial config loading.
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

_config = None
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/default.yaml')

def load_config(config_path=DEFAULT_CONFIG_PATH):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    global _config
    if _config is not None:
        # Config is already loaded, return the existing one
        return _config

    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        sys.exit(1) # Exit if config file is essential and not found

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            _config = yaml.safe_load(f)
        logger.info(f"Configuration loaded successfully from {config_path}")
        return _config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading config {config_path}: {e}")
        sys.exit(1)

def get_config():
    """
    Returns the loaded configuration. Loads it if not already loaded.

    Returns:
        dict: The configuration dictionary.
    """
    if _config is None:
        return load_config()
    return _config

# Example usage (optional, for testing the module directly)
if __name__ == "__main__":
    # This block will only run if you execute config.py directly
    print("Attempting to load configuration...")
    try:
        config = get_config()
        print("Config loaded:")
        # print(yaml.dump(config, indent=2)) # Uncomment to print the full config
        print(f"DB Path: {config['paths']['db']}")
        print(f"Default Timeframes: {config['timeframes']['default']}")
        print(f"Top 8 Symbols: {config['symbol_groups']['top8']}")
    except Exception as e:
        print(f"Failed to load or access config: {e}")