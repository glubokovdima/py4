# src/utils/logging_setup.py

import logging.config
import yaml
import os
import sys

# Import the main config loader to get paths
from src.utils.config import get_config

# Define the default path to the logging config YAML
DEFAULT_LOGGING_CONFIG_PATH = os.path.join(os.path.dirname(__file__), '../../config/logging.yaml')

def setup_logging(logging_config_path=DEFAULT_LOGGING_CONFIG_PATH):
    """
    Sets up logging based on the configuration in the specified YAML file.
    Uses paths from the main application config (default.yaml).

    Args:
        logging_config_path (str): Path to the logging configuration YAML file.
    """
    if not os.path.exists(logging_config_path):
        print(f"Error: Logging configuration file not found at {logging_config_path}", file=sys.stderr)
        # Fallback to basic logging if config file is missing
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [LoggingSetup] - Logging config file not found. Falling back to basic logging.')
        logging.warning(f"Logging configuration file not found at {logging_config_path}. Falling back to basic logging.")
        return

    try:
        with open(logging_config_path, 'r', encoding='utf-8') as f:
            logging_config = yaml.safe_load(f)

        # --- Dynamically update file paths using paths from default.yaml ---
        main_config = get_config() # Load main config to get paths
        logs_dir = main_config['paths']['logs_dir']
        data_dir = main_config['paths']['data_dir']

        # Ensure log directories exist
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True) # update_log.txt goes here

        # Update paths in file handlers in the loaded config dictionary
        for handler_name, handler_config in logging_config.get('handlers', {}).items():
            if 'class' in handler_config and ('FileHandler' in handler_config['class'] or 'RotatingFileHandler' in handler_config['class']):
                # Assuming the filename is relative to the project root as defined in default.yaml
                # Update the filename to be an absolute path or relative to where the script is run
                # Let's make it relative to the script's cwd where the logs/data dirs are expected to be
                original_filename = handler_config.get('filename')
                if original_filename:
                    # Check if the path is intended for logs_dir or data_dir based on the original placeholder
                    if 'logs/' in original_filename.replace('\\', '/'): # Use replace for cross-platform path check
                         handler_config['filename'] = os.path.join(logs_dir, os.path.basename(original_filename))
                    elif 'data/' in original_filename.replace('\\', '/'):
                         handler_config['filename'] = os.path.join(data_dir, os.path.basename(original_filename))
                    # If it's neither, leave it as is or handle as needed. Assuming logs/ or data/ are the only file targets for now.


        # --- Apply the configuration ---
        logging.config.dictConfig(logging_config)
        logging.info("Logging configured successfully.")

    except yaml.YAMLError as e:
        print(f"Error parsing logging configuration file {logging_config_path}: {e}", file=sys.stderr)
        # Fallback to basic logging on parse error
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [LoggingSetup] - Error parsing logging config. Falling back to basic logging.')
        logging.error(f"Error parsing logging configuration file {logging_config_path}: {e}. Falling back to basic logging.")
    except Exception as e:
        print(f"An unexpected error occurred during logging setup from {logging_config_path}: {e}", file=sys.stderr)
        # Fallback to basic logging on other errors
        logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [LoggingSetup] - Unexpected error during logging setup. Falling back to basic logging.')
        logging.error(f"An unexpected error occurred during logging setup from {logging_config_path}: {e}. Falling back to basic logging.")

# Example usage (optional, for testing the module directly)
if __name__ == "__main__":
    # Ensure the config is loaded first for setup_logging
    # get_config() # setup_logging will call get_config internally

    print("Attempting to setup logging...")
    setup_logging()

    # Get loggers by name
    main_logger = logging.getLogger(__name__)
    update_logger = logging.getLogger("update_data") # Get the specific logger

    # Test logging levels and handlers
    main_logger.debug("This is a debug message from main_logger.") # Should appear if root level is DEBUG
    main_logger.info("This is an info message from main_logger.")   # Should go to console and file_pipeline
    main_logger.warning("This is a warning message from main_logger.") # Should go to console and file_pipeline
    main_logger.error("This is an error message from main_logger.")   # Should go to console and file_pipeline

    update_logger.debug("This is a debug message from update_data logger.") # Should appear if update_data level is DEBUG (set in yaml)
    update_logger.info("This is an info message from update_data logger.") # Should go to console and file_update
    update_logger.warning("This is a warning message from update_data logger.") # Should go to console and file_update
    update_logger.error("This is an error message from update_data logger.") # Should go to console and file_update

    # Example of a logger not explicitly configured (inherits from root)
    other_logger = logging.getLogger("some_other_module")
    other_logger.info("This is an info message from another logger.") # Should go to console and file_pipeline

    print("Logging test complete.")