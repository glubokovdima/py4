# core/helpers/__init__.py

"""
Helpers Submodule.

This submodule provides a collection of utility functions and classes
that are reused across different parts of the 'core' application.
It includes helpers for:
- Configuration loading
- Command-line interactions and script execution
- Binance API communication
- Data input/output operations (e.g., loading/saving features)
- Database operations (SQLite interaction)
- Model loading and saving operations
- Common prediction logic and calculations
- General utility functions (e.g., time conversions, classification)
"""

from .utils import (
    load_config,
    print_header,
    run_script,
    select_timeframes_interactive,
    clear_training_artifacts_interactive,
    ensure_base_directories,
    get_tf_ms,
    classify_delta_value  # Moved from preprocess_features to be more general
)

from .binance_api import (
    get_valid_symbols_binance,
    fetch_klines_binance
)

from .data_io import (
    load_features_pkl,
    save_features_pkl,    # Added for completeness
    save_sample_csv,      # Added for completeness
    load_feature_list_from_txt, # Added for completeness
    save_feature_list_to_txt    # Added for completeness
)

from .db_ops import (
    init_db,
    insert_klines_sqlite,
    get_last_timestamp_sqlite,
    load_candles_from_sqlite # Renamed for clarity (was load_candles_from_db)
)

from .model_ops import (
    load_model_with_fallback,
    load_model_simple, # For loading a single model file
    save_model_joblib # Added for completeness
)

from .prediction_logic import (
    compute_final_delta,
    get_signal_strength,
    is_conflict,
    get_confidence_hint,
    calculate_trade_levels,
    similarity_analysis
)


# Define __all__ to specify what is exported when 'from core.helpers import *' is used.
# This helps in keeping the namespace clean and explicit.
__all__ = [
    # From utils.py
    "load_config", "print_header", "run_script", "select_timeframes_interactive",
    "clear_training_artifacts_interactive", "ensure_base_directories", "get_tf_ms",
    "classify_delta_value",

    # From binance_api.py
    "get_valid_symbols_binance", "fetch_klines_binance",

    # From data_io.py
    "load_features_pkl", "save_features_pkl", "save_sample_csv",
    "load_feature_list_from_txt", "save_feature_list_to_txt",

    # From db_ops.py
    "init_db", "insert_klines_sqlite", "get_last_timestamp_sqlite", "load_candles_from_sqlite",

    # From model_ops.py
    "load_model_with_fallback", "load_model_simple", "save_model_joblib",

    # From prediction_logic.py
    "compute_final_delta", "get_signal_strength", "is_conflict", "get_confidence_hint",
    "calculate_trade_levels", "similarity_analysis",
]
