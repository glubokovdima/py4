# core/features/__init__.py

"""
Feature Engineering Submodule.

This submodule is responsible for:
1. Building initial features from raw candle data (builder.py, from build_features.py).
2. Preprocessing loaded candle data to compute technical indicators,
   target variables, and other relevant features for model training
   and prediction (preprocessor.py, from preprocess_features.py).
"""

# From builder.py (original build_features.py)
# This script seemed to be a standalone preprocessor for specific CSVs,
# so its main function might not be directly exposed here unless it's generalized.
# If it's meant to be a step in a larger feature pipeline, its function can be imported.
# For now, let's assume it's more of a utility script or its logic is merged elsewhere.
# from .builder import preprocess_all_csv_features # Example if it were generalized

# From preprocessor.py (original preprocess_features.py)
from .preprocessor import (
    main_preprocess_logic,
    # load_candles_from_db_for_features, # Exposing for potential reuse
    compute_and_prepare_features       # Exposing for potential reuse
)

__all__ = [
    "main_preprocess_logic",
    # "load_candles_from_db_for_features",
    "compute_and_prepare_features",
    # "preprocess_all_csv_features", # If exposed from builder
]
