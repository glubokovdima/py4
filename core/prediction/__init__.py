# core/prediction/__init__.py

"""
Prediction Submodule.

This submodule is responsible for generating predictions using trained models.
It includes functionalities for:
- Predicting on the latest available features (predictor.py, from predict_all.py).
- Generating "live" or single-instance predictions (live_predictor.py).
- Generating predictions across multiple timeframes for a single symbol
  (multiframe_predictor.py).
"""

# From predictor.py (original predict_all.py)
from .predictor import (
    main_predict_all_logic,
    # Expose internal functions if they are useful standalone and refactored for it
    # e.g., calculate_final_predictions_for_symbol, if it's made reusable
)

# From live_predictor.py (original predict_live.py)
from .live_predictor import main_live_prediction_logic

# From multiframe_predictor.py (original predict_multiframe.py)
from .multiframe_predictor import main_multiframe_prediction_logic


__all__ = [
    "main_predict_all_logic",
    "main_live_prediction_logic",
    "main_multiframe_prediction_logic",
]
