# core/prediction/live_predictor.py
"""
Generates "live" or single-instance predictions for a specified timeframe
based on the latest available features and trained models.
This module is based on the logic from the original predict_live.py.
"""
import pandas as pd
import argparse
import os
import sys
import logging

from ..helpers.utils import load_config
from ..helpers.data_io import load_features_pkl, load_feature_list_from_txt
from ..helpers.model_ops import load_model_simple  # Using simple load as it's specific TF models

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global config dictionary, to be loaded
CONFIG = {}

# --- Default Configuration (can be overridden by config.yaml) ---
# These paths are relative to the project root if not absolute.
# The keys '5m', '15m', etc. should match keys in config.yaml's live_prediction_features_paths
DEFAULT_FEATURES_PATH_CONFIG = {
    '5m': 'data/features_5m.pkl',  # Assumes generic features file per TF
    '15m': 'data/features_15m.pkl',
    '30m': 'data/features_30m.pkl'
}
DEFAULT_MODEL_PATH_TEMPLATE = "models/{tf}_{model_type}.pkl"
DEFAULT_MODEL_FEATURES_LIST_TEMPLATE = "models/{tf}_features_selected.txt"  # Changed from _features.txt


def _get_live_pred_config_value(key, timeframe=None, default_value=None):
    """Safely retrieves a value from CONFIG, prioritizing live_prediction specific sections."""
    global CONFIG
    if not CONFIG:  # Ensure config is loaded
        CONFIG = load_config()
        if not CONFIG:
            logger.error("LivePredictor: Configuration not loaded. Using hardcoded defaults.")
            # Fallback to hardcoded defaults if config system fails entirely
            if key == "features_paths" and timeframe:
                return DEFAULT_FEATURES_PATH_CONFIG.get(timeframe)
            if key == "model_path_template":
                return DEFAULT_MODEL_PATH_TEMPLATE
            if key == "model_features_list_template":
                return DEFAULT_MODEL_FEATURES_LIST_TEMPLATE
            return default_value

    if key == "features_paths" and timeframe:
        live_pred_cfg = CONFIG.get('live_prediction_features_paths', {})
        return live_pred_cfg.get(timeframe, DEFAULT_FEATURES_PATH_CONFIG.get(timeframe))

    # For model templates, they might be more general
    if key == "model_path_template":
        # Could have a specific live_prediction.model_path_template in config,
        # or fall back to a general one if defined elsewhere (e.g. model_training section)
        return CONFIG.get('live_prediction', {}).get('model_path_template',
                                                     CONFIG.get('model_training', {}).get('model_path_template', DEFAULT_MODEL_PATH_TEMPLATE))

    if key == "model_features_list_template":
        return CONFIG.get('live_prediction', {}).get('model_features_list_template',
                                                     CONFIG.get('model_training', {}).get('model_features_list_template', DEFAULT_MODEL_FEATURES_LIST_TEMPLATE))

    return CONFIG.get(key, default_value)


def _load_latest_features_for_live(timeframe_key, symbol_filter=None):
    """
    Loads the latest row of features for a given timeframe, optionally filtered by symbol.
    Assumes features files are pre-generated and contain data for multiple symbols,
    and are sorted by timestamp if filtering for the "latest".
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    features_file_rel_path = _get_live_pred_config_value("features_paths", timeframe=timeframe_key)

    if not features_file_rel_path:
        logger.error(f"Path to features file for timeframe '{timeframe_key}' not defined in config or defaults.")
        return None

    features_file_abs_path = os.path.join(project_root, features_file_rel_path) \
        if not os.path.isabs(features_file_rel_path) else features_file_rel_path

    df_all_features = load_features_pkl(features_file_abs_path)  # Uses helper
    if df_all_features is None or df_all_features.empty:
        logger.warning(f"No features loaded from {features_file_abs_path} for timeframe {timeframe_key}.")
        return None

    df_to_process = df_all_features
    if symbol_filter:
        df_to_process = df_all_features[df_all_features['symbol'].astype(str).str.upper() == symbol_filter.upper()]
        if df_to_process.empty:
            logger.warning(f"No features found for symbol '{symbol_filter}' in {features_file_abs_path} for TF {timeframe_key}.")
            return None
        logger.info(f"Filtered features for symbol '{symbol_filter}' for TF {timeframe_key}.")

    # Get the latest row (assuming data is sorted by timestamp ascending during feature generation)
    # If multiple symbols, this gets the latest row overall, which might not be what's desired
    # unless a symbol_filter is applied.
    # If no symbol filter, this will take the absolute last row in the file.
    if not df_to_process.empty:
        # Ensure sorting by timestamp to get the true latest
        if 'timestamp' in df_to_process.columns:
            try:
                df_to_process['timestamp'] = pd.to_datetime(df_to_process['timestamp'])
                latest_row_df = df_to_process.sort_values('timestamp').iloc[-1:]
                logger.info(f"Using latest features from timestamp: {latest_row_df['timestamp'].iloc[0]} "
                            f"for symbol: {latest_row_df['symbol'].iloc[0] if 'symbol' in latest_row_df else 'N/A'}")
                return latest_row_df
            except Exception as e:
                logger.error(f"Error processing timestamp for latest features: {e}. Returning last row as is.")
                return df_to_process.iloc[-1:]  # Fallback
        else:
            logger.warning("No 'timestamp' column in features, returning the last available row.")
            return df_to_process.iloc[-1:]
    else:
        logger.warning(f"DataFrame became empty after potential filtering for TF {timeframe_key}. No latest features.")
        return None


def main_live_prediction_logic(timeframe_to_predict, symbol_to_predict=None):
    """
    Main logic for generating a live prediction for a single timeframe and optionally a symbol.

    Args:
        timeframe_to_predict (str): The timeframe to predict for (e.g., '15m').
        symbol_to_predict (str, optional): The specific symbol to predict for.
            If None, prediction is based on the absolute latest data in the feature file
            for the given timeframe (could be any symbol).
    """
    global CONFIG
    if not CONFIG:  # Ensure config is loaded
        CONFIG = load_config()
        if not CONFIG:
            logger.critical("LivePredictor: Configuration not loaded. Aborting.")
            return 1  # Error

    logger.info(f"🚀 Starting live prediction for timeframe: {timeframe_to_predict}"
                f"{f', symbol: {symbol_to_predict}' if symbol_to_predict else ''}")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # --- Load Model Paths from Config or Defaults ---
    model_path_template = _get_live_pred_config_value("model_path_template", default_value=DEFAULT_MODEL_PATH_TEMPLATE)
    # For live prediction, we typically use generic models per timeframe, not symbol-specific ones
    # unless the setup is very advanced.
    # The original predict_live.py used 'models/{tf}_clf_up.pkl', etc.
    # We assume clf_class, reg_delta, reg_vol from trainer.py

    # Construct model paths using the timeframe as the "filter_suffix" part of the template
    # e.g., models/15m_clf_class.pkl
    path_clf_class = os.path.join(project_root, model_path_template.format(tf=timeframe_to_predict, model_type='clf_class'))
    path_reg_delta = os.path.join(project_root, model_path_template.format(tf=timeframe_to_predict, model_type='reg_delta'))
    path_reg_vol = os.path.join(project_root, model_path_template.format(tf=timeframe_to_predict, model_type='reg_vol'))

    models_found = True
    for model_p in [path_clf_class, path_reg_delta, path_reg_vol]:
        if not os.path.exists(model_p):
            logger.error(f"Required model file not found: {model_p}")
            models_found = False
    if not models_found:
        logger.error(f"One or more models missing for timeframe {timeframe_to_predict}. "
                     f"Consider training models using: python train_model.py --tf {timeframe_to_predict}")
        return 1

    # --- Load Latest Features ---
    df_latest_features = _load_latest_features_for_live(timeframe_to_predict, symbol_to_predict)
    if df_latest_features is None or df_latest_features.empty:
        logger.error(f"No latest features available for TF {timeframe_to_predict}"
                     f"{f', symbol {symbol_to_predict}' if symbol_to_predict else ''}. Prediction aborted.")
        return 1

    # --- Load Feature List Used by Models ---
    feature_list_template = _get_live_pred_config_value("model_features_list_template", default_value=DEFAULT_MODEL_FEATURES_LIST_TEMPLATE)
    # Assuming generic feature list per timeframe model: models/15m_features_selected.txt
    feature_list_path = os.path.join(project_root, feature_list_template.format(tf=timeframe_to_predict))

    selected_feature_cols = load_feature_list_from_txt(feature_list_path)  # Uses helper
    if not selected_feature_cols:
        logger.error(f"Failed to load feature list from {feature_list_path} or list is empty. Prediction aborted.")
        return 1

    # Ensure all selected features are present in the loaded data
    missing_features = [col for col in selected_feature_cols if col not in df_latest_features.columns]
    if missing_features:
        logger.error(f"Live features data is missing columns required by the model: {missing_features}. Prediction aborted.")
        return 1

    X_live_data = df_latest_features[selected_feature_cols]
    if X_live_data.isnull().values.any():
        nan_cols = X_live_data.columns[X_live_data.isnull().any()].tolist()
        logger.warning(f"NaN values found in live features data for prediction: {nan_cols}. "
                       "Prediction might be inaccurate. Consider imputation if this is common.")
        # Basic imputation for robustness (e.g., fill with zero or mean), though ideally features are clean.
        # X_live_data = X_live_data.fillna(0) # Example: fill with 0

    # --- Load Models ---
    model_classifier = load_model_simple(path_clf_class)
    model_delta_reg = load_model_simple(path_reg_delta)
    model_vol_reg = load_model_simple(path_reg_vol)

    if not all([model_classifier, model_delta_reg, model_vol_reg]):
        logger.error("Failed to load one or more models. Prediction aborted.")
        return 1

    # --- Make Predictions ---
    try:
        # Classifier prediction (e.g., for UP/DOWN/NEUTRAL based on clf_class)
        # Original predict_live.py had a simplified UP/DOWN logic based on proba_up_combined.
        # Using clf_class, we get probabilities for multiple classes.
        # TARGET_CLASS_NAMES needs to be consistent with training (e.g. from config)
        # For live_predictor, it might be simpler or use a specific binary "UP" model.
        # Let's assume clf_class is binary [DOWN, UP] or [0, 1] where 1 is UP.

        proba_all_classes = model_classifier.predict_proba(X_live_data)[0]  # Get probabilities for the first (only) row

        # Determine positive class index (e.g., 'UP' or class 1)
        # This depends on how clf_class was trained and its .classes_ attribute
        model_classes_ = getattr(model_classifier, 'classes_', None)
        positive_class_label_for_proba = 1  # Default if binary 0/1
        if model_classes_ is not None:
            try:
                # Try to find 'UP' if classes are strings, or 1 if numeric
                if 'UP' in model_classes_: positive_class_label_for_proba = 'UP'
                positive_class_idx = list(model_classes_).index(positive_class_label_for_proba)
            except ValueError:
                logger.warning(f"Positive class '{positive_class_label_for_proba}' not in model classes {model_classes_}. Assuming index 1 for UP probability.")
                positive_class_idx = 1  # Fallback if 'UP' or 1 not explicitly found but model is binary-like
                if len(model_classes_) <= positive_class_idx:  # Further fallback if index 1 is out of bounds
                    logger.error(f"Cannot determine positive class index from {model_classes_}. Prediction may be incorrect.")
                    positive_class_idx = 0  # Safest fallback, but likely wrong
        else:  # model.classes_ not available
            logger.warning("model.classes_ not available. Assuming index 1 for UP probability.")
            positive_class_idx = 1
            if len(proba_all_classes) <= positive_class_idx:
                logger.error(f"Probability array too short for index {positive_class_idx}. Prediction may be incorrect.")
                positive_class_idx = 0

        proba_up_metric = proba_all_classes[positive_class_idx] if positive_class_idx < len(proba_all_classes) else 0.0

        # Simplified signal logic from original predict_live.py
        # Thresholds should ideally come from config
        signal_threshold_up = 0.55
        signal_threshold_down = 0.45
        if proba_up_metric > signal_threshold_up:
            predicted_signal_simple = 'UP'
        elif proba_up_metric < signal_threshold_down:
            predicted_signal_simple = 'DOWN'
        else:
            predicted_signal_simple = 'NEUTRAL'

        # Regression predictions
        predicted_delta = model_delta_reg.predict(X_live_data)[0]
        predicted_volatility = model_vol_reg.predict(X_live_data)[0]

    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        return 1

    # --- Display Results ---
    timestamp_val_str = "N/A"
    symbol_val_str = "N/A"
    if 'timestamp' in df_latest_features.columns:
        try:
            ts_obj = pd.to_datetime(df_latest_features['timestamp'].iloc[0])
            timestamp_val_str = ts_obj.strftime('%Y-%m-%d %H:%M:%S')
        except:
            pass  # Ignore errors if timestamp is not convertible
    if 'symbol' in df_latest_features.columns:
        symbol_val_str = df_latest_features['symbol'].iloc[0]

    print(f"\n--- Live Прогноз для {symbol_val_str} [{timeframe_to_predict}] на {timestamp_val_str} ---")
    print(f"  Вероятность UP (class {positive_class_label_for_proba}): {proba_up_metric:.2%}")
    # For debugging, show all probabilities:
    # if model_classes_ is not None:
    #     proba_dict_display = {model_classes_[i]: proba_all_classes[i] for i in range(len(proba_all_classes))}
    #     print(f"  Все вероятности классов: {proba_dict_display}")
    # else:
    #     print(f"  Все вероятности классов (raw): {proba_all_classes.tolist()}")
    print(f"  ↗️  Ожидаемое изменение (delta): {predicted_delta:.2%}")
    print(f"  ⚡ Ожидаемая волатильность: {predicted_volatility:.2%}")  # Original had .2f, assuming %
    print(f"  🚦 Сигнал (упрощенный): {predicted_signal_simple}")

    logger.info(f"Live prediction for {symbol_val_str} [{timeframe_to_predict}] at {timestamp_val_str}: "
                f"P(UP)={proba_up_metric:.2%}, Delta={predicted_delta:.2%}, Vol={predicted_volatility:.2%}, Signal='{predicted_signal_simple}'")
    return 0  # Success


if __name__ == "__main__":
    # Load config first for defaults, then parse args
    CONFIG = load_config()
    if not CONFIG:
        # Basic logging if config fails, for CLI run
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')
        logger.warning("Running live_predictor.py with default config values as config.yaml failed to load.")
    else:
        # Setup logging properly if config loaded (e.g. if it defines log levels/paths)
        # For now, just basic if __main__
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')

    # Get available timeframes for choices from config or use defaults
    available_tfs_for_live = list(
        CONFIG.get('live_prediction_features_paths', DEFAULT_FEATURES_PATH_CONFIG).keys()
    )

    parser = argparse.ArgumentParser(description="Generate a live prediction for a single timeframe.")
    parser.add_argument('--tf', type=str, default='15m', choices=available_tfs_for_live,
                        help=f"Timeframe for prediction. Available: {', '.join(available_tfs_for_live)}. Default: 15m")
    parser.add_argument('--symbol', type=str, default=None,
                        help="Optional: Specific symbol to predict for (e.g., BTCUSDT). "
                             "If not provided, uses the latest data for any symbol in the feature file.")

    args = parser.parse_args()

    exit_code = 1  # Default to error
    try:
        exit_code = main_live_prediction_logic(args.tf, args.symbol)
    except KeyboardInterrupt:
        logger.info("\n[LivePredictor] 🛑 Live prediction interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"[LivePredictor] 💥 Unexpected critical error: {e}", exc_info=True)
        sys.exit(1)

    sys.exit(exit_code)
