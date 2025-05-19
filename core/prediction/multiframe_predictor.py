# core/prediction/multiframe_predictor.py
"""
Generates predictions for a single symbol across multiple specified timeframes,
aggregating the results to provide an overall trend assessment.
This module is based on the logic from the original predict_multiframe.py.
"""
import pandas as pd
import argparse
import os
import sys
import logging

from ..helpers.utils import load_config
from ..helpers.data_io import load_features_pkl, load_feature_list_from_txt
from ..helpers.model_ops import load_model_simple

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global config dictionary, to be loaded
CONFIG = {}

# --- Default Configuration (can be overridden by config.yaml) ---
DEFAULT_MODEL_DIR_MULTI = "models"  # Used if model path template is relative
DEFAULT_FEATURES_PATH_TEMPLATE = "data/features_{tf}.pkl"  # Generic features per TF
DEFAULT_MODEL_PATH_TEMPLATE = "models/{tf}_{model_type}.pkl"
DEFAULT_MODEL_FEATURES_LIST_TEMPLATE = "models/{tf}_features_selected.txt"
DEFAULT_MULTIFRAME_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

# Thresholds for simplified signal generation (can be moved to config)
PROB_UP_THRESHOLD_LONG = 0.70  # Original script had 0.7
DELTA_THRESHOLD_LONG = 0.002
PROB_UP_THRESHOLD_SHORT = 0.30  # Original script had 0.3 (meaning P(UP) < 0.3 for SHORT)
DELTA_THRESHOLD_SHORT = -0.002


def _get_multiframe_config_value(key, default_value=None):
    """Safely retrieves a value from CONFIG, prioritizing multiframe_prediction specific sections if any."""
    global CONFIG
    if not CONFIG:  # Ensure config is loaded
        CONFIG = load_config()
        if not CONFIG:
            logger.error("MultiFramePredictor: Configuration not loaded. Using hardcoded defaults.")
            if key == "timeframes_list": return DEFAULT_MULTIFRAME_TIMEFRAMES
            if key == "model_path_template": return DEFAULT_MODEL_PATH_TEMPLATE
            if key == "features_path_template": return DEFAULT_FEATURES_PATH_TEMPLATE
            if key == "model_features_list_template": return DEFAULT_MODEL_FEATURES_LIST_TEMPLATE
            return default_value

    # Example: if you had a 'multiframe_prediction' section in config.yaml
    # multiframe_cfg = CONFIG.get('multiframe_prediction', {})
    # val = multiframe_cfg.get(key)
    # if val is not None: return val

    # Fallback to general config values or defaults
    if key == "timeframes_list":
        return CONFIG.get('predict_multiframe_timeframes', DEFAULT_MULTIFRAME_TIMEFRAMES)
    if key == "model_path_template":
        return CONFIG.get('model_training', {}).get('model_path_template', DEFAULT_MODEL_PATH_TEMPLATE)
    if key == "features_path_template":  # Assuming generic features per TF for multiframe
        return CONFIG.get('feature_engineering', {}).get('features_path_template', DEFAULT_FEATURES_PATH_TEMPLATE)
    if key == "model_features_list_template":
        return CONFIG.get('model_training', {}).get('model_features_list_template', DEFAULT_MODEL_FEATURES_LIST_TEMPLATE)

    return CONFIG.get(key, default_value)


def _predict_for_single_symbol_tf(symbol_arg, timeframe_arg, project_root_path):
    """
    Generates predictions for a single symbol on a single timeframe.
    This function encapsulates the core prediction logic for one TF.
    """
    logger.info(f"Generating prediction for {symbol_arg} on timeframe {timeframe_arg}")

    # --- Load Paths from Config or Defaults ---
    model_path_template = _get_multiframe_config_value("model_path_template")
    features_path_template = _get_multiframe_config_value("features_path_template")
    model_features_list_template = _get_multiframe_config_value("model_features_list_template")

    # --- Construct full paths ---
    # Models are assumed to be generic per timeframe for this predictor
    path_clf_class = os.path.join(project_root_path, model_path_template.format(tf=timeframe_arg, model_type='clf_class'))
    path_reg_delta = os.path.join(project_root_path, model_path_template.format(tf=timeframe_arg, model_type='reg_delta'))
    path_reg_vol = os.path.join(project_root_path, model_path_template.format(tf=timeframe_arg, model_type='reg_vol'))

    # Check if all model files exist
    models_exist = True
    for model_p in [path_clf_class, path_reg_delta, path_reg_vol]:
        if not os.path.exists(model_p):
            logger.warning(f"Model file not found: {model_p} for {symbol_arg} [{timeframe_arg}]. Skipping this TF.")
            models_exist = False
            break
    if not models_exist:
        return None

    # Load features for the specified timeframe
    # The original predict_multiframe.py tried to load data_binance/{symbol}/{tf}.csv
    # and then call compute_features_stub.
    # Here, we assume pre-generated features_{tf}.pkl exist, containing multiple symbols.
    features_file_rel_path = features_path_template.format(tf=timeframe_arg)  # e.g. data/features_15m.pkl
    features_file_abs_path = os.path.join(project_root_path, features_file_rel_path) \
        if not os.path.isabs(features_file_rel_path) else features_file_rel_path

    df_all_tf_features = load_features_pkl(features_file_abs_path)
    if df_all_tf_features is None or df_all_tf_features.empty:
        logger.warning(f"No features loaded from {features_file_abs_path} for TF {timeframe_arg}. Skipping.")
        return None

    # Filter for the specific symbol and get the latest row
    df_symbol_features = df_all_tf_features[df_all_tf_features['symbol'].astype(str).str.upper() == symbol_arg.upper()]
    if df_symbol_features.empty:
        logger.warning(f"No features found for symbol '{symbol_arg}' in {features_file_abs_path} for TF {timeframe_arg}. Skipping.")
        return None

    # Get the latest features for this symbol on this TF
    if 'timestamp' in df_symbol_features.columns:
        try:
            df_symbol_features['timestamp'] = pd.to_datetime(df_symbol_features['timestamp'])
            latest_feature_row = df_symbol_features.sort_values('timestamp').iloc[-1:].copy()
        except Exception as e:
            logger.error(f"Error processing timestamp for {symbol_arg} [{timeframe_arg}]: {e}. Using last row as is.")
            latest_feature_row = df_symbol_features.iloc[-1:].copy()
    else:
        logger.warning(f"No 'timestamp' column for {symbol_arg} [{timeframe_arg}]. Using last available row.")
        latest_feature_row = df_symbol_features.iloc[-1:].copy()

    if latest_feature_row.empty:  # Should not happen if df_symbol_features was not empty
        logger.warning(f"Could not get latest feature row for {symbol_arg} [{timeframe_arg}]. Skipping.")
        return None

    # Load the list of features the model was trained on
    feature_list_rel_path = model_features_list_template.format(tf=timeframe_arg)  # e.g. models/15m_features_selected.txt
    feature_list_abs_path = os.path.join(project_root_path, feature_list_rel_path) \
        if not os.path.isabs(feature_list_rel_path) else feature_list_rel_path

    selected_feature_cols = load_feature_list_from_txt(feature_list_abs_path)
    if not selected_feature_cols:
        logger.warning(f"Failed to load feature list from {feature_list_abs_path} for TF {timeframe_arg} or list is empty. Skipping.")
        return None

    missing_features = [col for col in selected_feature_cols if col not in latest_feature_row.columns]
    if missing_features:
        logger.warning(f"Latest features for {symbol_arg} [{timeframe_arg}] are missing columns required by the model: {missing_features}. Skipping.")
        return None

    X_predict_data = latest_feature_row[selected_feature_cols]
    if X_predict_data.isnull().values.any():
        nan_cols = X_predict_data.columns[X_predict_data.isnull().any()].tolist()
        logger.warning(f"NaN values found in features for {symbol_arg} [{timeframe_arg}]: {nan_cols}. Prediction may be inaccurate.")
        # Consider filling NaNs if this is a common issue: X_predict_data = X_predict_data.fillna(0)

    # Load models
    model_clf = load_model_simple(path_clf_class)
    model_delta = load_model_simple(path_reg_delta)
    model_vol = load_model_simple(path_reg_vol)

    if not all([model_clf, model_delta, model_vol]):
        logger.warning(f"Failed to load one or more models for {symbol_arg} [{timeframe_arg}]. Skipping.")
        return None

    # Make predictions
    try:
        proba_all_classes = model_clf.predict_proba(X_predict_data)[0]

        # Determine positive class index (e.g., 'UP' or class 1)
        model_classes_ = getattr(model_clf, 'classes_', None)
        positive_class_label_for_proba = 1  # Default for binary 0/1
        positive_class_idx = 1  # Default index
        if model_classes_ is not None:
            try:
                if 'UP' in model_classes_: positive_class_label_for_proba = 'UP'
                positive_class_idx = list(model_classes_).index(positive_class_label_for_proba)
            except ValueError:  # Fallback
                if len(model_classes_) > 1:
                    positive_class_idx = 1
                else:
                    positive_class_idx = 0
        elif len(proba_all_classes) > 1:
            positive_class_idx = 1
        else:
            positive_class_idx = 0

        prob_up_value = proba_all_classes[positive_class_idx] if positive_class_idx < len(proba_all_classes) else 0.0

        delta_value = model_delta.predict(X_predict_data)[0]
        vol_value = model_vol.predict(X_predict_data)[0]
    except Exception as e:
        logger.error(f"Error during prediction for {symbol_arg} [{timeframe_arg}]: {e}", exc_info=True)
        return None

    momentum_value = delta_value / vol_value if vol_value > 1e-9 else 0.0  # Avoid division by zero

    # Simplified signal logic from original predict_multiframe.py
    signal_text_val = "NEUTRAL"
    if prob_up_value > PROB_UP_THRESHOLD_LONG and delta_value > DELTA_THRESHOLD_LONG:
        signal_text_val = "LONG"
    elif prob_up_value < PROB_UP_THRESHOLD_SHORT and delta_value < DELTA_THRESHOLD_SHORT:
        # Note: if prob_up is P(UP), then P(UP) < 0.3 means P(DOWN) > 0.7 (for binary)
        signal_text_val = "SHORT"

    return {
        'timeframe': timeframe_arg,
        'prob_up_pct': round(prob_up_value * 100, 2),
        'delta_pct': round(delta_value * 100, 2),
        'volatility_pct': round(vol_value * 100, 2),
        'momentum': round(momentum_value, 2),
        'signal': signal_text_val,
        'timestamp': latest_feature_row['timestamp'].iloc[0] if 'timestamp' in latest_feature_row else pd.NaT
    }


def _summarize_multiframe_results(results_list, long_count_threshold=3, short_count_threshold=3):
    """Summarizes results from multiple timeframes to give an overall trend."""
    if not results_list:
        return "Нет результатов для суммирования."

    valid_results = [r for r in results_list if r and isinstance(r, dict) and 'signal' in r]
    if not valid_results:
        return "Нет валидных результатов для суммирования."

    long_count = sum(1 for r_item in valid_results if r_item['signal'].upper() == 'LONG')
    short_count = sum(1 for r_item in valid_results if r_item['signal'].upper() == 'SHORT')

    if long_count >= long_count_threshold:
        return '📈  Итоговый тренд: ВВЕРХ'
    elif short_count >= short_count_threshold:
        return '📉  Итоговый тренд: ВНИЗ'
    else:
        return '⚖️  Итоговый тренд: НЕЙТРАЛЬНО'


def main_multiframe_prediction_logic(symbol_to_predict, timeframes_list=None):
    """
    Main logic for generating multi-timeframe predictions for a single symbol.

    Args:
        symbol_to_predict (str): The symbol to predict for (e.g., 'BTCUSDT').
        timeframes_list (list, optional): A list of timeframe strings to process.
            If None, uses timeframes from config or defaults.
    """
    global CONFIG
    if not CONFIG:  # Ensure config is loaded
        CONFIG = load_config()
        if not CONFIG:
            logger.critical("MultiFramePredictor: Configuration not loaded. Aborting.")
            return 1  # Error

    # Determine project root for constructing absolute paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    symbol_upper = symbol_to_predict.upper()
    logger.info(f"\n--- Мультифрейм-прогноз для {symbol_upper} ---")
    print(f"\n--- Мультифрейм-прогноз для {symbol_upper} ---")

    if timeframes_list is None:
        timeframes_list = _get_multiframe_config_value("timeframes_list", DEFAULT_MULTIFRAME_TIMEFRAMES)

    if not timeframes_list:
        logger.error("No timeframes specified for multi-frame prediction. Aborting.")
        print("Ошибка: Список таймфреймов для обработки пуст.")
        return 1

    all_tf_predictions = []
    for tf_main_item in timeframes_list:
        try:
            prediction_result = _predict_for_single_symbol_tf(symbol_upper, tf_main_item, project_root)
            if prediction_result:
                all_tf_predictions.append(prediction_result)
                ts_str = pd.to_datetime(prediction_result['timestamp']).strftime('%Y-%m-%d %H:%M') if pd.notna(prediction_result['timestamp']) else "N/A"
                print(f"  {prediction_result['timeframe']:>4s} | {prediction_result['signal']:<7s} | "
                      f"P(Up): {prediction_result['prob_up_pct']:>6.2f}% | "
                      f"Δ: {prediction_result['delta_pct']:>6.2f}% | "
                      f"σ: {prediction_result['volatility_pct']:>6.2f}% | "
                      f"Mom: {prediction_result['momentum']:>5.2f} | "
                      f"TS: {ts_str}")
        except Exception as e_inner:
            logger.error(f"Критическая ошибка при обработке {tf_main_item} для {symbol_upper}: {e_inner}", exc_info=True)
            # Continue to next timeframe

    summary_trend = _summarize_multiframe_results(all_tf_predictions)
    print("\n" + summary_trend)
    logger.info(f"Итоговый мультифрейм тренд для {symbol_upper}: {summary_trend.split(': ')[-1]}")

    logger.info(f"--- Мультифрейм-прогноз для {symbol_upper} завершён ---")
    return 0  # Success


if __name__ == "__main__":
    # Load config and setup logging for standalone execution
    CONFIG = load_config()
    if not CONFIG:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')
        logger.warning("Running multiframe_predictor.py with default config values as config.yaml failed to load.")
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')

    default_tfs = _get_multiframe_config_value("timeframes_list", DEFAULT_MULTIFRAME_TIMEFRAMES)

    parser = argparse.ArgumentParser(description="Generate multi-timeframe predictions for a single symbol.")
    parser.add_argument("--symbol", type=str, required=True,
                        help="Symbol to predict for (e.g., BTCUSDT).")
    parser.add_argument('--tf', nargs='*', default=default_tfs,
                        help=f"Timeframes for prediction (e.g., 1m 5m 15m). Default: {' '.join(default_tfs)}")

    args = parser.parse_args()

    exit_code = 1  # Default to error
    try:
        exit_code = main_multiframe_prediction_logic(args.symbol, args.tf)
    except KeyboardInterrupt:
        logger.info(f"\n[MultiFramePredictor] 🛑 Multi-frame prediction for {args.symbol} interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"[MultiFramePredictor] 💥 Unexpected critical error for {args.symbol}: {e}", exc_info=True)
        sys.exit(1)

    sys.exit(exit_code)
