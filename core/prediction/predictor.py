# core/prediction/predictor.py
"""
Generates predictions for multiple symbols across various timeframes using trained models.
It loads features, applies models, calculates final signals by incorporating historical
similarity analysis, and saves results (predictions, trade plan, alerts).
This module is based on the logic from the original predict_all.py.
"""
import pandas as pd
import joblib  # Still used by model_ops if not directly here
import os
import argparse
import sys
import logging
import numpy as np  # For NaN and array operations
from tabulate import tabulate  # For pretty printing tables

from ..helpers.utils import load_config
from ..helpers.data_io import load_features_pkl, save_features_pkl, save_sample_csv, load_feature_list_from_txt  # Not saving features here
from ..helpers.model_ops import load_model_with_fallback
from ..helpers.prediction_logic import (
    compute_final_delta,
    get_signal_strength,
    is_conflict,
    get_confidence_hint,
    calculate_trade_levels,
    similarity_analysis
)

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global config dictionary, to be loaded
CONFIG = {}

# --- Default Configuration (can be overridden by config.yaml) ---
DEFAULT_PREDICT_ALL_TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']
DEFAULT_TARGET_CLASS_NAMES = ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']
DEFAULT_LOG_DIR_PREDICT = "logs"  # Output directory for predictions, trade plan, alerts
DEFAULT_DATA_DIR_FEATURES = "data"  # Input directory for features_*.pkl files

# For model loading fallback
DEFAULT_MODEL_PATH_TEMPLATE = "models/{filter_suffix}_{tf}_{model_type}.pkl"
DEFAULT_MODEL_FEATURES_LIST_TEMPLATE = "models/{filter_suffix}_{tf}_features_selected.txt"
DEFAULT_GENERIC_MODEL_SUFFIX = "all"  # Used in load_model_with_fallback

# Thresholds for decision logic (can be fine-tuned in config.yaml)
DEFAULT_THRESHOLD_DELTA_FINAL_ABS = 0.002
DEFAULT_THRESHOLD_SIGMA_HIST_MAX = 0.015
DEFAULT_THRESHOLD_TP_HIT_PROBA_MIN = 0.60
DEFAULT_MIN_CONFIDENCE_FOR_TRADE = 0.08
DEFAULT_TRADE_PLAN_CONFIDENCE_THRESHOLD = 0.08
DEFAULT_TRADE_PLAN_TP_HIT_THRESHOLD = 0.55


def _get_predictor_config_value(key, sub_section=None, default_value=None):
    """Safely retrieves a value from CONFIG, optionally from a sub-section."""
    global CONFIG
    if not CONFIG:
        CONFIG = load_config()
        if not CONFIG:
            logger.error("Predictor: Configuration not loaded. Using hardcoded defaults.")
            # Provide direct defaults if config system fails entirely
            if key == "timeframes_list": return DEFAULT_PREDICT_ALL_TIMEFRAMES
            if key == "target_class_names": return DEFAULT_TARGET_CLASS_NAMES
            # Add more direct defaults as needed for critical configs
            return default_value

    source_dict = CONFIG
    if sub_section:
        source_dict = CONFIG.get(sub_section, {})

    return source_dict.get(key, default_value)


def _get_decision_threshold(key, default_value):
    """Helper to get decision thresholds from config."""
    return _get_predictor_config_value(key, sub_section="prediction.decision_thresholds", default_value=default_value)


def _get_trade_plan_threshold(key, default_value):
    """Helper to get trade plan thresholds from config."""
    return _get_predictor_config_value(key, sub_section="prediction.trade_plan_thresholds", default_value=default_value)


def _determine_feature_file_path(timeframe, project_root, data_dir, filter_suffix="all"):
    """
    Determines the path to the feature file, trying suffixed version first, then generic.
    Example: Tries data/features_top8_15m.pkl, then data/features_15m.pkl
    """
    # Path for potentially filtered features (e.g., features_top8_15m.pkl)
    suffixed_features_filename = f"features_{filter_suffix}_{timeframe}.pkl"
    suffixed_features_path = os.path.join(project_root, data_dir, suffixed_features_filename)

    if os.path.exists(suffixed_features_path):
        logger.debug(f"Using suffixed features file: {suffixed_features_path}")
        return suffixed_features_path

    # Path for generic features (e.g., features_15m.pkl)
    generic_features_filename = f"features_{timeframe}.pkl"
    generic_features_path = os.path.join(project_root, data_dir, generic_features_filename)

    if os.path.exists(generic_features_path):
        logger.info(f"Suffixed features file not found. Using generic features file: {generic_features_path}")
        return generic_features_path

    logger.warning(f"Neither suffixed ({suffixed_features_path}) nor generic "
                   f"({generic_features_path}) features file found for TF {timeframe} "
                   f"and filter '{filter_suffix}'.")
    return None


def _determine_feature_list_path(timeframe, project_root, models_dir, filter_suffix="all", generic_model_suffix="all"):
    """
    Determines the path to the feature list file (.txt), trying suffixed then generic.
    Example: Tries models/top8_15m_features_selected.txt, then models/all_15m_features_selected.txt
    """
    # Path for features list corresponding to a specific/group model
    suffixed_list_filename = f"{filter_suffix}_{timeframe}_features_selected.txt"
    suffixed_list_path = os.path.join(project_root, models_dir, suffixed_list_filename)

    if os.path.exists(suffixed_list_path):
        logger.debug(f"Using suffixed feature list file: {suffixed_list_path}")
        return suffixed_list_path

    # Path for features list corresponding to a generic model (e.g., "all")
    # This is a fallback if the model being used is generic.
    if filter_suffix.lower() != generic_model_suffix.lower():  # Avoid re-checking "all" if it was the primary
        generic_list_filename = f"{generic_model_suffix}_{timeframe}_features_selected.txt"
        generic_list_path = os.path.join(project_root, models_dir, generic_list_filename)
        if os.path.exists(generic_list_path):
            logger.info(f"Suffixed feature list file not found. Using generic feature list: {generic_list_path}")
            return generic_list_path

    logger.warning(f"Neither suffixed ({suffixed_list_path}) nor generic feature list "
                   f"found for TF {timeframe} and filter '{filter_suffix}'.")
    return None


def _is_trade_worthy_logic(prediction_entry_dict, thresholds):
    """Applies the advanced logic to determine if a signal is trade-worthy."""
    signal_candidate = prediction_entry_dict.get('signal')
    delta_final = prediction_entry_dict.get('delta_final')
    sigma_hist = prediction_entry_dict.get('std_delta_similar')  # sigma_history_std in original
    tp_hit_proba = prediction_entry_dict.get('tp_hit_proba')
    confidence_score = prediction_entry_dict.get('confidence_score')

    is_worthy = False  # Default

    # Common conditions: confidence, historical stability (sigma_hist), TP hit probability
    passes_common_filters = (
            pd.notna(confidence_score) and confidence_score > thresholds['min_confidence_for_trade'] and
            (pd.isna(sigma_hist) or sigma_hist < thresholds['sigma_hist_max']) and  # If sigma_hist is NaN, assume it passes (or handle differently)
            (pd.isna(tp_hit_proba) or tp_hit_proba > thresholds['tp_hit_proba_min'])  # If tp_hit_proba is NaN, assume it passes
    )

    if passes_common_filters:
        if signal_candidate in ['UP', 'STRONG UP', 'LONG']:  # Allow 'LONG' as well
            if pd.notna(delta_final) and delta_final > thresholds['delta_final_abs']:
                is_worthy = True
        elif signal_candidate in ['DOWN', 'STRONG DOWN', 'SHORT']:  # Allow 'SHORT'
            if pd.notna(delta_final) and delta_final < -thresholds['delta_final_abs']:  # delta must be negative
                is_worthy = True

    return is_worthy


def main_predict_all_logic(
        save_output_flag=False,
        symbol_filter=None,  # e.g., "BTCUSDT"
        group_filter_key=None  # e.g., "top8" (already lowercased)
):
    """
    Main logic for generating predictions for all relevant symbols and timeframes.

    Args:
        save_output_flag (bool): If True, save predictions, trade plan, and alerts to files.
        symbol_filter (str, optional): Uppercased symbol to filter by.
        group_filter_key (str, optional): Lowercased group key to filter by.
    """
    global CONFIG
    if not CONFIG:
        CONFIG = load_config()
        if not CONFIG:
            logger.critical("Predictor: Configuration not loaded. Aborting.")
            return 1

    # Setup logging for this specific run if not already configured by a higher level
    if not logging.getLogger().hasHandlers() and not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')

    # --- Get Configuration Values ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Directories
    log_dir_pred_rel = _get_predictor_config_value("logs_dir", default_value=DEFAULT_LOG_DIR_PREDICT)
    log_dir_pred_abs = os.path.join(project_root, log_dir_pred_rel)
    os.makedirs(log_dir_pred_abs, exist_ok=True)

    data_dir_rel = _get_predictor_config_value("data_dir", default_value=DEFAULT_DATA_DIR_FEATURES)
    models_dir_rel = _get_predictor_config_value("models_dir", default_value="models")  # From global config

    # Timeframes and Symbols/Groups
    timeframes_for_prediction = _get_predictor_config_value("timeframes_list", sub_section="predict_all", default_value=DEFAULT_PREDICT_ALL_TIMEFRAMES)
    target_class_names_list = _get_predictor_config_value("target_class_names", sub_section="prediction", default_value=DEFAULT_TARGET_CLASS_NAMES)
    all_symbol_groups_dict = _get_predictor_config_value("symbol_groups", default_value={})

    # Model path templates
    model_path_template_str = _get_predictor_config_value("model_path_template", default_value=DEFAULT_MODEL_PATH_TEMPLATE)
    model_features_list_template_str = _get_predictor_config_value("model_features_list_template", default_value=DEFAULT_MODEL_FEATURES_LIST_TEMPLATE)
    generic_model_suffix_str = _get_predictor_config_value("generic_model_suffix", default_value=DEFAULT_GENERIC_MODEL_SUFFIX)

    # Decision thresholds
    decision_thresholds_map = {
        'delta_final_abs': _get_decision_threshold('delta_final_abs', DEFAULT_THRESHOLD_DELTA_FINAL_ABS),
        'sigma_hist_max': _get_decision_threshold('sigma_hist_max', DEFAULT_THRESHOLD_SIGMA_HIST_MAX),
        'tp_hit_proba_min': _get_decision_threshold('tp_hit_proba_min', DEFAULT_THRESHOLD_TP_HIT_PROBA_MIN),
        'min_confidence_for_trade': _get_decision_threshold('min_confidence_for_trade', DEFAULT_MIN_CONFIDENCE_FOR_TRADE)
    }
    # Trade plan thresholds
    trade_plan_conf_thresh = _get_trade_plan_threshold('confidence', DEFAULT_TRADE_PLAN_CONFIDENCE_THRESHOLD)
    trade_plan_tp_hit_thresh = _get_trade_plan_threshold('tp_hit_proba', DEFAULT_TRADE_PLAN_TP_HIT_THRESHOLD)

    # --- Determine symbols to process and file suffix for outputs ---
    # file_output_suffix is used for naming output files like latest_predictions_{suffix}.csv
    symbols_to_process_explicit = None  # List of actual symbols to iterate over
    file_output_suffix = "all"  # Default output suffix

    if symbol_filter:  # Single symbol takes precedence
        symbols_to_process_explicit = [symbol_filter]
        file_output_suffix = symbol_filter
        logger.info(f"🛠️ Prediction filter: Single symbol '{symbol_filter}'")
    elif group_filter_key:
        if group_filter_key in all_symbol_groups_dict:
            symbols_to_process_explicit = all_symbol_groups_dict[group_filter_key]
            if not symbols_to_process_explicit:  # Group exists but is empty
                logger.warning(f"Symbol group '{group_filter_key}' is empty in configuration. No symbols to process for this group.")
                return 0  # Or 1 if this is an error condition
            file_output_suffix = group_filter_key
            logger.info(f"🛠️ Prediction filter: Group '{group_filter_key}' ({len(symbols_to_process_explicit)} symbols)")
        else:  # Should have been caught by CLI, but double check
            logger.error(f"❌ Unknown group key '{group_filter_key}' for prediction. Aborting.")
            return 1
    else:
        logger.info("🛠️ Prediction filter: Processing all available symbols from feature files.")
        # In this case, symbols_to_process_explicit remains None, and we discover symbols per TF file.

    # Define output file paths using the determined suffix
    latest_predictions_file_path = os.path.join(log_dir_pred_abs, f'latest_predictions_{file_output_suffix}.csv')
    trade_plan_file_path = os.path.join(log_dir_pred_abs, f'trade_plan_{file_output_suffix}.csv')
    alerts_file_path = os.path.join(log_dir_pred_abs, f'alerts_{file_output_suffix}.txt')

    logger.info(f"🚀 Starting prediction generation (Filter: {file_output_suffix}). Output suffix: '{file_output_suffix}'")
    all_predictions_collector = []  # List to store dicts for each prediction
    trade_plan_collector = []  # List for trade plan entries

    # --- Main Loop: Iterate over Timeframes ---
    for tf in timeframes_for_prediction:
        logger.info(f"\n--- Processing Timeframe: {tf} ---")

        # Determine which feature file to load (suffixed for group/symbol, or generic)
        # If symbols_to_process_explicit is set (symbol or group filter active), use that suffix for features.
        # Otherwise, `file_output_suffix` is "all", implying generic features.
        current_feature_filter_suffix = file_output_suffix  # This suffix is for feature file lookup

        features_file = _determine_feature_file_path(tf, project_root, data_dir_rel, current_feature_filter_suffix)
        if not features_file:
            logger.warning(f"No feature file found for TF {tf} with filter '{current_feature_filter_suffix}'. Skipping this timeframe.")
            continue

        df_tf_features = load_features_pkl(features_file)
        if df_tf_features is None or df_tf_features.empty:
            logger.warning(f"Feature file {features_file} is empty or failed to load. Skipping TF {tf}.")
            continue

        # Determine which feature list (.txt) to load
        # This should align with the models being loaded (e.g., if using "top8" models, use "top8" feature list)
        # The `load_model_with_fallback` will try different model types, so the feature list
        # should ideally match the *actually loaded* model. This is tricky.
        # For now, assume feature list matches the `current_feature_filter_suffix`.
        feature_list_file = _determine_feature_list_path(tf, project_root, models_dir_rel, current_feature_filter_suffix, generic_model_suffix_str)
        if not feature_list_file:
            logger.error(f"Feature list file not found for TF {tf} with filter '{current_feature_filter_suffix}'. Cannot proceed with this TF.")
            continue

        selected_feature_cols_for_model = load_feature_list_from_txt(feature_list_file)
        if not selected_feature_cols_for_model:
            logger.error(f"Feature list loaded from {feature_list_file} is empty. Cannot proceed with TF {tf}.")
            continue

        # Determine which symbols to iterate over for this timeframe's feature file
        if symbols_to_process_explicit:
            # If a symbol or group filter is active, only process those symbols
            # that are ALSO present in the currently loaded df_tf_features.
            symbols_in_current_df = df_tf_features['symbol'].unique()
            actual_symbols_for_tf = [s for s in symbols_to_process_explicit if s in symbols_in_current_df]
            if not actual_symbols_for_tf:
                logger.info(f"No symbols from the filter list {symbols_to_process_explicit} found in feature file {features_file} for TF {tf}.")
                continue
        else:
            # No explicit filter, process all unique symbols found in this feature file
            actual_symbols_for_tf = df_tf_features['symbol'].unique().tolist()
            if not actual_symbols_for_tf:
                logger.info(f"No symbols found in feature file {features_file} for TF {tf}.")
                continue

        actual_symbols_for_tf.sort()  # For consistent processing order

        # --- Inner Loop: Iterate over Symbols for the current Timeframe ---
        for symbol in actual_symbols_for_tf:
            logger.info(f"--- Predicting for Symbol: {symbol} on TF: {tf} ---")

            df_symbol_tf_features = df_tf_features[df_tf_features['symbol'] == symbol].copy()
            if df_symbol_tf_features.empty:  # Should not happen if symbol came from df_tf_features.unique()
                logger.warning(f"Unexpected: No data for symbol {symbol} in df_tf_features after filtering for TF {tf}.")
                continue

            # Ensure data is sorted by timestamp to get the latest row correctly
            if 'timestamp' in df_symbol_tf_features.columns:
                try:
                    df_symbol_tf_features['timestamp'] = pd.to_datetime(df_symbol_tf_features['timestamp'])
                    df_symbol_tf_features.sort_values('timestamp', inplace=True)
                except Exception as e:
                    logger.error(f"Error processing timestamp for {symbol} [{tf}]: {e}. Sorting skipped.")

            if len(df_symbol_tf_features) < 1:  # Need at least one row for prediction
                logger.warning(f"Not enough data ({len(df_symbol_tf_features)} rows) for {symbol} on TF {tf} to make prediction.")
                continue

            # Data for prediction (latest row) and history (all but latest)
            live_row_df = df_symbol_tf_features.iloc[-1:].copy()  # Latest row for prediction
            historical_df = df_symbol_tf_features.iloc[:-1].copy()  # All other rows for similarity

            # Validate that the live row has the necessary features
            missing_live_cols = [col for col in selected_feature_cols_for_model if col not in live_row_df.columns]
            if missing_live_cols:
                logger.error(f"Live data for {symbol} [{tf}] is missing model features: {missing_live_cols}. Skipping.")
                continue
            if live_row_df[selected_feature_cols_for_model].isnull().values.any():
                nan_live_features = live_row_df[selected_feature_cols_for_model].columns[live_row_df[selected_feature_cols_for_model].isnull().any()].tolist()
                logger.warning(f"Live data for {symbol} [{tf}] has NaNs in features: {nan_live_features}. Prediction may be unreliable.")
                # Consider skipping or imputing if critical: continue

            X_live_for_model = live_row_df[selected_feature_cols_for_model]

            # --- Load Models (with fallback) ---
            # The filter_suffix for model loading should be the symbol itself,
            # and load_model_with_fallback will handle group/generic fallbacks.
            model_clf_class = load_model_with_fallback(model_path_template_str, symbol, tf, "clf_class", all_symbol_groups_dict, generic_model_suffix_str)
            model_reg_delta = load_model_with_fallback(model_path_template_str, symbol, tf, "reg_delta", all_symbol_groups_dict, generic_model_suffix_str)
            model_reg_vol = load_model_with_fallback(model_path_template_str, symbol, tf, "reg_vol", all_symbol_groups_dict, generic_model_suffix_str)
            model_clf_tp_hit = load_model_with_fallback(model_path_template_str, symbol, tf, "clf_tp_hit", all_symbol_groups_dict, generic_model_suffix_str)  # Optional

            if not all([model_clf_class, model_reg_delta, model_reg_vol]):
                logger.warning(f"One or more core models (class, delta, vol) not found for {symbol} on TF {tf} after fallback. Skipping symbol.")
                continue
            if model_clf_tp_hit:
                logger.debug(f"TP-hit model loaded for {symbol} [{tf}]. Classes: {getattr(model_clf_tp_hit, 'classes_', 'N/A')}")
            else:
                logger.debug(f"TP-hit model NOT loaded or found for {symbol} [{tf}]. TP Hit% will be NaN.")

            # --- Similarity Analysis ---
            avg_delta_hist_similar, std_delta_hist_similar, similarity_hint_text = np.nan, np.nan, "Нет данных/ошибка анализа"
            if not historical_df.empty and 'delta' in historical_df.columns:
                # Ensure historical_df for similarity also uses only selected_feature_cols_for_model
                historical_features_for_sim = historical_df[selected_feature_cols_for_model]
                historical_deltas_for_sim = historical_df['delta']

                # Check for NaNs in historical_features_for_sim before passing
                # similarity_analysis handles internal NaN cleaning.
                avg_delta_hist_similar, std_delta_hist_similar, similarity_hint_text = similarity_analysis(
                    X_live_for_model, historical_features_for_sim, historical_deltas_for_sim
                )  # Uses defaults for top_n, min_samples from prediction_logic
            else:
                similarity_hint_text = "Недостаточно исторических данных для анализа схожести"
                logger.debug(f"Skipping similarity analysis for {symbol} [{tf}]: historical data empty or 'delta' column missing.")

            # --- Make Predictions ---
            try:
                # Classifier
                proba_raw_all_classes = model_clf_class.predict_proba(X_live_for_model)[0]
                model_actual_classes = getattr(model_clf_class, 'classes_', None)  # Actual classes from the model [0, 1] or ['DOWN', 'UP'] etc.

                # Create a dictionary of probabilities keyed by actual model class labels
                proba_dict_model_order = {}
                if model_actual_classes is not None and len(model_actual_classes) == len(proba_raw_all_classes):
                    proba_dict_model_order = {model_actual_classes[i]: proba_raw_all_classes[i] for i in range(len(proba_raw_all_classes))}
                else:  # Fallback if classes_ is weird or unavailable
                    logger.warning(f"Model classes attribute issue for {symbol} [{tf}]. Using indexed probabilities.")
                    proba_dict_model_order = {i: proba_raw_all_classes[i] for i in range(len(proba_raw_all_classes))}

                # Determine predicted class label and simplified signal
                # This part needs careful handling of how model_actual_classes are mapped to target_class_names_list
                # For now, let's assume model_actual_classes are like [0, 1] or ['DOWN', 'UP']
                # and target_class_names_list is ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']

                # Simplified: find max proba class from model's own classes
                pred_class_idx_in_model = proba_raw_all_classes.argmax()
                predicted_label_from_model = model_actual_classes[pred_class_idx_in_model] if model_actual_classes is not None else str(pred_class_idx_in_model)

                # Map this to a simplified signal (UP, DOWN, NEUTRAL)
                # This mapping needs to be robust based on your model's output
                # If clf_class is binary (0 for DOWN, 1 for UP):
                if predicted_label_from_model == 1 or str(predicted_label_from_model).upper() == 'UP':
                    main_signal = "UP"
                elif predicted_label_from_model == 0 or str(predicted_label_from_model).upper() == 'DOWN':
                    main_signal = "DOWN"
                else:  # If multiclass and not clearly UP/DOWN
                    main_signal = str(predicted_label_from_model).upper()  # Could be NEUTRAL, STRONG UP, etc.

                # Confidence score
                if len(proba_raw_all_classes) >= 2:
                    sorted_probas = np.sort(proba_raw_all_classes)  # Ascending
                    confidence = float(sorted_probas[-1] - sorted_probas[-2])  # diff between best and second best
                elif len(proba_raw_all_classes) == 1:
                    confidence = float(proba_raw_all_classes[0])
                else:
                    confidence = 0.0

                confidence_text_hint = get_confidence_hint(confidence)

                # Regressors
                delta_model_pred = model_reg_delta.predict(X_live_for_model)[0]
                vol_model_pred = model_reg_vol.predict(X_live_for_model)[0]
                # Ensure predicted volatility is non-negative
                if pd.notna(vol_model_pred) and vol_model_pred < 0:
                    logger.debug(f"Predicted volatility for {symbol} [{tf}] was {vol_model_pred:.4f}, using abs value.")
                    vol_model_pred = abs(vol_model_pred)
                if pd.notna(vol_model_pred) and vol_model_pred < 1e-9:  # Effectively zero
                    vol_model_pred = 0.0

                # TP-Hit Probability (Optional Model)
                tp_hit_probability_val = np.nan
                if model_clf_tp_hit:
                    try:
                        tp_probas_all = model_clf_tp_hit.predict_proba(X_live_for_model)[0]
                        tp_model_classes = getattr(model_clf_tp_hit, 'classes_', None)
                        if tp_model_classes is not None and (1 in tp_model_classes or '1' in tp_model_classes):
                            # Find index of class '1' (TP hit)
                            try:
                                class_1_tp_idx = list(tp_model_classes).index(1)
                            except ValueError:
                                class_1_tp_idx = list(tp_model_classes).index('1')  # try string '1'
                            tp_hit_probability_val = tp_probas_all[class_1_tp_idx]
                        elif len(tp_probas_all) > 1:  # Assume binary, class 1 is at index 1
                            tp_hit_probability_val = tp_probas_all[1]
                            logger.debug(f"TP-hit model classes unclear ({tp_model_classes}), assuming positive class at index 1.")
                    except Exception as e_tp:
                        logger.warning(f"Error getting TP-hit probability for {symbol} [{tf}]: {e_tp}")

            except Exception as e_pred:
                logger.error(f"Error during prediction for {symbol} [{tf}]: {e_pred}", exc_info=True)
                continue  # Skip to next symbol

            # --- Consolidate Prediction Info ---
            entry_price_val = live_row_df['close'].iloc[0] if 'close' in live_row_df else np.nan
            # Determine direction for SL/TP calculation from main_signal
            # This needs to map your main_signal (which could be 'UP', 'STRONG UP', etc.) to 'long' or 'short'
            trade_direction_for_levels = 'none'
            if main_signal in ['UP', 'STRONG UP']:
                trade_direction_for_levels = 'long'
            elif main_signal in ['DOWN', 'STRONG DOWN']:
                trade_direction_for_levels = 'short'

            sl_level, tp_level = calculate_trade_levels(entry_price_val, trade_direction_for_levels, vol_model_pred)  # Uses helper

            final_delta_val = compute_final_delta(delta_model_pred, avg_delta_hist_similar, std_delta_hist_similar)  # Uses helper
            signal_strength_text = get_signal_strength(final_delta_val, confidence, std_delta_hist_similar)  # Uses helper
            conflict_flag_val = is_conflict(delta_model_pred, avg_delta_hist_similar)  # Uses helper

            # Prepare proba_dict for output, ensuring all target_class_names_list are keys
            # This requires mapping from model_actual_classes to target_class_names_list
            # This part is complex and depends heavily on how clf_class outputs relate to the 5 target classes.
            # Simplified: if model is binary [0,1] for [DOWN,UP], we map these to the broader 5 classes.
            # For predict_all.py, the original proba_dict was {TARGET_CLASS_NAMES[i]: proba_raw[i]}
            # This assumes proba_raw has 5 elements. If clf_class is binary, this needs adaptation.
            # Placeholder:
            output_proba_dict = {cls_name: 0.0 for cls_name in target_class_names_list}
            if model_actual_classes is not None:
                # Example: if model_actual_classes = [0,1] (DOWN, UP)
                # and target_class_names_list = ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']
                # We need a mapping rule. For simplicity, assume direct mapping if names match or map by index if numeric.
                for i, class_label_model in enumerate(model_actual_classes):
                    # Try to find a match in target_class_names_list
                    # This is a naive mapping, might need refinement
                    if str(class_label_model).upper() in target_class_names_list:
                        output_proba_dict[str(class_label_model).upper()] = proba_raw_all_classes[i]
                    elif isinstance(class_label_model, int) and class_label_model < len(target_class_names_list):
                        # If model class is int (0,1) and target_class_names_list is longer, this mapping is ambiguous.
                        # E.g. where does model class 0 (DOWN) map in the 5 classes?
                        # This needs a clear strategy. For now, let's assume if model is binary 0/1,
                        # it maps to 'DOWN' and 'UP' from the 5-class list.
                        if class_label_model == 0 and 'DOWN' in output_proba_dict:  # Assuming 0 maps to 'DOWN'
                            output_proba_dict['DOWN'] = proba_raw_all_classes[i]
                        elif class_label_model == 1 and 'UP' in output_proba_dict:  # Assuming 1 maps to 'UP'
                            output_proba_dict['UP'] = proba_raw_all_classes[i]
            else:  # Fallback if model_actual_classes is None
                if len(proba_raw_all_classes) == len(target_class_names_list):  # If raw probas match length of target names
                    for i_cls, cls_name_target in enumerate(target_class_names_list):
                        output_proba_dict[cls_name_target] = proba_raw_all_classes[i_cls]
                elif len(proba_raw_all_classes) == 2:  # Binary model, map to DOWN and UP
                    if 'DOWN' in output_proba_dict: output_proba_dict['DOWN'] = proba_raw_all_classes[0]
                    if 'UP' in output_proba_dict: output_proba_dict['UP'] = proba_raw_all_classes[1]

            prediction_entry = {
                'symbol': symbol, 'tf': tf,
                'timestamp_obj': pd.to_datetime(live_row_df['timestamp'].iloc[0]) if 'timestamp' in live_row_df else pd.NaT,
                'signal': main_signal,  # Simplified signal
                'confidence_score': confidence,
                'confidence_hint': confidence_text_hint,
                'proba_dict': output_proba_dict,  # Probabilities for each of the 5 target classes
                'predicted_delta_model': delta_model_pred,  # Renamed for clarity
                'predicted_volatility': vol_model_pred,
                'entry_price': entry_price_val,  # Renamed from 'entry'
                'sl': sl_level, 'tp': tp_level,
                'direction_for_levels': trade_direction_for_levels,  # Renamed from 'direction'
                'avg_delta_similar_hist': avg_delta_hist_similar,  # Renamed
                'std_delta_similar_hist': std_delta_hist_similar,  # Renamed
                'similarity_hint': similarity_hint_text,
                'delta_final': final_delta_val,
                'signal_strength': signal_strength_text,
                'conflict_model_hist': conflict_flag_val,  # Renamed
                'tp_hit_proba': tp_hit_probability_val,
                'error_msg': None  # For any specific error during this symbol's processing
            }
            prediction_entry['is_trade_worthy'] = _is_trade_worthy_logic(prediction_entry, decision_thresholds_map)
            all_predictions_collector.append(prediction_entry)

            # Add to trade plan if conditions are met
            if prediction_entry['direction_for_levels'] != 'none' and \
                    prediction_entry['confidence_score'] >= trade_plan_conf_thresh and \
                    (pd.isna(prediction_entry['tp_hit_proba']) or prediction_entry['tp_hit_proba'] >= trade_plan_tp_hit_thresh):

                rr_val = np.nan
                if pd.notna(sl_level) and pd.notna(tp_level) and pd.notna(entry_price_val) and abs(entry_price_val - sl_level) > 1e-9:
                    try:
                        rr_val = round(abs(tp_level - entry_price_val) / abs(entry_price_val - sl_level), 2)
                    except ZeroDivisionError:
                        pass  # rr_val remains NaN

                trade_plan_collector.append({
                    'symbol': symbol, 'tf': tf, 'entry_price': entry_price_val,
                    'direction': prediction_entry['direction_for_levels'],
                    'sl': sl_level, 'tp': tp_level, 'rr': rr_val,
                    'confidence': prediction_entry['confidence_score'],
                    'signal_raw': prediction_entry['signal'],  # The raw signal from classifier
                    'timestamp_utc': prediction_entry['timestamp_obj'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(prediction_entry['timestamp_obj']) else "N/A",
                    'confidence_hint': prediction_entry['confidence_hint'],
                    'tp_hit_proba': prediction_entry['tp_hit_proba']
                })

    # --- End of Loops: Save and Display Results ---
    if not all_predictions_collector:
        logger.info("No predictions were generated in this run.")
        return 0  # Success, but no predictions

    # Convert collected predictions to DataFrame
    df_all_predictions_out = pd.DataFrame(all_predictions_collector)

    # Add string timestamp for display
    df_all_predictions_out['timestamp_str_display'] = df_all_predictions_out['timestamp_obj'].dt.strftime('%Y-%m-%d %H:%M')
    df_all_predictions_out['timestamp_str_log'] = df_all_predictions_out['timestamp_obj'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Save results if flag is set
    if save_output_flag:
        logger.info("💾  Saving prediction results...")
        # Prepare DataFrame for CSV (flatten proba_dict, select columns)
        df_to_save_list = []
        for _, row in df_all_predictions_out.iterrows():
            item_to_save = {
                'symbol': row['symbol'], 'tf': row['tf'], 'timestamp': row['timestamp_str_log'],
                'signal': row['signal'], 'confidence_score': row['confidence_score'],
                'tp_hit_proba': row['tp_hit_proba'],
                'predicted_delta_model': row['predicted_delta_model'],
                'avg_delta_similar_hist': row['avg_delta_similar_hist'],
                'std_delta_similar_hist': row['std_delta_similar_hist'],
                'delta_final': row['delta_final'],
                'predicted_volatility': row['predicted_volatility'],
                'entry_price': row['entry_price'], 'sl': row['sl'], 'tp': row['tp'],
                'direction': row['direction_for_levels'],  # Use the one for levels
                'signal_strength': row['signal_strength'],
                'conflict_model_hist': row['conflict_model_hist'],
                'confidence_hint': row['confidence_hint'],
                'similarity_hint': row['similarity_hint'],
                'is_trade_worthy': row['is_trade_worthy']
            }
            # Add probabilities from proba_dict (keys are from target_class_names_list)
            if isinstance(row['proba_dict'], dict):
                item_to_save.update(row['proba_dict'])
            df_to_save_list.append(item_to_save)

        df_csv_output = pd.DataFrame(df_to_save_list)

        # Define column order for CSV
        csv_cols_ordered = [
                               'symbol', 'tf', 'timestamp', 'signal', 'confidence_score', 'tp_hit_proba',
                               'predicted_delta_model', 'avg_delta_similar_hist', 'std_delta_similar_hist', 'delta_final',
                               'predicted_volatility', 'entry_price', 'sl', 'tp', 'direction',
                               'signal_strength', 'conflict_model_hist', 'is_trade_worthy',
                               'confidence_hint', 'similarity_hint'
                           ] + target_class_names_list  # Add probability columns at the end

        # Ensure all columns exist, reorder, and save
        final_cols_for_csv = [col for col in csv_cols_ordered if col in df_csv_output.columns]
        # Add any other columns from df_csv_output not in the predefined order
        for col_df in df_csv_output.columns:
            if col_df not in final_cols_for_csv:
                final_cols_for_csv.append(col_df)

        if not df_csv_output.empty:
            try:
                df_csv_output = df_csv_output[final_cols_for_csv]  # Reorder
                df_csv_output.to_csv(latest_predictions_file_path, index=False, float_format='%.6f')
                logger.info(f"📄 Prediction signals saved to: {latest_predictions_file_path}")
            except Exception as e_csv:
                logger.error(f"❌ Failed to save predictions CSV to {latest_predictions_file_path}: {e_csv}")
        else:
            logger.info(f"🤷 No data to save for predictions CSV: {latest_predictions_file_path}")

        # Save Trade Plan
        if trade_plan_collector:
            df_trade_plan_out = pd.DataFrame(trade_plan_collector)
            trade_plan_cols_ordered = [
                'symbol', 'tf', 'timestamp_utc', 'direction', 'signal_raw', 'confidence', 'tp_hit_proba',
                'entry_price', 'sl', 'tp', 'rr', 'confidence_hint'
            ]
            final_cols_tp = [col for col in trade_plan_cols_ordered if col in df_trade_plan_out.columns]
            for col_df_tp in df_trade_plan_out.columns:
                if col_df_tp not in final_cols_tp: final_cols_tp.append(col_df_tp)

            try:
                df_trade_plan_out = df_trade_plan_out[final_cols_tp]
                df_trade_plan_out.to_csv(trade_plan_file_path, index=False, float_format='%.6f')
                logger.info(f"📈 Trade plan saved to: {trade_plan_file_path}")
            except Exception as e_tp_csv:
                logger.error(f"❌ Failed to save trade plan CSV to {trade_plan_file_path}: {e_tp_csv}")
        else:
            logger.info(f"🤷 Trade plan is empty. File {trade_plan_file_path} not created.")

        # Save Alerts File
        try:
            with open(alerts_file_path, "w", encoding="utf-8") as alert_f:
                for _, row_alert in df_all_predictions_out.iterrows():
                    if row_alert.get('is_trade_worthy'):
                        strength_emoji = row_alert['signal_strength'].replace(" Сильный", "🟢").replace(" Умеренный", "🟡").replace(" Слабый", "⚪")
                        direction_alert = row_alert['direction_for_levels'].upper() if pd.notna(row_alert['direction_for_levels']) else "N/A"
                        delta_f_alert = f"{row_alert['delta_final']:.2%}" if pd.notna(row_alert['delta_final']) else "N/A"
                        conf_alert = f"{row_alert['confidence_score']:.2f}" if pd.notna(row_alert['confidence_score']) else "N/A"
                        tp_hit_alert = f"{row_alert['tp_hit_proba']:.1%}" if pd.notna(row_alert['tp_hit_proba']) else "N/A"

                        alert_line = (
                            f"{row_alert['symbol']} {row_alert['tf']} {strength_emoji} "
                            f"{direction_alert} Δ:{delta_f_alert} "
                            f"Conf:{conf_alert} TP Hit:{tp_hit_alert}\n"
                        )
                        alert_f.write(alert_line)
            logger.info(f"📣 Alerts file saved to: {alerts_file_path}")
        except Exception as e_alert:
            logger.error(f"❌ Failed to save alerts file to {alerts_file_path}: {e_alert}")

    # --- Display Predictions in Console ---
    headers_console = ["TF", "Timestamp", "Сигнал", "Conf.", "ΔModel", "ΔHist", "σHist", "ΔFinal", "Конф?", "Сила", "TP Hit%", "Trade?"]
    grouped_by_symbol_display = df_all_predictions_out.groupby('symbol')

    sorted_symbols_display = sorted(list(grouped_by_symbol_display.groups.keys()))

    for symbol_key_display in sorted_symbols_display:
        df_sym_display = grouped_by_symbol_display.get_group(symbol_key_display)
        # Sort rows by timeframe order then by confidence
        try:
            df_sym_display_sorted = df_sym_display.sort_values(
                by=['tf', 'confidence_score'],
                ascending=[True, False],  # Sort TF ascending, confidence descending
                key=lambda series: series.map(lambda x: timeframes_for_prediction.index(x)) if series.name == 'tf' else series
            )
        except Exception as e_sort_display:  # Fallback sort
            logger.warning(f"Error sorting display rows for {symbol_key_display}: {e_sort_display}. Using default sort.")
            df_sym_display_sorted = df_sym_display.sort_values(by='confidence_score', ascending=False)

        table_data_console = []
        for _, row_disp in df_sym_display_sorted.iterrows():
            table_data_console.append([
                row_disp['tf'],
                row_disp['timestamp_str_display'],
                row_disp['signal'],
                f"{row_disp['confidence_score']:.3f}" if pd.notna(row_disp['confidence_score']) else "N/A",
                f"{row_disp['predicted_delta_model']:.2%}" if pd.notna(row_disp['predicted_delta_model']) else "N/A",
                f"{row_disp['avg_delta_similar_hist']:.2%}" if pd.notna(row_disp['avg_delta_similar_hist']) else "N/A",
                f"{row_disp['std_delta_similar_hist']:.2%}" if pd.notna(row_disp['std_delta_similar_hist']) else "N/A",
                f"{row_disp['delta_final']:.2%}" if pd.notna(row_disp['delta_final']) else "N/A",
                "❗" if row_disp['conflict_model_hist'] else " ",
                row_disp['signal_strength'].replace(" Сильный", "🟢").replace(" Умеренный", "🟡").replace(" Слабый", "⚪"),
                f"{row_disp['tp_hit_proba']:.1%}" if pd.notna(row_disp['tp_hit_proba']) else "N/A",
                "✅" if row_disp.get('is_trade_worthy') else "❌"
            ])

        print(f"\n📊  Прогноз по символу: {symbol_key_display}")
        try:
            print(tabulate(table_data_console, headers=headers_console, tablefmt="pretty", stralign="right", numalign="right"))
        except Exception as e_tab:
            logger.error(f"❌ Ошибка при выводе таблицы для {symbol_key_display}: {e_tab}")
            print("Не удалось отформатировать таблицу. Данные:")
            for r_print in table_data_console: print(r_print)  # Fallback print

    # Display TOP SIGNALS
    print("\n📈 ТОП СИГНАЛЫ (Сильные и Уверенные)")
    top_signals_display_list = []
    for _, row_top in df_all_predictions_out.iterrows():
        # Using criteria from original predict_all.py for top signals
        if (row_top['direction_for_levels'] != 'none' and
                row_top['signal_strength'] in ['🟢 Сильный', '🟡 Умеренный', '🟢  Сильный', '🟡  Умеренный'] and  # Handle potential extra space
                pd.notna(row_top['confidence_score']) and row_top['confidence_score'] > 0.05 and
                pd.notna(row_top['delta_final']) and abs(row_top['delta_final']) > 0.002):
            top_signals_display_list.append({
                'Symbol': row_top['symbol'], 'TF': row_top['tf'],
                'Direction': row_top['direction_for_levels'].upper(),
                'ΔFinal': f"{row_top['delta_final']:.2%}",
                'Conf.': f"{row_top['confidence_score']:.2f}",
                'Сила': row_top['signal_strength'].replace(" Сильный", "🟢").replace(" Умеренный", "🟡").replace(" Слабый", "⚪"),
                'TP Hit%': f"{row_top['tp_hit_proba']:.1%}" if pd.notna(row_top['tp_hit_proba']) else "N/A",
                'Trade?': "✅" if row_top.get('is_trade_worthy') else "❌"
            })

    if top_signals_display_list:
        strength_order_map = {'🟢': 0, '🟡': 1, '⚪': 2}  # For sorting by strength emoji
        top_signals_display_list_sorted = sorted(
            top_signals_display_list,
            key=lambda x: (
                strength_order_map.get(x['Сила'][0] if x['Сила'] else '⚪', 99),  # Sort by emoji part of strength
                -float(x['Conf.']),  # Descending confidence
                timeframes_for_prediction.index(x['TF']) if x['TF'] in timeframes_for_prediction else float('inf')  # By TF order
            )
        )
        try:
            print(tabulate(top_signals_display_list_sorted, headers="keys", tablefmt="pretty"))
        except Exception as e_tab_top:
            logger.error(f"❌ Ошибка при выводе таблицы ТОП СИГНАЛОВ: {e_tab_top}")
            print("Не удалось отформатировать таблицу ТОП СИГНАЛОВ. Данные:")
            for r_top_print in top_signals_display_list_sorted: print(r_top_print)
    else:
        print("🤷 Нет сильных сигналов по текущим условиям.")

    logger.info("✅  Процесс генерации прогнозов завершён.")
    return 0  # Success


if __name__ == "__main__":
    CONFIG = load_config()
    if not CONFIG:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')
        logger.warning("Running predictor.py with default config values as config.yaml failed to load.")
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')

    parser = argparse.ArgumentParser(description="Generate predictions based on trained models.")
    parser.add_argument('--save', action='store_true', help="Save prediction results to CSV files.")
    parser.add_argument('--symbol', type=str, default=None,
                        help="Single symbol to process (e.g., BTCUSDT).")
    parser.add_argument('--symbol-group', type=str, default=None,
                        help="Symbol group key to process (e.g., top8, meme).")

    args = parser.parse_args()

    # --- Logic from original predict_all.py __main__ to determine filters ---
    parsed_symbol_filter = None
    parsed_group_filter_key = None
    all_groups_cfg = _get_predictor_config_value("symbol_groups", default_value={})

    if args.symbol and args.symbol_group:
        logger.error("❌ Cannot specify both --symbol and --symbol-group. Exiting.")
        sys.exit(1)
    elif args.symbol_group:
        parsed_group_filter_key = args.symbol_group.lower()
        if parsed_group_filter_key not in all_groups_cfg:
            logger.error(f"❌ Unknown symbol group: '{args.symbol_group}'. Available: {list(all_groups_cfg.keys())}. Exiting.")
            sys.exit(1)
    elif args.symbol:
        # If --symbol value matches a group key, treat it as a group filter
        if args.symbol.lower() in all_groups_cfg:
            parsed_group_filter_key = args.symbol.lower()
            logger.info(f"Interpreting --symbol '{args.symbol}' as group filter '{parsed_group_filter_key}'.")
        else:
            parsed_symbol_filter = args.symbol.upper()  # Treat as a single symbol

    exit_code = 1  # Default to error
    try:
        exit_code = main_predict_all_logic(
            save_output_flag=args.save,
            symbol_filter=parsed_symbol_filter,
            group_filter_key=parsed_group_filter_key
        )
    except KeyboardInterrupt:
        logger.info("\n[Predictor] 🛑 Prediction generation interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"[Predictor] 💥 Unexpected critical error: {e}", exc_info=True)
        sys.exit(1)

    sys.exit(exit_code)
