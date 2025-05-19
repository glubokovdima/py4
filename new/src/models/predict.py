# src/models/predict.py

import pandas as pd
import numpy as np
import os
import logging
import joblib
from datetime import datetime, timezone # Import timezone
from tabulate import tabulate # Keep tabulate for console output formatting
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler # Keep scaler for similarity analysis
import sys # For sys.exit

# Import config and logging setup
from src.utils.config import get_config

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Load Configuration ---
config = get_config()
PATHS_CONFIG = config['paths']
SYMBOL_GROUPS = config['symbol_groups'] # Need symbol groups for fallback logic
TIMEFRAMES_CONFIG = config['timeframes'] # Need default timeframes list
PREDICT_CONFIG = config['predict'] # Prediction-specific thresholds and settings
TRADE_PLAN_CONFIG = config['trade_plan'] # Trade plan specific thresholds and settings
FEATURE_BUILDER_CONFIG = config['feature_builder'] # Need target_shift


# --- Constants from Config ---
# Prediction Thresholds
MIN_CONFIDENCE_FOR_PREDICT = PREDICT_CONFIG.get('min_confidence', 0.08)
TP_HIT_PROBA_MIN = PREDICT_CONFIG.get('tp_hit_proba_min', 0.60)
SIGMA_HISTORY_MAX = PREDICT_CONFIG.get('sigma_history_max', 0.015)
DELTA_FINAL_ABS_MIN = PREDICT_CONFIG.get('delta_final_abs_min', 0.002)

# Similarity Analysis Settings
SIMILARITY_TOP_N = PREDICT_CONFIG.get('similarity_top_n', 15)
SIMILARITY_MIN_HISTORY = PREDICT_CONFIG.get('similarity_min_history', 100)

# Trade Plan Settings
TRADE_PLAN_CONFIDENCE_THRESHOLD = TRADE_PLAN_CONFIG.get('confidence_threshold', 0.08)
TRADE_PLAN_TP_HIT_THRESHOLD = TRADE_PLAN_CONFIG.get('tp_hit_threshold', 0.55)
TRADE_LEVELS_RR = TRADE_PLAN_CONFIG.get('trade_levels_rr', 2.0)
TRADE_LEVELS_MIN_VOLATILITY = TRADE_PLAN_CONFIG.get('trade_levels_min_volatility', 1e-9)

# Other Constants
TARGET_SHIFT = FEATURE_BUILDER_CONFIG['target_shift']


# Target names mapping - should be consistent with training/feature building
# In src/models/train.py, we mapped {'DOWN': 0, 'UP': 1}
# The prediction can result in ['DOWN', 'UP'] or other values if the model was trained differently.
# If the model predicts 0/1, we need to map back.
# The original predict_all.txt had TARGET_CLASS_NAMES = ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']
# but the model only predicted 'UP'/'DOWN'/'NaN' or 0/1. The 'STRONG' and 'NEUTRAL' came from post-processing.
# Let's define the possible *output* signals here for clarity in reports.
OUTPUT_SIGNALS = ['SHORT', 'LONG', 'NEUTRAL', 'UNKNOWN']


# --- Helper Functions (Moved from old predict_all.txt) ---

def load_model_with_fallback(symbol_or_group_key, tf, model_type):
    """
    Loads a joblib model file with fallback logic.
    Tries: 1. specific_{tf}_{type}.pkl, 2. group_{tf}_{type}.pkl (if symbol in group), 3. all_{tf}_{type}.pkl
    """
    model_output_dir = PATHS_CONFIG['models_dir']

    # 1. Specific model (symbol or group key)
    specific_model_path = os.path.join(model_output_dir, f"{symbol_or_group_key}_{tf}_{model_type}.pkl")
    if os.path.exists(specific_model_path):
        logger.debug(f"✅ Using specific model for {symbol_or_group_key} on {tf}: {model_type}")
        try:
            return joblib.load(specific_model_path)
        except Exception as e:
            logger.error(f"Error loading specific model {specific_model_path}: {e}")
            # Fallback to next option

    # 2. Group model (only if symbol_or_group_key is actually a symbol)
    # If the key is a group key, we already tried the specific path in step 1.
    # If the key is 'all', there's no group fallback.
    if symbol_or_group_key != 'all':
        # Check if the provided symbol_or_group_key (which is a symbol) belongs to any group
        for group_name, symbol_list in SYMBOL_GROUPS.items():
            if symbol_or_group_key in symbol_list:
                group_model_path = os.path.join(model_output_dir, f"{group_name}_{tf}_{model_type}.pkl")
                if os.path.exists(group_model_path):
                    logger.debug(f"✅ Using group model ({group_name}) for {symbol_or_group_key} on {tf}: {model_type}")
                    try:
                        return joblib.load(group_model_path)
                    except Exception as e:
                        logger.error(f"Error loading group model {group_model_path}: {e}")
                        # Fallback to next option
                    # Found group model, no need to check other groups for this symbol
                    break # Exit the group loop
        # Note: If symbol_or_group_key was a group key itself (e.g. 'top8'), step 1 already handled
        # loading models/top8_tf_model_type.pkl. This step only applies when the key is a single symbol.


    # 3. General 'all' model
    all_model_path = os.path.join(model_output_dir, f"all_{tf}_{model_type}.pkl")
    if os.path.exists(all_model_path):
        logger.debug(f"⚠️ Using general 'all' model for {symbol_or_group_key} on {tf}: {model_type}")
        try:
            return joblib.load(all_model_path)
        except Exception as e:
            logger.error(f"Error loading general 'all' model {all_model_path}: {e}")
            return None

    logger.warning(f"❌ Model not found for key '{symbol_or_group_key}', group fallback, or 'all': {model_type} on {tf}. Paths checked: {specific_model_path}, group paths, {all_model_path}")
    return None


def compute_final_delta(delta_model_pred, delta_history_avg, sigma_history_std):
    """
    Computes the final delta prediction by combining model prediction and history average,
    weighted by historical volatility.
    """
    # Handle cases where history data is missing or invalid
    if pd.isna(delta_history_avg) or pd.isna(sigma_history_std) or sigma_history_std < TRADE_LEVELS_MIN_VOLATILITY:
        # If history is unreliable or missing, rely solely on the model prediction
        # Check if model prediction is valid
        if pd.notna(delta_model_pred):
            return round(delta_model_pred, 5)
        else:
            return np.nan # Cannot compute if both are invalid

    # Simple linear interpolation weights based on sigma_history_std
    # Adjust these thresholds and weights based on desired behavior
    min_sigma = 0.005
    max_sigma = 0.020
    weight_hist_at_min_sigma = 0.6 # Max weight for history when sigma is low
    weight_hist_at_max_sigma = 0.2 # Min weight for history when sigma is high

    if sigma_history_std <= min_sigma:
        w_hist = weight_hist_at_min_sigma
    elif sigma_history_std >= max_sigma:
        w_hist = weight_hist_at_max_sigma
    else:
        # Linear interpolation: w_hist decreases as sigma_history_std increases
        alpha = (sigma_history_std - min_sigma) / (max_sigma - min_sigma)
        w_hist = weight_hist_at_min_sigma - alpha * (weight_hist_at_min_sigma - weight_hist_at_max_sigma)

    w_model = 1.0 - w_hist

    # Ensure model prediction is valid before using it
    if pd.isna(delta_model_pred):
         # If model prediction is NaN, rely solely on history (if history is valid)
         return round(delta_history_avg, 5)
    else:
         # Both model and history are considered
         return round(w_model * delta_model_pred + w_hist * delta_history_avg, 5)


def get_signal_strength(delta_final, confidence_score, sigma_history_std):
    """
    Determines the strength of the signal based on final delta, confidence,
    and historical volatility. Uses thresholds from config.
    """
    # Use config thresholds
    delta_threshold_strong = PREDICT_CONFIG.get('delta_threshold_strong', 0.025)
    delta_threshold_moderate = PREDICT_CONFIG.get('delta_threshold_moderate', 0.010)
    conf_threshold_strong = PREDICT_CONFIG.get('conf_threshold_strong', 0.15)
    conf_threshold_moderate = PREDICT_CONFIG.get('conf_threshold_moderate', 0.05)
    sigma_threshold_reliable = PREDICT_CONFIG.get('sigma_threshold_reliable', 0.010)
    sigma_threshold_unreliable = PREDICT_CONFIG.get('sigma_threshold_unreliable', 0.020)


    if pd.isna(delta_final) or pd.isna(confidence_score):
        return "⚪ Слабый" # Cannot determine strength without delta or confidence

    # Treat missing sigma history as potentially unreliable for strength determination
    sigma_hist_reliable = pd.notna(sigma_history_std) and sigma_history_std < sigma_threshold_reliable
    sigma_hist_unreliable = pd.notna(sigma_history_std) and sigma_history_std >= sigma_threshold_unreliable

    abs_delta = abs(delta_final)

    # Check for STRONG signal potential
    if abs_delta > delta_threshold_strong and confidence_score > conf_threshold_strong:
        if sigma_hist_reliable:
            return "🟢 Сильный" # High delta, high confidence, reliable history
        elif sigma_hist_unreliable:
             return "🟡 Умеренный" # High delta, high confidence, but unreliable history
        else: # sigma_history_std is between reliable and unreliable, or NaN
             return "🟡 Умеренный" # Treat as moderately reliable history

    # Check for MODERATE signal potential
    elif abs_delta > delta_threshold_moderate and confidence_score > conf_threshold_moderate:
        if sigma_hist_unreliable:
            return "⚪ Слабый" # Moderate signal, but unreliable history
        else:
            return "🟡 Умеренный" # Moderate signal, reliable or moderately reliable history

    # If neither STRONG nor MODERATE potential is met based on delta/confidence
    else:
        return "⚪ Слабый"


def is_conflict(delta_model_pred, delta_history_avg):
    """Checks if model prediction and history average conflict in direction."""
    if pd.isna(delta_model_pred) or pd.isna(delta_history_avg):
        return False # No conflict if either is NaN

    # Define a threshold for "close to zero" to avoid reporting conflict on tiny values
    zero_threshold = FEATURE_BUILDER_CONFIG.get('delta_zero_threshold', 0.005) # Use a config threshold

    # Only report conflict if both are not close to zero and have opposite signs
    if abs(delta_model_pred) < zero_threshold or abs(delta_history_avg) < zero_threshold:
         return False # No conflict if either is near zero

    return (delta_model_pred > 0 and delta_history_avg < 0) or \
           (delta_model_pred < 0 and delta_history_avg > 0)


def get_confidence_hint(confidence_score):
    """Provides a text hint based on the confidence score."""
    # Confidence hint thresholds - should align with signal strength logic or be distinct
    # Using some example thresholds; ideally, these would be in config
    if pd.isna(confidence_score): return "Не определено"

    if confidence_score > 0.20:
        return "Очень высокая уверенность"
    elif confidence_score > 0.10:
        return "Высокая уверенность"
    elif confidence_score > 0.05:
        return "Умеренная уверенность"
    elif confidence_score > 0.02: # Below this, maybe flag for caution
         return "Низкая уверенность - будьте осторожны"
    else:
        return "Очень низкая уверенность - пропустить"


def calculate_trade_levels(entry_price, direction, predicted_volatility):
    """
    Calculates Stop Loss (SL) and Take Profit (TP) based on entry, direction,
    predicted volatility (used as base risk/reward unit), and configured RR ratio.
    """
    rr = TRADE_LEVELS_RR # Get RR from config
    min_vol = TRADE_LEVELS_MIN_VOLATILITY # Min volatility from config

    # Predicted volatility must be positive and above the minimum threshold
    if pd.isna(predicted_volatility) or predicted_volatility < min_vol:
        logger.debug(f"Predicted volatility ({predicted_volatility:.6f}) is below minimum ({min_vol:.6f}) or NaN. Cannot calculate trade levels.")
        return np.nan, np.nan

    # Use predicted volatility as the base unit for SL distance
    sl_distance = predicted_volatility
    tp_distance = predicted_volatility * rr

    if direction == 'long':
        sl = entry_price - sl_distance
        tp = entry_price + tp_distance
    elif direction == 'short':
        sl = entry_price + sl_distance
        tp = entry_price - tp_distance
    else:
        logger.debug(f"Direction is '{direction}'. Cannot calculate trade levels.")
        return np.nan, np.nan

    # Ensure SL/TP are not negative (especially for short trades on low-priced assets)
    sl = max(0, sl)
    tp = max(0, tp)

    # Rounding: Determine precision based on the asset's price scale?
    # For now, use a fixed reasonable number of decimal places.
    # A more robust approach would dynamically determine precision based on price.
    precision = 6 # Example precision
    sl = round(sl, precision)
    tp = round(tp, precision)

    # Basic validation: SL should not be equal to entry (or very close)
    if abs(entry_price - sl) < 1e-9:
         logger.warning(f"Calculated SL ({sl}) is too close to Entry ({entry_price}). Setting SL/TP to NaN.")
         return np.nan, np.nan

    return sl, tp


def similarity_analysis(X_live_df, X_hist_df, deltas_hist_series):
    """
    Performs cosine similarity analysis between the live feature vector
    and historical feature vectors to find similar past situations.
    Returns average delta and std deviation of deltas for top similar situations.
    """
    top_n = SIMILARITY_TOP_N # Get top_n from config
    min_history_for_sim = SIMILARITY_MIN_HISTORY # Get min history from config

    if X_hist_df.empty or len(X_hist_df) < min_history_for_sim:
        logger.debug(f"Insufficient historical data ({len(X_hist_df)} < {min_history_for_sim}) for similarity analysis.")
        return np.nan, np.nan, "Недостаточно исторических данных для анализа схожести"

    # Ensure X_live_df is a DataFrame (should be 1 row) and has the same columns as X_hist_df
    if not isinstance(X_live_df, pd.DataFrame) or X_live_df.shape[0] != 1:
         logger.error("X_live_df must be a single-row DataFrame for similarity analysis.")
         return np.nan, np.nan, "Неверный формат входных данных (X_live_df)"

    # Ensure column names match and are in the same order
    if not X_live_df.columns.equals(X_hist_df.columns):
         logger.error("Column mismatch between X_live_df and X_hist_df for similarity analysis.")
         # Attempt to intersect columns, but ideally this should not happen if features are built correctly
         common_cols = X_live_df.columns.intersection(X_hist_df.columns)
         if len(common_cols) == 0:
              logger.error("No common features for similarity analysis.")
              return np.nan, np.nan, "Нет общих признаков для анализа схожести"
         logger.warning(f"Using only {len(common_cols)} common features for similarity analysis.")
         X_live_df_common = X_live_df[common_cols]
         X_hist_df_common = X_hist_df[common_cols]
    else:
         X_live_df_common = X_live_df
         X_hist_df_common = X_hist_df


    # Align historical deltas with historical features based on index
    if not deltas_hist_series.index.equals(X_hist_df_common.index):
         try:
              hist_deltas_aligned = deltas_hist_series.reindex(X_hist_df_common.index).copy()
              logger.debug("Indices of historical deltas aligned with features for similarity.")
         except Exception as e:
              logger.error(f"Error aligning historical deltas index: {e}")
              return np.nan, np.nan, "Ошибка выравнивания данных для схожести"
    else:
        hist_deltas_aligned = deltas_hist_series.copy()


    # Drop rows with NaN in features OR deltas before scaling/similarity
    # This ensures we only use complete historical examples with known outcomes
    nan_in_features = X_hist_df_common.isnull().any(axis=1)
    nan_in_deltas = hist_deltas_aligned.isnull()
    valid_indices = X_hist_df_common.index[~(nan_in_features | nan_in_deltas)]

    hist_df_cleaned = X_hist_df_common.loc[valid_indices]
    hist_deltas_cleaned = hist_deltas_aligned.loc[valid_indices]

    if len(hist_df_cleaned) < top_n:
         logger.debug(f"Not enough clean historical data ({len(hist_df_cleaned)} < {top_n}) for similarity analysis after dropping NaNs.")
         return np.nan, np.nan, f"Недостаточно чистых исторических данных ({len(hist_df_cleaned)} < {top_n}) для анализа схожести"


    # Scale features
    scaler = StandardScaler()
    try:
        X_hist_scaled = scaler.fit_transform(hist_df_cleaned)
        # Ensure X_live_df_common has the same columns and scale it using the same scaler
        x_live_scaled = scaler.transform(X_live_df_common)

    except ValueError as e:
        logger.error(f"Scaling error in similarity_analysis: {e}")
        return np.nan, np.nan, "Ошибка масштабирования признаков"
    except Exception as e:
         logger.error(f"Unexpected scaling error in similarity_analysis: {e}", exc_info=True)
         return np.nan, np.nan, "Неизвестная ошибка масштабирования признаков"


    # Compute cosine similarity
    # Ensure x_live_scaled is 2D array (shape (1, n_features))
    if x_live_scaled.ndim == 1:
         x_live_scaled = x_live_scaled.reshape(1, -1)

    try:
        sims = cosine_similarity(X_hist_scaled, x_live_scaled).flatten()
    except Exception as e:
         logger.error(f"Error computing cosine similarity: {e}", exc_info=True)
         return np.nan, np.nan, "Ошибка при расчете схожести"


    # Find indices of top N similar historical points
    actual_top_n = min(top_n, len(sims))
    if actual_top_n <= 0:
         return np.nan, np.nan, "Нет данных для топ-N после чистки и масштабирования"

    # Get the indices in the *cleaned* historical data
    top_indices_in_cleaned_hist = sims.argsort()[-actual_top_n:][::-1]

    # Get the original indices from the cleaned data index
    original_top_indices = hist_df_cleaned.iloc[top_indices_in_cleaned_hist].index

    if len(original_top_indices) == 0:
         # Should not happen if actual_top_n > 0, but safeguard
         return np.nan, np.nan, "Не найдено достаточно схожих ситуаций после фильтрации"

    # Get the deltas for these original indices from the cleaned deltas series
    similar_deltas = hist_deltas_cleaned.loc[original_top_indices]

    # Calculate mean and std of these deltas
    avg_delta = similar_deltas.mean()
    std_delta = similar_deltas.std()

    # Ensure std_delta is not NaN if there's only one point (std is 0)
    if len(similar_deltas) == 1:
        std_delta = 0.0

    hint = (
        "Высокий разброс" if pd.notna(std_delta) and std_delta > PREDICT_CONFIG.get('sigma_history_high_threshold', 0.02) else # Use config thresholds
        "Умеренно стабильно" if pd.notna(std_delta) and std_delta > PREDICT_CONFIG.get('sigma_history_moderate_threshold', 0.01) else
        "Стабильный паттерн" if pd.notna(std_delta) else
        "Не удалось определить разброс"
    )
    if pd.isna(avg_delta) or pd.isna(std_delta):
        hint = "Не удалось рассчитать статистику схожести"

    return avg_delta, std_delta, hint


# --- Main Prediction Function ---

def generate_predictions(tf_list, symbol_filter=None, group_filter=None):
    """
    Generates predictions for the specified timeframes and filtered symbols.

    Args:
        tf_list (list): List of timeframe keys to process.
        symbol_filter (str, optional): Single symbol to filter by.
        group_filter (str, optional): Group name to filter by.

    Returns:
        list: A list of dictionaries, where each dictionary is a prediction entry.
    """
    logger.info(f"🚀  Запуск генерации прогнозов...")

    # --- Determine target symbols and file suffix ---
    target_syms = None
    files_suffix = "all" # Default suffix for files
    log_filter_details = "для всех символов"

    if symbol_filter:
        target_syms = [symbol_filter.upper()] # Ensure uppercase
        log_filter_details = f"для символа: {target_syms[0]}"
        files_suffix = target_syms[0]
    elif group_filter:
        group_key = group_filter.lower() # Ensure lowercase
        if group_key not in SYMBOL_GROUPS:
            logger.error(f"❌ Неизвестная группа символов: '{group_filter}'. Доступные группы: {list(SYMBOL_GROUPS.keys())}")
            return [] # Return empty list if group is invalid
        target_syms = SYMBOL_GROUPS[group_key]
        log_filter_details = f"для группы: '{group_filter}' ({len(target_syms)} символов)"
        files_suffix = group_key
    else:
        log_filter_details = "для всех символов"
        files_suffix = "all" # Explicitly 'all' suffix for all symbols

    logger.info(f"Генерация прогнозов {log_filter_details} на ТФ: {', '.join(tf_list)}. Суффикс файлов: {files_suffix}")


    all_predictions_data = []
    # Trade plan data is generated here and can be returned or processed later
    # trade_plan_data = [] # Decide if this should be returned by this function or generated elsewhere

    for tf in tf_list:
        logger.info(f"\n--- Обработка таймфрейма: {tf} ---")

        # --- Load Features ---
        # Attempt to load features file based on determined suffix
        features_path = os.path.join(PATHS_CONFIG['data_dir'], f"features_{files_suffix}_{tf}.pkl")

        # Fallback to generic 'all' features file if specific/group one doesn't exist
        if not os.path.exists(features_path) and files_suffix != "all":
            fallback_features_path = os.path.join(PATHS_CONFIG['data_dir'], f"features_all_{tf}.pkl")
            if os.path.exists(fallback_features_path):
                features_path = fallback_features_path
                logger.warning(f"Файл признаков для '{files_suffix}' не найден ({os.path.basename(features_path)}). Использую общий файл признаков: {os.path.basename(fallback_features_path)}.")
                # Note: If we use the 'all' features file here, the DataFrame `df` will contain
                # data for ALL symbols, regardless of symbol_filter/group_filter.
                # We will need to filter `df` by `target_syms` below.
            else:
                 logger.error(f"❌ Ни фильтрованная ({os.path.basename(features_path)}), ни общая ({os.path.basename(fallback_features_path)}) версия файла признаков не найдены для {tf}. Пропуск таймфрейма.")
                 continue # Skip this TF if no features file is found
        elif not os.path.exists(features_path): # Case where files_suffix == 'all' but file doesn't exist
             logger.error(f"❌ Общий файл признаков не найден: {os.path.basename(features_path)} для {tf}. Пропуск таймфрейма.")
             continue # Skip this TF

        try:
            df = pd.read_pickle(features_path)
            logger.info(f"Загружены признаки из {os.path.basename(features_path)}. Размер: {df.shape}")
        except Exception as e:
            logger.error(f"❌ Не удалось прочитать файл признаков {features_path}: {e}. Пропуск таймфрейма {tf}.")
            continue

        if df.empty:
            logger.warning(f"⚠️ Файл признаков пуст: {os.path.basename(features_path)}. Пропуск таймфрейма {tf}.")
            continue

        # --- Filter DataFrame by Target Symbols (if target_syms is specified) ---
        # This step is necessary if the loaded features file contains more symbols
        # than requested by symbol_filter or group_filter.
        if target_syms is not None:
            original_rows_count = len(df)
            df = df[df['symbol'].isin(target_syms)].copy() # Use .copy() after filtering
            after_filter_count = len(df)

            if df.empty:
                 logger.info(f"🤷 После фильтрации по символам/группе нет данных на {tf}. Пропуск таймфрейма {tf}.")
                 continue # Переходим к следующему таймрейму

            if original_rows_count > after_filter_count:
                 logger.info(f"DataFrame отфильтрован по символам. {original_rows_count} → {after_filter_count} строк для TF {tf}.")
            # else: # Log if no filtering happened (e.g., loaded a group-specific file that matched the filter)
            #      logger.debug(f"DataFrame did not require filtering by symbols for TF {tf}.")


        # --- Load Selected Features List ---
        # Attempt to load the selected features file based on the *original* files_suffix
        # used to name the models, falling back to the 'all' suffix if needed.
        features_list_path_specific = os.path.join(PATHS_CONFIG['models_dir'], f"{files_suffix}_{tf}_features_selected.txt")
        features_list_path_all = os.path.join(PATHS_CONFIG['models_dir'], f"all_{tf}_features_selected.txt")

        features_list_path = None
        if os.path.exists(features_list_path_specific):
             features_list_path = features_list_path_specific
             logger.debug(f"Использую список признаков: {os.path.basename(features_list_path)}")
        elif os.path.exists(features_list_path_all):
             features_list_path = features_list_path_all
             logger.warning(f"Список признаков '{os.path.basename(features_list_path_specific)}' не найден. Использую общий список признаков: {os.path.basename(features_list_path_all)}.")
        else:
            logger.error(
                f"❌ Ни специфичный ('{os.path.basename(features_list_path_specific)}') ни общий ('{os.path.basename(features_list_path_all)}') список признаков не найден для {tf}. Невозможно безопасно продолжить. Пропуск.")
            continue

        try:
            with open(features_list_path, "r", encoding="utf-8") as f:
                feature_cols_from_file = [line.strip() for line in f if line.strip()]
            if not feature_cols_from_file:
                logger.error(f"❌ Файл со списком признаков '{os.path.basename(features_list_path)}' пуст. Пропуск {tf}.")
                continue
            logger.debug(f"Загружено {len(feature_cols_from_file)} признаков из списка.")
        except Exception as e:
            logger.error(f"❌ Ошибка чтения файла со списком признаков '{os.path.basename(features_list_path)}': {e}. Пропуск {tf}.")
            continue

        # --- Prepare Data for Prediction ---
        # For each symbol in the (potentially filtered) DataFrame, get the last row for X_live
        # and the rest of the data for history (for similarity analysis).

        available_symbols_in_data = df['symbol'].unique()
        if len(available_symbols_in_data) == 0:
             logger.warning(f"Нет символов в данных для обработки на {tf} после фильтрации.")
             continue # Move to next TF

        # Sort symbols for predictable processing order
        symbols_to_process_this_tf = sorted(available_symbols_in_data.tolist())

        for symbol in symbols_to_process_this_tf:

            df_sym = df[df['symbol'] == symbol].sort_values('timestamp').copy()

            # Ensure enough data for processing (at least 1 row for prediction, more for history/sim)
            if len(df_sym) < 2: # Need at least 2 rows: 1 for X_live, 1 for history (for sim)
                 logger.warning(f"⚠️ Недостаточно данных ({len(df_sym)} строк) для {symbol} {tf}. Пропуск символа.")
                 continue

            logger.debug(f"--- Обработка {symbol} на {tf} ({len(df_sym)} строк) ---")

            # Get the last row for the live prediction input
            row_df = df_sym.iloc[-1:].copy()
            # Get the rest of the data for historical analysis (e.g., similarity)
            hist_df_full = df_sym.iloc[:-1].copy()

            if 'close' not in row_df.columns or 'timestamp' not in row_df.columns:
                logger.warning(f"⚠️ В последних данных для {symbol} {tf} отсутствует 'close' или 'timestamp'. Пропуск символа.")
                continue

            # --- Select features for prediction ---
            # Check that all selected features exist in the data slice for this symbol
            missing_cols_in_df_for_symbol = [col for col in feature_cols_from_file if col not in row_df.columns]
            if missing_cols_in_df_for_symbol:
                logger.error(
                    f"❌ В DataFrame для {symbol} на {tf} из {os.path.basename(features_path)} отсутствуют столбцы {missing_cols_in_df_for_symbol}, "
                    f"необходимые для модели (согласно {os.path.basename(features_list_path)}). Пропуск символа {symbol} на этом таймфрейме."
                )
                continue

            X_live = row_df[feature_cols_from_file].copy() # Select features and make a copy

            # Check for NaN in X_live features before prediction
            if X_live.isnull().values.any():
                 nan_features = X_live.columns[X_live.isnull().any()].tolist()
                 logger.warning(f"⚠️ В X_live для {symbol} {tf} есть NaN в признаках: {nan_features}. Пропуск символа {symbol}.")
                 continue # Skip this symbol/TF

            # --- Load Models for the Symbol/Key and TF ---
            # Use the original files_suffix to load models, as preprocess should have
            # generated features files with that suffix, and train should have saved models with it.
            # load_model_with_fallback handles finding group/all models if specific isn't found.
            model_class = load_model_with_fallback(files_suffix, tf, "clf_class")
            model_delta = load_model_with_fallback(files_suffix, tf, "reg_delta")
            model_vol = load_model_with_fallback(files_suffix, tf, "reg_vol")
            model_tp_hit = load_model_with_fallback(files_suffix, tf, "clf_tp_hit") # TP-hit model is optional

            # Check that essential models are loaded
            if not all([model_class, model_delta, model_vol]):
                logger.warning(
                    f"⚠️ Одна или несколько основных моделей для {symbol} на {tf} не загружены. Пропуск символа {symbol} на этом таймфrame.")
                continue # Skip this symbol for this tf


            # --- Similarity Analysis ---
            # Need delta column in historical data for similarity target
            avg_delta_similar, std_delta_similar, similarity_hint = np.nan, np.nan, "Нет данных для анализа схожести"

            if 'delta' not in hist_df_full.columns:
                logger.debug(f"Столбец 'delta' отсутствует в исторических данных hist_df_full для {symbol} {tf}. Анализ схожести невозможен.")
                similarity_hint = "Отсутствует столбец 'delta' в истории"
            else:
                 # Select the same features used for prediction from history, plus the delta target
                required_hist_cols = feature_cols_from_file + ['delta']
                missing_hist_cols = [col for col in required_hist_cols if col not in hist_df_full.columns]

                if missing_hist_cols:
                    logger.debug(f"В hist_df_full для {symbol} {tf} отсутствуют столбцы {missing_hist_cols} для анализа схожести.")
                    similarity_hint = "Отсутствуют необходимые исторические признаки/цель"
                else:
                    hist_for_sim_features = hist_df_full[feature_cols_from_file].copy()
                    hist_for_sim_deltas = hist_df_full['delta'].copy()

                    # Similarity analysis function handles cleaning NaNs and min history check
                    avg_delta_similar, std_delta_similar, similarity_hint = similarity_analysis(
                        X_live, hist_for_sim_features, hist_for_sim_deltas)


            # --- Make Predictions ---
            try:
                # Classification prediction (direction)
                proba_raw = model_class.predict_proba(X_live)[0]
                # Get model's learned classes to map probabilities correctly
                model_classes_clf = getattr(model_class, 'classes_', [0, 1]) # Default to [0, 1] if .classes_ not found

                # Map raw probabilities to class labels
                proba_dict_model_order = {model_classes_clf[i]: proba_raw[i] for i in range(len(proba_raw))}

                # Determine predicted class label based on highest probability
                if len(proba_raw) > 0:
                    pred_class_label_raw = model_classes_clf[proba_raw.argmax()]
                    # Map raw label (0/1) to 'DOWN'/'UP' or keep as is if different
                    if pred_class_label_raw == 1:
                        signal = "LONG" # Convention: 1 is UP/LONG
                    elif pred_class_label_raw == 0:
                        signal = "SHORT" # Convention: 0 is DOWN/SHORT
                    else:
                        signal = str(pred_class_label_raw) # Use raw label if not 0 or 1
                else:
                    signal = "UNKNOWN"
                    logger.warning(f"model_class.predict_proba returned empty for {symbol} {tf}.")


                # Calculate confidence score (difference between highest and second highest probability)
                if len(proba_raw) < 2:
                    confidence_score = float(proba_raw.max()) if len(proba_raw) == 1 else 0.0
                else:
                    sorted_probas = np.sort(proba_raw)
                    confidence_score = float(sorted_probas[-1] - sorted_probas[-2])

                confidence_hint = get_confidence_hint(confidence_score)


                # Regression predictions (delta and volatility)
                predicted_delta = model_delta.predict(X_live)[0]
                predicted_volatility = model_vol.predict(X_live)[0]

                # Ensure predicted volatility is non-negative
                if pd.notna(predicted_volatility) and predicted_volatility < 0:
                     logger.warning(f"Predicted volatility is negative ({predicted_volatility:.6f}) for {symbol} {tf}. Setting to 0.")
                     predicted_volatility = 0.0


                # TP-hit probability prediction (optional model)
                tp_hit_proba = np.nan
                if model_tp_hit:
                    try:
                        tp_hit_proba_all_classes = model_tp_hit.predict_proba(X_live)[0]
                        model_classes_tp_hit = getattr(model_tp_hit, 'classes_', None)

                        # Find probability for the positive class (usually 1)
                        if model_classes_tp_hit is not None and 1 in model_classes_tp_hit:
                            try:
                                class_1_idx = list(model_classes_tp_hit).index(1)
                                tp_hit_proba = tp_hit_proba_all_classes[class_1_idx]
                            except ValueError:
                                logger.error(f"Internal error: TP Hit class '1' not found in model.classes_ ({model_classes_tp_hit}) for {symbol} {tf}. Setting tp_hit_proba to NaN.")
                                tp_hit_proba = np.nan # Stay NaN
                        elif len(tp_hit_proba_all_classes) > 1:
                            # Fallback: Assume binary classification where index 1 is the positive class
                            tp_hit_proba = tp_hit_proba_all_classes[1]
                            logger.warning(f"TP Hit model classes ({model_classes_tp_hit}) do not contain '1'. Using probability at index 1 for {symbol} {tf}.")
                        else:
                            logger.warning(f"TP Hit predict_proba output has unexpected shape ({len(tp_hit_proba_all_classes)}) for {symbol} {tf}. Setting tp_hit_proba to NaN.")
                            tp_hit_proba = np.nan # Stay NaN

                    except Exception as e_tp_pred:
                        logger.error(f"Error during TP Hit predict_proba for {symbol} {tf}: {e_tp_pred}. Setting tp_hit_proba to NaN.")
                        tp_hit_proba = np.nan # Stay NaN


            except Exception as e:
                logger.error(f"❌ Error during model prediction for {symbol} {tf} with X_live shape {X_live.shape}: {e}. Skipping symbol.", exc_info=True)
                continue # Skip this symbol if prediction fails

            # --- Post-processing and Decision Logic ---

            # Determine trade direction based on the signal string
            direction = 'long' if signal == 'LONG' else 'short' if signal == 'SHORT' else 'none'

            # Compute final delta by blending model prediction and history average
            delta_final = compute_final_delta(predicted_delta, avg_delta_similar, std_delta_similar)

            # Determine signal strength
            signal_strength_val = get_signal_strength(delta_final, confidence_score, std_delta_similar)

            # Check for conflict between model and history delta directions
            conflict_flag = is_conflict(predicted_delta, avg_delta_similar)

            # Calculate Trade Levels (SL/TP)
            # Use predicted volatility for SL/TP calculation
            entry_price = float(row_df['close'].values[0])
            sl, tp = calculate_trade_levels(entry_price, direction, predicted_volatility)


            # --- Determine if trade is worthy based on config thresholds ---
            is_trade_worthy = False
            # Check common filters first (confidence, sigma_hist, tp_hit_proba)
            passes_common_filters = (
                confidence_score >= MIN_CONFIDENCE_FOR_PREDICT and
                (pd.isna(std_delta_similar) or std_delta_similar < SIGMA_HISTORY_MAX) and # Check sigma_history_std against max threshold
                (pd.isna(tp_hit_proba) or tp_hit_proba >= TP_HIT_PROBA_MIN) # Check tp_hit_proba against min threshold
            )

            if passes_common_filters:
                # Then check directional delta threshold
                if direction == 'long':
                    # For LONG, delta_final must be positive and greater than threshold
                    if pd.notna(delta_final) and delta_final >= DELTA_FINAL_ABS_MIN:
                        is_trade_worthy = True
                elif direction == 'short':
                    # For SHORT, delta_final must be negative and its absolute value greater than threshold
                    if pd.notna(delta_final) and delta_final <= -DELTA_FINAL_ABS_MIN:
                        is_trade_worthy = True
                # If direction is 'none' or UNKNOWN, it's not trade worthy by definition


            # --- Store Prediction Results ---
            ts_obj = pd.to_datetime(row_df['timestamp'].values[0]) # Ensure it's a datetime object

            prediction_entry = {
                'symbol': symbol,
                'tf': tf,
                'timestamp_obj': ts_obj, # Store object for sorting/manipulation
                'timestamp_str_log': ts_obj.strftime('%Y-%m-%d %H:%M:%S'), # String format for logging
                'timestamp_str_display': ts_obj.strftime('%Y-%m-%d %H:%M'), # String format for console/CSV
                'signal': signal, # 'LONG', 'SHORT', 'NEUTRAL', 'UNKNOWN'
                'confidence_score': confidence_score, # 0 to 1, p_max - p_2nd
                'confidence_hint': confidence_hint, # Text hint for confidence
                'proba_dict': proba_dict_model_order, # Probabilities mapped to model class labels (e.g., {0: 0.6, 1: 0.4})
                'predicted_delta': predicted_delta, # Raw delta from regression model
                'predicted_volatility': predicted_volatility, # Raw volatility from regression model
                'avg_delta_similar': avg_delta_similar, # Avg delta from similar history
                'std_delta_similar': std_delta_similar, # Std dev of deltas from similar history
                'similarity_hint': similarity_hint, # Text hint for similarity analysis
                'delta_final': delta_final, # Blended final delta
                'signal_strength': signal_strength_val, # '🟢 Сильный', '🟡 Умеренный', '⚪ Слабый'
                'conflict': conflict_flag, # Boolean if model and history deltas conflict
                'tp_hit_proba': tp_hit_proba, # TP-hit probability from optional model
                'is_trade_worthy': is_trade_worthy, # Boolean based on thresholds
                'entry': entry_price, # Current close price
                'sl': sl, # Calculated Stop Loss
                'tp': tp, # Calculated Take Profit
                'direction': direction, # 'long', 'short', 'none' (based on signal string)
                'error': None # Placeholder for any symbol-specific errors encountered
            }

            all_predictions_data.append(prediction_entry)

    logger.info("✅  Процесс генерации прогнозов для всех символов/ТФ завершён.")
    return all_predictions_data


# --- Output Functions (Can be called by the script after getting data) ---

def save_predictions_and_plan(predictions_data, files_suffix):
    """Saves prediction results and trade plan to CSV and generates alert file."""
    logger.info("💾  Сохранение результатов...")

    # Define output paths using the files_suffix
    predictions_file = os.path.join(PATHS_CONFIG['logs_dir'], f'latest_predictions_{files_suffix}.csv')
    trade_plan_file = os.path.join(PATHS_CONFIG['logs_dir'], f'trade_plan_{files_suffix}.csv')
    alerts_file = os.path.join(PATHS_CONFIG['logs_dir'], f'alerts_{files_suffix}.txt')


    if not predictions_data:
        logger.info("🤷 Нет данных прогнозов для сохранения.")
        return

    # --- Prepare Predictions DataFrame for CSV ---
    df_out_list = []
    # Determine column order for CSV, including probabilities
    # Get all unique keys from proba_dict across all prediction entries
    all_proba_keys = sorted(list(set(key for entry in predictions_data for key in entry['proba_dict'].keys())))

    csv_columns_order = [
        'symbol', 'tf', 'timestamp', 'signal', 'confidence_score', 'tp_hit_proba',
        'predicted_delta', 'avg_delta_similar', 'std_delta_similar', 'delta_final',
        'predicted_volatility', 'entry', 'sl', 'tp', 'direction',
        'signal_strength', 'conflict', 'confidence_hint', 'similarity_hint',
        'is_trade_worthy', # New field
    ]
    # Add probability columns after fixed columns
    csv_columns_order.extend(all_proba_keys)
    # Add any other keys that might be in the dict but not in the predefined list
    all_entry_keys = set(key for entry in predictions_data for key in entry.keys())
    for key in all_entry_keys:
        if key not in csv_columns_order and key not in ['timestamp_obj', 'timestamp_str_log', 'timestamp_str_display', 'proba_dict', 'error']: # Exclude internal/redundant keys
            csv_columns_order.append(key)


    for r_item in predictions_data:
        # Create a flat dictionary for each row, extracting data and merging proba_dict
        item = {key: r_item.get(key) for key in csv_columns_order if key not in all_proba_keys} # Get fixed cols
        item['timestamp'] = r_item['timestamp_str_display'] # Use display format for CSV
        # Add probabilities, defaulting to NaN if a class wasn't present in the model/dict for this entry
        for proba_key in all_proba_keys:
             item[proba_key] = r_item['proba_dict'].get(proba_key, np.nan)

        df_out_list.append(item)

    df_out = pd.DataFrame(df_out_list)

    # Ensure final columns match the determined order
    final_csv_columns = [col for col in csv_columns_order if col in df_out.columns]
    # Add any columns in df_out not in the desired order (shouldn't happen if logic is correct, but safeguard)
    for col in df_out.columns:
        if col not in final_csv_columns:
            final_csv_columns.append(col)

    if not df_out.empty:
        try:
            df_out = df_out[final_csv_columns] # Reorder columns
            df_out.to_csv(predictions_file, index=False, float_format='%.6f')
            logger.info(f"📄  Сигналы сохранены в: {predictions_file}")
        except Exception as e:
            logger.error(f"❌ Не удалось сохранить {predictions_file}: {e}")
    else:
        logger.info(f"🤷 Нет данных для сохранения в {predictions_file}.")

    # --- Prepare Trade Plan DataFrame for CSV ---
    trade_plan_data = []
    for row in predictions_data:
        # Check if trade is worthy based on the flag determined in generate_predictions
        if row.get('is_trade_worthy'): # Use the pre-computed flag
            # Calculate RR ratio only if SL/TP are valid
            rr_value = np.nan
            if pd.notna(row['sl']) and pd.notna(row['tp']) and pd.notna(row['entry']):
                 if abs(row['entry'] - row['sl']) > 1e-9: # Avoid division by zero/near-zero
                     try:
                         rr_value = round(abs(row['tp'] - row['entry']) / abs(row['entry'] - row['sl']), 2)
                     except ZeroDivisionError:
                         logger.warning(f"⚠️ Деление на ноль при расчете RR для {row['symbol']} {row['tf']} (Entry={row['entry']}, SL={row['sl']}).")
                         rr_value = np.nan
                 else:
                      logger.warning(f"⚠️ SL очень близко к цене входа для {row['symbol']} {row['tf']}. Невозможно рассчитать RR.")


            trade_plan_data.append({
                'symbol': row['symbol'],
                'tf': row['tf'],
                'timestamp': row['timestamp_str_display'], # Use display format for CSV
                'direction': row['direction'],
                'signal': row['signal'], # Use the raw signal string ('LONG'/'SHORT')
                'confidence': row['confidence_score'],
                'tp_hit_proba': row['tp_hit_proba'],
                'delta_final': row['delta_final'], # Include final delta in trade plan
                'signal_strength': row['signal_strength'], # Include strength
                'entry': row['entry'],
                'sl': row['sl'],
                'tp': row['tp'],
                'rr': rr_value,
                'hint': row['confidence_hint'], # Use confidence hint
                'similarity_hint': row['similarity_hint'], # Include similarity hint
                'conflict': row['conflict'], # Include conflict flag
            })

    if trade_plan_data:
        df_trade_plan = pd.DataFrame(trade_plan_data)
        # Define column order for trade plan CSV
        trade_plan_cols_order = [
            'symbol', 'tf', 'timestamp', 'direction', 'signal', 'confidence', 'tp_hit_proba',
            'delta_final', 'signal_strength', 'conflict', 'entry', 'sl', 'tp', 'rr',
            'hint', 'similarity_hint'
        ]
        # Ensure all columns are included
        final_trade_plan_cols = [col for col in trade_plan_cols_order if col in df_trade_plan.columns]
        for col in df_trade_plan.columns:
            if col not in final_trade_plan_cols:
                final_trade_plan_cols.append(col)

        if not df_trade_plan.empty:
            try:
                df_trade_plan = df_trade_plan[final_trade_plan_cols] # Reorder columns
                df_trade_plan.to_csv(trade_plan_file, index=False, float_format='%.6f')
                logger.info(f"📈  Торговый план сохранён в: {trade_plan_file}")
            except Exception as e:
                logger.error(f"❌ Не удалось сохранить {trade_plan_file}: {e}")
        else:
            logger.info(f"🤷 Торговый план пуст, файл {trade_plan_file} не создан.")
    else:
        logger.info("🤷 Нет данных для торгового плана (ни один сигнал не прошел фильтр is_trade_worthy).")

    # --- Generate Alerts File ---
    # Generate alerts only for signals marked as 'is_trade_worthy'
    worthy_signals = [row for row in predictions_data if row.get('is_trade_worthy')]

    if worthy_signals:
         try:
            with open(alerts_file, "w", encoding="utf-8") as alert_file:
                for row in worthy_signals:
                    # Format the alert line
                    # Use short icons for strength
                    strength_icon = row['signal_strength'].replace(" Сильный","🟢").replace(" Умеренный","🟡").replace(" Слабый","⚪") # Should be 🟢 or 🟡 here
                    alert_line = (
                        f"{row['symbol']} {row['tf']} {strength_icon} "
                        f"{row['direction'].upper()} Δ:{row['delta_final']:.2%} "
                        f"Conf:{row['confidence_score']:.2f} "
                        f"TP:{row['tp_hit_proba']:.1%}\n" if pd.notna(row['tp_hit_proba'])
                        else f"{row['symbol']} {row['tf']} {strength_icon} "
                             f"{row['direction'].upper()} Δ:{row['delta_final']:.2%} "
                             f"Conf:{row['confidence_score']:.2f} TP:N/A\n"
                    )
                    alert_file.write(alert_line)
            logger.info(f"📣 Alert-файл сохранён: {alerts_file} ({len(worthy_signals)} сигналов)")
         except Exception as e:
             logger.error(f"❌ Не удалось сохранить alert-файл {alerts_file}: {e}")
    else:
         logger.info(f"🤷 Нет торговых сигналов, прошедших фильтр is_trade_worthy. Alert-файл {alerts_file} не создан.")


# --- Console Output Function ---

def print_predictions_to_console(predictions_data):
    """Prints prediction results to the console, grouped by symbol."""
    if not predictions_data:
        logger.info("🤷 Нет данных прогнозов для вывода в консоль.")
        return

    logger.info("\n📊  Результаты прогнозов:")

    headers_tabulate = ["TF", "Timestamp", "Сигнал", "Conf.", "ΔModel", "ΔHist", "σHist", "ΔFinal", "Конф?", "Сила",
                        "TP Hit%", "Trade?"]

    # Group data by symbol
    grouped_by_symbol = {}
    for row_data_item in predictions_data:
        grouped_by_symbol.setdefault(row_data_item['symbol'], []).append(row_data_item)

    # Sort symbols for alphabetical output
    sorted_symbols_for_display = sorted(grouped_by_symbol.keys())

    for symbol_key in sorted_symbols_for_display:
        rows_list = grouped_by_symbol[symbol_key]
        try:
            # Sort within each symbol group: first by timeframe order, then by confidence descending
            # Use default timeframes order from config
            tf_order = TIMEFRAMES_CONFIG.get('default', [])
            tf_order_map = {tf: i for i, tf in enumerate(tf_order)}

            sorted_rows = sorted(rows_list,
                                 key=lambda r_item_sort: (tf_order_map.get(r_item_sort['tf'], len(tf_order)), # Use index in default list, or put unknown TFs at the end
                                                          -r_item_sort['confidence_score'])) # Sort by confidence descending

        except Exception as e:
            logger.warning(f"⚠️ Ошибка сортировки для {symbol_key}: {e}. Сортировка только по уверенности.")
            sorted_rows = sorted(rows_list, key=lambda r_item_sort: (-r_item_sort['confidence_score']))

        table_data_tabulate = []
        for r_tab in sorted_rows:
            # Format data for tabulate table
            table_data_tabulate.append([
                r_tab['tf'],
                r_tab['timestamp_str_display'],
                r_tab['signal'], # 'LONG', 'SHORT', etc.
                f"{r_tab['confidence_score']:.3f}",
                f"{r_tab['predicted_delta']:.2%}" if pd.notna(r_tab['predicted_delta']) else "N/A",
                f"{r_tab['avg_delta_similar']:.2%}" if pd.notna(r_tab['avg_delta_similar']) else "N/A",
                f"{r_tab['std_delta_similar']:.2%}" if pd.notna(r_tab['std_delta_similar']) else "N/A",
                f"{r_tab['delta_final']:.2%}" if pd.notna(r_tab['delta_final']) else "N/A",
                "❗" if r_tab['conflict'] else " ",
                r_tab['signal_strength'].replace(" Сильный", "🟢").replace(" Умеренный", "🟡").replace(" Слабый", "⚪"),
                f"{r_tab['tp_hit_proba']:.1%}" if pd.notna(r_tab['tp_hit_proba']) else "N/A",
                "✅" if r_tab.get('is_trade_worthy') else "❌" # Display trade_worthy flag
            ])

        print(f"\n📊  Прогноз по символу: {symbol_key}")
        try:
            print(tabulate(table_data_tabulate, headers=headers_tabulate, tablefmt="pretty", stralign="right", numalign="right"))
        except Exception as e:
            logger.error(f"❌ Ошибка при выводе таблицы для {symbol_key}: {e}")
            # Fallback to simple print if tabulate fails
            print("Не удалось отформатировать таблицу. Данные:")
            for row in table_data_tabulate:
                print(row)

    # --- Console Output for Top Signals (Worthy Trades) ---
    print("\n📈 ТОП ТОРГОВЫЕ СИГНАЛЫ (Trade Worthy)")
    worthy_signals_for_display = [row for row in predictions_data if row.get('is_trade_worthy')]

    if worthy_signals_for_display:
        # Sort worthy signals: strength > confidence > TF order
        strength_order_map = {"🟢 Сильный": 0, "🟡 Умеренный": 1} # Map strength strings to sort order
        tf_order_map = {tf: i for i, tf in enumerate(tf_order)} # Map TF strings to sort order

        try:
            sorted_worthy_signals = sorted(worthy_signals_for_display,
                                           key=lambda x: (strength_order_map.get(x['signal_strength'], 99),
                                                          -x['confidence_score'], # Sort confidence descending
                                                          tf_order_map.get(x['tf'], len(tf_order))
                                                         ))
        except Exception as e:
             logger.warning(f"Ошибка при сортировке топ торговых сигналов: {e}. Сортировка только по уверенности.")
             sorted_worthy_signals = sorted(worthy_signals_for_display, key=lambda x: (-x['confidence_score']))


        top_signals_table_data = []
        for row in sorted_worthy_signals:
            top_signals_table_data.append({
                'Symbol': row['symbol'],
                'TF': row['tf'],
                'Direction': row['direction'].upper(),
                'ΔFinal': f"{row['delta_final']:.2%}" if pd.notna(row['delta_final']) else "N/A",
                'Conf.': f"{row['confidence_score']:.2f}",
                'Сила': row['signal_strength'].replace(" Сильный", "🟢").replace(" Умеренный", "🟡").replace(" Слабый", "⚪"), # Use icons
                'TP Hit%': f"{row['tp_hit_proba']:.1%}" if pd.notna(row['tp_hit_proba']) else "N/A",
                'Entry': f"{row['entry']:.6f}" if pd.notna(row['entry']) else "N/A", # Add Entry, SL, TP
                'SL': f"{row['sl']:.6f}" if pd.notna(row['sl']) else "N/A",
                'TP': f"{row['tp']:.6f}" if pd.notna(row['tp']) else "N/A",
                'RR': f"{row['rr']:.2f}" if pd.notna(row['rr']) else "N/A", # RR should be calculated here or stored
                'Hint': row['confidence_hint'],
                'Sim Hint': row['similarity_hint'], # Add similarity hint
            })

        if top_signals_table_data:
            # Calculate RR for the display table if needed (could store it in prediction_entry)
            # Let's add RR calculation to generate_predictions and store it in prediction_entry
            # Add RR calculation to generate_predictions and ensure it's stored.
            # Assuming RR is now in the prediction_entry dict.
            # Updated the prediction_entry dict definition above to include 'rr'.
            # Need to calculate RR inside the main loop.

            # Ensure RR is calculated for worthy signals
            for row in worthy_signals_for_display:
                 rr_value_display = np.nan
                 if pd.notna(row['sl']) and pd.notna(row['tp']) and pd.notna(row['entry']):
                     if abs(row['entry'] - row['sl']) > 1e-9:
                         try:
                             rr_value_display = round(abs(row['tp'] - row['entry']) / abs(row['entry'] - row['sl']), 2)
                         except ZeroDivisionError:
                             rr_value_display = np.nan
                     else:
                          rr_value_display = np.nan
                 row['rr'] = rr_value_display # Add or update RR in the dict for display

            # Regenerate top_signals_table_data with RR included
            top_signals_table_data = []
            for row in sorted_worthy_signals:
                top_signals_table_data.append({
                    'Symbol': row['symbol'],
                    'TF': row['tf'],
                    'Direction': row['direction'].upper(),
                    'ΔFinal': f"{row['delta_final']:.2%}" if pd.notna(row['delta_final']) else "N/A",
                    'Conf.': f"{row['confidence_score']:.2f}",
                    'Сила': row['signal_strength'].replace(" Сильный", "🟢").replace(" Умеренный", "🟡").replace(" Слабый", "⚪"), # Use icons
                    'TP Hit%': f"{row['tp_hit_proba']:.1%}" if pd.notna(row['tp_hit_proba']) else "N/A",
                    'Entry': f"{row['entry']:.6f}" if pd.notna(row['entry']) else "N/A", # Add Entry, SL, TP
                    'SL': f"{row['sl']:.6f}" if pd.notna(row['sl']) else "N/A",
                    'TP': f"{row['tp']:.6f}" if pd.notna(row['tp']) else "N/A",
                    'RR': f"{row['rr']:.2f}" if pd.notna(row['rr']) else "N/A", # Use the calculated RR
                    'Hint': row['confidence_hint'],
                    'Sim Hint': row['similarity_hint'],
                })


            try:
                print(tabulate(top_signals_table_data, headers="keys", tablefmt="pretty"))
            except Exception as e:
                 logger.error(f"❌ Ошибка при выводе таблицы ТОП ТОРГОВЫХ СИГНАЛОВ: {e}")
                 # Fallback to simple print
                 print("Не удалось отформатировать таблицу ТОП ТОРГОВЫХ СИГНАЛОВ. Данные:")
                 for row in top_signals_table_data:
                     print(row)

        else:
            print("🤷 Нет торговых сигналов, прошедших фильтр is_trade_worthy.")


    # --- Console Output for Trade Plan (Simplified) ---
    # The trade plan is effectively the same as the 'worthy signals' for console
    # We can just print the 'TOP TRADE SIGNALS' section as the console trade plan.
    # Or have a separate, potentially shorter format if needed.
    # For now, the "ТОП ТОРГОВЫЕ СИГНАЛЫ" section serves as the console trade plan.


# Example usage (optional, for testing the module directly)
if __name__ == "__main__":
    # In a real run, logging would be set up by the entry script (cli.py)
    # For direct testing, set it up here
    setup_logging()
    logger = logging.getLogger(__name__) # Re-get logger after setup

    print("Testing src/models/predict.py")

    # To test, you need feature files and model files.
    # This test requires significant setup (dummy models, dummy features).
    # It's more practical to test predict.py via the scripts/predict.py entry point
    # after running preprocess.py and train.py with dummy data.

    # --- Dummy Data and Model Creation for Testing ---
    # This section is complex and often better handled by dedicated unit tests
    # or a robust integration test setup.
    # For a simple test, we can try to create minimal dummy data and models.

    tf_test = '15m'
    key_test = 'test_predict'
    dummy_feature_count = 20
    dummy_history_len = 500 # Needs > SIMILARITY_MIN_HISTORY (100)
    dummy_live_row = 1
    n_samples_total = dummy_history_len + dummy_live_row

    dummy_features_path = os.path.join(PATHS_CONFIG['data_dir'], f"features_{key_test}_{tf_test}.pkl")

    if not os.path.exists(dummy_features_path):
        logger.warning(f"Dummy features file not found at {dummy_features_path}. Creating minimal dummy data for testing.")
        # Create a dummy DataFrame
        dummy_data = {
            'symbol': ['TESTUSDT'] * n_samples_total,
            'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=n_samples_total, freq='min')),
            'open': np.random.rand(n_samples_total) * 100 + 1000,
            'high': lambda x: x['open'] + np.random.rand(len(x)) * 10,
            'low': lambda x: x['open'] - np.random.rand(len(x)) * 10,
            'close': lambda x: x['open'] + np.random.randn(len(x)) * 5,
            'volume': np.random.rand(n_samples_total) * 10000,
            'delta': np.random.randn(n_samples_total) * 0.01, # Required for history/similarity
            'volatility': np.random.rand(n_samples_total) * 0.02, # Required for training targets
             'target_class': np.random.choice(['UP', 'DOWN', np.nan], n_samples_total, p=[0.45, 0.45, 0.1]), # Required for training targets
             'target_tp_hit': np.random.choice([0, 1, np.nan], n_samples_total, p=[0.6, 0.3, 0.1]), # Required for training targets
        }
        # Add dummy numerical features
        feature_cols = [f'feature_{i}' for i in range(dummy_feature_count)]
        for col in feature_cols:
            dummy_data[col] = np.random.randn(n_samples_total) * random.uniform(0.1, 1.0)

        df_dummy = pd.DataFrame(dummy_data)
        df_dummy['high'] = df_dummy[['open', 'close']].max(axis=1) + np.random.rand(n_samples_total) * 5
        df_dummy['low'] = df_dummy[['open', 'close']].min(axis=1) - np.random.rand(n_samples_total) * 5


        os.makedirs(PATHS_CONFIG['data_dir'], exist_ok=True)
        try:
            df_dummy.to_pickle(dummy_features_path)
            logger.info(f"Dummy features file created at {dummy_features_path}")
        except Exception as e:
            logger.error(f"Failed to save dummy features file: {e}")
            # sys.exit(1) # Don't exit, maybe just warn

    dummy_models_dir = PATHS_CONFIG['models_dir']
    os.makedirs(dummy_models_dir, exist_ok=True)

    # Create dummy models (minimal CatBoost instances)
    dummy_feature_names = feature_cols if 'feature_cols' in locals() else [f'feature_{i}' for i in range(dummy_feature_count)]
    dummy_model_params = {'iterations': 1, 'verbose': 0, 'random_seed': RANDOM_STATE} # Minimal params

    models_to_create = ['clf_class', 'reg_delta', 'reg_vol', 'clf_tp_hit']
    created_models = {}

    for model_type in models_to_create:
         model_path = os.path.join(dummy_models_dir, f"{key_test}_{tf_test}_{model_type}.pkl")
         if not os.path.exists(model_path):
             logger.warning(f"Dummy model {model_type} not found. Creating minimal dummy model at {model_path}.")
             try:
                 if 'clf' in model_type:
                      # Need a dummy X and y for fit
                      dummy_X = pd.DataFrame(np.random.rand(10, len(dummy_feature_names)), columns=dummy_feature_names)
                      dummy_y = np.random.randint(0, 2, 10) # Binary target
                      model = CatBoostClassifier(**dummy_model_params, loss_function='Logloss', eval_metric='Accuracy')
                      model.fit(dummy_X, dummy_y)
                      # Add dummy classes attribute if not present
                      if not hasattr(model, 'classes_'):
                           model.classes_ = np.array([0, 1])

                 else: # Regressor
                      dummy_X = pd.DataFrame(np.random.rand(10, len(dummy_feature_names)), columns=dummy_feature_names)
                      dummy_y = np.random.rand(10)
                      model = CatBoostRegressor(**dummy_model_params, loss_function='RMSE', eval_metric='RMSE')
                      model.fit(dummy_X, dummy_y)

                 joblib.dump(model, model_path)
                 created_models[model_type] = model
                 logger.info(f"Dummy model {model_type} created.")
             except Exception as e:
                 logger.error(f"Failed to create dummy model {model_type}: {e}")
                 created_models[model_type] = None # Set to None if creation failed
         else:
              logger.info(f"Dummy model {model_type} already exists.")
              try:
                  created_models[model_type] = joblib.load(model_path)
              except Exception as e:
                   logger.error(f"Failed to load existing dummy model {model_type}: {e}")
                   created_models[model_type] = None


    # Create a dummy selected features file
    dummy_features_list_path = os.path.join(dummy_models_dir, f"{key_test}_{tf_test}_features_selected.txt")
    if not os.path.exists(dummy_features_list_path):
         logger.warning(f"Dummy features list file not found at {dummy_features_list_path}. Creating one.")
         try:
             with open(dummy_features_list_path, "w", encoding="utf-8") as f:
                 f.write("\n".join(dummy_feature_names))
             logger.info("Dummy features list file created.")
         except Exception as e:
             logger.error(f"Failed to create dummy features list file: {e}")


    # Run the prediction process with the dummy data/models
    print("\nRunning generate_predictions with dummy data...")
    try:
        # We need to simulate the df structure that generate_predictions expects
        # which is a DataFrame loaded from features_{key}_{tf}.pkl
        # Let's load the dummy file we created
        if os.path.exists(dummy_features_path):
             df_for_predict_test = pd.read_pickle(dummy_features_path)
             # generate_predictions processes one TF at a time
             predictions = generate_predictions([tf_test], symbol_filter='TESTUSDT') # Test with the dummy symbol
             print("\nSaving dummy predictions and plan...")
             save_predictions_and_plan(predictions, key_test) # Use key_test as suffix
             print("\nPrinting dummy predictions to console...")
             print_predictions_to_console(predictions)
        else:
             logger.error("Dummy features file not available, skipping prediction test.")


    except Exception as e:
        logger.error(f"Error during test prediction run: {e}", exc_info=True)
        # sys.exit(1) # Don't exit, just report error

    print("\nPrediction module test finished.")