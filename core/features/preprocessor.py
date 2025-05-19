# core/features/preprocessor.py
"""
Loads candle data from SQLite database, computes various technical indicators,
target variables (delta, volatility, class), and other features.
Saves the processed features to a pickle file and a sample CSV.
This module is based on the logic from the original preprocess_features.py.
"""
import sqlite3
import pandas as pd
import numpy as np
import ta
import os
import argparse
from tqdm import tqdm
import sys
import logging
from ta.volatility import BollingerBands  # Explicit import for clarity

from ..helpers.utils import load_config, classify_delta_value
from ..helpers.db_ops import load_candles_from_sqlite  # Renamed from load_candles_from_db

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Configuration Constants (Defaults, can be overridden by config.yaml) ---
DEFAULT_DB_PATH = 'database/market_data.db'
DEFAULT_OUTPUT_DIR = 'data'  # For features_*.pkl and sample_*.csv

# Feature engineering parameters (from config or defaults)
DEFAULT_TARGET_SHIFT_CANDLES = 5
DEFAULT_TP_THRESHOLD_FOR_TARGET_TP_HIT = 0.005  # Example, should come from config if used

# Global config dictionary, to be loaded
CONFIG = {}

# Symbol groups (could also be loaded from config if they become very dynamic)
# For now, keeping it similar to original preprocess_features.py for direct use
# These are also defined in config.yaml, this is a fallback or for direct script run.
DEFAULT_SYMBOL_GROUPS = {
    "top8": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"],
    "meme": ["DOGEUSDT", "PEPEUSDT", "FLOKIUSDT", "WIFUSDT", "SHIBUSDT"],
    "defi": ["UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "COMPUSDT"]
}


def _get_feature_eng_config_value(key, default_value=None):
    """Safely retrieves a value from the CONFIG['feature_engineering'] section."""
    global CONFIG
    if not CONFIG:
        CONFIG = load_config()
        if not CONFIG:
            logger.error("Configuration not loaded. Using defaults for feature engineering config.")
            if key == "target_shift_candles": return DEFAULT_TARGET_SHIFT_CANDLES
            if key == "tp_threshold": return DEFAULT_TP_THRESHOLD_FOR_TARGET_TP_HIT
            return default_value

    feature_eng_config = CONFIG.get('feature_engineering', {})
    return feature_eng_config.get(key, default_value)


def _compute_ta_features_for_symbol(df_group):
    """
    Computes Technical Analysis (TA) features for a single symbol's DataFrame.
    The input df_group is expected to be indexed by timestamp for that symbol.
    """
    # Ensure 'close' and 'volume' columns exist
    if 'close' not in df_group.columns or 'volume' not in df_group.columns:
        logger.warning(f"Symbol data missing 'close' or 'volume'. TA features might be incomplete.")
        # Add NaN columns for all expected features to maintain structure
        # This list should match all features created below
        ta_feature_names = [
            'rsi', 'ema_20', 'ema_50', 'macd', 'macd_signal', 'macd_diff', 'obv', 'atr',
            'rsi_shift1', 'rsi_shift2', 'rsi_shift3',
            'ema_20_shift1', 'ema_20_shift2', 'ema_20_shift3',
            'ema_50_shift1', 'ema_50_shift2', 'ema_50_shift3',
            'target_class_shift1', 'volume_z', 'candle_body_size', 'candle_hl_range', 'is_doji',
            'hour', 'minute', 'dayofweek', 'dayofmonth', 'weekofyear',
            'ema_20_slope', 'ema_50_slope', 'rsi_cross_50', 'pin_bar',
            'ema_diff', 'rsi_change_3', 'volume_mean', 'volume_std', 'volume_spike',
            'hour_sin', 'dayofweek_sin', 'bb_width',
            'close_rolling_mean_5', 'close_rolling_std_5', 'volume_rolling_mean_5', 'volume_rolling_std_5', 'returns_rolling_std_5',
            'close_rolling_mean_10', 'close_rolling_std_10', 'volume_rolling_mean_10', 'volume_rolling_std_10', 'returns_rolling_std_10',
            'close_rolling_mean_20', 'close_rolling_std_20', 'volume_rolling_mean_20', 'volume_rolling_std_20', 'returns_rolling_std_20',
            'consecutive_up', 'consecutive_down'
        ]
        for feat_name in ta_feature_names:
            if feat_name not in df_group.columns:
                df_group[feat_name] = np.nan
        return df_group

    # RSI
    df_group['rsi'] = ta.momentum.RSIIndicator(close=df_group['close'], window=14, fillna=False).rsi()
    # EMA
    df_group['ema_20'] = ta.trend.EMAIndicator(close=df_group['close'], window=20, fillna=False).ema_indicator()
    df_group['ema_50'] = ta.trend.EMAIndicator(close=df_group['close'], window=50, fillna=False).ema_indicator()
    # MACD
    macd_indicator = ta.trend.MACD(close=df_group['close'], window_slow=26, window_fast=12, window_sign=9, fillna=False)
    df_group['macd'] = macd_indicator.macd()
    df_group['macd_signal'] = macd_indicator.macd_signal()
    df_group['macd_diff'] = macd_indicator.macd_diff()
    # OBV
    if 'volume' in df_group.columns:
        df_group['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df_group['close'], volume=df_group['volume'], fillna=False).on_balance_volume()
    else:
        df_group['obv'] = np.nan
    # ATR
    if all(col in df_group.columns for col in ['high', 'low', 'close']):
        df_group['atr'] = ta.volatility.AverageTrueRange(high=df_group['high'], low=df_group['low'], close=df_group['close'], window=14, fillna=False).average_true_range()
    else:
        df_group['atr'] = np.nan

    # Lag features for RSI and EMA
    for shift_val in [1, 2, 3]:
        df_group[f'rsi_shift{shift_val}'] = df_group['rsi'].shift(shift_val)
        df_group[f'ema_20_shift{shift_val}'] = df_group['ema_20'].shift(shift_val)
        df_group[f'ema_50_shift{shift_val}'] = df_group['ema_50'].shift(shift_val)

    # Lag target_class (must be computed before this function if used as feature)
    if 'target_class' in df_group.columns:  # target_class is defined based on future, so shift(1) is a valid lag
        df_group['target_class_shift1'] = df_group['target_class'].shift(1)
    else:
        df_group['target_class_shift1'] = np.nan  # Or 0, or some other placeholder

    # Volume Z-score
    if 'volume' in df_group.columns:
        vol_roll_mean = df_group['volume'].rolling(window=20, min_periods=1).mean()
        vol_roll_std = df_group['volume'].rolling(window=20, min_periods=1).std().replace(0, np.nan)  # Avoid division by zero
        df_group['volume_z'] = (df_group['volume'] - vol_roll_mean) / vol_roll_std
        df_group['volume_z'] = df_group['volume_z'].clip(-5, 5)  # Clip outliers
    else:
        df_group['volume_z'] = np.nan

    # Candle metrics
    if all(col in df_group.columns for col in ['open', 'close', 'high', 'low']):
        df_group['candle_body_size'] = abs(df_group['close'] - df_group['open'])
        df_group['candle_hl_range'] = df_group['high'] - df_group['low']
        df_group['is_doji'] = ((df_group['candle_body_size'] / (df_group['candle_hl_range'].replace(0, np.nan) + 1e-9)) < 0.1).astype(int)
        # Pin bar
        wick_up = df_group['high'] - df_group[['open', 'close']].max(axis=1)
        wick_down = df_group[['open', 'close']].min(axis=1) - df_group['low']
        body = abs(df_group['close'] - df_group['open'])
        df_group['pin_bar'] = ((wick_up > (body * 2 + 1e-9)) | (wick_down > (body * 2 + 1e-9))).astype(int)
    else:
        df_group['candle_body_size'] = np.nan
        df_group['candle_hl_range'] = np.nan
        df_group['is_doji'] = 0
        df_group['pin_bar'] = 0

    # Time features (if index is DatetimeIndex)
    if isinstance(df_group.index, pd.DatetimeIndex):
        df_group['hour'] = df_group.index.hour.astype(np.int8)
        df_group['minute'] = df_group.index.minute.astype(np.int8)
        df_group['dayofweek'] = df_group.index.dayofweek.astype(np.int8)  # Monday=0, Sunday=6
        df_group['dayofmonth'] = df_group.index.day.astype(np.int8)
        df_group['weekofyear'] = df_group.index.isocalendar().week.astype(np.int8)
        df_group['hour_sin'] = np.sin(2 * np.pi * df_group['hour'] / 24)
        df_group['dayofweek_sin'] = np.sin(2 * np.pi * df_group['dayofweek'] / 7)
    else:
        for col in ['hour', 'minute', 'dayofweek', 'dayofmonth', 'weekofyear', 'hour_sin', 'dayofweek_sin']:
            df_group[col] = np.nan

    # Slopes and crosses
    df_group['ema_20_slope'] = df_group['ema_20'].diff()
    df_group['ema_50_slope'] = df_group['ema_50'].diff()
    df_group['rsi_cross_50'] = ((df_group['rsi'] > 50) & (df_group['rsi'].shift(1) <= 50)).astype(int)

    # EMA diff
    df_group['ema_diff'] = df_group['ema_20'] - df_group['ema_50']
    # RSI change
    df_group['rsi_change_3'] = df_group['rsi'].diff(3)

    # Volume spike
    if 'volume' in df_group.columns:
        df_group['volume_mean'] = df_group['volume'].rolling(window=20, min_periods=1).mean()
        df_group['volume_std'] = df_group['volume'].rolling(window=20, min_periods=1).std().replace(0, np.nan)
        df_group['volume_spike'] = ((df_group['volume'] - df_group['volume_mean']) > 2 * df_group['volume_std']).astype(int)
    else:
        df_group['volume_mean'] = np.nan
        df_group['volume_std'] = np.nan
        df_group['volume_spike'] = 0

    # Bollinger Bands Width
    if 'close' in df_group.columns and len(df_group['close']) >= 20:
        try:
            bb_indicator = BollingerBands(close=df_group['close'], window=20, window_dev=2, fillna=False)
            df_group['bb_width'] = bb_indicator.bollinger_wband()  # Correct method for width
            # Or manually: bb_h = bb_indicator.bollinger_hband(); bb_l = bb_indicator.bollinger_lband(); df_group['bb_width'] = bb_h - bb_l
        except Exception as e:
            logger.debug(f"Could not calculate Bollinger Bands for a group: {e}")
            df_group['bb_width'] = np.nan
    else:
        df_group['bb_width'] = np.nan

    # Rolling statistics (from R1)
    rolling_windows = [5, 10, 20]
    if 'log_return_1' in df_group.columns:  # Ensure log_return_1 is pre-calculated
        for window in rolling_windows:
            df_group[f'close_rolling_mean_{window}'] = df_group['close'].rolling(window, min_periods=1).mean()
            df_group[f'close_rolling_std_{window}'] = df_group['close'].rolling(window, min_periods=1).std().replace(0, np.nan)
            if 'volume' in df_group.columns:
                df_group[f'volume_rolling_mean_{window}'] = df_group['volume'].rolling(window, min_periods=1).mean()
                df_group[f'volume_rolling_std_{window}'] = df_group['volume'].rolling(window, min_periods=1).std().replace(0, np.nan)
            else:  # Ensure columns exist even if volume is missing
                df_group[f'volume_rolling_mean_{window}'] = np.nan
                df_group[f'volume_rolling_std_{window}'] = np.nan
            df_group[f'returns_rolling_std_{window}'] = df_group['log_return_1'].rolling(window, min_periods=1).std().replace(0, np.nan)
    else:  # Ensure columns exist if log_return_1 is missing
        for window in rolling_windows:
            df_group[f'close_rolling_mean_{window}'] = np.nan
            df_group[f'close_rolling_std_{window}'] = np.nan
            df_group[f'volume_rolling_mean_{window}'] = np.nan
            df_group[f'volume_rolling_std_{window}'] = np.nan
            df_group[f'returns_rolling_std_{window}'] = np.nan

    # Consecutive trend (from R2)
    if 'close' in df_group.columns:
        trend_up = (df_group['close'] > df_group['close'].shift(1)).astype(int)
        trend_down = (df_group['close'] < df_group['close'].shift(1)).astype(int)
        df_group['consecutive_up'] = trend_up * (trend_up.groupby((trend_up != trend_up.shift()).cumsum()).cumcount() + 1)
        df_group['consecutive_down'] = trend_down * (trend_down.groupby((trend_down != trend_down.shift()).cumsum()).cumcount() + 1)
    else:
        df_group['consecutive_up'] = 0
        df_group['consecutive_down'] = 0

    return df_group


def compute_and_prepare_features(df_all_candles_input, tf_name, btc_features_df=None):
    """
    Computes features and target variables for all symbols in the input DataFrame.

    Args:
        df_all_candles_input (pd.DataFrame): DataFrame with candle data for all symbols.
                                     Expected columns: 'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume'.
        tf_name (str): Timeframe identifier (e.g., '15m').
        btc_features_df (pd.DataFrame, optional): Pre-calculated BTC features, indexed by timestamp.

    Returns:
        pd.DataFrame: DataFrame with all computed features and targets.
    """
    logger.info(f"Starting feature computation for {df_all_candles_input['symbol'].nunique()} symbols on TF {tf_name}...")

    if df_all_candles_input.empty:
        logger.warning(f"Input DataFrame for feature computation is empty for TF {tf_name}.")
        return pd.DataFrame()

    all_symbols_processed_features = []

    target_shift = _get_feature_eng_config_value("target_shift_candles", DEFAULT_TARGET_SHIFT_CANDLES)
    tp_hit_threshold = _get_feature_eng_config_value("tp_threshold", DEFAULT_TP_THRESHOLD_FOR_TARGET_TP_HIT)

    unique_symbols_in_df = df_all_candles_input['symbol'].unique()
    if len(unique_symbols_in_df) == 0:
        logger.warning(f"No unique symbols found in input data for TF {tf_name}.")
        return pd.DataFrame()

    for symbol_val in tqdm(unique_symbols_in_df, desc=f"Processing features for TF {tf_name}", unit="symbol"):
        df_sym_original = df_all_candles_input[df_all_candles_input['symbol'] == symbol_val].copy()

        if len(df_sym_original) < 100:  # Increased min_periods for some TA indicators
            logger.info(f"Symbol {symbol_val} on TF {tf_name} has insufficient data ({len(df_sym_original)} < 100). Skipping.")
            continue

        # Set timestamp as index for easier processing within _compute_ta_features_for_symbol
        # and for merging with BTC features.
        df_sym = df_sym_original.set_index('timestamp', drop=False)  # Keep timestamp column for later reset_index

        # --- Calculate Target Variables ---
        # Ensure 'close' is present before calculating targets
        if 'close' not in df_sym.columns:
            logger.warning(f"Symbol {symbol_val} missing 'close' column. Cannot compute targets. Skipping.")
            continue

        df_sym['log_return_1'] = np.log(df_sym['close'] / df_sym['close'].shift(1))  # Used in TA features

        df_sym['future_close'] = df_sym['close'].shift(-target_shift)
        df_sym['delta'] = (df_sym['future_close'] / df_sym['close']) - 1

        # Volatility target
        if all(col in df_sym.columns for col in ['high', 'low', 'close']):
            future_max_high = df_sym['high'].rolling(window=target_shift, min_periods=1).max().shift(-target_shift + 1)
            future_min_low = df_sym['low'].rolling(window=target_shift, min_periods=1).min().shift(-target_shift + 1)
            df_sym['volatility'] = (future_max_high - future_min_low) / df_sym['close']
        else:
            df_sym['volatility'] = np.nan

        # Classification targets
        df_sym['target_up'] = (df_sym['delta'] > 0).astype(int)  # Simple binary UP/NOT_UP
        df_sym['target_class'] = df_sym['delta'].apply(classify_delta_value)  # 'UP', 'DOWN', or NaN (from helpers)

        # TP-hit target (example)
        df_sym[f'target_tp_hit_{int(tp_hit_threshold * 10000)}bps'] = np.where(df_sym['delta'] > tp_hit_threshold, 1, 0)
        # Renaming to a generic name for easier use in trainer.py
        # The specific threshold used is now part of the column name for clarity if multiple are generated.
        # For the trainer, we'll likely pick one, e.g., 'target_tp_hit'
        df_sym.rename(columns={f'target_tp_hit_{int(tp_hit_threshold * 10000)}bps': 'target_tp_hit'}, inplace=True)

        # --- Compute TA Features ---
        df_sym_ta = _compute_ta_features_for_symbol(df_sym.copy())  # Pass copy to avoid modifying df_sym inplace issues

        # --- Merge with BTC Features (if provided) ---
        if btc_features_df is not None and not btc_features_df.empty:
            # Ensure both DataFrames have timezone-naive DatetimeIndex for merging, or same timezone
            df_sym_ta_index_orig_tz = df_sym_ta.index.tz
            btc_features_index_orig_tz = btc_features_df.index.tz

            df_sym_ta_for_merge = df_sym_ta.copy()
            btc_features_for_merge = btc_features_df.copy()

            if df_sym_ta_for_merge.index.tz is not None:
                df_sym_ta_for_merge.index = df_sym_ta_for_merge.index.tz_localize(None)
            if btc_features_for_merge.index.tz is not None:
                btc_features_for_merge.index = btc_features_for_merge.index.tz_localize(None)

            df_sym_merged = df_sym_ta_for_merge.merge(
                btc_features_for_merge,
                left_index=True,
                right_index=True,
                how='left',
                suffixes=('', '_btc')
            )
            # Forward-fill BTC features to handle potential misalignments or missing BTC data points
            btc_cols_to_ffill = [col for col in df_sym_merged.columns if '_btc' in col]
            if btc_cols_to_ffill:
                df_sym_merged[btc_cols_to_ffill] = df_sym_merged[btc_cols_to_ffill].ffill()

            # Restore original timezone if it existed, or keep tz-naive
            if df_sym_ta_index_orig_tz is not None:
                df_sym_merged.index = df_sym_merged.index.tz_localize(df_sym_ta_index_orig_tz)

            df_sym_final = df_sym_merged
        else:
            df_sym_final = df_sym_ta

        # --- Filter low activity periods (optional) ---
        # This was in original script, keep for consistency
        if 'atr' in df_sym_final.columns and 'volume' in df_sym_final.columns:
            min_periods_rolling_filter = min(20, len(df_sym_final))
            if min_periods_rolling_filter > 0:
                mean_vol = df_sym_final['volume'].rolling(window=20, min_periods=min_periods_rolling_filter).mean()
                mean_atr = df_sym_final['atr'].rolling(window=20, min_periods=min_periods_rolling_filter).mean()

                condition_volume = df_sym_final['volume'] > mean_vol
                condition_atr = df_sym_final['atr'] > mean_atr

                # Fill NaN in conditions with False so they are not dropped by condition
                condition_volume = condition_volume.fillna(False)
                condition_atr = condition_atr.fillna(False)

                df_sym_filtered_activity = df_sym_final[condition_volume & condition_atr]

                if len(df_sym_filtered_activity) < len(df_sym_final):
                    logger.debug(f"Symbol {symbol_val} TF {tf_name}: Filtered out {len(df_sym_final) - len(df_sym_filtered_activity)} rows due to low activity.")
                df_sym_final = df_sym_filtered_activity
            else:
                logger.debug(f"Symbol {symbol_val} TF {tf_name}: Not enough data for low activity filtering.")
        else:
            logger.debug(f"Symbol {symbol_val} TF {tf_name}: 'atr' or 'volume' missing, skipping low activity filter.")

        if not df_sym_final.empty:
            # df_sym_final already has 'symbol' from the original df_all_candles_input structure if index was reset
            # If 'symbol' column was lost due to set_index without keeping the column, re-add it.
            # Since we did df_sym = df_sym_original.set_index('timestamp', drop=False), 'symbol' and 'timestamp' are still columns.
            # After all operations, we reset index to make timestamp a column again for concat.
            all_symbols_processed_features.append(df_sym_final.reset_index(drop=True))  # drop=True if timestamp is already a column
        else:
            logger.info(f"Symbol {symbol_val} on TF {tf_name} resulted in an empty DataFrame after processing. Not added.")

    if not all_symbols_processed_features:
        logger.warning(f"No features computed for any symbol on TF {tf_name}.")
        return pd.DataFrame()

    full_features_df = pd.concat(all_symbols_processed_features, ignore_index=True)

    # --- Final Cleanup ---
    # Drop rows where essential target or key features are NaN
    # These are typically due to shift operations or insufficient history for TA calculation at the start/end of series.
    cols_for_dropna = ['delta', 'volatility', 'target_class', 'rsi', 'target_tp_hit', 'ema_20_slope', 'ema_50_slope']
    # Ensure all columns in cols_for_dropna actually exist in full_features_df before trying to dropna
    existing_cols_for_dropna = [col for col in cols_for_dropna if col in full_features_df.columns]

    len_before_dropna = len(full_features_df)
    if existing_cols_for_dropna:
        full_features_df = full_features_df.dropna(subset=existing_cols_for_dropna)
    len_after_dropna = len(full_features_df)

    logger.info(f"Dropped {len_before_dropna - len_after_dropna} rows due to NaNs in key columns: {existing_cols_for_dropna}.")
    logger.info(f"Feature computation for TF {tf_name} complete. Final DataFrame shape: {full_features_df.shape}.")

    return full_features_df


def _prepare_btc_features(df_all_candles_for_tf, tf_name):
    """Prepares lagged BTC features (RSI, log return, volume) to be merged with other symbols."""
    if 'BTCUSDT' not in df_all_candles_for_tf['symbol'].unique():
        logger.info("BTCUSDT data not present in the current set. No BTC features will be added.")
        return None

    df_btc_raw = df_all_candles_for_tf[df_all_candles_for_tf['symbol'] == 'BTCUSDT'].copy()
    if df_btc_raw.empty:
        logger.info("BTCUSDT DataFrame is empty. No BTC features will be added.")
        return None

    df_btc_raw = df_btc_raw.set_index('timestamp')  # Set index for TA and shift

    if len(df_btc_raw) < 50:  # Min data for reliable TA and lags
        logger.warning(f"Insufficient data for BTCUSDT on TF {tf_name} ({len(df_btc_raw)} rows) to calculate its features.")
        return None

    logger.info(f"Preparing BTC features for TF {tf_name}...")
    # Calculate features for BTC that will be used for other symbols
    if 'close' in df_btc_raw.columns:
        df_btc_raw['log_return_1_btc'] = np.log(df_btc_raw['close'] / df_btc_raw['close'].shift(1))
        df_btc_raw['rsi_btc'] = ta.momentum.RSIIndicator(close=df_btc_raw['close'], window=14, fillna=False).rsi()
    else:
        df_btc_raw['log_return_1_btc'] = np.nan
        df_btc_raw['rsi_btc'] = np.nan

    if 'volume' in df_btc_raw.columns:
        df_btc_raw['volume_btc'] = df_btc_raw['volume']  # Just renaming for clarity if needed
    else:
        df_btc_raw['volume_btc'] = np.nan

    # Select and lag these BTC features (we want BTC's past state to predict other symbol's future)
    btc_features_to_lag = ['log_return_1_btc', 'rsi_btc', 'volume_btc']
    existing_btc_features = [col for col in btc_features_to_lag if col in df_btc_raw.columns]

    if not existing_btc_features:
        logger.warning(f"None of the designated BTC features ({btc_features_to_lag}) could be calculated for TF {tf_name}.")
        return None

    btc_lagged_features = df_btc_raw[existing_btc_features].shift(1)  # Lag by 1 period
    btc_lagged_features.dropna(inplace=True)  # Drop NaNs created by lagging

    logger.info(f"BTC features prepared for TF {tf_name}. Shape: {btc_lagged_features.shape}")
    return btc_lagged_features


def main_preprocess_logic(timeframe, symbol_filter=None, symbol_group_filter=None):
    """
    Main logic for preprocessing features for a given timeframe and optional filters.

    Args:
        timeframe (str): Timeframe to process (e.g., '15m').
        symbol_filter (str, optional): A single symbol to filter data for.
        symbol_group_filter (str, optional): Name of a symbol group to filter data for.
    """
    global CONFIG
    CONFIG = load_config()
    if not CONFIG:
        logger.critical("Failed to load configuration. Preprocessing cannot proceed.")
        return 1

    # Setup logging for this run (if not already configured by a higher level)
    # Basic config if no handlers are present on the root logger or this logger
    if not logging.getLogger().hasHandlers() and not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    db_path_cfg = CONFIG.get('db_path', DEFAULT_DB_PATH)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db_path = os.path.join(project_root, db_path_cfg) if not os.path.isabs(db_path_cfg) else db_path_cfg

    output_dir_cfg = CONFIG.get('data_dir', DEFAULT_OUTPUT_DIR)  # 'data' dir from config
    output_dir_path = os.path.join(project_root, output_dir_cfg) if not os.path.isabs(output_dir_cfg) else output_dir_cfg
    os.makedirs(output_dir_path, exist_ok=True)

    symbol_groups_cfg = CONFIG.get('symbol_groups', DEFAULT_SYMBOL_GROUPS)

    # Determine the actual list of symbols to process and the suffix for output files
    symbols_to_process_list = None
    file_suffix = "all"  # Default suffix if no filters
    log_processing_details = " (all symbols)"

    if symbol_group_filter:
        group_key = symbol_group_filter.lower()
        if group_key in symbol_groups_cfg:
            symbols_to_process_list = symbol_groups_cfg[group_key]
            file_suffix = group_key
            log_processing_details = f" (group: {group_key})"
            if not symbols_to_process_list:
                logger.warning(f"Symbol group '{group_key}' is empty in config. Processing all symbols.")
                symbols_to_process_list = None  # Fallback to all
                file_suffix = "all"
                log_processing_details = " (all symbols, as group was empty)"
        else:
            logger.error(f"Unknown symbol group: {symbol_group_filter}. Processing all symbols.")
            # Fallback to all symbols
            file_suffix = "all"
            log_processing_details = " (all symbols, as group was unknown)"
    elif symbol_filter:
        symbols_to_process_list = [symbol_filter.upper()]
        file_suffix = symbol_filter.upper()
        log_processing_details = f" (symbol: {symbol_filter.upper()})"

    logger.info(f"⚙️ Starting feature preprocessing for TF: {timeframe}{log_processing_details}")
    logger.info(f"Output file suffix will be: '{file_suffix}'")

    # 1. Load all candle data for the timeframe
    df_all_tf_candles = load_candles_from_sqlite(db_path, timeframe)
    if df_all_tf_candles.empty:
        logger.error(f"No candle data loaded from DB for timeframe {timeframe}. Preprocessing aborted.")
        return 1

    # 2. Filter by symbol list if provided
    if symbols_to_process_list:
        logger.info(f"Filtering candle data for symbols: {symbols_to_process_list}")
        df_all_tf_candles = df_all_tf_candles[df_all_tf_candles['symbol'].isin(symbols_to_process_list)]
        if df_all_tf_candles.empty:
            logger.error(f"No data remains after filtering for symbols {symbols_to_process_list} on TF {timeframe}. Aborting.")
            return 1
        logger.info(f"Data filtered. {len(df_all_tf_candles)} rows remain for {df_all_tf_candles['symbol'].nunique()} symbols.")

    # 3. Prepare BTC features (if BTC is in the data to be processed or generally available)
    # Pass the (potentially filtered) df_all_tf_candles to _prepare_btc_features
    btc_features = _prepare_btc_features(df_all_tf_candles, timeframe)

    # 4. Compute features for the selected symbols
    df_final_features = compute_and_prepare_features(df_all_tf_candles, timeframe, btc_features)

    if df_final_features.empty:
        logger.error(f"Feature computation resulted in an empty DataFrame for TF {timeframe}{log_processing_details}. No output files generated.")
        return 1

    # 5. Save processed features
    output_pickle_filename = f"features_{file_suffix}_{timeframe}.pkl"
    output_pickle_path = os.path.join(output_dir_path, output_pickle_filename)
    output_sample_csv_filename = f"sample_{file_suffix}_{timeframe}.csv"
    output_sample_csv_path = os.path.join(output_dir_path, output_sample_csv_filename)

    try:
        df_final_features.to_pickle(output_pickle_path)
        logger.info(f"💾 Processed features saved to: {output_pickle_path} (Shape: {df_final_features.shape})")
    except Exception as e:
        logger.error(f"Error saving features pickle file to {output_pickle_path}: {e}")
        return 1  # Indicate failure

    try:
        sample_size = min(1000, len(df_final_features))
        if sample_size > 0:
            df_sample_to_save = df_final_features.head(sample_size).copy()
            # Ensure timestamp is string for CSV if it's datetime
            if 'timestamp' in df_sample_to_save.columns and pd.api.types.is_datetime64_any_dtype(df_sample_to_save['timestamp']):
                df_sample_to_save['timestamp'] = df_sample_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_sample_to_save.to_csv(output_sample_csv_path, index=False)
            logger.info(f"📄 Sample data saved to: {output_sample_csv_path} ({sample_size} rows)")
    except Exception as e:
        logger.error(f"Error saving sample CSV file to {output_sample_csv_path}: {e}")
        # Continue even if sample saving fails, main pickle is more important

    logger.info(f"✅ Feature preprocessing for TF: {timeframe}{log_processing_details} completed.")
    return 0  # Success


if __name__ == '__main__':
    # This block allows running the preprocessor directly.
    # It mimics the argument parsing of the original preprocess_features.py

    # Load config to get defaults for symbol groups if not passed via CLI
    # For direct run, basicConfig is fine. If imported, it uses existing config.
    if not logging.getLogger().hasHandlers() and not logger.hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    parser = argparse.ArgumentParser(description="Preprocess candle data from SQLite to generate features.")
    parser.add_argument('--tf', type=str, required=True,
                        help='Timeframe to process (e.g., 5m, 15m).')
    parser.add_argument('--symbol', type=str, default=None,
                        help="A single symbol to process (e.g., BTCUSDT). Overrides --symbol-group.")
    parser.add_argument('--symbol-group', type=str, default=None,
                        help="Name of a symbol group to process (e.g., top8, meme). Defined in config or defaults.")

    args = parser.parse_args()

    exit_code = 1  # Default to error
    try:
        # If --symbol is given, it takes precedence over --symbol-group for filtering.
        # If neither is given, processes all symbols found in the DB for that timeframe.
        exit_code = main_preprocess_logic(
            timeframe=args.tf,
            symbol_filter=args.symbol,
            symbol_group_filter=args.symbol_group
        )
    except KeyboardInterrupt:
        logger.warning(f"\n[Preprocessor] 🛑 Preprocessing for TF {args.tf} interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"[Preprocessor] 💥 Unexpected critical error during preprocessing for TF {args.tf}: {e}", exc_info=True)
        sys.exit(1)

    sys.exit(exit_code)
