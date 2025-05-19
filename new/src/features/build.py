# src/features/build.py

import pandas as pd
import numpy as np
import ta # Assuming ta-lib or python-ta is installed
import logging

# Import config and logging setup (config needs to be loaded to get paths)
from src.utils.config import get_config

# Get logger for this module
logger = logging.getLogger(__name__)

# Load configuration
config = get_config()
FEATURE_BUILDER_CONFIG = config['feature_builder']

# --- Feature Computation Functions (Moved from scripts/preprocess.py) ---

def classify_binary_target(delta_val):
    """Classifies delta into binary 'UP'/'DOWN'/'NaN' based on thresholds."""
    if pd.isna(delta_val): return np.nan
    # Use config thresholds for classification
    up_threshold = FEATURE_BUILDER_CONFIG.get('target_up_threshold', 0.002) # Default if not in config
    down_threshold = FEATURE_BUILDER_CONFIG.get('target_down_threshold', -0.002) # Default if not in config

    if delta_val > up_threshold:
        return 'UP'
    elif delta_val < down_threshold:
        return 'DOWN'
    else:
        return np.nan # Neutral or small moves are NaN for target_class


def compute_technical_indicators(df_input_group):
    """Computes various technical indicators using the 'ta' library."""
    df_group = df_input_group.copy() # Work on a copy

    # --- Get TA Parameters from Config ---
    rsi_window = FEATURE_BUILDER_CONFIG.get('rsi_window', 14)
    ema_fast_window = FEATURE_BUILDER_CONFIG.get('ema_fast_window', 20)
    ema_slow_window = FEATURE_BUILDER_CONFIG.get('ema_slow_window', 50)
    macd_slow_window = FEATURE_BUILDER_CONFIG.get('macd_slow_window', 26)
    macd_fast_window = FEATURE_BUILDER_CONFIG.get('macd_fast_window', 12)
    macd_sign_window = FEATURE_BUILDER_CONFIG.get('macd_sign_window', 9)
    atr_window = FEATURE_BUILDER_CONFIG.get('atr_window', 14)
    bollinger_window = FEATURE_BUILDER_CONFIG.get('bollinger_window', 20)
    bollinger_dev = FEATURE_BUILDER_CONFIG.get('bollinger_dev', 2)


    try:
        # RSI
        if len(df_group) >= rsi_window:
             df_group['rsi'] = ta.momentum.RSIIndicator(close=df_group['close'], window=rsi_window).rsi()
        else: df_group['rsi'] = np.nan

        # EMA
        if len(df_group) >= ema_fast_window:
             df_group['ema_20'] = ta.trend.EMAIndicator(close=df_group['close'], window=ema_fast_window).ema_indicator()
        else: df_group['ema_20'] = np.nan
        if len(df_group) >= ema_slow_window:
             df_group['ema_50'] = ta.trend.EMAIndicator(close=df_group['close'], window=ema_slow_window).ema_indicator()
        else: df_group['ema_50'] = np.nan

        # MACD
        if len(df_group) >= macd_slow_window:
            macd_indicator = ta.trend.MACD(close=df_group['close'],
                                           window_slow=macd_slow_window,
                                           window_fast=macd_fast_window,
                                           window_sign=macd_sign_window)
            df_group['macd'] = macd_indicator.macd()
            df_group['macd_signal'] = macd_indicator.macd_signal()
            df_group['macd_diff'] = macd_indicator.macd_diff()
        else:
            df_group['macd'] = np.nan
            df_group['macd_signal'] = np.nan
            df_group['macd_diff'] = np.nan


        # OBV (On-Balance Volume)
        if len(df_group) >= 1: # OBV requires at least one candle
             df_group['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df_group['close'], volume=df_group['volume']).on_balance_volume()
        else: df_group['obv'] = np.nan

        # ATR (Average True Range)
        if len(df_group) >= atr_window:
            df_group['atr'] = ta.volatility.AverageTrueRange(high=df_group['high'], low=df_group['low'],
                                               close=df_group['close'], window=atr_window).average_true_range()
        else: df_group['atr'] = np.nan

        # Bollinger Bands Width
        if len(df_group) >= bollinger_window:
             try:
                 bb = ta.volatility.BollingerBands(close=df_group['close'], window=bollinger_window, window_dev=bollinger_dev)
                 df_group['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
             except Exception as e:
                 logger.warning(f"({df_group['symbol'].iloc[0] if not df_group.empty else 'N/A'}): Failed to compute Bollinger Bands ({e}). Setting bb_width to NaN.")
                 df_group['bb_width'] = np.nan
        else:
             df_group['bb_width'] = np.nan


    except ImportError:
        logger.error("TA-Lib or TA library not installed. Skipping TA features.")
        # Add NaN columns for all TA features if library is missing
        ta_cols = ['rsi', 'ema_20', 'ema_50', 'macd', 'macd_signal', 'macd_diff', 'obv', 'atr', 'bb_width']
        for col in ta_cols:
            if col not in df_group.columns:
                 df_group[col] = np.nan
    except Exception as e:
        logger.error(f"Error computing TA features ({df_group['symbol'].iloc[0] if not df_group.empty else 'N/A'}): {e}", exc_info=True)
        # Set TA features to NaN on error
        ta_cols = ['rsi', 'ema_20', 'ema_50', 'macd', 'macd_signal', 'macd_diff', 'obv', 'atr', 'bb_width']
        for col in ta_cols:
            if col not in df_group.columns:
                 df_group[col] = np.nan

    return df_group


def compute_custom_features(df_input_group):
    """Computes various custom features."""
    df_group = df_input_group.copy() # Work on a copy

    # --- Get Custom Feature Parameters from Config ---
    rolling_windows = FEATURE_BUILDER_CONFIG.get('rolling_windows', [5, 10, 20])
    volume_z_window = FEATURE_BUILDER_CONFIG.get('volume_z_window', 20)


    # Volume Z-score and clipping
    if 'volume' in df_group.columns and len(df_group) >= volume_z_window:
        # Replace 0 with NaN in std for volume_z calculation
        volume_std = df_group['volume'].rolling(window=volume_z_window, min_periods=1).std()
        df_group['volume_z'] = (df_group['volume'] - df_group['volume'].rolling(window=volume_z_window, min_periods=1).mean()) / volume_std.replace(0, np.nan)
        # Clip outliers - clip limits are hardcoded, could be in config
        df_group['volume_z'] = df_group['volume_z'].clip(-5, 5)
    else:
         df_group['volume_z'] = np.nan


    # Candle body/range features
    if all(col in df_group.columns for col in ['close', 'open', 'high', 'low']):
         df_group['candle_body_size'] = abs(df_group['close'] - df_group['open'])
         df_group['candle_hl_range'] = df_group['high'] - df_group['low']
         # Avoid division by zero or near-zero
         df_group['is_doji'] = ((df_group['candle_body_size'] / (df_group['candle_hl_range'] + 1e-9)) < 0.1).astype(int)

         # Pin bar features - check if 'candle_body_size' is already computed
         if 'candle_body_size' in df_group.columns:
              wick_up = df_group['high'] - df_group[['open', 'close']].max(axis=1)
              wick_down = df_group[['open', 'close']].min(axis=1) - df_group['low']
              body = df_group['candle_body_size']
              # Avoid division by zero or near-zero body size when checking for pin bar
              df_group['pin_bar'] = (((wick_up / (body + 1e-9)) > 2) | ((wick_down / (body + 1e-9)) > 2)).astype(int)
         else:
              df_group['pin_bar'] = 0 # Default to 0 if candle_body_size is missing
              logger.warning(f"({df_group['symbol'].iloc[0] if not df_group.empty else 'N/A'}): Cannot compute pin_bar, candle_body_size is missing.")
    else:
         df_group['candle_body_size'] = np.nan
         df_group['candle_hl_range'] = np.nan
         df_group['is_doji'] = 0 # Default to 0 if cols missing
         df_group['pin_bar'] = 0 # Default to 0 if cols missing


    # Time-based features (ensure index is datetime)
    if isinstance(df_group.index, pd.DatetimeIndex):
        df_group['hour'] = df_group.index.hour
        df_group['minute'] = df_group.index.minute
        df_group['dayofweek'] = df_group.index.dayofweek
        df_group['dayofmonth'] = df_group.index.day
        try:
             df_group['weekofyear'] = df_group.index.isocalendar().week.astype(int)
        except Exception: # Handle potential errors with isocalendar
             df_group['weekofyear'] = np.nan # Set to NaN if calculation fails
             logger.warning(f"({df_group['symbol'].iloc[0] if not df_group.empty else 'N/A'}): Could not compute weekofyear.")

        # Cyclical time features
        if pd.notna(df_group['hour']).any():
             df_group['hour_sin'] = np.sin(2 * np.pi * df_group['hour'].astype(float) / 24)
        else: df_group['hour_sin'] = np.nan
        if pd.notna(df_group['dayofweek']).any():
             df_group['dayofweek_sin'] = np.sin(2 * np.pi * df_group['dayofweek'].astype(float) / 7)
        else: df_group['dayofweek_sin'] = np.nan
    else:
        logger.warning(f"({df_group['symbol'].iloc[0] if not df_group.empty else 'N/A'}): Index is not DatetimeIndex. Skipping time-based features.")
        for col in ['hour', 'minute', 'dayofweek', 'dayofmonth', 'weekofyear', 'hour_sin', 'dayofweek_sin']:
            if col not in df_group.columns:
                 df_group[col] = np.nan


    # EMA difference and slope (requires EMA columns from technical indicators)
    if 'ema_20' in df_group.columns and 'ema_50' in df_group.columns:
        df_group['ema_diff'] = df_group['ema_20'] - df_group['ema_50']
        # Slopes
        df_group['ema_20_slope'] = df_group['ema_20'].diff()
        df_group['ema_50_slope'] = df_group['ema_50'].diff()
    else:
        df_group['ema_diff'] = np.nan
        df_group['ema_20_slope'] = np.nan
        df_group['ema_50_slope'] = np.nan
        if not df_group.empty:
             logger.warning(f"({df_group['symbol'].iloc[0]}): EMA columns missing for ema_diff/slope calculation.")


    # RSI change over 3 periods (requires rsi column)
    if 'rsi' in df_group.columns:
        df_group['rsi_change_3'] = df_group['rsi'].diff(3)
    else:
        df_group['rsi_change_3'] = np.nan
        if not df_group.empty:
             logger.warning(f"({df_group['symbol'].iloc[0]}): RSI column missing for rsi_change_3 calculation.")


    # Volume rolling stats and spike
    if 'volume' in df_group.columns:
        for window in rolling_windows:
             if len(df_group) >= window:
                 df_group[f'volume_rolling_mean_{window}'] = df_group['volume'].rolling(window).mean()
                 df_group[f'volume_rolling_std_{window}'] = df_group['volume'].rolling(window).std()
             else:
                 df_group[f'volume_rolling_mean_{window}'] = np.nan
                 df_group[f'volume_rolling_std_{window}'] = np.nan

        # Volume spike based on the largest window
        largest_window = max(rolling_windows) if rolling_windows else 20
        volume_mean_col = f'volume_rolling_mean_{largest_window}'
        volume_std_col = f'volume_rolling_std_{largest_window}'

        if volume_mean_col in df_group.columns and volume_std_col in df_group.columns:
            # Avoid division by zero in the spike calculation condition
            volume_std_safe = df_group[volume_std_col].replace(0, np.nan)
            df_group['volume_spike'] = ((df_group['volume'] - df_group[volume_mean_col]) > 2 * volume_std_safe).fillna(False).astype(int)
        else:
            df_group['volume_spike'] = 0 # Default to 0 if rolling stats not computed
            if not df_group.empty:
                 logger.warning(f"({df_group['symbol'].iloc[0]}): Volume rolling stats missing for volume_spike.")


    else: # If 'volume' column is missing entirely
        for window in rolling_windows:
             df_group[f'volume_rolling_mean_{window}'] = np.nan
             df_group[f'volume_rolling_std_{window}'] = np.nan
        df_group['volume_spike'] = 0


    # Returns rolling std (volatility based on price changes)
    if 'log_return_1' in df_group.columns:
        for window in rolling_windows:
             if len(df_group) >= window:
                df_group[f'returns_rolling_std_{window}'] = df_group['log_return_1'].rolling(window).std()
             else:
                df_group[f'returns_rolling_std_{window}'] = np.nan
    else:
        for window in rolling_windows:
             df_group[f'returns_rolling_std_{window}'] = np.nan
        if not df_group.empty:
             logger.warning(f"({df_group['symbol'].iloc[0]}): log_return_1 missing for returns_rolling_std.")


    # Length of consecutive trend up/down
    if 'close' in df_group.columns and len(df_group) > 1:
        df_group['trend_up_flag'] = (df_group['close'] > df_group['close'].shift(1)).astype(int)
        df_group['trend_down_flag'] = (df_group['close'] < df_group['close'].shift(1)).astype(int)

        # Compute consecutive counts
        df_group['consecutive_up'] = df_group['trend_up_flag'] * (df_group['trend_up_flag'].groupby((df_group['trend_up_flag'] != df_group['trend_up_flag'].shift()).cumsum()).cumcount() + 1)
        df_group['consecutive_down'] = df_group['trend_down_flag'] * (df_group['trend_down_flag'].groupby((df_group['trend_down_flag'] != df_group['trend_down_flag'].shift()).cumsum()).cumcount() + 1)

        df_group.drop(['trend_up_flag', 'trend_down_flag'], axis=1, inplace=True)
    else:
        df_group['consecutive_up'] = 0
        df_group['consecutive_down'] = 0
        if not df_group.empty:
             logger.warning(f"({df_group['symbol'].iloc[0]}): Close column missing or not enough data for consecutive trend.")


    return df_group


def compute_btc_features(df_btc_raw):
    """Computes relevant features from BTCUSDT data, shifted by 1 period."""
    if df_btc_raw.empty:
        logger.warning("BTCUSDT DataFrame is empty, cannot compute BTC features.")
        return pd.DataFrame()

    # Ensure index is datetime and sorted
    df_btc_raw = df_btc_raw.set_index('timestamp').sort_index().copy()

    # Get minimum data requirement from config or use a default
    min_data_for_btc_features = FEATURE_BUILDER_CONFIG.get('min_data_for_btc_features', 50) # Default if not in config

    if len(df_btc_raw) < min_data_for_btc_features:
        logger.warning(f"Not enough data for BTCUSDT ({len(df_btc_raw)} < {min_data_for_btc_features}) to compute its features.")
        return pd.DataFrame()

    # Compute features for BTC
    if 'close' in df_btc_raw.columns:
         df_btc_raw['log_return_1_btc'] = np.log(df_btc_raw['close'] / df_btc_raw['close'].shift(1))
    else:
         df_btc_raw['log_return_1_btc'] = np.nan

    # Use a consistent window for RSI, e.g., from FEATURE_BUILDER_CONFIG
    rsi_window = FEATURE_BUILDER_CONFIG.get('rsi_window', 14)
    try:
        import ta
        if 'close' in df_btc_raw.columns and len(df_btc_raw) >= rsi_window:
            df_btc_raw['rsi_btc'] = ta.momentum.RSIIndicator(close=df_btc_raw['close'], window=rsi_window).rsi()
        else: df_btc_raw['rsi_btc'] = np.nan
    except ImportError:
        logger.warning("TA-Lib or TA library not installed. Skipping BTC RSI feature.")
        df_btc_raw['rsi_btc'] = np.nan
    except Exception as e:
        logger.warning(f"Error computing BTC RSI: {e}. Setting rsi_btc to NaN.")
        df_btc_raw['rsi_btc'] = np.nan

    if 'volume' in df_btc_raw.columns:
         df_btc_raw['volume_btc'] = df_btc_raw['volume'] # Keep raw volume
    else:
         df_btc_raw['volume_btc'] = np.nan


    # Collect all computed BTC features
    btc_feature_cols = ['log_return_1_btc', 'rsi_btc', 'volume_btc']
    # Filter out columns that might not have been computed due to errors/missing data
    btc_features_df = df_btc_raw[[col for col in btc_feature_cols if col in df_btc_raw.columns]]

    # Shift BTC features by 1 to avoid look-ahead bias
    btc_features_prepared = btc_features_df.shift(1)

    # Drop rows where shifted features are NaN
    btc_features_prepared.dropna(inplace=True)

    logger.info(f"BTCUSDT features computed and shifted. Rows: {len(btc_features_prepared)}")
    return btc_features_prepared


def apply_flat_filter(df_input):
    """
    Applies a filter to remove periods of low volume and ATR (flat market).
    Requires 'atr' and 'volume' columns.
    Based on the logic from the old script.
    """
    df_filtered = df_input.copy() # Work on a copy

    if 'atr' not in df_filtered.columns or 'volume' not in df_filtered.columns:
        logger.warning("Columns 'atr' or 'volume' missing for flat filter. Skipping filter.")
        return df_filtered

    # Get filter window from config or use a default
    filter_window = FEATURE_BUILDER_CONFIG.get('flat_filter_window', 20) # Default if not in config

    if len(df_filtered) < filter_window:
        logger.debug(f"Not enough data ({len(df_filtered)} < {filter_window}) for flat filter rolling stats. Skipping filter.")
        return df_filtered

    try:
        min_periods_rolling = min(filter_window, len(df_filtered))
        if min_periods_rolling > 0:
            condition_volume = df_filtered['volume'] > df_filtered['volume'].rolling(window=filter_window, min_periods=min_periods_rolling).mean()
            condition_atr = df_filtered['atr'] > df_filtered['atr'].rolling(window=filter_window, min_periods=min_periods_rolling).mean()
            # Fill NaNs from rolling calculation (at the start of the series) with False, so these initial rows are filtered out
            condition_volume = condition_volume.fillna(False)
            condition_atr = condition_atr.fillna(False)

            # Apply the filter condition
            df_filtered = df_filtered[condition_volume & condition_atr].copy() # Use .copy() after filtering

            logger.debug(f"Applied flat filter (volume > rolling_mean, atr > rolling_mean). Remaining rows: {len(df_filtered)}")
        else:
             logger.warning("Min_periods for flat filter is 0. Skipping filter application.")

    except Exception as e:
        logger.error(f"Error applying flat filter: {e}", exc_info=True)
        # In case of error, return the original DataFrame to not block the process entirely
        # logger.info("Returning original DataFrame due to flat filter error.") # Maybe log this
        return df_input # Return input df on error

    return df_filtered


def compute_all_features_for_symbol(df_sym_input, tf_name, btc_features_df):
    """
    Computes all features for a single symbol.
    This function orchestrates the feature calculation steps.
    """
    df_sym = df_sym_input.copy() # Work on a copy
    symbol_val = df_sym['symbol'].iloc[0] if not df_sym.empty else "N/A"

    logger.debug(f"Computing features for {symbol_val} on {tf_name}...")

    # Ensure sorted by timestamp and set as index before TA/custom features
    df_sym = df_sym.sort_values('timestamp').set_index('timestamp')

    # Get minimum data requirement from config or use a default
    min_data_for_features = FEATURE_BUILDER_CONFIG.get('min_data_for_features', 100) # Default if not in config

    if len(df_sym) < min_data_for_features:
        logger.debug(
            f"Sufficient data not available for {symbol_val} on {tf_name} ({len(df_sym)} < {min_data_for_features}). Skipping feature computation for this symbol.")
        return pd.DataFrame() # Return empty DataFrame

    # Compute Technical Indicators
    df_sym_features = compute_technical_indicators(df_sym)

    # Compute Custom Features (may rely on TA features)
    df_sym_features = compute_custom_features(df_sym_features) # Pass the result from TA features

    # Merge BTC features if available
    if btc_features_df is not None and not btc_features_df.empty:
        # Ensure both indexes are DatetimeIndex and timezone-naive for merging
        if isinstance(df_sym_features.index, pd.DatetimeIndex) and isinstance(btc_features_df.index, pd.DatetimeIndex):
             if df_sym_features.index.tz is not None:
                 df_sym_features.index = df_sym_features.index.tz_localize(None)
             if btc_features_df.index.tz is not None:
                 btc_features_df_local = btc_features_df.copy() # Avoid modifying original BTC df if it came from somewhere else
                 btc_features_df_local.index = btc_features_df_local.index.tz_localize(None)
             else:
                 btc_features_df_local = btc_features_df # No tz conversion needed
        else:
             logger.warning(f"({symbol_val}): Index is not DatetimeIndex for merging BTC features. Skipping BTC merge.")
             btc_features_df_local = pd.DataFrame() # Cannot merge if indexes aren't DatetimeIndex

        if not btc_features_df_local.empty:
            original_cols = set(df_sym_features.columns)
            # Perform the merge
            df_sym_features = df_sym_features.merge(btc_features_df_local, left_index=True, right_index=True, how='left', suffixes=('', '_btc'))

            # Forward fill BTC features after merge, as they might be sparse if TF is different
            btc_cols_added = [col for col in df_sym_features.columns if col not in original_cols]
            if btc_cols_added:
                df_sym_features[btc_cols_added] = df_sym_features[btc_cols_added].ffill()
                logger.debug(f"({symbol_val}): Merged and ffilled {len(btc_cols_added)} BTC features.")
            else:
                 logger.debug(f"({symbol_val}): No BTC features columns added during merge.")
        else:
            logger.debug(f"({symbol_val}): BTC features DataFrame is empty after tz handling. Skipping BTC merge.")


    # Drop rows where target variables are NaN - these cannot be used for training
    # This dropna is usually applied *after* computing all features but *before* the final filter/dropna
    # on key features for training/prediction.
    # Let's apply it here to remove rows that definitively lack a target.
    # Note: The original script applied dropna *after* the flat filter.
    # Let's stick to the original script's order for now for consistency,
    # applying the main dropna subset later in the script.
    # However, if the target depends on shifted future data, those rows will naturally be NaN
    # at the end of the series, and dropna will handle them.

    logger.debug(f"Finished computing features for {symbol_val} on {tf_name}.")
    return df_sym_features # Return DataFrame with timestamp as index


# Example usage (optional, for testing the module directly)
if __name__ == "__main__":
    # Set up logging for the test
    from src.utils.logging_setup import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__) # Re-get logger after setup

    print("Testing src/features/build.py")

    # Create a dummy DataFrame for testing
    data = {
        'symbol': ['TESTUSDT'] * 1000 + ['ANOTHERUSDT'] * 1000 + ['BTCUSDT'] * 1000,
        'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=1000, freq='min').tolist() * 3),
        'open': np.random.rand(3000) * 100 + 1000,
        'high': lambda x: x['open'] + np.random.rand(len(x)) * 10,
        'low': lambda x: x['open'] - np.random.rand(len(x)) * 10,
        'close': lambda x: x['open'] + np.random.randn(len(x)) * 5,
        'volume': np.random.rand(3000) * 10000
    }
    # Make close prices trend slightly for target features
    data['close'] = pd.Series(data['close']).add(np.linspace(0, 50, 3000)).tolist()
    df_test = pd.DataFrame(data)
    df_test['high'] = df_test[['open', 'close']].max(axis=1) + np.random.rand(3000) * 5
    df_test['low'] = df_test[['open', 'close']].min(axis=1) - np.random.rand(3000) * 5


    # Prepare BTC features from the test data slice
    df_btc_raw_test = df_test[df_test['symbol'] == 'BTCUSDT'].copy()
    btc_features_test = compute_btc_features(df_btc_raw_test)
    print(f"Computed BTC features shape: {btc_features_test.shape}")
    if not btc_features_test.empty:
        print("BTC features head:")
        print(btc_features_test.head().to_string())


    # Compute features for a non-BTC symbol
    df_sym_test = df_test[df_test['symbol'] == 'TESTUSDT'].copy()
    print("\nComputing features for TESTUSDT...")
    df_features_test = compute_all_features_for_symbol(df_sym_test, '1min', btc_features_test)
    print(f"Computed features shape for TESTUSDT: {df_features_test.shape}")
    if not df_features_test.empty:
        print("TESTUSDT features head:")
        # Convert index back to column for display if needed, or keep as index
        print(df_features_test.reset_index().head().to_string())
        print("\nCheck for key columns:")
        print(df_features_test[['delta', 'volatility', 'target_class', 'target_tp_hit', 'rsi', 'ema_20', 'bb_width', 'volume_z', 'consecutive_up', 'log_return_1_btc', 'rsi_btc']].head().to_string())


    # Test flat filter
    print("\nTesting flat filter on TESTUSDT features...")
    len_before_filter = len(df_features_test)
    df_filtered_test = apply_flat_filter(df_features_test.reset_index()) # Pass df with timestamp as column
    len_after_filter = len(df_filtered_test)
    print(f"Original rows: {len_before_filter}, Rows after flat filter: {len_after_filter}")
    if not df_filtered_test.empty:
         print("Filtered features head:")
         print(df_filtered_test.head().to_string())

    print("\nFeature building module test finished.")