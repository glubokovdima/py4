# core/features/builder.py
"""
Builds features from raw CSV data sources.
This module is intended for scenarios where features are generated
directly from CSV files (e.g., from 'data_binance/*/{timeframe}.csv')
and saved as a pickle file.

This is distinct from preprocessor.py which works from the SQLite database.
"""
import pandas as pd
import os
import argparse
import logging
from glob import glob  # For finding files matching a pattern

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Configuration Constants (Defaults, can be overridden by config or args) ---
DEFAULT_INPUT_CSV_DIR_TEMPLATE = "data_binance/{symbol}"  # Example, assuming symbol-specific folders
DEFAULT_OUTPUT_DIR = "data"
DEFAULT_OUTPUT_FILENAME_TEMPLATE = "features_built_{timeframe}.pkl"


def _load_and_combine_symbol_timeframe_csvs(input_dir_pattern, timeframe):
    """
    Loads and combines CSV files for a given timeframe from a directory pattern.
    Example pattern: 'data_binance/*/{timeframe}.csv'
    """
    # Construct the specific file pattern for the given timeframe
    # Example: data_binance/*/15m.csv
    file_pattern = input_dir_pattern.format(timeframe=timeframe)

    csv_files = glob(file_pattern)

    if not csv_files:
        logger.warning(f"No CSV files found for pattern: {file_pattern}")
        return pd.DataFrame()

    all_dataframes = []
    logger.info(f"Found {len(csv_files)} CSV files for timeframe {timeframe} using pattern '{file_pattern}'.")

    for file_path in csv_files:
        try:
            # Extract symbol name from path, e.g., from 'data_binance/BTCUSDT/15m.csv'
            # This logic might need adjustment based on your actual directory structure
            parts = file_path.split(os.sep)
            symbol = "UNKNOWN_SYMBOL"
            if len(parts) >= 3 and parts[-1] == f"{timeframe}.csv":  # Expects .../SYMBOL/TF.csv
                symbol = parts[-2]

            df_symbol = pd.read_csv(file_path)
            # --- TODO: Standardize columns if necessary ---
            # e.g., df_symbol.rename(columns={'old_ts_col': 'timestamp', ...}, inplace=True)
            # Ensure 'timestamp' column exists and is in datetime format
            # df_symbol['timestamp'] = pd.to_datetime(df_symbol['timestamp'], unit='ms') # Example

            df_symbol['symbol'] = symbol  # Add symbol column
            all_dataframes.append(df_symbol)
            logger.debug(f"Loaded {len(df_symbol)} rows from {file_path} for symbol {symbol}")
        except Exception as e:
            logger.error(f"Error loading or processing CSV file {file_path}: {e}")
            continue

    if not all_dataframes:
        logger.warning(f"No dataframes were successfully loaded for timeframe {timeframe}.")
        return pd.DataFrame()

    combined_df = pd.concat(all_dataframes, ignore_index=True)
    logger.info(f"Combined data for timeframe {timeframe}. Total rows: {len(combined_df)}, Symbols: {combined_df['symbol'].nunique()}")
    return combined_df


def _calculate_features_from_raw(df_raw):
    """
    Calculates features from the raw combined DataFrame.
    This is where the core feature engineering logic for CSV data would go.

    Args:
        df_raw (pd.DataFrame): Combined dataframe from CSVs, expected to have
                               columns like 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol'.

    Returns:
        pd.DataFrame: DataFrame with calculated features.
    """
    if df_raw.empty:
        return pd.DataFrame()

    logger.info(f"Starting feature calculation for {df_raw['symbol'].nunique()} symbols from raw CSV data.")

    # --- TODO: Implement actual feature engineering logic here ---
    # This could involve:
    # 1. Sorting data by symbol and timestamp
    #       df_raw = df_raw.sort_values(['symbol', 'timestamp'])
    # 2. Calculating technical indicators (RSI, MACD, Bollinger Bands, etc.) per symbol
    #       Example (conceptual, requires proper implementation for groupby):
    #       def calculate_symbol_features(df_group):
    #           df_group['rsi'] = ta.momentum.RSIIndicator(close=df_group['close'], window=14).rsi()
    #           # ... more features
    #           return df_group
    #       df_features = df_raw.groupby('symbol').apply(calculate_symbol_features)
    # 3. Creating target variables (e.g., 'target_up', 'delta', 'volatility')
    #       This would require shifting future data, similar to preprocessor.py
    # 4. Handling NaNs
    # 5. Selecting final features

    # For now, as a placeholder, let's just return a copy with a dummy feature
    df_features = df_raw.copy()
    df_features['dummy_feature_builder'] = 1

    logger.info("Placeholder feature calculation complete.")
    return df_features


def build_features_from_csv(
        timeframe,
        input_csv_pattern="data_binance/*/{timeframe}.csv",  # Pattern to find CSVs
        output_dir=DEFAULT_OUTPUT_DIR,
        output_filename_template=DEFAULT_OUTPUT_FILENAME_TEMPLATE
):
    """
    Main logic to build features from CSV files for a given timeframe.
    - Reads CSV files based on the input_csv_pattern (e.g., 'data_binance/*/{timeframe}.csv').
    - Combines data if multiple symbol CSVs are found.
    - Calculates features.
    - Saves the resulting features to a pickle file.

    Args:
        timeframe (str): The timeframe to process (e.g., '15m').
        input_csv_pattern (str): Glob pattern to find input CSV files.
                                 The '{timeframe}' placeholder will be replaced.
                                 The '*' placeholder can be used for symbol directories.
        output_dir (str): Directory to save the output pickle file.
        output_filename_template (str): Filename template for the output.
                                        '{timeframe}' will be replaced.
    """
    logger.info(f"Starting feature building process for timeframe: {timeframe} from CSVs.")

    # 1. Load and combine raw data from CSVs
    # The pattern 'data_binance/*/{timeframe}.csv' implies that for each symbol,
    # there's a CSV named like '15m.csv' inside a folder named after the symbol.
    # e.g., data_binance/BTCUSDT/15m.csv, data_binance/ETHUSDT/15m.csv
    df_raw_combined = _load_and_combine_symbol_timeframe_csvs(input_csv_pattern, timeframe)

    if df_raw_combined.empty:
        logger.warning(f"No raw data loaded for timeframe {timeframe}. Aborting feature building.")
        return False

    # 2. Calculate features
    df_with_features = _calculate_features_from_raw(df_raw_combined)

    if df_with_features.empty:
        logger.warning(f"Feature calculation resulted in an empty DataFrame for {timeframe}. Not saving.")
        return False

    # 3. Save features to pickle
    os.makedirs(output_dir, exist_ok=True)
    output_filename = output_filename_template.format(timeframe=timeframe)
    output_path = os.path.join(output_dir, output_filename)

    try:
        df_with_features.to_pickle(output_path)
        logger.info(f"Features built from CSVs saved to: {output_path} (Shape: {df_with_features.shape})")
        return True
    except Exception as e:
        logger.error(f"Error saving features pickle file to {output_path}: {e}")
        return False


if __name__ == '__main__':
    # Setup basic logging for direct script execution
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    parser = argparse.ArgumentParser(description="Build features from raw CSV data sources.")
    parser.add_argument('--tf', type=str, required=True,
                        help='Timeframe to process (e.g., 15m). This will be used in input_csv_pattern and output_filename.')
    parser.add_argument('--input-pattern', type=str, default="data_binance/*/{timeframe}.csv",
                        help="Glob pattern for input CSV files. Use '{timeframe}' as placeholder. Example: 'my_data/{symbol}/{timeframe}_data.csv'")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save the output pickle file (default: {DEFAULT_OUTPUT_DIR}).")

    args = parser.parse_args()

    try:
        success = build_features_from_csv(
            timeframe=args.tf,
            input_csv_pattern=args.input_pattern,
            output_dir=args.output_dir
        )
        if success:
            logger.info(f"Feature building for timeframe {args.tf} completed successfully.")
        else:
            logger.error(f"Feature building for timeframe {args.tf} failed or produced no output.")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info(f"\nFeature building for timeframe {args.tf} interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"An unexpected error occurred during feature building for {args.tf}: {e}", exc_info=True)
        sys.exit(1)
