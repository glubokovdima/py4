# core/analysis/symbol_selector.py
"""
Selects top symbols based on trading activity (volume, ATR, volatility)
across specified timeframes.
"""
import sqlite3
import pandas as pd
import numpy as np
import os
import argparse
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)
# If you want to control logging level specifically for this module:
# logger.setLevel(logging.INFO)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# logger.addHandler(handler)
# logger.propagate = False # Prevents double logging if root logger is also configured

# --- Configuration Constants ---
# In a more advanced setup, these would come from config.yaml
DB_PATH_DEFAULT = "database/market_data.db"
MIN_OBS_DEFAULT = 5000  # Minimum number of candles per TF for a symbol
OUTPUT_CSV_DEFAULT = "top_symbols_selected.csv"  # Output in project root for now
ROLLING_WINDOW_STATS = 50  # Window for rolling stats like avg_volume, avg_atr


def _load_data_for_tf(tf, db_path):
    """Loads candle data for a specific timeframe from the SQLite database."""
    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        # Check if table exists
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='candles_{tf}';")
        if cursor.fetchone() is None:
            logger.warning(f"Table 'candles_{tf}' not found in database {db_path}.")
            return pd.DataFrame()

        df = pd.read_sql_query(f"SELECT * FROM candles_{tf}", conn)
        conn.close()
        if df.empty:
            logger.info(f"No data found for timeframe {tf} in {db_path}.")
            return pd.DataFrame()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["tf"] = tf
        return df
    except sqlite3.Error as e:
        logger.error(f"SQLite error loading data for {tf} from {db_path}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading data for {tf} from {db_path}: {e}")
        return pd.DataFrame()


def _evaluate_symbols_on_tf(df_tf, tf_name, min_observations):
    """Evaluates symbols on a given timeframe's data."""
    results = []
    if df_tf.empty:
        return pd.DataFrame(results)

    grouped = df_tf.groupby("symbol")
    for symbol, df_sym in grouped:
        if len(df_sym) < min_observations:
            logger.debug(f"Symbol {symbol} on {tf_name} has {len(df_sym)} observations, less than min {min_observations}. Skipping.")
            continue

        df_sym = df_sym.sort_values("timestamp").copy()  # Ensure correct order for rolling/shift

        # ATR (Average True Range) - simple high-low for this context
        df_sym["atr_hl"] = df_sym["high"] - df_sym["low"]
        # Log returns
        df_sym["log_return"] = np.log(df_sym["close"] / df_sym["close"].shift(1))

        # Calculate rolling stats safely
        # Ensure there are enough non-NaN values for .iloc[-1] after rolling
        if len(df_sym) >= ROLLING_WINDOW_STATS:
            avg_volume = df_sym["volume"].rolling(ROLLING_WINDOW_STATS, min_periods=1).mean().iloc[-1]
            avg_atr = df_sym["atr_hl"].rolling(ROLLING_WINDOW_STATS, min_periods=1).mean().iloc[-1]
        else:  # Not enough data for full rolling window, use available data mean
            avg_volume = df_sym["volume"].mean()
            avg_atr = df_sym["atr_hl"].mean()

        volatility = df_sym["log_return"].std()

        # Handle cases where metrics might be NaN (e.g., single data point for std)
        if pd.isna(avg_volume) or pd.isna(avg_atr) or pd.isna(volatility):
            logger.warning(f"NaN metric for {symbol} on {tf_name}. Skipping.")
            continue

        # Score: simple product; can be refined
        score = avg_volume * avg_atr * volatility

        results.append({
            "symbol": symbol,
            "tf": tf_name,
            "avg_volume": avg_volume,
            "avg_atr": avg_atr,
            "volatility": volatility,
            "score": score
        })
    return pd.DataFrame(results)


def main_symbol_selection_logic(
        timeframes_list,
        db_path=DB_PATH_DEFAULT,
        min_observations=MIN_OBS_DEFAULT,
        output_csv_path=OUTPUT_CSV_DEFAULT,
        top_n=20
):
    """
    Main logic for selecting top symbols.

    Args:
        timeframes_list (list): List of timeframes to analyze (e.g., ['1m', '5m']).
        db_path (str): Path to the SQLite database.
        min_observations (int): Minimum number of observations for a symbol on a TF.
        output_csv_path (str): Path to save the CSV of top symbols.
        top_n (int): Number of top symbols to select.

    Returns:
        pd.DataFrame: DataFrame of top N symbols, or empty DataFrame if no data.
    """
    logger.info(f"Starting symbol selection for timeframes: {timeframes_list}")
    logger.info(f"Database: {db_path}, Min observations: {min_observations}, Output: {output_csv_path}")

    all_tf_results = []

    for tf_item in timeframes_list:
        logger.info(f"📊 Processing timeframe: {tf_item}")
        df_current_tf = _load_data_for_tf(tf_item, db_path)
        if df_current_tf.empty:
            logger.warning(f"⚠️ No data loaded for timeframe {tf_item}. Skipping.")
            continue

        df_eval_current_tf = _evaluate_symbols_on_tf(df_current_tf, tf_item, min_observations)
        if not df_eval_current_tf.empty:
            all_tf_results.append(df_eval_current_tf)
        else:
            logger.info(f"No symbols met criteria for timeframe {tf_item}.")

    if not all_tf_results:
        logger.error("❌ No suitable data found for analysis across any timeframe.")
        return pd.DataFrame()

    df_combined_results = pd.concat(all_tf_results)
    if df_combined_results.empty:
        logger.error("❌ Combined results are empty after processing all timeframes.")
        return pd.DataFrame()

    # Aggregate scores across timeframes for each symbol
    # Using mean of scores as the aggregation logic here. Could be sum, weighted mean, etc.
    df_summary = df_combined_results.groupby("symbol").agg(
        avg_volume_across_tfs=("avg_volume", "mean"),
        avg_atr_across_tfs=("avg_atr", "mean"),
        avg_volatility_across_tfs=("volatility", "mean"),
        final_score=("score", "mean"),  # Or sum, or more complex aggregation
        num_tfs_present=("tf", "nunique")
    ).sort_values("final_score", ascending=False)

    df_top_symbols = df_summary.head(top_n).reset_index()

    if df_top_symbols.empty:
        logger.info("ℹ️ No symbols made it to the top N list.")
    else:
        logger.info(f"\n🔥 Top-{top_n} most active symbols (aggregated across TFs):")
        try:
            # For cleaner console output
            print(df_top_symbols.round(4).to_string(index=False))
        except Exception as e:
            logger.warning(f"Could not print top symbols table: {e}")
            print(df_top_symbols)

        try:
            output_dir = os.path.dirname(output_csv_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            df_top_symbols["symbol"].to_csv(output_csv_path, index=False, header=False)  # Save only symbol names
            logger.info(f"💾 Top symbols list saved to: {output_csv_path}")
        except IOError as e:
            logger.error(f"Error saving top symbols CSV to {output_csv_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error saving CSV: {e}")

    return df_top_symbols


if __name__ == "__main__":
    # Setup basic logging for direct script execution
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    parser = argparse.ArgumentParser(description="Analyze cryptocurrency pair activity across multiple timeframes.")
    parser.add_argument('--tf', nargs='+', required=True,
                        help='Specify timeframes, e.g., 1m 5m 15m')
    parser.add_argument('--db-path', type=str, default=DB_PATH_DEFAULT,
                        help=f'Path to the SQLite database (default: {DB_PATH_DEFAULT})')
    parser.add_argument('--min-obs', type=int, default=MIN_OBS_DEFAULT,
                        help=f'Minimum observations per symbol per TF (default: {MIN_OBS_DEFAULT})')
    parser.add_argument('--output-csv', type=str, default=OUTPUT_CSV_DEFAULT,
                        help=f'Path to save the output CSV of top symbols (default: {OUTPUT_CSV_DEFAULT})')
    parser.add_argument('--top-n', type=int, default=20,
                        help='Number of top symbols to select (default: 20)')

    args = parser.parse_args()

    main_symbol_selection_logic(
        timeframes_list=args.tf,
        db_path=args.db_path,
        min_observations=args.min_obs,
        output_csv_path=args.output_csv,
        top_n=args.top_n
    )
