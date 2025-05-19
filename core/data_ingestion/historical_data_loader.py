# core/data_ingestion/historical_data_loader.py
"""
Loads historical K-line (candle) data from Binance API and stores it
in a local SQLite database. This module is based on the logic from
the original old_update_binance_data.py.
"""
import requests
import sqlite3
import time
from datetime import datetime, timezone
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys

from ..helpers.utils import load_config, get_tf_ms
from ..helpers.binance_api import get_valid_symbols_binance, fetch_klines_binance
from ..helpers.db_ops import init_db, insert_klines_sqlite, get_last_timestamp_sqlite

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Configuration Constants (Defaults, can be overridden by config.yaml) ---
# These will be primarily loaded from config.yaml via load_config()
# but we can set some fallbacks or structure here.

DEFAULT_DB_PATH = 'database/market_data.db'
DEFAULT_UPDATE_LOG_FILE = "data/update_log.txt"
DEFAULT_BINANCE_URL = 'https://api.binance.com/api/v3/klines'
DEFAULT_MAX_CANDLES_PER_SYMBOL = 50000
# Default start timestamp (Binance launch for many pairs, e.g., BTCUSDT ~2017-08-17)
# Corresponds to datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000
DEFAULT_START_DATE_MS = 1502928000000
DEFAULT_CORE_SYMBOLS = ['BTCUSDT', 'ETHUSDT']  # Example, should come from config
DEFAULT_TIMEFRAMES_MAP = {
    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '4h': '4h', '1d': '1d'
}

# Global config dictionary, to be loaded
CONFIG = {}


def _get_ingestion_config_value(key, default_value=None):
    """Safely retrieves a value from the CONFIG['binance_api_config'] section."""
    global CONFIG
    if not CONFIG:  # Ensure config is loaded
        CONFIG = load_config()
        if not CONFIG:
            logger.error("Configuration not loaded. Using defaults for ingestion config.")
            # Fallback to direct default if config system fails entirely
            if key == "url": return DEFAULT_BINANCE_URL
            if key == "max_candles_per_symbol": return DEFAULT_MAX_CANDLES_PER_SYMBOL
            if key == "default_start_date_ms": return DEFAULT_START_DATE_MS
            if key == "timeframes_map": return DEFAULT_TIMEFRAMES_MAP
            if key == "core_symbols_list": return DEFAULT_CORE_SYMBOLS
            return default_value

    binance_config = CONFIG.get('binance_api_config', {})
    return binance_config.get(key, default_value)


def setup_historical_loader_logging():
    """
    Sets up logging specifically for this historical data loader.
    Logs to both console and a dedicated update_log.txt file.
    """
    global CONFIG
    if not CONFIG:
        CONFIG = load_config()

    # Determine project root to correctly place data/update_log.txt
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    data_dir_name = CONFIG.get('data_dir', 'data')
    log_file_name = CONFIG.get('update_log_file', DEFAULT_UPDATE_LOG_FILE).split('/')[-1]  # Get just filename

    data_dir_path = os.path.join(project_root, data_dir_name)
    log_file_path = os.path.join(data_dir_path, log_file_name)

    os.makedirs(data_dir_path, exist_ok=True)

    # Clear existing handlers for this logger to avoid duplicate logs if re-run
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Get root logger's handlers and remove them if they are StreamHandlers to avoid double console output
    # This is a bit aggressive; a better way is to manage logger hierarchy carefully.
    # For now, if this script is run standalone, it might conflict with CLI's root logger.
    # root_logger = logging.getLogger()
    # for handler in root_logger.handlers[:]:
    #     if isinstance(handler, logging.StreamHandler):
    #         root_logger.removeHandler(handler)

    log_level = logging.INFO  # Or from config
    logger.setLevel(log_level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [HistLoader] — %(message)s')

    # File Handler for update_log.txt
    try:
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')  # Append mode
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"Error setting up file logger for {log_file_path}: {e}")  # Use print if logger fails

    # Console Handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info(f"Historical Loader logging configured. Log file: {log_file_path}")
    # logger.propagate = False # To prevent messages from going to the root logger if it's configured differently


def _process_symbol_for_tf_historical(
        symbol, tf_key, interval_str, step_ms, db_path_local,
        start_date_ms, max_candles_total, binance_api_url, pbar_instance=None
):
    """
    Fetches and inserts K-lines for a single symbol and timeframe.
    This function is designed to be run in a thread.
    """
    # Create a new SQLite connection for each thread
    try:
        conn_local = sqlite3.connect(db_path_local, timeout=60)  # Increased timeout
    except sqlite3.Error as e:
        logger.error(f"[{symbol}-{tf_key}] Failed to connect to DB in thread: {e}")
        return 0  # Return 0 downloaded candles

    downloaded_for_symbol_tf = 0
    try:
        last_ts = get_last_timestamp_sqlite(conn_local, tf_key, symbol)
        next_ts = (last_ts + step_ms) if last_ts else start_date_ms

        while downloaded_for_symbol_tf < max_candles_total:
            current_time_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            if next_ts >= current_time_ms - step_ms:  # Don't request current (possibly incomplete) candle
                logger.info(f"[{symbol}-{tf_key}] Reached current data or very recent. Halting for this symbol.")
                break

            klines = fetch_klines_binance(symbol, interval_str, binance_api_url, start_time=next_ts)
            if not klines:
                logger.debug(f"[{symbol}-{tf_key}] No klines returned from Binance for timestamp "
                             f"{datetime.fromtimestamp(next_ts / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M')}. "
                             f"Possibly end of history or API issue.")
                break

            rows_inserted, _, _ = insert_klines_sqlite(conn_local, tf_key, symbol, klines)
            downloaded_for_symbol_tf += rows_inserted

            if pbar_instance and rows_inserted > 0:
                pbar_instance.update(rows_inserted)
            # elif pbar_instance and not rows_inserted and klines: # Update progress even if no new rows (already exists)
            #     pbar_instance.update(len(klines))

            if len(klines) < 1000:  # Binance usually returns 1000 if more data exists
                logger.info(f"[{symbol}-{tf_key}] Received {len(klines)} klines (less than limit), "
                            f"likely end of available historical data for this period.")
                break

            if not klines[-1] or not str(klines[-1][0]).isdigit():  # Basic check for valid last candle timestamp
                logger.error(f"[{symbol}-{tf_key}] Invalid last kline timestamp: {klines[-1]}. Stopping.")
                break

            next_ts = int(klines[-1][0]) + step_ms
            time.sleep(0.1)  # Small delay to be polite to the API

        if downloaded_for_symbol_tf > 0:
            logger.info(f"[{symbol}-{tf_key}] Downloaded {downloaded_for_symbol_tf} new klines.")
        return downloaded_for_symbol_tf
    except Exception as e:
        logger.error(f"[{symbol}-{tf_key}] Error in thread: {e}", exc_info=True)
        return 0  # Return 0 in case of any error in the thread
    finally:
        if 'conn_local' in locals() and conn_local:
            conn_local.close()


def update_timeframe_parallel_historical(tf_key, db_path, core_symbols_list, timeframes_map):
    """
    Updates a single timeframe by fetching data for core symbols in parallel.
    """
    logger.info(f"[Historical Update] ⏳ Starting parallel download for timeframe: {tf_key}")

    timeframes_map_cfg = _get_ingestion_config_value("timeframes_map", timeframes_map)
    if tf_key not in timeframes_map_cfg:
        logger.error(f"Invalid timeframe key: {tf_key}. Not in timeframes_map.")
        return

    # init_db is called before starting any timeframe processing in main_historical_load_logic
    # os.makedirs(os.path.dirname(db_path), exist_ok=True) # Ensure DB directory exists

    step_ms = get_tf_ms(tf_key)  # Uses helper
    if step_ms == 0:
        logger.error(f"Could not determine millisecond step for timeframe {tf_key}.")
        return

    interval_str = timeframes_map_cfg[tf_key]
    start_date_ms = _get_ingestion_config_value("default_start_date_ms", DEFAULT_START_DATE_MS)
    max_candles_sym = _get_ingestion_config_value("max_candles_per_symbol", DEFAULT_MAX_CANDLES_PER_SYMBOL)
    binance_api_url = _get_ingestion_config_value("url", DEFAULT_BINANCE_URL)

    valid_binance_symbols = get_valid_symbols_binance()  # Uses helper
    if not valid_binance_symbols:
        logger.warning("Failed to get valid symbols from Binance. Historical update for {tf_key} might be incomplete.")
        # Proceed with core_symbols_list, hoping they are valid

    active_symbols_to_process = [s for s in core_symbols_list if not valid_binance_symbols or s in valid_binance_symbols]
    if not active_symbols_to_process:
        logger.info(f"No active symbols from core_symbols_list to process for {tf_key}.")
        return

    logger.info(f"Processing {len(active_symbols_to_process)} symbols for {tf_key}. First 5: {', '.join(active_symbols_to_process[:5])}...")

    total_downloaded_for_tf = 0
    # Using ThreadPoolExecutor for parallel downloads per symbol
    # max_workers=5 is a sensible default to avoid overwhelming Binance API or local resources
    with ThreadPoolExecutor(max_workers=5) as executor, \
            tqdm(total=len(active_symbols_to_process) * max_candles_sym,  # Approximate total for progress bar
                 desc=f"Updating {tf_key}", unit=" klines", mininterval=1.0, smoothing=0.1) as pbar:

        future_to_symbol = {
            executor.submit(
                _process_symbol_for_tf_historical,
                symbol, tf_key, interval_str, step_ms, db_path,
                start_date_ms, max_candles_sym, binance_api_url, pbar
            ): symbol for symbol in active_symbols_to_process
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                downloaded_count = future.result()
                total_downloaded_for_tf += downloaded_count
            except Exception as e:
                logger.error(f"Error processing future for symbol {symbol} [{tf_key}]: {e}", exc_info=True)

    logger.info(f"[Historical Update] ✅ Timeframe {tf_key} update complete. Total klines downloaded: {total_downloaded_for_tf}.")


def update_single_symbol_tf_historical(symbol_to_update, tf_key_to_update, db_path, timeframes_map):
    """
    Updates a single symbol for a single timeframe.
    """
    logger.info(f"[Historical Update] ⏳ Updating single: {symbol_to_update} [{tf_key_to_update}]")

    timeframes_map_cfg = _get_ingestion_config_value("timeframes_map", timeframes_map)
    if tf_key_to_update not in timeframes_map_cfg:
        logger.error(f"Invalid timeframe key: {tf_key_to_update}.")
        return

    symbol_to_update_upper = symbol_to_update.upper()
    valid_binance_symbols = get_valid_symbols_binance()
    if not valid_binance_symbols:
        logger.warning("Failed to get valid symbols from Binance. Cannot verify symbol for single update.")
    elif symbol_to_update_upper not in valid_binance_symbols:
        logger.error(f"Symbol {symbol_to_update_upper} is not valid or not actively trading on Binance (USDT pairs).")
        return

    # init_db is called before this in main_historical_load_logic
    # os.makedirs(os.path.dirname(db_path), exist_ok=True)

    step_ms = get_tf_ms(tf_key_to_update)
    if step_ms == 0: return
    interval_str = timeframes_map_cfg[tf_key_to_update]
    start_date_ms = _get_ingestion_config_value("default_start_date_ms", DEFAULT_START_DATE_MS)
    max_candles_sym = _get_ingestion_config_value("max_candles_per_symbol", DEFAULT_MAX_CANDLES_PER_SYMBOL)
    binance_api_url = _get_ingestion_config_value("url", DEFAULT_BINANCE_URL)

    # For a single symbol, no need for ThreadPoolExecutor, run sequentially
    with tqdm(total=max_candles_sym, desc=f"Downloading {symbol_to_update_upper} [{tf_key_to_update}]", unit=" klines", mininterval=1.0) as pbar_single:
        downloaded_count = _process_symbol_for_tf_historical(
            symbol_to_update_upper, tf_key_to_update, interval_str, step_ms, db_path,
            start_date_ms, max_candles_sym, binance_api_url, pbar_single
        )

    logger.info(f"[Historical Update] ✅ Update for {symbol_to_update_upper} [{tf_key_to_update}] complete. Klines downloaded: {downloaded_count}.")


def main_historical_load_logic(
        timeframes_to_process=None,  # List of TFs, or None for all from config
        specific_symbol=None,  # Single symbol to update, or None for core list
        run_all_core_symbols=False  # Flag to run for all core symbols across specified TFs
):
    """
    Main orchestrator for historical data loading.

    Args:
        timeframes_to_process (list, optional): Specific timeframes to process.
            If None, processes all timeframes defined in config's timeframes_map.
        specific_symbol (str, optional): A single symbol to update (e.g., "BTCUSDT").
            If provided, only this symbol is updated for the specified timeframes.
            `run_all_core_symbols` is ignored if this is set.
        run_all_core_symbols (bool): If True and `specific_symbol` is None,
            updates all symbols from `core_symbols_list` in config for the specified timeframes.
    """
    global CONFIG
    CONFIG = load_config()
    if not CONFIG:
        print("CRITICAL: Failed to load configuration (config.yaml). Aborting historical data load.")
        logger.critical("Failed to load configuration (config.yaml). Aborting historical data load.")
        return 1  # Error code

    setup_historical_loader_logging()  # Setup logging using config values

    # Get database path from config or use default
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db_dir_name = CONFIG.get('database_dir', 'database')
    db_file_name = CONFIG.get('db_path', DEFAULT_DB_PATH).split('/')[-1]
    db_path = os.path.join(project_root, db_dir_name, db_file_name)

    os.makedirs(os.path.join(project_root, db_dir_name), exist_ok=True)
    if not init_db(db_path, _get_ingestion_config_value("timeframes_map", DEFAULT_TIMEFRAMES_MAP)):
        logger.error(f"Failed to initialize database at {db_path}. Aborting.")
        return 1

    core_symbols = _get_ingestion_config_value("core_symbols_list", DEFAULT_CORE_SYMBOLS)
    timeframes_map_cfg = _get_ingestion_config_value("timeframes_map", DEFAULT_TIMEFRAMES_MAP)

    if timeframes_to_process is None:
        timeframes_to_process = list(timeframes_map_cfg.keys())

    logger.info(f"Historical data loader starting. DB: {db_path}")
    logger.info(f"Timeframes to process: {timeframes_to_process}")

    if specific_symbol:
        logger.info(f"Mode: Updating specific symbol '{specific_symbol}' for specified timeframes.")
        for tf_key in timeframes_to_process:
            if tf_key in timeframes_map_cfg:
                update_single_symbol_tf_historical(specific_symbol, tf_key, db_path, timeframes_map_cfg)
            else:
                logger.warning(f"Timeframe {tf_key} for symbol {specific_symbol} not in config. Skipping.")
    elif run_all_core_symbols:
        logger.info(f"Mode: Updating all core symbols ({len(core_symbols)}) for specified timeframes.")
        for tf_key in timeframes_to_process:
            if tf_key in timeframes_map_cfg:
                update_timeframe_parallel_historical(tf_key, db_path, core_symbols, timeframes_map_cfg)
            else:
                logger.warning(f"Timeframe {tf_key} for core symbols not in config. Skipping.")
    else:
        logger.warning("No specific mode chosen (single symbol or all core symbols). Please specify arguments.")
        # Print help or guidance here if needed
        return 1  # Indicate an issue / no action taken

    logger.info("[Historical Update] All requested operations finished.")
    return 0  # Success


if __name__ == "__main__":
    # This block allows running the historical loader directly.
    # It mimics the argument parsing of the original old_update_binance_data.py

    # Config is loaded within main_historical_load_logic, so no need to load here explicitly for defaults.
    # Setup logging here if running standalone, otherwise it's handled by main_historical_load_logic
    # or the calling CLI. For direct run, let main_historical_load_logic handle its own logging setup.

    parser = argparse.ArgumentParser(description="Historical K-line Data Loader for Binance.")
    parser.add_argument('--symbol', type=str,
                        help='Specific symbol to update (e.g., BTCUSDT).')
    parser.add_argument('--tf', type=str,  # Changed from list to single, or list if main_logic handles it
                        help='Specific timeframe to update (e.g., 15m). If --symbol is also given, updates that symbol for this TF. If only --tf, updates all core symbols for this TF.')
    parser.add_argument('--all-tf-all-core', action='store_true',
                        help='Update all core symbols for all configured timeframes.')
    # Original script had --all which meant all core symbols for all TFs.
    # And --sequential-all-tf which is now just how --all-tf-all-core works (TFs sequentially, symbols in parallel within TF)

    args = parser.parse_args()
    print(f"Running historical_data_loader.py with arguments: {args}")  # Basic print for direct run

    exit_code = 1  # Default to error
    try:
        if args.all_tf_all_core:
            logger.info("Mode: --all-tf-all-core. Updating all core symbols for all configured timeframes.")
            exit_code = main_historical_load_logic(timeframes_to_process=None, run_all_core_symbols=True)
        elif args.symbol and args.tf:
            logger.info(f"Mode: --symbol {args.symbol} --tf {args.tf}. Updating specific symbol and timeframe.")
            exit_code = main_historical_load_logic(timeframes_to_process=[args.tf], specific_symbol=args.symbol)
        elif args.tf:  # Only --tf is given
            logger.info(f"Mode: --tf {args.tf}. Updating all core symbols for timeframe {args.tf}.")
            exit_code = main_historical_load_logic(timeframes_to_process=[args.tf], run_all_core_symbols=True)
        elif args.symbol:  # Only --symbol is given
            logger.warning("Mode: --symbol specified, but --tf is missing. To update a specific symbol, provide --tf as well.")
            logger.info("To update a specific symbol for ALL timeframes, consider a loop or use --all-tf-all-core and filter later.")
            parser.print_help()
        else:
            logger.info("No specific arguments provided. Use --help for options.")
            parser.print_help()

    except KeyboardInterrupt:
        logger.warning("\n[HistLoader] 🛑 Historical data loading interrupted by user (Ctrl+C).")
        sys.exit(130)  # Standard exit code for Ctrl+C
    except Exception as e:
        logger.critical(f"[HistLoader] 💥 Unexpected critical error: {e}", exc_info=True)
        sys.exit(1)  # General error

    logger.info(f"Historical data loader script finished with exit code {exit_code}.")
    sys.exit(exit_code)
