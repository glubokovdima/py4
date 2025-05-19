# core/data_ingestion/incremental_data_loader.py
"""
Loads recent K-line (candle) data from Binance API for incremental updates
and stores it in a local SQLite database. Based on incremental_data_loader.py.
"""
import requests
import sqlite3
import time
from datetime import datetime, timedelta, timezone
import os
import argparse
from tqdm import tqdm
import logging
import sys

from ..helpers.utils import load_config, get_tf_ms
from ..helpers.binance_api import fetch_klines_binance  # Using the refactored one
from ..helpers.db_ops import init_db, insert_klines_sqlite

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global config dictionary, to be loaded
CONFIG = {}

# --- Default Configuration Fallbacks (should primarily come from config.yaml) ---
DEFAULT_DB_PATH_INC = 'database/market_data.db'
DEFAULT_UPDATE_LOG_FILE_INC = "data/update_log.txt"  # Note: historical_loader also uses this
DEFAULT_BINANCE_URL_INC = 'https://api.binance.com/api/v3/klines'
DEFAULT_CORE_SYMBOLS_INC = ['BTCUSDT', 'ETHUSDT']
DEFAULT_TIMEFRAMES_MAP_INC = {
    '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
    '1h': '1h', '4h': '4h', '1d': '1d'
}
DEFAULT_TF_HOURS_BACK_INC = {  # How many hours of recent data to fetch
    '1m': 5 * 24, '5m': 14 * 24, '15m': 30 * 24, '30m': 30 * 24,
    '1h': 90 * 24, '4h': 180 * 24, '1d': 365 * 3 * 24  # 3 years for daily
}
DEFAULT_CANDLE_LIMIT_INC = 1000


def _get_incremental_config_value(key, timeframe_key=None, default_value=None):
    """Safely retrieves a value from CONFIG for incremental loader."""
    global CONFIG
    if not CONFIG:
        CONFIG = load_config()
        if not CONFIG:
            logger.error("IncrementalLoader: Configuration not loaded. Using hardcoded defaults.")
            # Provide direct defaults if config system fails entirely
            if key == "db_path": return DEFAULT_DB_PATH_INC
            if key == "update_log_file": return DEFAULT_UPDATE_LOG_FILE_INC
            if key == "binance_api_url": return DEFAULT_BINANCE_URL_INC
            if key == "core_symbols_list": return DEFAULT_CORE_SYMBOLS_INC
            if key == "timeframes_map": return DEFAULT_TIMEFRAMES_MAP_INC
            if key == "tf_hours_back" and timeframe_key: return DEFAULT_TF_HOURS_BACK_INC.get(timeframe_key)
            if key == "tf_hours_back_map": return DEFAULT_TF_HOURS_BACK_INC
            if key == "candle_limit": return DEFAULT_CANDLE_LIMIT_INC
            return default_value

    # Try to get from a specific 'incremental_loader' section if it exists
    inc_config = CONFIG.get('incremental_loader', {})
    val = inc_config.get(key)
    if val is not None:
        if key == "tf_hours_back" and timeframe_key: return val.get(timeframe_key, default_value)
        if key == "tf_hours_back_map": return val  # Return the whole map
        return val

    # Fallback to general or other relevant sections
    if key == "db_path": return CONFIG.get('db_path', default_value)
    if key == "update_log_file": return CONFIG.get('update_log_file', default_value)
    if key == "binance_api_url": return CONFIG.get('binance_api_config', {}).get('url', default_value)
    if key == "core_symbols_list": return CONFIG.get('binance_api_config', {}).get('core_symbols_list', default_value)
    if key == "timeframes_map": return CONFIG.get('binance_api_config', {}).get('timeframes_map', default_value)
    if key == "tf_hours_back_map": return CONFIG.get('incremental_loader_tf_hours_back', default_value)  # Example specific key
    if key == "tf_hours_back" and timeframe_key:
        hours_back_map = CONFIG.get('incremental_loader_tf_hours_back', DEFAULT_TF_HOURS_BACK_INC)
        return hours_back_map.get(timeframe_key, default_value)
    if key == "candle_limit": return CONFIG.get('binance_api_config', {}).get('limit', default_value)  # Assuming 'limit' might be general

    return default_value


def setup_incremental_loader_logging():
    """Sets up logging for incremental data loader."""
    global CONFIG
    if not CONFIG: CONFIG = load_config()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir_name = CONFIG.get('data_dir', 'data')
    log_file_name_rel = _get_incremental_config_value("update_log_file", default_value=DEFAULT_UPDATE_LOG_FILE_INC)
    log_file_name = log_file_name_rel.split('/')[-1]

    data_dir_path = os.path.join(project_root, data_dir_name)
    log_file_path = os.path.join(data_dir_path, log_file_name)
    os.makedirs(data_dir_path, exist_ok=True)

    # Avoid duplicate handlers if this function is called multiple times or by CLI
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file_path for h in logger.handlers):
        logger.debug("File handler for incremental log already exists. Skipping setup.")
    else:
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        fh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s [IncLoader] — %(message)s'))
        logger.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s [IncLoader] — %(message)s'))
        logger.addHandler(sh)

    logger.setLevel(logging.INFO)  # Or from config
    logger.info(f"Incremental Loader logging configured. Log file: {log_file_path}")


def _update_recent_data_for_tf(timeframe_key):
    """
    Core logic to update recent data for a single timeframe.
    (Previously `update_recent` from incremental_data_loader.py)
    """
    logger.info(f"[Incremental Update] ⏳ Starting for timeframe: {timeframe_key}")

    # Config values
    db_path_rel = _get_incremental_config_value("db_path", default_value=DEFAULT_DB_PATH_INC)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    db_path_abs = os.path.join(project_root, db_path_rel) if not os.path.isabs(db_path_rel) else db_path_rel

    core_symbols_list = _get_incremental_config_value("core_symbols_list", default_value=DEFAULT_CORE_SYMBOLS_INC)
    timeframes_map = _get_incremental_config_value("timeframes_map", default_value=DEFAULT_TIMEFRAMES_MAP_INC)
    tf_hours_back_map = _get_incremental_config_value("tf_hours_back_map", default_value=DEFAULT_TF_HOURS_BACK_INC)
    binance_api_url = _get_incremental_config_value("binance_api_url", default_value=DEFAULT_BINANCE_URL_INC)  # For fetch_klines_binance
    candle_fetch_limit = _get_incremental_config_value("candle_limit", default_value=DEFAULT_CANDLE_LIMIT_INC)

    # Ensure DB and tables exist (init_db uses timeframes_map from config)
    # init_db is typically called once before any updates. If called per TF, it's just a check.
    if not init_db(db_path_abs, timeframes_map):  # init_db is from helpers
        logger.error(f"Failed to initialize database at {db_path_abs} for TF {timeframe_key}. Aborting this TF.")
        return

    try:
        conn = sqlite3.connect(db_path_abs)
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database {db_path_abs} for TF {timeframe_key}: {e}")
        return

    api_interval_str = timeframes_map.get(timeframe_key)
    if not api_interval_str:
        logger.error(f"Invalid timeframe key '{timeframe_key}' not found in timeframes_map. Skipping.")
        conn.close()
        return

    hours_to_fetch_back = tf_hours_back_map.get(timeframe_key, 48)  # Default 48 hours if not in map
    start_datetime_utc = datetime.now(timezone.utc) - timedelta(hours=hours_to_fetch_back)
    start_timestamp_ms = int(start_datetime_utc.timestamp() * 1000)

    logger.info(f"Updating {timeframe_key} for the last {hours_to_fetch_back} hours (since {start_datetime_utc.strftime('%Y-%m-%d %H:%M:%S %Z')})")

    for symbol in tqdm(core_symbols_list, desc=f"Updating {timeframe_key}", unit="symbol", leave=False):
        try:
            # fetch_klines_binance is from helpers, already handles retries and rate limits
            klines = fetch_klines_binance(
                symbol, api_interval_str, api_url_override=binance_api_url,  # Pass full klines URL
                start_time_ms=start_timestamp_ms, limit=candle_fetch_limit
            )
            if klines:
                # insert_klines_sqlite is from helpers
                inserted_count, _, _ = insert_klines_sqlite(conn, timeframe_key, symbol, klines)
                if inserted_count > 0:
                    logger.info(f"{symbol} [{timeframe_key}] — Added: {inserted_count} new klines.")
                # else:
                #    logger.debug(f"{symbol} [{timeframe_key}] — No new unique klines to add.")
            else:
                logger.debug(f"{symbol} [{timeframe_key}] — No klines returned from API for the recent period.")
        except Exception as e_sym:
            logger.error(f"Critical error processing symbol {symbol} [{timeframe_key}]: {e_sym}", exc_info=True)
            continue  # Continue with the next symbol

    conn.close()
    logger.info(f"[Incremental Update] ✅ Finished for timeframe: {timeframe_key}")


def main_incremental_load_logic(timeframes_to_process=None):
    """
    Main orchestrator for incremental data loading.

    Args:
        timeframes_to_process (list, optional): List of timeframe strings to process.
            If None, processes all timeframes defined in config's timeframes_map.
    """
    global CONFIG
    CONFIG = load_config()
    if not CONFIG:
        print("CRITICAL: Failed to load configuration (config.yaml). Aborting incremental data load.")
        logger.critical("Failed to load configuration (config.yaml). Aborting incremental data load.")
        return 1  # Error code

    setup_incremental_loader_logging()  # Setup logging using config values

    timeframes_map_cfg = _get_incremental_config_value("timeframes_map", default_value=DEFAULT_TIMEFRAMES_MAP_INC)

    if timeframes_to_process is None:
        timeframes_to_process = list(timeframes_map_cfg.keys())
        logger.info(f"No specific timeframes provided, processing all configured: {timeframes_to_process}")

    if not timeframes_to_process:
        logger.warning("No timeframes selected for incremental update.")
        return 0  # No work to do

    logger.info(f"Starting incremental update for timeframes: {', '.join(timeframes_to_process)}")

    for tf_key in timeframes_to_process:
        if tf_key in timeframes_map_cfg:
            _update_recent_data_for_tf(tf_key)
        else:
            logger.error(f"Unknown timeframe: {tf_key}. Skipping. Available in config: {', '.join(timeframes_map_cfg.keys())}")

    logger.info("[Incremental Update] All specified timeframes processed.")
    return 0  # Success


if __name__ == "__main__":
    # This block allows running the incremental loader directly.
    CONFIG = load_config()  # Load config for defaults needed by arg parser
    if not CONFIG:
        # Basic logging if config fails, for CLI run
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')
        logger.warning("Running incremental_data_loader.py with default config values as config.yaml failed to load.")
    else:
        # Setup logging properly if config loaded
        # main_incremental_load_logic will call setup_incremental_loader_logging
        pass

    default_tfs_list = list(_get_incremental_config_value("timeframes_map", default_value=DEFAULT_TIMEFRAMES_MAP_INC).keys())

    parser = argparse.ArgumentParser(description="Incrementally update recent candle data from Binance.")
    parser.add_argument('--tf', nargs='*', default=default_tfs_list,
                        help=f"Timeframes to process (e.g., 1m 5m 15m). Default: all configured. Available: {', '.join(default_tfs_list)}")
    args = parser.parse_args()

    exit_code = 1  # Default to error
    try:
        if not args.tf:  # Should not happen with default set, but good check
            logger.warning("No timeframes specified for update. Use --tf or run without args for all.")
            parser.print_help()
        else:
            print(f"Running incremental update for timeframes: {', '.join(args.tf)}")
            exit_code = main_incremental_load_logic(timeframes_to_process=args.tf)

    except KeyboardInterrupt:
        logger.warning("\n[IncLoader] 🛑 Incremental update interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"[IncLoader] 💥 Unexpected critical error: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Incremental data loader script finished with exit code {exit_code}.")
    sys.exit(exit_code)
