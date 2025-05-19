# scripts/update_data.py

import requests
import time
from datetime import datetime, timezone
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys

# Ensure the src directory is in the Python path
# This is a common pattern in this structure, but might need adjustment
# depending on how you run the scripts (e.g., from project root vs scripts dir)
# A more robust approach is to run from the project root: python scripts/update_data.py ...
# If running from scripts/, uncomment the following lines:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


# Import config, logging setup, and db module
from src.utils.config import get_config
from src.utils.logging_setup import setup_logging
from src import db # Import the db module

# --- Initial Setup ---
# Load configuration
config = get_config()
DATA_UPDATE_CONFIG = config['data_update']
PATHS_CONFIG = config['paths']
TIMEFRAMES_CONFIG = config['timeframes']

# Configure logging for this script
# Note: In a pipeline or via cli.py, logging might already be set up.
# Calling setup_logging again is generally safe if it uses dictConfig,
# but logging.basicConfig should be avoided if calling setup_logging.
# Let's assume setup_logging handles idempotency or is called once by the entry point.
setup_logging()
logger = logging.getLogger(__name__) # Use logger specific to this module

# --- Constants from Config ---
BINANCE_API_URL = DATA_UPDATE_CONFIG['binance_api_url']
MAX_CANDLES_PER_REQUEST = DATA_UPDATE_CONFIG['max_candles_per_request']
MAX_CANDLES_PER_SYMBOL_FULL = DATA_UPDATE_CONFIG['max_candles_per_symbol']
DEFAULT_START_DATE_MS = DATA_UPDATE_CONFIG['default_start_date_ms']
REQUEST_RETRIES = DATA_UPDATE_CONFIG['request_retries']
REQUEST_DELAY_SEC = DATA_UPDATE_CONFIG['request_delay_sec']
REQUEST_TIMEOUT_SEC = DATA_UPDATE_CONFIG['request_timeout_sec']
RATE_LIMIT_BAN_SEC = DATA_UPDATE_CONFIG['rate_limit_ban_sec']
CORE_SYMBOLS = DATA_UPDATE_CONFIG['core_symbols']
MAX_WORKERS = config['database']['max_workers'] # Re-using max_workers from database config


# --- Utility Functions ---

def get_tf_ms(tf_key):
    """Converts timeframe key (e.g., '1m', '1h', '1d') to milliseconds."""
    mult = {'m': 60, 'h': 3600, 'd': 86400}
    num_str = ''.join(filter(str.isdigit, tf_key))
    if not num_str:
        logger.error(f"Invalid timeframe format: {tf_key}. Cannot extract number.")
        return 0
    try:
        num = int(num_str)
    except ValueError:
         logger.error(f"Invalid timeframe format: {tf_key}. Cannot convert number part to int.")
         return 0

    unit = ''.join(filter(str.isalpha, tf_key))
    if not unit or unit not in mult:
        logger.error(f"Invalid timeframe format: {tf_key}. Unknown unit '{unit}'.")
        return 0
    return num * mult[unit] * 1000


def get_valid_symbols():
    """Fetches list of active USDT symbols from Binance Exchange Info."""
    url = 'https://api.binance.com/api/v3/exchangeInfo'
    try:
        logger.info("Fetching list of active USDT symbols from Binance...")
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SEC)
        response.raise_for_status()
        data = response.json()
        symbols = {s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'}
        logger.info(f"Found {len(symbols)} active USDT symbols.")
        return symbols
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching symbol list from Binance: {e}")
        return set()
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching symbol list: {e}")
        return set()

def fetch_klines(symbol, interval_str, start_time=None, end_time=None, limit=MAX_CANDLES_PER_REQUEST, retries=REQUEST_RETRIES, delay=REQUEST_DELAY_SEC):
    """Fetches klines from Binance API with retry and rate limit handling."""
    params = {
        'symbol': symbol,
        'interval': interval_str,
        'limit': limit
    }
    if start_time:
        params['startTime'] = start_time
    if end_time:
        params['endTime'] = end_time # Note: endTime is exclusive for startTime queries

    for attempt in range(retries):
        try:
            # logger.debug(f"Requesting {symbol} {interval_str} from {start_time} to {end_time} (attempt {attempt + 1}/{retries})")
            response = requests.get(BINANCE_API_URL, params=params, timeout=REQUEST_TIMEOUT_SEC)

            if response.status_code in [429, 418]: # Rate limit exceeded or IP banned
                ban_duration = RATE_LIMIT_BAN_SEC
                try:
                    retry_after = int(response.headers.get("Retry-After", ban_duration))
                    ban_duration = max(retry_after, ban_duration)
                    logger.warning(f"Rate limit hit (HTTP {response.status_code}) for {symbol} {interval_str}. Waiting {ban_duration} seconds...")
                except (ValueError, TypeError):
                    logger.warning(f"Rate limit hit (HTTP {response.status_code}) for {symbol} {interval_str}. No Retry-After header, waiting default {ban_duration} seconds...")
                time.sleep(ban_duration)
                continue # Retry the request

            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            klines = response.json()

            # Check if klines data structure is as expected (list of lists)
            if not isinstance(klines, list) or (klines and not isinstance(klines[0], list)):
                 logger.error(f"Unexpected response format for {symbol} {interval_str}: {klines}")
                 return [] # Return empty list on unexpected format

            # Convert numerical fields to float/int and handle potential errors
            cleaned_klines = []
            for k in klines:
                 try:
                     # Expected format: [ timestamp, open, high, low, close, volume, close_time, quote_asset_volume, number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore ]
                     # We need: timestamp (0), open (1), high (2), low (3), close (4), volume (5)
                     cleaned_klines.append([
                        int(k[0]), # timestamp (ms, int)
                        float(k[1]), # open (real)
                        float(k[2]), # high (real)
                        float(k[3]), # low (real)
                        float(k[4]), # close (real)
                        float(k[5])  # volume (real)
                     ])
                 except (ValueError, TypeError, IndexError) as parse_err:
                     logger.warning(f"Failed to parse kline data for {symbol} {interval_str} at timestamp {k[0] if len(k)>0 else 'N/A'}: {parse_err}. Skipping this kline.")
                     continue # Skip this malformed kline

            return cleaned_klines # Return list of cleaned klines

        except requests.exceptions.HTTPError as http_err:
            logger.warning(f"HTTP error fetching {symbol} {interval_str} (attempt {attempt + 1}/{retries}): {http_err}. Response text: {response.text if response else 'N/A'}")
        except requests.exceptions.ConnectionError as conn_err:
             logger.warning(f"Connection error fetching {symbol} {interval_str} (attempt {attempt + 1}/{retries}): {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
             logger.warning(f"Timeout error fetching {symbol} {interval_str} (attempt {attempt + 1}/{retries}): {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logger.warning(f"General Request error fetching {symbol} {interval_str} (attempt {attempt + 1}/{retries}): {req_err}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during fetch_klines for {symbol} {interval_str} (attempt {attempt + 1}/{retries}): {e}", exc_info=True)


        if attempt < retries - 1:
            current_delay = delay * (attempt + 1)
            logger.info(f"Retrying {symbol} {interval_str} in {current_delay:.1f} seconds...")
            time.sleep(current_delay)
        else:
            logger.error(f"Failed to fetch klines for {symbol} {interval_str} after {retries} attempts.")
            return [] # Return empty list after all retries fail

    return [] # Should not be reached if retries > 0 and no exception, but as a safeguard


def process_symbol_for_tf(symbol, tf_key, step_ms, update_mode, pbar_instance=None):
    """
    Fetches and inserts candles for a single symbol and timeframe.
    Handles full or incremental updates.
    Designed to be run in a separate thread/process.

    Args:
        symbol (str): The trading symbol.
        tf_key (str): The timeframe key.
        step_ms (int): Milliseconds duration of one candle.
        update_mode (str): 'full' or 'mini'.
        pbar_instance (tqdm, optional): A shared tqdm instance to update progress.
    """
    # Each thread/process needs its own DB connection
    # Using db.get_db_connection() ensures proper connection handling per thread
    conn_local = None
    downloaded_count = 0
    interval_str = tf_key # Binance uses the same string

    try:
        conn_local = db.get_db_connection()

        if update_mode == 'full':
            start_ts = DEFAULT_START_DATE_MS
            logger.debug(f"[{symbol}-{tf_key}] Starting full update from {datetime.fromtimestamp(start_ts/1000, timezone.utc)}")
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000) # Fetch up to current time
        else: # 'mini' update
            last_ts = db.get_last_timestamp(tf_key, symbol)
            # Start from the timestamp AFTER the last recorded candle
            # If last_ts is None, start from a recent date (e.g., last year or a default mini-update start)
            # For mini update, let's start from the last timestamp + step_ms if available,
            # otherwise maybe a fixed recent past (e.g., 30 days ago)? Or just the default start?
            # Sticking to last_ts + step_ms for simplicity, effectively only fetching NEW candles.
            # If last_ts is None, it means no data exists, mini-update won't fetch anything unless we fallback.
            # Let's fallback to a very recent date for mini-update if no data exists.
            # A safer mini-update fallback might be config['mini_update_start_date_ms']
            # For now, let's strictly fetch from last_ts+step_ms for mini. If last_ts is None, no update happens.
            start_ts = (last_ts + step_ms) if last_ts else None # No fallback for mini if empty table
            end_ts = int(datetime.now(timezone.utc).timestamp() * 1000) # Fetch up to current time

            if start_ts is None:
                 logger.debug(f"[{symbol}-{tf_key}] No existing data, mini update skipped.")
                 return 0 # No data to update if table is empty

            logger.debug(f"[{symbol}-{tf_key}] Starting mini update from {datetime.fromtimestamp(start_ts/1000, timezone.utc)}")


        current_fetch_ts = start_ts
        total_fetched_count = 0 # Track total fetched in this call

        while True:
            # In 'full' mode, fetch batches from historical start up to end_ts.
            # In 'mini' mode, fetch from last_ts+step_ms up to end_ts.
            # Binance API limit=1000 means we get 1000 candles starting from startTime.
            # The *next* fetch should start from the timestamp *after* the last one received.

            klines = fetch_klines(symbol, interval_str, start_time=current_fetch_ts, end_time=end_ts)

            if not klines:
                logger.debug(f"[{symbol}-{tf_key}] No new klines fetched starting from {datetime.fromtimestamp(current_fetch_ts/1000, timezone.utc)}. Update finished for this symbol/TF.")
                break

            # Insert fetched klines
            # insert_candles returns the number of rows PROCESSED for insertion
            processed_count = db.insert_candles(conn_local, tf_key, symbol, klines)
            downloaded_count += processed_count # Sum processed count

            if pbar_instance:
                pbar_instance.update(processed_count)

            total_fetched_count += len(klines)

            # Determine the timestamp for the next request
            # The next request should start from the timestamp of the last candle + step_ms
            # Ensure klines[-1] and klines[-1][0] are valid
            if klines and len(klines[-1]) > 0 and isinstance(klines[-1][0], int):
                 current_fetch_ts = klines[-1][0] + step_ms
            else:
                 logger.warning(f"[{symbol}-{tf_key}] Last kline data invalid: {klines[-1] if klines else 'N/A'}. Stopping fetch loop.")
                 break # Stop if the last kline is malformed

            # Stop condition: If the last fetched candle's timestamp is >= end_ts (current time)
            # or if the number of fetched klines is less than the limit (usually indicates end of available data)
            if (klines and klines[-1][0] >= end_ts) or len(klines) < MAX_CANDLES_PER_REQUEST:
                 logger.debug(f"[{symbol}-{tf_key}] Reached end_ts or fetched less than limit ({len(klines)} < {MAX_CANDLES_PER_REQUEST}). Update finished.")
                 break

            # Stop condition for full update: if we've downloaded a lot of candles
            if update_mode == 'full' and downloaded_count >= MAX_CANDLES_PER_SYMBOL_FULL:
                 logger.warning(f"[{symbol}-{tf_key}] Reached max candles limit ({MAX_CANDLES_PER_SYMBOL_FULL}) for full update. Stopping.")
                 break

            time.sleep(0.1) # Small delay between requests per symbol

        if downloaded_count > 0:
             logger.info(f"[{symbol}-{tf_key}] Finished update. Processed {downloaded_count} potential new candles.")
        else:
             logger.debug(f"[{symbol}-{tf_key}] Finished update. No new candles processed.")

    except Exception as e:
        logger.error(f"[{symbol}-{tf_key}] Error during processing: {e}", exc_info=True)
    finally:
        if conn_local:
            conn_local.close()
    return downloaded_count # Return total count processed for insertion

# --- Main Update Logic ---

def run_update(timeframe_keys, symbols_to_update, update_mode):
    """
    Orchestrates the data update process for specified timeframes and symbols.

    Args:
        timeframe_keys (list): List of timeframe keys to update.
        symbols_to_update (list): List of symbols to update.
        update_mode (str): 'full' or 'mini'.
    """
    logger.info(f"Starting '{update_mode}' update for TFs: {', '.join(timeframe_keys)} and {len(symbols_to_update)} symbols...")

    # Ensure tables exist for all target timeframes
    db.init_db(timeframe_keys)

    # Get the list of valid trading symbols from Binance
    valid_binance_symbols = get_valid_symbols()
    if not valid_binance_symbols:
        logger.error("Failed to get valid symbols from Binance. Update aborted.")
        return

    # Filter symbols_to_update to only include those actively trading on Binance
    symbols_to_process = [s for s in symbols_to_update if s in valid_binance_symbols]
    if not symbols_to_process:
        logger.warning("None of the specified symbols are currently trading USDT pairs on Binance. Update skipped.")
        return
    logger.info(f"Processing {len(symbols_to_process)} active symbols: {', '.join(symbols_to_process[:10])}{'...' if len(symbols_to_process) > 10 else ''}")


    total_downloaded_across_all_tfs = 0

    # Process each timeframe sequentially, but symbols within each TF in parallel
    for tf_key in timeframe_keys:
        logger.info(f"\n--- Updating timeframe: {tf_key} ({update_mode}) ---")
        step_ms = get_tf_ms(tf_key)
        if step_ms == 0:
            logger.error(f"Skipping {tf_key} due to invalid format.")
            continue

        # Use tqdm with a shared progress bar for all symbols in this TF
        # The total for tqdm is an estimate (number of symbols * max candles per symbol per fetch loop)
        # A more accurate total is hard to get upfront. Using a base number per symbol is okay for visual feedback.
        estimated_total_steps = len(symbols_to_process) * (MAX_CANDLES_PER_SYMBOL_FULL // MAX_CANDLES_PER_REQUEST + 1) if update_mode == 'full' else len(symbols_to_process) * 10 # Rough estimate for mini

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor, \
             tqdm(total=estimated_total_steps, desc=f"Overall {tf_key}", unit="req", smoothing=0.1) as pbar: # Using 'req' unit as we update per fetch loop result

            futures = {executor.submit(process_symbol_for_tf, symbol, tf_key, step_ms, update_mode, pbar): symbol for symbol in symbols_to_process}

            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    downloaded_count = future.result()
                    total_downloaded_across_all_tfs += downloaded_count
                    # pbar.set_postfix_str(f"{symbol} done ({downloaded_count})") # Optional: show symbol status
                except Exception as e:
                    logger.error(f"Error in thread for symbol {symbol} [{tf_key}]: {e}")

        logger.info(f"--- Finished update for timeframe: {tf_key} ---")


    logger.info(f"\n✅ Data update process completed. Total candles processed for insertion across all TFs: {total_downloaded_across_all_tfs}")


# --- Command Line Interface ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Update historical and live Binance Klines data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--full', action='store_true', help='Perform a full historical update for CORE_SYMBOLS across all default timeframes.')
    group.add_argument('--mini', action='store_true', help='Perform an incremental update for CORE_SYMBOLS across specified or default timeframes (fetches new candles since last record).')
    group.add_argument('--symbol', type=str, help='Update data for a single specified symbol (requires --tf).')

    parser.add_argument('--tf', nargs='+', help='Specify one or more timeframes (e.g., 15m 1h). Required with --symbol, optional with --mini (defaults to all default TFs if not specified).')
    # Note: --symbol-group from old update_data.py is now handled by cli.py passing a list of symbols via --symbol-list if needed, or by filtering *after* loading data in preprocess/predict. For update, we either update CORE_SYMBOLS or a single specified symbol.

    args = parser.parse_args()

    # Determine the list of timeframes to process
    timeframes_to_process = []
    if args.tf:
        # Validate specified timeframes against config defaults or a known list
        allowed_timeframes = TIMEFRAMES_CONFIG['default'] # Or define a separate list of all possible Binance TFs
        timeframes_to_process = [t for t in args.tf if t in allowed_timeframes]
        invalid_tfs = [t for t in args.tf if t not in allowed_timeframes]
        if invalid_tfs:
            logger.warning(f"Ignoring invalid timeframes: {', '.join(invalid_tfs)}. Allowed: {', '.join(allowed_timeframes)}")
        if not timeframes_to_process:
            logger.error("No valid timeframes specified. Exiting.")
            sys.exit(1)
    elif args.full or args.mini:
        # Use default timeframes if --tf is not specified with --full or --mini
        timeframes_to_process = TIMEFRAMES_CONFIG['default']
        logger.info(f"No timeframes specified with --full/--mini, using defaults: {', '.join(timeframes_to_process)}")
    # If --symbol is used without --tf, the mutual exclusion group handles it (parser requires --tf with --symbol)


    # Determine the list of symbols to process and the update mode
    symbols_to_process = []
    update_mode = None

    if args.full:
        update_mode = 'full'
        symbols_to_process = CORE_SYMBOLS # Use the predefined list for full update
        logger.info(f"Full update mode selected. Processing {len(symbols_to_process)} CORE_SYMBOLS.")
    elif args.mini:
        update_mode = 'mini'
        symbols_to_process = CORE_SYMBOLS # Use the predefined list for mini update
        logger.info(f"Mini update mode selected. Processing {len(symbols_to_process)} CORE_SYMBOLS.")
    elif args.symbol:
        update_mode = 'mini' # Updating a single symbol is typically an incremental update
        symbols_to_process = [args.symbol.upper()] # Ensure symbol is uppercase
        logger.info(f"Single symbol update mode selected: {symbols_to_process[0]} for {', '.join(timeframes_to_process)}.")
        if len(timeframes_to_process) > 1:
            logger.warning("Updating a single symbol (--symbol) across multiple timeframes (--tf) sequentially.")


    # --- Run the update ---
    try:
        if args.symbol:
            # If updating a single symbol across multiple TFs, run sequentially for TFs
            for tf_key in timeframes_to_process:
                 logger.info(f"Starting update for {args.symbol.upper()} on {tf_key}...")
                 step_ms = get_tf_ms(tf_key)
                 if step_ms == 0: continue
                 # Use a simple tqdm for visual feedback for a single symbol/TF
                 with tqdm(total=MAX_CANDLES_PER_SYMBOL_FULL, desc=f"Updating {args.symbol.upper()} {tf_key}", unit="candles", mininterval=1.0) as pbar_single:
                     process_symbol_for_tf(args.symbol.upper(), tf_key, step_ms, update_mode, pbar_single)
                 logger.info(f"Finished update for {args.symbol.upper()} on {tf_key}.")
        elif symbols_to_process and timeframes_to_process and update_mode:
             # Run the main parallel update for full/mini modes
             run_update(timeframes_to_process, symbols_to_process, update_mode)
        else:
             logger.error("Insufficient arguments provided. Use --help for usage.")

    except KeyboardInterrupt:
        logger.warning("\nData update process interrupted by user (Ctrl+C).")
        sys.exit(130) # Standard Unix exit code for Ctrl+C
    except Exception as e:
        logger.error(f"An unexpected error occurred during the update process: {e}", exc_info=True)
        sys.exit(1) # Standard Unix exit code for general error