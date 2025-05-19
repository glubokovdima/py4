# core/helpers/binance_api.py
"""
Helper functions for interacting with the Binance API.
Mainly focused on fetching exchange information and K-line (candle) data.
"""
import requests
import time
import logging
from datetime import datetime, timezone  # For logging timestamps

logger = logging.getLogger(__name__)

# --- Default Configuration (can be overridden by global config if needed) ---
DEFAULT_BINANCE_API_URL = 'https://api.binance.com/api/v3'  # Base URL
DEFAULT_KLINES_ENDPOINT = '/klines'
DEFAULT_EXCHANGE_INFO_ENDPOINT = '/exchangeInfo'
DEFAULT_REQUEST_TIMEOUT = 15  # seconds
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 5  # seconds
DEFAULT_RATE_LIMIT_PAUSE = 60  # seconds for 429/418 errors


def get_valid_symbols_binance(base_asset_filter='USDT', quote_asset_filter=None, status_filter='TRADING'):
    """
    Fetches a list of actively trading symbols from Binance, optionally filtered.

    Args:
        base_asset_filter (str, optional): Filter by base asset (e.g., 'USDT').
                                           If None, no base asset filter is applied. Defaults to 'USDT'.
        quote_asset_filter (str, optional): Filter by quote asset (e.g., 'BTC').
                                           If None, no quote asset filter is applied. Defaults to None.
        status_filter (str, optional): Filter by trading status (e.g., 'TRADING').
                                       If None, no status filter applied. Defaults to 'TRADING'.

    Returns:
        set: A set of valid symbol strings (e.g., {'BTCUSDT', 'ETHUSDT'}),
             or an empty set if an error occurs.
    """
    url = f"{DEFAULT_BINANCE_API_URL}{DEFAULT_EXCHANGE_INFO_ENDPOINT}"
    try:
        response = requests.get(url, timeout=DEFAULT_REQUEST_TIMEOUT)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        data = response.json()

        symbols = set()
        for s_info in data.get('symbols', []):
            symbol_name = s_info.get('symbol')

            passes_base_filter = True
            if base_asset_filter and s_info.get('quoteAsset') != base_asset_filter:  # common mistake: baseAsset vs quoteAsset
                passes_base_filter = False  # For pairs like BTCUSDT, USDT is quoteAsset
                # If we want USDT pairs, we filter by quoteAsset='USDT'

            passes_quote_filter = True  # If filtering by quote, e.g. BTC in BTCUSDT, ETHBTC
            if quote_asset_filter and s_info.get('baseAsset') != quote_asset_filter:
                passes_quote_filter = False

            passes_status_filter = True
            if status_filter and s_info.get('status') != status_filter:
                passes_status_filter = False

            if symbol_name and passes_base_filter and passes_quote_filter and passes_status_filter:
                symbols.add(symbol_name)

        if symbols:
            logger.info(f"Fetched {len(symbols)} symbols from Binance matching criteria "
                        f"(base_asset: {base_asset_filter}, quote_asset: {quote_asset_filter}, status: {status_filter}).")
        else:
            logger.warning(f"No symbols found matching criteria on Binance "
                           f"(base_asset: {base_asset_filter}, quote_asset: {quote_asset_filter}, status: {status_filter}).")
        return symbols
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching exchange info from Binance: {e}")
        return set()
    except Exception as e:
        logger.error(f"Unexpected error parsing exchange info from Binance: {e}")
        return set()


def fetch_klines_binance(
        symbol,
        interval_str,
        api_url_override=None,  # Allows using a different klines endpoint if needed
        start_time_ms=None,
        end_time_ms=None,
        limit=1000,
        max_retries=DEFAULT_MAX_RETRIES,
        retry_delay_s=DEFAULT_RETRY_DELAY,
        rate_limit_pause_s=DEFAULT_RATE_LIMIT_PAUSE
):
    """
    Fetches K-line (candle) data from Binance API for a given symbol and interval.

    Args:
        symbol (str): The trading symbol (e.g., 'BTCUSDT').
        interval_str (str): The K-line interval (e.g., '1m', '5m', '1h').
        api_url_override (str, optional): Full URL for klines if not using default.
        start_time_ms (int, optional): Start time in milliseconds since epoch.
        end_time_ms (int, optional): End time in milliseconds since epoch.
        limit (int, optional): Max number of K-lines to return (Binance API limit is 1000).
                               Defaults to 1000.
        max_retries (int, optional): Maximum number of retries on request failure.
        retry_delay_s (int, optional): Delay in seconds between retries.
        rate_limit_pause_s (int, optional): Pause duration in seconds if a rate limit error (429/418) occurs.

    Returns:
        list: A list of K-line data, where each K-line is a list of values
              (timestamp, open, high, low, close, volume, ...).
              Returns an empty list if an error occurs or no data is found.
    """
    klines_url = api_url_override if api_url_override else f"{DEFAULT_BINANCE_API_URL}{DEFAULT_KLINES_ENDPOINT}"

    params = {
        'symbol': symbol.upper(),  # Ensure symbol is uppercase
        'interval': interval_str,
        'limit': min(limit, 1000)  # Adhere to Binance API limit
    }
    if start_time_ms is not None:
        params['startTime'] = int(start_time_ms)
    if end_time_ms is not None:
        params['endTime'] = int(end_time_ms)

    for attempt in range(max_retries):
        try:
            # logger.debug(f"Requesting K-lines: {klines_url} with params {params} (Attempt {attempt + 1})")
            response = requests.get(klines_url, params=params, timeout=DEFAULT_REQUEST_TIMEOUT)

            # Handle rate limiting (429 Too Many Requests, 418 I'm a teapot - used by Binance for IP bans)
            if response.status_code == 429 or \
                    (response.status_code == 418 and "ban" in response.text.lower()):  # Check for ban message

                pause_duration = rate_limit_pause_s
                try:  # Try to get Retry-After header
                    retry_after_header = int(response.headers.get("Retry-After", rate_limit_pause_s))
                    pause_duration = max(retry_after_header, pause_duration)
                except (ValueError, TypeError):
                    pass  # Use default pause_duration

                logger.warning(
                    f"[{symbol}-{interval_str}] Rate limit hit (HTTP {response.status_code}). "
                    f"Pausing for {pause_duration} seconds. Attempt {attempt + 1}/{max_retries}."
                )
                time.sleep(pause_duration)
                continue  # Retry the same request after pause

            response.raise_for_status()  # Raise HTTPError for other bad responses (4xx or 5xx)
            klines_data = response.json()

            # Log if empty data is returned for a successful request
            if not klines_data and start_time_ms:
                logger.debug(f"[{symbol}-{interval_str}] No klines data returned by Binance for start_time "
                             f"{datetime.fromtimestamp(start_time_ms / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M')}.")
            elif klines_data:
                logger.debug(f"[{symbol}-{interval_str}] Fetched {len(klines_data)} klines.")

            return klines_data

        except requests.exceptions.HTTPError as http_err:
            # Specific handling for common client errors if needed, e.g., 400 Bad Request (invalid symbol/interval)
            if response and response.status_code == 400:
                logger.error(f"[{symbol}-{interval_str}] HTTP 400 Bad Request: {http_err} - {response.text}. Likely invalid symbol or parameters. Halting retries for this request.")
                return []  # Do not retry on 400 errors for this specific request
            logger.warning(
                f"[{symbol}-{interval_str}] HTTP error: {http_err} - "
                f"{response.text if response else 'No response object'}. Attempt {attempt + 1}/{max_retries}."
            )
        except requests.exceptions.RequestException as req_err:  # Includes ConnectionError, Timeout, etc.
            logger.warning(
                f"[{symbol}-{interval_str}] Request error: {req_err}. Attempt {attempt + 1}/{max_retries}."
            )
        except Exception as e:  # Catch any other unexpected errors during request or JSON parsing
            logger.error(
                f"[{symbol}-{interval_str}] Unexpected error during kline fetch: {e}. Attempt {attempt + 1}/{max_retries}.",
                exc_info=True
            )

        # If not the last attempt, wait and retry
        if attempt < max_retries - 1:
            current_actual_delay = retry_delay_s * (2 ** attempt)  # Exponential backoff
            logger.info(
                f"[{symbol}-{interval_str}] Retrying in {current_actual_delay:.1f} seconds... "
                f"(Attempt {attempt + 2}/{max_retries})"
            )
            time.sleep(current_actual_delay)
        else:
            logger.error(
                f"[{symbol}-{interval_str}] Failed to fetch klines after {max_retries} attempts."
            )

    return []  # Return empty list if all retries fail


# Example usage (if run directly for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    # Test get_valid_symbols_binance
    print("\n--- Testing get_valid_symbols_binance (USDT pairs) ---")
    usdt_symbols = get_valid_symbols_binance(base_asset_filter='USDT')  # Corrected: USDT is quoteAsset
    if usdt_symbols:
        print(f"Found {len(usdt_symbols)} USDT pairs. First 5: {list(usdt_symbols)[:5]}")
    else:
        print("No USDT pairs found or error occurred.")

    print("\n--- Testing get_valid_symbols_binance (BTC quote pairs) ---")
    btc_quote_symbols = get_valid_symbols_binance(base_asset_filter=None, quote_asset_filter='BTC')
    if btc_quote_symbols:
        print(f"Found {len(btc_quote_symbols)} BTC quote pairs. First 5: {list(btc_quote_symbols)[:5]}")
    else:
        print("No BTC quote pairs found or error occurred.")

    # Test fetch_klines_binance
    print("\n--- Testing fetch_klines_binance (BTCUSDT 1m) ---")
    # Fetch a few recent klines for BTCUSDT on 1m timeframe
    # To get recent klines, don't specify startTime or endTime, just limit
    btcusdt_klines_recent = fetch_klines_binance('BTCUSDT', '1m', limit=5)
    if btcusdt_klines_recent:
        print(f"Fetched {len(btcusdt_klines_recent)} recent 1m klines for BTCUSDT:")
        for kline in btcusdt_klines_recent:
            # Timestamp, Open, High, Low, Close, Volume, ...
            ts = datetime.fromtimestamp(kline[0] / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  Time: {ts}, Open: {kline[1]}, Close: {kline[4]}, Volume: {kline[5]}")
    else:
        print("Failed to fetch recent BTCUSDT 1m klines.")

    print("\n--- Testing fetch_klines_binance (ETHUSDT 15m, specific start time) ---")
    # Example start time: 2 days ago
    two_days_ago_ms = int((datetime.now(timezone.utc) - pd.Timedelta(days=2)).timestamp() * 1000)
    ethusdt_klines_historical = fetch_klines_binance('ETHUSDT', '15m', start_time_ms=two_days_ago_ms, limit=3)
    if ethusdt_klines_historical:
        print(f"Fetched {len(ethusdt_klines_historical)} historical 15m klines for ETHUSDT:")
        for kline in ethusdt_klines_historical:
            ts = datetime.fromtimestamp(kline[0] / 1000, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  Time: {ts}, Open: {kline[1]}, Close: {kline[4]}")
    else:
        print("Failed to fetch historical ETHUSDT 15m klines.")

    print("\n--- Testing fetch_klines_binance (Invalid Symbol) ---")
    invalid_symbol_klines = fetch_klines_binance('INVALIDCOINXYZ', '1m', limit=5)
    if not invalid_symbol_klines:
        print("Correctly returned empty list for invalid symbol.")
    else:
        print(f"ERROR: Unexpectedly got data for invalid symbol: {invalid_symbol_klines}")
