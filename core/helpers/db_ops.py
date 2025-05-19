# core/helpers/db_ops.py
"""
Helper functions for database operations, primarily interacting with an SQLite database
to store and retrieve market candle data.
"""
import sqlite3
import os
import logging
from datetime import datetime, timezone  # For logging and type hints

logger = logging.getLogger(__name__)


def init_db(db_path, timeframes_map):
    """
    Initializes the SQLite database and creates tables for each timeframe if they don't exist.

    Args:
        db_path (str): The path to the SQLite database file.
        timeframes_map (dict): A dictionary mapping timeframe keys (e.g., '1m', '5m')
                               to their string representations (used for table names).
                               Example: {'1m': '1m', '5m': '5m', ...}

    Returns:
        bool: True if initialization was successful or DB already setup, False on error.
    """
    try:
        # Ensure the directory for the database exists
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        for tf_key in timeframes_map.keys():  # Use keys from the map for table names
            table_name = f"candles_{tf_key}"
            # SQL to create table with primary key on (symbol, timestamp) to prevent duplicates
            # and ensure efficient lookups.
            # Using REAL for open, high, low, close, volume as they can be floating point.
            # INTEGER for timestamp (milliseconds since epoch).
            create_table_sql = f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            '''
            cursor.execute(create_table_sql)
            logger.debug(f"Table '{table_name}' checked/created successfully.")

        conn.commit()
        logger.info(f"Database '{db_path}' initialized/verified successfully for timeframes: {list(timeframes_map.keys())}.")
        return True
    except sqlite3.Error as e:
        logger.error(f"SQLite error during database initialization at '{db_path}': {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error during database initialization at '{db_path}': {e}", exc_info=True)
        return False
    finally:
        if 'conn' in locals() and conn:
            conn.close()


def insert_klines_sqlite(connection, timeframe_key, symbol, klines_data):
    """
    Inserts or ignores K-line data into the specified timeframe table in the SQLite database.
    Uses 'INSERT OR IGNORE' to avoid errors on duplicate entries.

    Args:
        connection (sqlite3.Connection): An active SQLite connection object.
        timeframe_key (str): The key for the timeframe (e.g., '1m', '15m'), used to determine table name.
        symbol (str): The trading symbol (e.g., 'BTCUSDT').
        klines_data (list): A list of K-line entries. Each entry is expected to be a list/tuple
                            where k[0] is timestamp (ms), k[1-4] are o,h,l,c, k[5] is volume.

    Returns:
        tuple: (count_inserted, first_inserted_ts, last_inserted_ts)
               count_inserted (int): Number of new rows actually inserted.
               first_inserted_ts (int or None): Timestamp of the first kline in the batch.
               last_inserted_ts (int or None): Timestamp of the last kline in the batch.
    """
    table_name = f"candles_{timeframe_key}"
    cursor = connection.cursor()

    rows_to_insert = []
    first_ts = None
    last_ts = None

    if klines_data:
        first_ts = int(klines_data[0][0])
        last_ts = int(klines_data[-1][0])
        for kline in klines_data:
            try:
                # Ensure data types are correct before insertion
                # kline format: [timestamp, open, high, low, close, volume, close_time, quote_asset_vol, num_trades, taker_buy_base, taker_buy_quote, ignore]
                row = (
                    symbol.upper(),
                    int(kline[0]),  # timestamp
                    float(kline[1]),  # open
                    float(kline[2]),  # high
                    float(kline[3]),  # low
                    float(kline[4]),  # close
                    float(kline[5])  # volume
                )
                rows_to_insert.append(row)
            except (IndexError, ValueError) as e:
                logger.warning(f"Skipping invalid kline data for {symbol} @ {kline[0] if kline else 'N/A'}: {kline}. Error: {e}")
                continue  # Skip this kline

    if not rows_to_insert:
        return 0, first_ts, last_ts

    insert_sql = f'''
        INSERT OR IGNORE INTO {table_name}
        (symbol, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    '''

    count_actually_inserted = 0
    try:
        # Using executemany for potentially better performance with many rows,
        # but rowcount might not be reliable across all SQLite versions/drivers for executemany.
        # A loop with individual execute and checking cursor.rowcount is more robust for counting.
        for row_data in rows_to_insert:
            cursor.execute(insert_sql, row_data)
            if cursor.rowcount > 0:
                count_actually_inserted += 1
        connection.commit()
        if count_actually_inserted > 0:
            logger.debug(f"Inserted {count_actually_inserted}/{len(rows_to_insert)} new klines for {symbol} into {table_name}.")
    except sqlite3.Error as e:
        logger.error(f"SQLite error inserting klines for {symbol} into {table_name}: {e}", exc_info=True)
        # Optionally rollback, though with INSERT OR IGNORE, partial success is possible.
        # connection.rollback()
        return 0, first_ts, last_ts  # Return 0 on error to signify failure for this batch
    except Exception as e:
        logger.error(f"Unexpected error inserting klines for {symbol} into {table_name}: {e}", exc_info=True)
        return 0, first_ts, last_ts

    return count_actually_inserted, first_ts, last_ts


def get_last_timestamp_sqlite(connection, timeframe_key, symbol):
    """
    Retrieves the most recent (max) timestamp for a given symbol and timeframe table.

    Args:
        connection (sqlite3.Connection): An active SQLite connection object.
        timeframe_key (str): The key for the timeframe, used to determine table name.
        symbol (str): The trading symbol.

    Returns:
        int or None: The last timestamp (milliseconds since epoch) or None if no data/error.
    """
    table_name = f"candles_{timeframe_key}"
    cursor = connection.cursor()
    query = f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = ?"
    try:
        cursor.execute(query, (symbol.upper(),))
        result = cursor.fetchone()
        if result and result[0] is not None:
            return int(result[0])
        return None
    except sqlite3.Error as e:
        logger.error(f"SQLite error fetching max timestamp for {symbol} from {table_name}: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching max timestamp for {symbol} from {table_name}: {e}", exc_info=True)
        return None


def load_candles_from_sqlite(db_path, timeframe_key, symbol_filter=None):
    """
    Loads candle data from the SQLite database for a specific timeframe,
    optionally filtered by symbol.

    Args:
        db_path (str): Path to the SQLite database file.
        timeframe_key (str): The key for the timeframe (e.g., '1m', '15m').
        symbol_filter (str or list, optional): A single symbol string or a list of symbol strings
                                              to filter the data. If None, loads data for all symbols.

    Returns:
        pd.DataFrame: A DataFrame containing the candle data, sorted by symbol and timestamp.
                      Returns an empty DataFrame if an error occurs, table/DB not found, or no data.
    """
    table_name = f"candles_{timeframe_key}"
    logger.info(f"Loading candles from DB table: {table_name}, Symbol filter: {symbol_filter or 'All'}")

    if not os.path.exists(db_path):
        logger.error(f"Database file not found: {db_path}")
        return pd.DataFrame()

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if table exists first
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if cursor.fetchone() is None:
            logger.error(f"Table '{table_name}' not found in database {db_path}.")
            conn.close()
            return pd.DataFrame()

        query = f"SELECT * FROM {table_name}"
        params = []

        if symbol_filter:
            if isinstance(symbol_filter, str):
                query += " WHERE symbol = ?"
                params.append(symbol_filter.upper())
            elif isinstance(symbol_filter, list) and symbol_filter:
                placeholders = ', '.join('?' for _ in symbol_filter)
                query += f" WHERE symbol IN ({placeholders})"
                params.extend([s.upper() for s in symbol_filter])
            else:
                logger.warning(f"Invalid symbol_filter type: {type(symbol_filter)}. Loading all symbols.")

        df = pd.read_sql_query(query, conn, params=params if params else None)

        if df.empty:
            logger.warning(f"No data found in table '{table_name}' matching filter '{symbol_filter}'.")
            return pd.DataFrame()

        # Standardize data types and sort
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        # Ensure numeric columns are indeed numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')  # Coerce errors to NaT/NaN

        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        logger.info(f"Loaded {len(df)} rows from '{table_name}' for {df['symbol'].nunique()} symbols.")
        return df

    except sqlite3.Error as e:
        logger.error(f"SQLite error loading candles from {table_name}: {e}", exc_info=True)
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading candles from {table_name}: {e}", exc_info=True)
        return pd.DataFrame()
    finally:
        if 'conn' in locals() and conn:
            conn.close()


# Example usage (if run directly for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    test_db_path = "temp_test_db_ops.sqlite"
    test_timeframes = {'1m': '1m', '5m_test': '5m_test'}  # Using a distinct name for 5m to avoid conflicts

    print(f"\n--- Testing Database Operations (DB: {test_db_path}) ---")

    # 1. Test init_db
    if os.path.exists(test_db_path): os.remove(test_db_path)  # Clean start

    if init_db(test_db_path, test_timeframes):
        print("Database initialized successfully.")

        # 2. Test insert_klines_sqlite
        conn_test = sqlite3.connect(test_db_path)

        # Sample kline data (timestamp, o, h, l, c, v)
        # Timestamps should be unique for a symbol to test INSERT OR IGNORE
        sample_klines_btc_1m = [
            [int(datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000), 20000, 20010, 19990, 20005, 10.5],
            [int(datetime(2023, 1, 1, 10, 1, 0, tzinfo=timezone.utc).timestamp() * 1000), 20005, 20015, 20000, 20010, 12.3],
            [int(datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000), 20000, 20010, 19990, 20005, 10.5],  # Duplicate
        ]
        sample_klines_eth_1m = [
            [int(datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc).timestamp() * 1000), 1500, 1502, 1498, 1501, 50.1],
        ]

        inserted_btc, _, _ = insert_klines_sqlite(conn_test, '1m', 'BTCUSDT', sample_klines_btc_1m)
        print(f"Inserted {inserted_btc} new klines for BTCUSDT 1m (expected 2 due to duplicate).")

        inserted_eth, _, _ = insert_klines_sqlite(conn_test, '1m', 'ETHUSDT', sample_klines_eth_1m)
        print(f"Inserted {inserted_eth} new klines for ETHUSDT 1m (expected 1).")

        # 3. Test get_last_timestamp_sqlite
        last_ts_btc = get_last_timestamp_sqlite(conn_test, '1m', 'BTCUSDT')
        expected_last_ts_btc = sample_klines_btc_1m[1][0]
        print(f"Last timestamp for BTCUSDT 1m: {last_ts_btc} (Expected: {expected_last_ts_btc})")
        assert last_ts_btc == expected_last_ts_btc

        last_ts_nonexist = get_last_timestamp_sqlite(conn_test, '1m', 'XYZUSDT')
        print(f"Last timestamp for XYZUSDT 1m: {last_ts_nonexist} (Expected: None)")
        assert last_ts_nonexist is None

        conn_test.close()  # Close connection before loading data with a new one

        # 4. Test load_candles_from_sqlite
        print("\n--- Loading all 1m data ---")
        df_1m_all = load_candles_from_sqlite(test_db_path, '1m')
        if not df_1m_all.empty:
            print(f"Loaded {len(df_1m_all)} rows for 1m (all symbols). Unique symbols: {df_1m_all['symbol'].unique()}")
            print(df_1m_all.head())
            assert len(df_1m_all) == 3  # 2 BTC + 1 ETH
        else:
            print("Failed to load 1m data or it's empty.")

        print("\n--- Loading BTCUSDT 1m data ---")
        df_1m_btc = load_candles_from_sqlite(test_db_path, '1m', symbol_filter='BTCUSDT')
        if not df_1m_btc.empty:
            print(f"Loaded {len(df_1m_btc)} rows for BTCUSDT 1m.")
            print(df_1m_btc.head())
            assert len(df_1m_btc) == 2
        else:
            print("Failed to load BTCUSDT 1m data.")

        print("\n--- Loading multiple symbols [BTCUSDT, ETHUSDT] 1m data ---")
        df_1m_multi = load_candles_from_sqlite(test_db_path, '1m', symbol_filter=['BTCUSDT', 'ETHUSDT'])
        if not df_1m_multi.empty:
            print(f"Loaded {len(df_1m_multi)} rows for BTCUSDT, ETHUSDT 1m. Unique symbols: {df_1m_multi['symbol'].unique()}")
            assert len(df_1m_multi) == 3
        else:
            print("Failed to load multi-symbol 1m data.")

        print("\n--- Loading data from non-existent table (candles_5m_test) ---")
        df_5m_empty = load_candles_from_sqlite(test_db_path, '5m_test')  # Table exists but is empty
        if df_5m_empty.empty:
            print("Correctly returned empty DataFrame for empty table 'candles_5m_test'.")
        else:
            print(f"ERROR: Expected empty DataFrame for 'candles_5m_test', got {len(df_5m_empty)} rows.")

    else:
        print("Database initialization failed. Further tests skipped.")

    # Clean up
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
        print(f"\nCleaned up test database: {test_db_path}")
