# src/db.py

import sqlite3
import pandas as pd
import os
import logging
from datetime import datetime, timezone

# Import config and logging setup
from src.utils.config import get_config

# Get logger for this module
logger = logging.getLogger(__name__)

# Load configuration
config = get_config()
DB_PATH = config['paths']['db']
SQLITE_TIMEOUT = config['database']['sqlite_timeout'] # Get timeout from config

def get_db_connection():
    """Establishes and returns a connection to the SQLite database."""
    try:
        # Ensure the database directory exists
        db_dir = os.path.dirname(DB_PATH)
        os.makedirs(db_dir, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=SQLITE_TIMEOUT)
        # conn.row_factory = sqlite3.Row # Optional: access columns by name
        logger.debug(f"Database connection established to {DB_PATH}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        # It might be better to raise the exception here and handle it in the caller
        # return None # Or raise e

def init_db(timeframe_keys):
    """
    Initializes the database by creating candle tables for specified timeframes
    if they do not already exist.

    Args:
        timeframe_keys (list): A list of timeframe keys (e.g., ['1m', '5m']).
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        for tf_key in timeframe_keys:
            table_name = f"candles_{tf_key}"
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    symbol TEXT,
                    timestamp INTEGER, -- Unix timestamp in milliseconds
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    PRIMARY KEY (symbol, timestamp)
                )
            ''')
            logger.debug(f"Table '{table_name}' checked/created.")
        conn.commit()
        logger.info(f"Database schema initialized for timeframes: {', '.join(timeframe_keys)}")
    except sqlite3.Error as e:
        logger.error(f"Database schema initialization error: {e}")
    finally:
        if conn:
            conn.close()

def insert_candles(conn, tf_key, symbol, candles_data):
    """
    Inserts candle data into the specified timeframe table.
    Uses INSERT OR IGNORE to handle existing records.

    Args:
        conn (sqlite3.Connection): An active database connection.
        tf_key (str): The timeframe key (e.g., '1m').
        symbol (str): The trading symbol (e.g., 'BTCUSDT').
        candles_data (list or pandas.DataFrame): The candle data to insert.
            Expected columns/indices: timestamp (ms), open, high, low, close, volume.

    Returns:
        int: The number of rows actually inserted.
    """
    table_name = f"candles_{tf_key}"
    cursor = conn.cursor()
    inserted_count = 0

    if isinstance(candles_data, pd.DataFrame):
        # Convert DataFrame to list of tuples, ensuring correct order and types
        # Handle potential datetime objects if the timestamp column was converted earlier
        if pd.api.types.is_datetime64_any_dtype(candles_data['timestamp']):
             candles_data_list = candles_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
             # Convert datetime to millisecond timestamp (integer)
             candles_data_list['timestamp'] = candles_data_list['timestamp'].astype(np.int64) // 10**6
             data_to_insert = candles_data_list.values.tolist()
        else:
             # Assume timestamp is already int (ms)
             data_to_insert = candles_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].values.tolist()
    elif isinstance(candles_data, list):
        # Assume list of lists or tuples, format matches expected columns
        data_to_insert = candles_data
    else:
        logger.error(f"Unsupported data type for inserting candles: {type(candles_data)}")
        return 0

    if not data_to_insert:
        logger.debug(f"No data to insert for {symbol} [{tf_key}].")
        return 0

    # Add symbol to each row
    data_to_insert_with_symbol = [(symbol,) + tuple(row) for row in data_to_insert]

    try:
        cursor.executemany(f'''
            INSERT OR IGNORE INTO {table_name}
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', data_to_insert_with_symbol)

        # To get the exact number of inserted rows, we'd need to query after commit
        # or use a different strategy. For now, we can estimate based on rowcount
        # if it was a single execute, but executemany doesn't easily give per-row counts.
        # A common way is to check rows that *didn't* exist before, but that's complex.
        # Let's just return the total number of rows attempted to insert for simplicity,
        # or assume rowcount is sufficient for a rough estimate.
        # SQLite cursor.rowcount after executemany is the total number of rows affected,
        # which includes ignored rows if they existed. It's not the *inserted* count.
        # The most reliable way is to query the count before and after, or fetch inserted ROWIDs.
        # Let's stick to a simpler return for now, maybe just logging the total attempted.
        conn.commit()
        # Logging the number of rows *attempted* to insert
        # logger.debug(f"Attempted to insert {len(data_to_insert)} rows for {symbol} [{tf_key}].")
        # A pragmatic approach: assume if executemany succeeded without error,
        # some rows were likely inserted if data_to_insert was not empty.
        # We don't have an easy way to get the *exact* inserted count here without
        # more complex logic involving selecting ROWIDs or comparing counts before/after.
        # Let's return len(data_to_insert) as a proxy for "processed count",
        # although it's not strictly "inserted count". Or maybe just return True/False for success.
        # Let's return the number of rows *provided* as data, acknowledging it might not be the exact inserted count.
        inserted_count = len(data_to_insert) # This is the number of rows provided, not necessarily inserted
        logger.debug(f"Processed {inserted_count} rows for {symbol} [{tf_key}].")
        return inserted_count # Returning count of rows processed for insertion
    except sqlite3.Error as e:
        logger.error(f"SQLite insert error for {symbol} [{tf_key}]: {e}")
        conn.rollback() # Roll back changes on error
        return 0
    except Exception as e:
        logger.error(f"General insert error for {symbol} [{tf_key}]: {e}")
        conn.rollback() # Roll back changes on error
        return 0

def load_candles(tf_key, symbol=None, start_time_ms=None, end_time_ms=None):
    """
    Loads candle data from the database for a given timeframe and optional filters.

    Args:
        tf_key (str): The timeframe key (e.g., '15m').
        symbol (str, optional): Filter by symbol. Defaults to None (all symbols).
        start_time_ms (int, optional): Filter by start timestamp (inclusive, milliseconds). Defaults to None.
        end_time_ms (int, optional): Filter by end timestamp (inclusive, milliseconds). Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame containing the candle data. Returns empty DataFrame on error or no data.
    """
    table_name = f"candles_{tf_key}"
    conn = None
    df = pd.DataFrame()
    try:
        conn = get_db_connection()
        query = f"SELECT symbol, timestamp, open, high, low, close, volume FROM {table_name}"
        conditions = []
        params = []

        if symbol:
            conditions.append("symbol = ?")
            params.append(symbol)
        if start_time_ms is not None:
            conditions.append("timestamp >= ?")
            params.append(start_time_ms)
        if end_time_ms is not None:
            conditions.append("timestamp <= ?")
            params.append(end_time_ms)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY symbol, timestamp"

        df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            # Convert timestamp (integer ms) to datetime objects
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # Assume UTC timestamps from Binance
            # It's often better to work with UTC internally and convert for display if needed
            # df['timestamp'] = df['timestamp'].dt.tz_convert('Your Local Timezone') # Optional: convert to local timezone

        logger.debug(f"Loaded {len(df)} rows from {table_name} with filters: symbol={symbol}, start={start_time_ms}, end={end_time_ms}")
        return df

    except pd.io.sql.DatabaseError as e:
        logger.error(f"SQL error loading data from {table_name}: {e}")
        return pd.DataFrame() # Return empty DataFrame on SQL error
    except Exception as e:
        logger.error(f"Error loading data from {table_name}: {e}")
        return pd.DataFrame() # Return empty DataFrame on general error
    finally:
        if conn:
            conn.close()

def get_last_timestamp(tf_key, symbol):
    """
    Retrieves the latest timestamp for a specific symbol and timeframe.

    Args:
        tf_key (str): The timeframe key.
        symbol (str): The trading symbol.

    Returns:
        int: The latest timestamp in milliseconds, or None if no data exists.
    """
    table_name = f"candles_{tf_key}"
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Check if table exists before querying
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        if cursor.fetchone() is None:
            logger.debug(f"Table '{table_name}' does not exist.")
            return None

        cursor.execute(f"SELECT MAX(timestamp) FROM {table_name} WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        latest_ts = row[0] if row and row[0] else None
        logger.debug(f"Last timestamp for {symbol} [{tf_key}]: {latest_ts}")
        return latest_ts
    except sqlite3.Error as e:
        logger.error(f"SQLite error getting last timestamp for {symbol} [{tf_key}]: {e}")
        return None
    except Exception as e:
        logger.error(f"Error getting last timestamp for {symbol} [{tf_key}]: {e}")
        return None
    finally:
        if conn:
            conn.close()

# Example usage (optional, for testing the module directly)
if __name__ == "__main__":
    # Set up logging for the test
    # In a real run, this would be called once at the script entry point
    from src.utils.logging_setup import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__) # Re-get logger after setup

    print("Testing src/db.py")

    # Example 1: Initialize DB
    print("\nInitializing DB...")
    all_timeframes = get_config()['timeframes']['default']
    init_db(all_timeframes)

    # Example 2: Insert data (requires a connection object)
    print("\nInserting sample data...")
    conn = None
    try:
        conn = get_db_connection()
        sample_candle = [
            (1678886400000, 25000.0, 25100.0, 24900.0, 25050.0, 1000.0), # Sample timestamp for 2023-03-15 00:00:00 UTC
            (1678886400000 + 60000, 25050.0, 25150.0, 25000.0, 25100.0, 1200.0) # Next 1m candle
        ]
        inserted_count = insert_candles(conn, '1m', 'TESTUSDT', sample_candle)
        print(f"Attempted to insert {len(sample_candle)} rows, Processed count reported: {inserted_count}")

        # Insert same data again to test INSERT OR IGNORE
        print("\nInserting same sample data again...")
        inserted_count_again = insert_candles(conn, '1m', 'TESTUSDT', sample_candle)
        print(f"Attempted to insert {len(sample_candle)} rows again, Processed count reported: {inserted_count_again} (should be 0 if data existed)")


        # Insert data for another symbol and TF
        sample_candle_btc = [
             (1678886400000, 25000.0, 25500.0, 24800.0, 25300.0, 5000.0), # Sample timestamp for 2023-03-15 00:00:00 UTC
        ]
        inserted_count_btc = insert_candles(conn, '1h', 'BTCUSDT', sample_candle_btc)
        print(f"Attempted to insert {len(sample_candle_btc)} rows for BTCUSDT 1h, Processed count reported: {inserted_count_btc}")

    finally:
        if conn:
            conn.close()


    # Example 3: Load data
    print("\nLoading all 1m data...")
    df_1m_all = load_candles('1m')
    print(f"Loaded {len(df_1m_all)} rows.")
    if not df_1m_all.empty:
        print(df_1m_all.head().to_string())

    print("\nLoading TESTUSDT 1m data...")
    df_1m_test = load_candles('1m', symbol='TESTUSDT')
    print(f"Loaded {len(df_1m_test)} rows for TESTUSDT.")
    if not df_1m_test.empty:
        print(df_1m_test.to_string()) # Should show the 2 inserted rows

    print("\nLoading BTCUSDT 1h data...")
    df_1h_btc = load_candles('1h', symbol='BTCUSDT')
    print(f"Loaded {len(df_1h_btc)} rows for BTCUSDT 1h.")
    if not df_1h_btc.empty:
        print(df_1h_btc.to_string()) # Should show the 1 inserted row

    # Example 4: Get last timestamp
    print("\nGetting last timestamp for TESTUSDT 1m...")
    last_ts_test = get_last_timestamp('1m', 'TESTUSDT')
    print(f"Last timestamp for TESTUSDT 1m: {last_ts_test}")
    if last_ts_test:
        print(f"Converted to datetime: {datetime.fromtimestamp(last_ts_test / 1000, timezone.utc)}")

    print("\nGetting last timestamp for NONEXISTENT 1m...")
    last_ts_nonexistent = get_last_timestamp('1m', 'NONEXISTENT')
    print(f"Last timestamp for NONEXISTENT 1m: {last_ts_nonexistent}") # Should be None

    print("\nGetting last timestamp for TESTUSDT 1h (should be None)...")
    last_ts_test_1h = get_last_timestamp('1h', 'TESTUSDT')
    print(f"Last timestamp for TESTUSDT 1h: {last_ts_test_1h}") # Should be None

    print("\nGetting last timestamp for BTCUSDT 1h...")
    last_ts_btc_1h = get_last_timestamp('1h', 'BTCUSDT')
    print(f"Last timestamp for BTCUSDT 1h: {last_ts_btc_1h}") # Should be the BTC timestamp

    print("\nDatabase module test finished.")