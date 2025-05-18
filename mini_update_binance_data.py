import requests
import sqlite3
import time
from datetime import datetime, timedelta, timezone
import os
import argparse
from tqdm import tqdm
import logging
import sys # Для sys.exit при Ctrl+C

# Создаем директорию data, если ее нет (для update_log.txt)
os.makedirs('data', exist_ok=True)
# Создаем директорию database, если ее нет
os.makedirs('database', exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s — %(message)s',
    handlers=[
        logging.FileHandler("data/update_log.txt", encoding='utf-8'), # Лог обновлений остается в data
        logging.StreamHandler()
    ]
)

CORE_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT',
    'BCHUSDT', 'LINKUSDT', 'DOTUSDT', 'ADAUSDT', 'SOLUSDT',
    'AVAXUSDT', 'TRXUSDT', 'UNIUSDT'
]

TIMEFRAMES = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '4h': '4h',
    '1d': '1d'
}

TF_HOURS_BACK = {
    '1m': 5 * 24,
    '5m': 14 * 24,
    '15m': 30 * 24,
    '30m': 30 * 24,
    '1h': 90 * 24,
    '4h': 180 * 24,
    '1d': 365 * 3 * 24
}

DB_PATH = 'database/market_data.db' # ИЗМЕНЕНО
BINANCE_URL = 'https://api.binance.com/api/v3/klines'
CANDLE_LIMIT = 1000

def get_tf_ms(tf):
    mult = {'m': 60, 'h': 3600, 'd': 86400}
    num_str = ''.join(filter(str.isdigit, tf))
    unit = ''.join(filter(str.isalpha, tf))
    return int(num_str) * mult[unit] * 1000

def init_db():
    # Убедимся, что директория для БД существует
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for tf_val in TIMEFRAMES: # Используем TIMEFRAMES.keys() или TIMEFRAMES.values() в зависимости от структуры
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS candles_{tf_val} (
                symbol TEXT,
                timestamp INTEGER,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                PRIMARY KEY (symbol, timestamp)
            )
        ''')
    conn.commit()
    conn.close()
    logging.info(f"База данных '{DB_PATH}' инициализирована/проверена.")

def fetch_klines(symbol, tf_interval, start_time): # tf переименован в tf_interval во избежание конфликта
    params = {
        'symbol': symbol,
        'interval': tf_interval,
        'startTime': start_time,
        'limit': CANDLE_LIMIT
    }
    try:
        response = requests.get(BINANCE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e: # Более общий обработчик для всех ошибок requests
        logging.error(f"Ошибка при запросе {symbol}-{tf_interval} (startTime: {start_time}): {e}")
        return []
    except Exception as e:
        logging.error(f"Неожиданная ошибка при запросе {symbol}-{tf_interval}: {e}")
        return []


def insert_klines(conn, tf_key, symbol, klines): # tf переименован в tf_key
    cursor = conn.cursor()
    count = 0
    for k in klines:
        try:
            cursor.execute(f'''
                INSERT OR IGNORE INTO candles_{tf_key}
                (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
            if cursor.rowcount > 0:
                count += 1
        except Exception as e:
            logging.error(f"Ошибка вставки свечи {symbol}-{tf_key}-{k[0]}: {e}")
    conn.commit()
    return count

def update_recent(tf_key): # tf переименован в tf_key
    logging.info(f"[Mini Update] ⏳ Начало инкрементального обновления для таймфрейма: {tf_key}")
    init_db() # Убедимся, что БД и таблицы существуют
    conn = sqlite3.connect(DB_PATH)
    # Используем tf_key для получения значения интервала из TIMEFRAMES
    interval_str = TIMEFRAMES.get(tf_key)
    if not interval_str:
        logging.error(f"Неверный ключ таймфрейма: {tf_key}. Пропуск.")
        conn.close()
        return

    step_ms = get_tf_ms(tf_key) # tf_key это '1m', '5m' и т.д.
    hours_back = TF_HOURS_BACK.get(tf_key, 48) # 48 часов по умолчанию, если tf_key не найден
    start_ts = int((datetime.utcnow() - timedelta(hours=hours_back)).timestamp() * 1000)

    logging.info(f"Обновление {tf_key} за последние {hours_back} часов (с {datetime.fromtimestamp(start_ts/1000, timezone.utc)} UTC)")

    for symbol in tqdm(CORE_SYMBOLS, desc=f"Обновление {tf_key}", unit="symbol"):
        try:
            klines = fetch_klines(symbol, interval_str, start_ts)
            if klines:
                inserted_count = insert_klines(conn, tf_key, symbol, klines) # Передаем tf_key для имени таблицы
                if inserted_count > 0:
                    logging.info(f"{symbol} [{tf_key}] — Добавлено: {inserted_count} свечей")
                # else:
                #     logging.debug(f"{symbol} [{tf_key}] — Нет новых уникальных свечей для добавления.")
            else:
                logging.warning(f"{symbol} [{tf_key}] — Нет данных от API (возможно, нет новых свечей или ошибка запроса)")
        except Exception as e:
            logging.error(f"Критическая ошибка при обработке {symbol} [{tf_key}]: {e}", exc_info=True)
            continue # Продолжаем со следующим символом

    conn.close()
    logging.info(f"[Mini Update] ✅ Инкрементальное обновление {tf_key} завершено.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Обновление только последних N часов по таймфреймам")
    parser.add_argument('--tf', nargs='*', default=list(TIMEFRAMES.keys()), # По умолчанию все таймфреймы
                        help=f"Таймфреймы для обработки (например: --tf 1m 5m 15m). Доступные: {', '.join(TIMEFRAMES.keys())}")
    args = parser.parse_args()

    selected_timeframes = args.tf
    if not selected_timeframes:
        logging.warning("Не указаны таймфреймы для обновления. Используйте --tf или запустите без аргументов для всех.")
        sys.exit(1)

    print(f"Запуск mini_update для таймфреймов: {', '.join(selected_timeframes)}")
    try:
        for tf_arg in selected_timeframes:
            if tf_arg in TIMEFRAMES:
                update_recent(tf_arg)
            else:
                logging.error(f"Неизвестный таймфрейм: {tf_arg}. Пропуск. Доступные: {', '.join(TIMEFRAMES.keys())}")
    except KeyboardInterrupt:
        print("\n[Mini Update] 🛑 Обновление прервано пользователем (Ctrl+C).")
        sys.exit(130) # Код для корректного перехвата в main_cli.py
    except Exception as e:
        logging.error(f"[Mini Update] 💥 Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)
    print("Все указанные таймфреймы обработаны.")