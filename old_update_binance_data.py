import requests
import sqlite3
import time
from datetime import datetime, timezone
import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import sys  # Для sys.exit при Ctrl+C

# Создаём директорию для данных (update_log.txt) и базы данных
os.makedirs('data', exist_ok=True)
os.makedirs('database', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s — %(message)s',
    handlers=[
        logging.FileHandler("data/update_log.txt", encoding='utf-8'),  # Лог обновлений остается в data
        logging.StreamHandler()
    ]
)

CORE_SYMBOLS = [

    'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'BCHUSDT', 'LTCUSDT',
    'BTCUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'XRPUSDT',
    'MATICUSDT', 'LINKUSDT', 'NEARUSDT', 'ATOMUSDT', 'TRXUSDT'
]

TIMEFRAMES = {

    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',  
    '4h': '4h',
    '1d': '1d'
}

DB_PATH = 'database/market_data.db'  # ИЗМЕНЕНО
BINANCE_URL = 'https://api.binance.com/api/v3/klines'
MAX_CANDLES_PER_SYMBOL = 100000
DEFAULT_START_DATE_MS = int(
    datetime(2017, 8, 17, tzinfo=timezone.utc).timestamp() * 1000)  # Binance launch date for many pairs


def get_valid_symbols():
    url = 'https://api.binance.com/api/v3/exchangeInfo'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        symbols = {s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT' and s['status'] == 'TRADING'}
        logging.info(f"Получено {len(symbols)} активных USDT символов с Binance.")
        return symbols
    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка при получении списка символов: {e}")
        return set()
    except Exception as e:
        logging.error(f"Неожиданная ошибка при получении списка символов: {e}")
        return set()


def get_tf_ms(tf_key):
    mult = {'m': 60, 'h': 3600, 'd': 86400}
    num_str = ''.join(filter(str.isdigit, tf_key))
    if not num_str:
        logging.error(f"Не удалось извлечь число из таймфрейма: {tf_key}")
        return 0
    num = int(num_str)
    unit = ''.join(filter(str.isalpha, tf_key))
    if unit not in mult:
        logging.error(f"Неизвестная единица измерения в таймфрейме: {tf_key}")
        return 0
    return num * mult[unit] * 1000


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for tf_val in TIMEFRAMES:
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


def fetch_klines(symbol, interval_str, start_time=None, retries=3, delay=5):
    params = {
        'symbol': symbol,
        'interval': interval_str,
        'limit': 1000  # Binance API limit per request
    }
    if start_time:
        params['startTime'] = start_time

    for attempt in range(retries):
        try:
            # logging.debug(f"Запрос: {BINANCE_URL} с параметрами {params}")
            response = requests.get(BINANCE_URL, params=params, timeout=15)  # Увеличил таймаут

            if response.status_code == 429 or \
                    (response.status_code == 418 and "Too many requests" in response.text):
                ban_duration = 60
                try:
                    # Попытка извлечь время бана из заголовка Retry-After
                    retry_after = int(response.headers.get("Retry-After", ban_duration))
                    ban_duration = max(retry_after, ban_duration)  # Берем большее из двух
                except (ValueError, TypeError):
                    pass  # Используем ban_duration по умолчанию
                logging.warning(
                    f"[{symbol}-{interval_str}] Превышен лимит запросов (HTTP {response.status_code}) — пауза {ban_duration} секунд")
                time.sleep(ban_duration)
                continue  # Повторяем тот же запрос после паузы

            response.raise_for_status()  # Вызовет исключение для 4xx/5xx ошибок, кроме 429/418
            return response.json()

        except requests.exceptions.HTTPError as http_err:
            logging.warning(
                f"[{symbol}-{interval_str}] HTTP ошибка: {http_err} - {response.text if response else 'No response object'}")
        except requests.exceptions.RequestException as e:  # ConnectionError, Timeout, etc.
            logging.warning(f"[{symbol}-{interval_str}] Ошибка запроса: {e}")

        if attempt < retries - 1:
            current_delay = delay * (attempt + 1)
            logging.info(
                f"[{symbol}-{interval_str}] Повторная попытка через {current_delay} сек. ({attempt + 2}/{retries})")
            time.sleep(current_delay)
        else:
            logging.error(f"[{symbol}-{interval_str}] Не удалось получить данные после {retries} попыток.")
    return []


def insert_klines(conn, tf_key, symbol, klines):
    cursor = conn.cursor()
    count = 0
    start_ts_val = None
    end_ts_val = None

    if klines:
        start_ts_val = int(klines[0][0])
        end_ts_val = int(klines[-1][0])

        for k in klines:
            try:
                cursor.execute(f'''
                    INSERT OR IGNORE INTO candles_{tf_key}
                    (symbol, timestamp, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])))
                if cursor.rowcount > 0:
                    count += 1
            except sqlite3.Error as e:  # Более специфичная ошибка для SQLite
                logging.error(
                    f"Ошибка вставки SQLite для {symbol} в {datetime.fromtimestamp(int(k[0]) / 1000, timezone.utc)}: {e}")
            except Exception as e:
                logging.error(
                    f"Общая ошибка вставки для {symbol} в {datetime.fromtimestamp(int(k[0]) / 1000, timezone.utc)}: {e}")
    conn.commit()
    return count, start_ts_val, end_ts_val


def get_last_timestamp(conn, tf_key, symbol):
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT MAX(timestamp) FROM candles_{tf_key} WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        return row[0] if row and row[0] else None
    except sqlite3.Error as e:
        logging.error(f"Ошибка получения MAX(timestamp) для {symbol} [{tf_key}]: {e}")
        return None


def process_symbol_for_tf(symbol, tf_key, interval_str, step_ms, db_path_local, pbar_instance=None):
    # Эта функция будет выполняться в каждом потоке
    # Создаем собственное подключение к БД для каждого потока
    conn_local = sqlite3.connect(db_path_local, timeout=60)
    downloaded_for_symbol = 0

    try:
        last_ts = get_last_timestamp(conn_local, tf_key, symbol)
        next_ts = (last_ts + step_ms) if last_ts else DEFAULT_START_DATE_MS

        # logging.info(f"Для {symbol} [{tf_key}]: last_ts={last_ts}, next_ts={datetime.fromtimestamp(next_ts/1000, timezone.utc)}")

        while downloaded_for_symbol < MAX_CANDLES_PER_SYMBOL:
            if next_ts >= int(datetime.now(timezone.utc).timestamp() * 1000):  # Не запрашивать данные из будущего
                logging.info(f"{symbol} [{tf_key}] достигнуты текущие данные.")
                break

            klines = fetch_klines(symbol, interval_str, next_ts)
            if not klines:
                # logging.info(f"{symbol} [{tf_key}] — Binance не вернул данные для {datetime.fromtimestamp(next_ts / 1000, timezone.utc)} или API limit.")
                break  # Если нет данных, выходим из цикла для этого символа

            rows_inserted, _, _ = insert_klines(conn_local, tf_key, symbol, klines)
            downloaded_for_symbol += rows_inserted

            if pbar_instance and rows_inserted > 0:
                pbar_instance.update(rows_inserted)
            # elif pbar_instance and not rows_inserted and klines: # Обновляем прогресс даже если свечи уже были
            #      pbar_instance.update(len(klines))

            if len(klines) < 1000:  # Если получили меньше, чем запрашивали, значит, это конец истории
                logging.info(f"{symbol} [{tf_key}] получено {len(klines)} свечей, вероятно, достигнут конец истории.")
                break

            # Проверка на корректность timestamp последней свечи
            if not klines[-1] or not str(klines[-1][0]).isdigit():
                logging.error(f"Некорректный timestamp в последней свече для {symbol} [{tf_key}]: {klines[-1]}")
                break

            next_ts = int(klines[-1][0]) + step_ms
            time.sleep(0.2)  # Небольшая задержка между запросами для одного символа

        if downloaded_for_symbol > 0:
            logging.info(f"{symbol} [{tf_key}] — Загружено {downloaded_for_symbol} новых свечей.")
        return downloaded_for_symbol
    except Exception as e:
        logging.error(f"Ошибка при обновлении {symbol} [{tf_key}] в потоке: {e}", exc_info=True)
        return 0  # Возвращаем 0 в случае ошибки
    finally:
        conn_local.close()


def update_timeframe_parallel(tf_key):
    logging.info(f"[Old Update] ⏳  Запуск параллельной загрузки для таймфрейма: {tf_key}")
    if tf_key not in TIMEFRAMES:
        logging.error(f"Неверный таймфрейм: {tf_key}")
        return

    init_db()  # Инициализируем БД перед началом работы с таймфреймом

    step_ms = get_tf_ms(tf_key)
    if step_ms == 0:
        return
    interval_str = TIMEFRAMES[tf_key]

    valid_binance_symbols = get_valid_symbols()
    if not valid_binance_symbols:
        logging.warning("Не удалось получить валидные символы Binance. Обновление прервано.")
        return

    active_symbols = [s for s in CORE_SYMBOLS if s in valid_binance_symbols]
    if not active_symbols:
        logging.info(f"Нет активных символов из CORE_SYMBOLS для {tf_key}.")
        return

    logging.info(f"Будет обработано {len(active_symbols)} символов для {tf_key}: {', '.join(active_symbols[:5])}...")

    total_downloaded_for_tf = 0
    # Используем ThreadPoolExecutor для параллельной загрузки по символам
    # max_workers=5 чтобы не перегружать API Binance и локальные ресурсы
    with ThreadPoolExecutor(max_workers=5) as executor, \
            tqdm(total=len(active_symbols) * MAX_CANDLES_PER_SYMBOL,  # Примерная общая оценка для tqdm
                 desc=f"Обновление {tf_key}", unit=" свечей", mininterval=1.0) as pbar:

        futures = {executor.submit(process_symbol_for_tf, symbol, tf_key, interval_str, step_ms, DB_PATH, pbar): symbol
                   for symbol in active_symbols}

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                downloaded_count = future.result()
                total_downloaded_for_tf += downloaded_count
                # pbar.set_postfix_str(f"{symbol} done") # Можно добавить информацию о последнем обработанном символе
            except Exception as e:
                logging.error(f"Ошибка в потоке для символа {symbol} [{tf_key}]: {e}")

    logging.info(
        f"[Old Update] ✅ Обновление таймфрейма {tf_key} завершено. Всего загружено: {total_downloaded_for_tf} свечей.")


def update_single_symbol_tf(symbol_to_update, tf_key_to_update):
    logging.info(f"[Old Update] ⏳ Обновление для одного символа/таймфрейма: {symbol_to_update} [{tf_key_to_update}]")
    if tf_key_to_update not in TIMEFRAMES:
        logging.error(f"Неверный таймфрейм: {tf_key_to_update}")
        return

    symbol_to_update = symbol_to_update.upper()
    valid_binance_symbols = get_valid_symbols()
    if not valid_binance_symbols:
        logging.warning("Не удалось получить валидные символы, обновление одиночного символа прервано.")
        return
    if symbol_to_update not in valid_binance_symbols:
        logging.error(f"Символ {symbol_to_update} не торгуется или невалиден на Binance (USDT).")
        return

    init_db()
    step_ms = get_tf_ms(tf_key_to_update)
    if step_ms == 0: return
    interval_str = TIMEFRAMES[tf_key_to_update]

    # Для одного символа нет смысла в ThreadPoolExecutor, делаем последовательно
    # Используем tqdm для отображения прогресса загрузки свечей для этого одного символа
    with tqdm(total=MAX_CANDLES_PER_SYMBOL, desc=f"Загрузка {symbol_to_update} [{tf_key_to_update}]", unit=" свечей",
              mininterval=1.0) as pbar_single:
        downloaded_count = process_symbol_for_tf(symbol_to_update, tf_key_to_update, interval_str, step_ms, DB_PATH,
                                                 pbar_single)

    logging.info(
        f"[Old Update] ✅ Обновление {symbol_to_update} [{tf_key_to_update}] завершено. Загружено: {downloaded_count} свечей.")


def update_all_tf_sequentially():
    logging.info("[Old Update] ⏳ Запуск полного последовательного обновления всех таймфреймов...")
    init_db()
    # Нет смысла получать valid_binance_symbols здесь, так как update_timeframe_parallel сделает это для каждого TF
    for tf_key_loop in TIMEFRAMES.keys():
        # Вместо последовательной логики здесь, вызываем параллельную для каждого ТФ
        update_timeframe_parallel(tf_key_loop)
    logging.info("[Old Update] ✅ Полное обновление всех таймфреймов завершено.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Загрузчик исторических данных Binance Klines.")
    parser.add_argument('--symbol', type=str, help='Символ для обновления (например: BTCUSDT)')
    parser.add_argument('--tf', type=str, choices=TIMEFRAMES.keys(), help='Таймфрейм (например: 15m)')
    parser.add_argument('--all', action='store_true',
                        help='Запуск полного обновления всех CORE_SYMBOLS и всех таймфреймов (параллельно по символам внутри каждого ТФ)')
    parser.add_argument('--sequential-all-tf', action='store_true',
                        help='Запуск полного обновления всех таймфреймов, обрабатывая каждый таймфрейм последовательно (но символы внутри ТФ параллельно)')

    args = parser.parse_args()
    print(f"Запуск old_update_binance_data.py с аргументами: {args}")

    try:
        if args.all:
            logging.info("Запуск полного обновления (все CORE_SYMBOLS, все таймфреймы).")
            # update_all_tf_sequentially() # Старый вариант - теперь update_timeframe_parallel обрабатывает символы параллельно
            for tf_key_main in TIMEFRAMES.keys():
                update_timeframe_parallel(tf_key_main)


        elif args.symbol and args.tf:
            logging.info(f"Обновление конкретного символа {args.symbol} и таймфрейма {args.tf}")
            update_single_symbol_tf(args.symbol, args.tf)

        elif args.tf:  # Если указан только таймфрейм, обновляем все CORE_SYMBOLS для этого таймфрейма
            logging.info(f"Обновление всех CORE_SYMBOLS по таймфрейму {args.tf}")
            update_timeframe_parallel(args.tf)

        elif args.sequential_all_tf:  # Отдельный флаг для этой логики
            logging.info(
                "Запуск полного обновления всех таймфреймов (последовательно по ТФ, параллельно по символам внутри ТФ)")
            update_all_tf_sequentially()

        elif args.symbol:
            logging.warning(f"Указан символ '{args.symbol}', но не указан таймфрейм. Используйте --tf.")
            logging.info(f"Доступные таймфреймы: {', '.join(TIMEFRAMES.keys())}")

        else:
            parser.print_help()
            logging.warning(
                "Не указаны достаточные аргументы. Используйте --all, или --symbol/--tf, или --tf, или --help.")

    except KeyboardInterrupt:
        print("\n[Old Update] 🛑 Обновление прервано пользователем (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logging.error(f"[Old Update] 💥 Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)
    print("Работа скрипта old_update_binance_data.py завершена.")