import sqlite3
import pandas as pd
import numpy as np
import ta
import os
import argparse
from tqdm import tqdm
import sys  # Для Ctrl+C
import logging
from ta.volatility import BollingerBands  # 📊 [2] Импорт для BollingerBands

# Настройка логирования для этого скрипта
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [Preprocess] - %(message)s',
                    stream=sys.stdout)

# 🔧 1.2 Добавь словарь групп вверху файла:
SYMBOL_GROUPS = {
    "top8": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "XRPUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"
    ],
    "meme": [
        "DOGEUSDT", "PEPEUSDT", "FLOKIUSDT", "WIFUSDT", "SHIBUSDT"
    ],
    "defi": [
        "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "COMPUSDT"
    ]
}

DB_PATH = 'database/market_data.db'
TARGET_SHIFT = 5
TP_THRESHOLD = 0.005


def load_candles_from_db(tf_key):
    logging.info(f"Загрузка свечей из candles_{tf_key}...")
    if not os.path.exists(DB_PATH):
        logging.error(f"Файл базы данных {DB_PATH} не найден!")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='candles_{tf_key}';")
        if cursor.fetchone() is None:
            logging.error(f"Таблица 'candles_{tf_key}' не найдена в базе данных {DB_PATH}.")
            return pd.DataFrame()

        df = pd.read_sql_query(f"SELECT * FROM candles_{tf_key}", conn)
        if df.empty:
            logging.warning(f"Таблица candles_{tf_key} пуста.")
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        logging.info(f"Загружено {len(df)} свечей из candles_{tf_key} для {df['symbol'].nunique()} символов.")
        return df
    except pd.io.sql.DatabaseError as e:
        logging.error(f"Ошибка SQL при чтении candles_{tf_key}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Общая ошибка при загрузке свечей для {tf_key}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def classify_delta_value(delta_val):
    if pd.isna(delta_val): return np.nan
    if delta_val > 0.002:
        return 'UP'
    elif delta_val < -0.002:
        return 'DOWN'
    else:
        return np.nan


def compute_ta_features(df_group):
    # Lag-фичи

    # Lag по целевой переменной
    df_group['target_class_shift1'] = df_group['target_class'].shift(1)
    # Индикаторы Технического Анализа
    # RSI
    df_group['rsi'] = ta.momentum.RSIIndicator(close=df_group['close'], window=14).rsi()
    # EMA
    df_group['ema_20'] = ta.trend.EMAIndicator(close=df_group['close'], window=20).ema_indicator()
    df_group['ema_50'] = ta.trend.EMAIndicator(close=df_group['close'], window=50).ema_indicator()
    # MACD
    macd_indicator = ta.trend.MACD(close=df_group['close'], window_slow=26, window_fast=12, window_sign=9)
    df_group['macd'] = macd_indicator.macd()
    df_group['macd_signal'] = macd_indicator.macd_signal()
    df_group['macd_diff'] = macd_indicator.macd_diff()
    # OBV (On-Balance Volume)
    df_group['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df_group['close'],
                                                         volume=df_group['volume']).on_balance_volume()
    # ATR (Average True Range)
    df_group['atr'] = ta.volatility.AverageTrueRange(high=df_group['high'], low=df_group['low'],
                                                     close=df_group['close'], window=14).average_true_range()

    # === Lag-фичи по RSI и EMA ===
    for shift_val in [1, 2, 3]:
        df_group[f'rsi_shift{shift_val}'] = df_group['rsi'].shift(shift_val)
        df_group[f'ema_20_shift{shift_val}'] = df_group['ema_20'].shift(shift_val)
        df_group[f'ema_50_shift{shift_val}'] = df_group['ema_50'].shift(shift_val)

    # Lag по целевой переменной
    df_group['target_class_shift1'] = df_group['target_class'].shift(1) if 'target_class' in df_group.columns else 0
    # Дополнительные признаки (существующие)
    df_group['volume_z'] = (df_group['volume'] - df_group['volume'].rolling(window=20, min_periods=1).mean()) / \
                           (df_group['volume'].rolling(window=20, min_periods=1).std().replace(0, np.nan))
    # 📉 [1] Ограничение выбросов volume_z
    df_group['volume_z'] = df_group['volume_z'].clip(-5, 5)

    df_group['candle_body_size'] = abs(df_group['close'] - df_group['open'])
    df_group['candle_hl_range'] = df_group['high'] - df_group['low']
    df_group['is_doji'] = ((df_group['candle_body_size'] / (df_group['candle_hl_range'] + 1e-9)) < 0.1).astype(int)

    if isinstance(df_group.index, pd.DatetimeIndex):
        df_group['hour'] = df_group.index.hour
        df_group['minute'] = df_group.index.minute
        df_group['dayofweek'] = df_group.index.dayofweek
        df_group['dayofmonth'] = df_group.index.day
        df_group['weekofyear'] = df_group.index.isocalendar().week.astype(int)
    else:
        for col in ['hour', 'dayofweek', 'minute', 'dayofmonth', 'weekofyear']:  # Added missing cols to default to NaN
            if col not in df_group.columns:
                df_group[col] = np.nan

    df_group['ema_20_slope'] = df_group['ema_20'].diff()
    df_group['ema_50_slope'] = df_group['ema_50'].diff()

    if 'rsi' in df_group.columns and len(df_group['rsi']) > 1:
        df_group['rsi_cross_50'] = ((df_group['rsi'] > 50) & (df_group['rsi'].shift(1) <= 50)).astype(int)
    else:
        df_group['rsi_cross_50'] = 0

    if all(col in df_group.columns for col in ['high', 'open', 'close', 'low']):
        wick_up = df_group['high'] - df_group[['open', 'close']].max(axis=1)
        wick_down = df_group[['open', 'close']].min(axis=1) - df_group['low']
        body = abs(df_group['close'] - df_group['open'])
        df_group['pin_bar'] = ((wick_up > (body * 2 + 1e-9)) | (wick_down > (body * 2 + 1e-9))).astype(int)
    else:
        df_group['pin_bar'] = 0

    # Новые признаки из предыдущего запроса
    if 'ema_20' in df_group.columns and 'ema_50' in df_group.columns:
        df_group['ema_diff'] = df_group['ema_20'] - df_group['ema_50']
    else:
        df_group['ema_diff'] = np.nan

    if 'rsi' in df_group.columns:
        df_group['rsi_change_3'] = df_group['rsi'].diff(3)
    else:
        df_group['rsi_change_3'] = np.nan

    if 'volume' in df_group.columns:
        df_group['volume_mean'] = df_group['volume'].rolling(window=20).mean()
        df_group['volume_std'] = df_group['volume'].rolling(window=20).std()
        df_group['volume_spike'] = ((df_group['volume'] - df_group['volume_mean']) > 2 * df_group['volume_std']).astype(
            int)
    else:
        df_group['volume_mean'] = np.nan
        df_group['volume_std'] = np.nan
        df_group['volume_spike'] = 0

    if 'hour' in df_group.columns and pd.notna(df_group['hour']).any():
        df_group['hour_sin'] = np.sin(2 * np.pi * df_group['hour'].astype(float) / 24)
    else:
        df_group['hour_sin'] = np.nan

    if 'dayofweek' in df_group.columns and pd.notna(df_group['dayofweek']).any():
        df_group['dayofweek_sin'] = np.sin(2 * np.pi * df_group['dayofweek'].astype(float) / 7)
    else:
        df_group['dayofweek_sin'] = np.nan

    # 📊 [2] Добавление bb_width (ширина Боллинджера)
    if 'close' in df_group.columns and len(df_group['close']) >= 20:  # BollingerBands требует достаточно данных
        try:
            bb = BollingerBands(close=df_group['close'], window=20, window_dev=2)
            df_group['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
        except Exception as e:
            logging.warning(f"Не удалось рассчитать Bollinger Bands: {e}")
            df_group['bb_width'] = np.nan
    else:
        df_group['bb_width'] = np.nan
    
        # === [R1] Rolling статистика по цене и объёму ===
    rolling_windows = [5, 10, 20]

    for window in rolling_windows:
        df_group[f'close_rolling_mean_{window}'] = df_group['close'].rolling(window).mean()
        df_group[f'close_rolling_std_{window}'] = df_group['close'].rolling(window).std()
        df_group[f'volume_rolling_mean_{window}'] = df_group['volume'].rolling(window).mean()
        df_group[f'volume_rolling_std_{window}'] = df_group['volume'].rolling(window).std()
        df_group[f'returns_rolling_std_{window}'] = df_group['log_return_1'].rolling(window).std()

    # === [R2] Длина последовательного тренда вверх/вниз ===
    df_group['trend_up'] = (df_group['close'] > df_group['close'].shift(1)).astype(int)
    df_group['trend_down'] = (df_group['close'] < df_group['close'].shift(1)).astype(int)

    df_group['consecutive_up'] = df_group['trend_up'] * (df_group['trend_up'].groupby((df_group['trend_up'] != df_group['trend_up'].shift()).cumsum()).cumcount() + 1)
    df_group['consecutive_down'] = df_group['trend_down'] * (df_group['trend_down'].groupby((df_group['trend_down'] != df_group['trend_down'].shift()).cumsum()).cumcount() + 1)

    df_group.drop(['trend_up', 'trend_down'], axis=1, inplace=True)


    return df_group


def compute_and_prepare_features(df_input, tf_name, btc_features_df):
    logging.info(f"Начало расчета признаков для {df_input['symbol'].nunique()} символов на ТФ {tf_name}...")
    all_symbols_features = []

    if df_input.empty:
        logging.warning(f"Входной DataFrame для compute_and_prepare_features пуст для ТФ {tf_name}.")
        return pd.DataFrame()

    unique_symbols = df_input['symbol'].unique()
    if len(unique_symbols) == 0:
        logging.warning(f"Нет уникальных символов для обработки в compute_and_prepare_features для ТФ {tf_name}.")
        return pd.DataFrame()

    for symbol_val in tqdm(unique_symbols, desc=f"Обработка {tf_name}", unit="symbol"):
        df_sym = df_input[df_input['symbol'] == symbol_val].copy()
        df_sym = df_sym.set_index('timestamp')

        if len(df_sym) < 100:
            logging.info(
                f"Символ {symbol_val} на ТФ {tf_name} имеет слишком мало данных ({len(df_sym)} < 100), пропускается.")
            continue

        df_sym['log_return_1'] = np.log(df_sym['close'] / df_sym['close'].shift(1))
        df_sym['future_close'] = df_sym['close'].shift(-TARGET_SHIFT)
        df_sym['delta'] = (df_sym['future_close'] / df_sym['close']) - 1

        future_max_high = df_sym['high'].rolling(window=TARGET_SHIFT).max().shift(-TARGET_SHIFT + 1)
        future_min_low = df_sym['low'].rolling(window=TARGET_SHIFT).min().shift(-TARGET_SHIFT + 1)
        df_sym['volatility'] = (future_max_high - future_min_low) / df_sym['close']

        df_sym['target_up'] = (df_sym['delta'] > 0).astype(int)
        df_sym['target_class'] = df_sym['delta'].apply(classify_delta_value)
        for thresh in [0.004, 0.005, 0.006]:
            df_sym['target_tp_hit'] = np.where(df_sym['delta'] > thresh, 1, 0)
            # сохранить файл с суффиксом _tpX, где X — значение
            # запустить обучение и сравнить F1/PR AUC
        df_sym = compute_ta_features(df_sym.copy())

        if btc_features_df is not None and not btc_features_df.empty:
            if isinstance(df_sym.index, pd.DatetimeIndex) and isinstance(btc_features_df.index, pd.DatetimeIndex):
                if df_sym.index.tz != btc_features_df.index.tz:
                    logging.debug(f"Normalizing timezone for merge with BTC features for {symbol_val}")
                    if df_sym.index.tz is not None:
                        df_sym.index = df_sym.index.tz_localize(None)
                    if btc_features_df.index.tz is not None:
                        btc_features_df = btc_features_df.copy()
                        btc_features_df.index = btc_features_df.index.tz_localize(None)

            df_sym = df_sym.merge(btc_features_df, left_index=True, right_index=True, how='left', suffixes=('', '_btc'))
            btc_cols_to_ffill = [col for col in df_sym.columns if '_btc' in col]
            if btc_cols_to_ffill:
                df_sym[btc_cols_to_ffill] = df_sym[btc_cols_to_ffill].ffill()

        if 'atr' in df_sym.columns and 'volume' in df_sym.columns:
            min_periods_rolling = min(20, len(df_sym))
            if min_periods_rolling > 0:
                condition_volume = df_sym['volume'] > df_sym['volume'].rolling(window=20,
                                                                               min_periods=min_periods_rolling).mean()
                condition_atr = df_sym['atr'] > df_sym['atr'].rolling(window=20, min_periods=min_periods_rolling).mean()
                condition_volume = condition_volume.fillna(False)
                condition_atr = condition_atr.fillna(False)
                df_sym_filtered = df_sym[condition_volume & condition_atr]

                if len(df_sym_filtered) < len(df_sym):
                    logging.debug(
                        f"Для {symbol_val} на {tf_name} отфильтровано {len(df_sym) - len(df_sym_filtered)} строк из-за флэта/слабых участков.")
                df_sym = df_sym_filtered
            else:
                logging.warning(f"Недостаточно данных для {symbol_val} на {tf_name} для фильтрации флэта.")
        else:
            logging.warning(
                f"Пропуск фильтрации флэта для {symbol_val} на {tf_name}: отсутствует колонка 'atr' или 'volume'.")

        if not df_sym.empty:
            df_sym['symbol'] = symbol_val
            all_symbols_features.append(df_sym.reset_index())
        else:
            logging.info(
                f"Символ {symbol_val} на ТФ {tf_name} стал пустым после фильтрации и не будет добавлен.")

    if not all_symbols_features:
        logging.warning(f"Не удалось рассчитать признаки ни для одного символа на ТФ {tf_name}.")
        return pd.DataFrame()

    full_features_df = pd.concat(all_symbols_features).reset_index(drop=True)

    cols_for_dropna = ['delta', 'volatility', 'target_class', 'rsi', 'target_tp_hit',
                       'ema_20_slope', 'ema_50_slope']
    len_before_dropna = len(full_features_df)
    full_features_df = full_features_df.dropna(subset=cols_for_dropna)
    len_after_dropna = len(full_features_df)
    logging.info(
        f"Удалено {len_before_dropna - len_after_dropna} строк из-за NaN в ключевых колонках.")
    logging.info(f"Расчет признаков для {tf_name} завершен. Итого строк: {len(full_features_df)}.")
    return full_features_df


def main_preprocess(args):
    tf_arg = args.tf
    
    # 🔧 1.4 Для model_name_suffix сделай:
    # Этот суффикс используется для имен файлов и определяется тем, какой из аргументов 
    # --symbol-group или --symbol был использован, или "all" по умолчанию.
    model_name_suffix_for_files = args.symbol_group or args.symbol or "all"

    # Загрузка свечей (всех для данного ТФ)
    df_all_candles = load_candles_from_db(tf_arg)
    if df_all_candles.empty:
        logging.error(f"Нет данных из БД для {tf_arg}. Построение признаков прервано.")
        return

    # Определение деталей для логгирования и фактической фильтрации
    log_processing_details = ""
    
    if args.symbol_list: # args.symbol_list мог быть установлен напрямую или через args.symbol_group
        if args.symbol_group:
            log_processing_details = f" группы '{args.symbol_group}' (символы: {', '.join(args.symbol_list)})"
        else:
            log_processing_details = f" списка символов ({', '.join(args.symbol_list)})"
        
        logging.info(f"Обработка для {tf_arg} для{log_processing_details}. Фильтрация по этому списку.")
        
        if 'symbol' not in df_all_candles.columns:
            logging.error("Колонка 'symbol' отсутствует в загруженных данных. Фильтрация невозможна.")
            return

        df_all_candles = df_all_candles[df_all_candles['symbol'].isin(args.symbol_list)]
        if df_all_candles.empty:
            logging.error(f"Нет данных по символам {args.symbol_list} для ТФ {tf_arg} после фильтрации.")
            return
        logging.info(f"Данные отфильтрованы по списку. Осталось {len(df_all_candles)} строк.")
    
    elif args.symbol: # Только если args.symbol_list не был активен
        log_processing_details = f" символа '{args.symbol}'"
        logging.info(f"Обработка для {tf_arg} для{log_processing_details}. Фильтрация по этому символу.")
        
        original_symbols_count = df_all_candles['symbol'].nunique()
        original_rows_count = len(df_all_candles)

        if 'symbol' not in df_all_candles.columns:
            logging.error(f"Колонка 'symbol' отсутствует. Фильтрация по {args.symbol} невозможна.")
            return

        df_all_candles = df_all_candles[df_all_candles['symbol'] == args.symbol]

        if df_all_candles.empty:
            logging.warning(f"После фильтрации по символу '{args.symbol}' для ТФ {tf_arg} данных не осталось. "
                            f"(Исходно было {original_rows_count} строк для {original_symbols_count} символов). "
                            f"Обработка не будет продолжена.")
            return
        else:
            logging.info(f"Отфильтровано по символу {args.symbol}. Осталось {len(df_all_candles)} строк "
                         f"(из {original_rows_count} строк, {original_symbols_count} символов).")
    
    else: # Ни --symbol-list, ни --symbol-group, ни --symbol не предоставлены
        log_processing_details = " (все символы из БД для этого ТФ)"
        logging.info(f"Обработка для {tf_arg} для{log_processing_details}. Используются все {df_all_candles['symbol'].nunique()} загруженных символа ({len(df_all_candles)} строк).")

    logging.info(f"⚙️  Начало расчета признаков для таймфрейма: {tf_arg} для{log_processing_details}")
    logging.info(f"Используемый суффикс для имен файлов: {model_name_suffix_for_files}")


    btc_features_prepared = None
    if 'BTCUSDT' in df_all_candles['symbol'].unique():
        logging.info("Подготовка признаков BTCUSDT...")
        df_btc_raw = df_all_candles[df_all_candles['symbol'] == 'BTCUSDT'].copy()

        if not df_btc_raw.empty:
            df_btc_raw = df_btc_raw.set_index('timestamp')
            if len(df_btc_raw) >= 50:
                df_btc_raw['log_return_1_btc'] = np.log(df_btc_raw['close'] / df_btc_raw['close'].shift(1))
                df_btc_raw['rsi_btc'] = ta.momentum.RSIIndicator(close=df_btc_raw['close'], window=14).rsi()
                df_btc_raw['volume_btc'] = df_btc_raw['volume']
                btc_features_prepared = df_btc_raw[['log_return_1_btc', 'rsi_btc', 'volume_btc']].shift(1)
                btc_features_prepared.dropna(inplace=True)
                logging.info(f"Признаки BTCUSDT подготовлены. Строк: {len(btc_features_prepared)}")
            else:
                logging.warning(
                    f"Недостаточно данных для BTCUSDT ({len(df_btc_raw)} строк) для расчета его признаков.")
        else:
            logging.info("Данные по BTCUSDT отсутствуют в текущем наборе свечей (возможно, отфильтрованы). Признаки BTC использоваться не будут.")
    else:
        logging.warning("Данные по BTCUSDT отсутствуют в (отфильтрованных) свечах. Признаки BTC использоваться не будут.")

    df_to_process = df_all_candles

    if df_to_process.empty:
        logging.error(
            f"Нет данных для обработки признаков на ТФ {tf_arg} для{log_processing_details} (df_to_process пуст).")
        return

    df_final_features = compute_and_prepare_features(df_to_process, tf_arg, btc_features_prepared)

    if df_final_features.empty:
        logging.error(f"Не удалось создать файл признаков для {tf_arg} для{log_processing_details}, так как итоговый DataFrame пуст.")
        return

    os.makedirs('data', exist_ok=True)
    
    output_pickle_path = f"data/features_{model_name_suffix_for_files}_{tf_arg}.pkl"
    output_sample_csv_path = f"data/sample_{model_name_suffix_for_files}_{tf_arg}.csv"

    try:
        df_final_features.to_pickle(output_pickle_path)
        logging.info(f"💾  Признаки сохранены: {output_pickle_path}, форма: {df_final_features.shape}")
    except Exception as e:
        logging.error(f"Ошибка сохранения Pickle файла {output_pickle_path}: {e}")

    try:
        sample_size = min(1000, len(df_final_features))
        if sample_size > 0:
            if 'timestamp' in df_final_features.columns and pd.api.types.is_datetime64_any_dtype(
                    df_final_features['timestamp']):
                df_sample = df_final_features.head(sample_size).copy()
                df_sample['timestamp'] = df_sample['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                df_sample.to_csv(output_sample_csv_path, index=False)
            else:
                df_final_features.head(sample_size).to_csv(output_sample_csv_path, index=False)
            logging.info(f"📄  Сэмпл данных сохранен: {output_sample_csv_path} ({sample_size} строк)")
    except Exception as e:
        logging.error(f"Ошибка сохранения CSV сэмпла {output_sample_csv_path}: {e}")

    logging.info(f"✅  Завершено построение признаков для таймфрейма: {tf_arg} для{log_processing_details}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Построение признаков на основе данных свечей из SQLite.")
    parser.add_argument('--tf', type=str, required=True, help='Таймфрейм для обработки (например: 5m, 15m)')
    parser.add_argument('--symbol', type=str, default=None,
                        help="Символ (например: BTCUSDT)")
    parser.add_argument('--symbol-list', nargs='+', help="Список символов через пробел (например: BTCUSDT ETHUSDT ...)")
    # 🔧 1.1 Добавь аргумент --symbol-group:
    parser.add_argument('--symbol-group', type=str, help="Псевдоним группы монет (например: top8, meme)")

    args = parser.parse_args()

    # 🔧 1.3 После args = parser.parse_args() добавь:
    if args.symbol_group:
        if args.symbol_group in SYMBOL_GROUPS:
            if args.symbol_list:
                 logging.warning(f"Аргумент --symbol-list ('{args.symbol_list}') будет перезаписан группой --symbol-group ('{args.symbol_group}').")
            args.symbol_list = SYMBOL_GROUPS[args.symbol_group]
            args.symbol = args.symbol_group
            logging.info(f"🧩 Используется группа символов '{args.symbol_group}': {args.symbol_list}")
        else:
            logging.error(f"Неизвестная группа символов: {args.symbol_group}. Доступные группы: {list(SYMBOL_GROUPS.keys())}")
            sys.exit(1)

    try:
        main_preprocess(args)
    except KeyboardInterrupt:
        log_details_exit = ""
        if args.symbol_group: log_details_exit = f" группы '{args.symbol_group}'"
        elif args.symbol_list: log_details_exit = f" списка символов ({', '.join(args.symbol_list)})"
        elif args.symbol: log_details_exit = f" символа '{args.symbol}'"
        else: log_details_exit = " (все символы)"
        print(f"\n[Preprocess] 🛑 Построение признаков для {args.tf} для{log_details_exit} прервано пользователем (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        log_details_exit = ""
        if args.symbol_group: log_details_exit = f" группы '{args.symbol_group}'"
        elif args.symbol_list: log_details_exit = f" списка символов ({', '.join(args.symbol_list)})"
        elif args.symbol: log_details_exit = f" символа '{args.symbol}'"
        else: log_details_exit = " (все символы)"
        logging.error(f"[Preprocess] 💥 Непредвиденная ошибка при обработке {args.tf} для{log_details_exit}: {e}",
                      exc_info=True)
        sys.exit(1)