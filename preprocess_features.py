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
        for col in ['hour', 'dayofweek']:
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

    if 'hour' in df_group.columns:
        df_group['hour_sin'] = np.sin(2 * np.pi * df_group['hour'] / 24)
    else:
        df_group['hour_sin'] = np.nan

    if 'dayofweek' in df_group.columns:
        df_group['dayofweek_sin'] = np.sin(2 * np.pi * df_group['dayofweek'] / 7)
    else:
        df_group['dayofweek_sin'] = np.nan

    # 📊 [2] Добавление bb_width (ширина Боллинджера)
    if 'close' in df_group.columns and len(df_group['close']) >= 20:  # BollingerBands требует достаточно данных
        try:
            bb = BollingerBands(close=df_group['close'], window=20, window_dev=2)
            df_group['bb_width'] = bb.bollinger_hband() - bb.bollinger_lband()
            # Также можно добавить индикатор положения цены относительно полос, если нужно
            # df_group['bb_pband'] = bb.bollinger_pband() # Процентное положение цены внутри полос
            # df_group['bb_wband'] = bb.bollinger_wband() # Ширина полос, нормированная на среднюю линию
        except Exception as e:
            logging.warning(f"Не удалось рассчитать Bollinger Bands: {e}")
            df_group['bb_width'] = np.nan
    else:
        df_group['bb_width'] = np.nan

    return df_group


def compute_and_prepare_features(df_input, tf_name, btc_features_df):
    logging.info(f"Начало расчета признаков для {df_input['symbol'].nunique()} символов на ТФ {tf_name}...")
    all_symbols_features = []

    for symbol_val in tqdm(df_input['symbol'].unique(), desc=f"Обработка {tf_name}", unit="symbol"):
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

        df_sym['target_tp_hit'] = (df_sym['delta'] > TP_THRESHOLD).astype(int)

        df_sym = compute_ta_features(df_sym.copy())

        if btc_features_df is not None and not btc_features_df.empty:
            df_sym = df_sym.merge(btc_features_df, left_index=True, right_index=True, how='left', suffixes=('', '_btc'))
            btc_cols_to_ffill = [col for col in df_sym.columns if '_btc' in col]
            if btc_cols_to_ffill:
                df_sym[btc_cols_to_ffill] = df_sym[btc_cols_to_ffill].ffill()

        if 'atr' in df_sym.columns and 'volume' in df_sym.columns:
            condition_volume = df_sym['volume'] > df_sym['volume'].rolling(window=20).mean()
            condition_atr = df_sym['atr'] > df_sym['atr'].rolling(window=20).mean()
            df_sym_filtered = df_sym[condition_volume & condition_atr]

            if len(df_sym_filtered) < len(df_sym):
                logging.debug(
                    f"Для {symbol_val} на {tf_name} отфильтровано {len(df_sym) - len(df_sym_filtered)} строк из-за флэта/слабых участков.")
            df_sym = df_sym_filtered
        else:
            logging.warning(
                f"Пропуск фильтрации флэта для {symbol_val} на {tf_name}: отсутствует колонка 'atr' или 'volume'.")

        if not df_sym.empty:
            df_sym['symbol'] = symbol_val
            all_symbols_features.append(df_sym)
        else:
            logging.info(
                f"Символ {symbol_val} на ТФ {tf_name} стал пустым после фильтрации флэта/слабых участков и не будет добавлен.")

    if not all_symbols_features:
        logging.warning(f"Не удалось рассчитать признаки ни для одного символа на ТФ {tf_name}.")
        return pd.DataFrame()

    full_features_df = pd.concat(all_symbols_features)

    cols_for_dropna = ['delta', 'volatility', 'target_class', 'rsi', 'target_tp_hit',
                       'ema_20_slope', 'ema_50_slope']
    # Если 'bb_width' критичен, его тоже можно добавить в cols_for_dropna,
    # но это приведет к удалению строк, где он не смог быть рассчитан (например, в начале датасета).
    # cols_for_dropna.append('bb_width')

    len_before_dropna = len(full_features_df)
    full_features_df = full_features_df.dropna(subset=cols_for_dropna)
    len_after_dropna = len(full_features_df)
    logging.info(
        f"Удалено {len_before_dropna - len_after_dropna} строк из-за NaN в ключевых колонках (включая target_class).")

    full_features_df = full_features_df.reset_index()
    logging.info(f"Расчет признаков для {tf_name} завершен. Итого строк: {len(full_features_df)}.")
    return full_features_df


def main_preprocess(tf_arg):
    logging.info(f"⚙️  Начало построения признаков для таймфрейма: {tf_arg}")

    df_all_candles = load_candles_from_db(tf_arg)
    if df_all_candles.empty:
        logging.error(f"Нет данных из БД для {tf_arg}. Построение признаков прервано.")
        return

    btc_features_prepared = None
    if 'BTCUSDT' in df_all_candles['symbol'].unique():
        logging.info("Подготовка признаков BTCUSDT для использования в других парах...")
        df_btc_raw = df_all_candles[df_all_candles['symbol'] == 'BTCUSDT'].copy()
        df_btc_raw = df_btc_raw.set_index('timestamp')

        if len(df_btc_raw) >= 50:
            df_btc_raw['log_return_1_btc'] = np.log(df_btc_raw['close'] / df_btc_raw['close'].shift(1))
            df_btc_raw['rsi_btc'] = ta.momentum.RSIIndicator(close=df_btc_raw['close'], window=14).rsi()
            df_btc_raw['volume_btc'] = df_btc_raw['volume']
            btc_features_prepared = df_btc_raw[['log_return_1_btc', 'rsi_btc', 'volume_btc']].shift(1)
            btc_features_prepared.dropna(inplace=True)
            logging.info(f"Признаки BTCUSDT подготовлены. Строк: {len(btc_features_prepared)}")
        else:
            logging.warning(f"Недостаточно данных для BTCUSDT ({len(df_btc_raw)} строк) для расчета его признаков.")
    else:
        logging.warning("Данные по BTCUSDT отсутствуют в загруженных свечах. Признаки BTC использоваться не будут.")

    df_to_process = df_all_candles

    if df_to_process.empty and btc_features_prepared is None:
        logging.error(
            f"Нет данных для обработки признаков на ТФ {tf_arg} (возможно, был только BTCUSDT с недостаточным количеством данных).")
        return

    df_final_features = compute_and_prepare_features(df_to_process, tf_arg, btc_features_prepared)

    if df_final_features.empty:
        logging.error(f"Не удалось создать файл признаков для {tf_arg}, так как итоговый DataFrame пуст.")
        return

    os.makedirs('data', exist_ok=True)
    output_pickle_path = f"data/features_{tf_arg}.pkl"
    output_sample_csv_path = f"data/sample_{tf_arg}.csv"

    try:
        df_final_features.to_pickle(output_pickle_path)
        logging.info(f"💾  Признаки сохранены: {output_pickle_path}, форма: {df_final_features.shape}")
    except Exception as e:
        logging.error(f"Ошибка сохранения Pickle файла {output_pickle_path}: {e}")

    try:
        sample_size = min(1000, len(df_final_features))
        if sample_size > 0:
            df_final_features.head(sample_size).to_csv(output_sample_csv_path, index=False)
            logging.info(f"📄  Сэмпл данных сохранен: {output_sample_csv_path} ({sample_size} строк)")
    except Exception as e:
        logging.error(f"Ошибка сохранения CSV сэмпла {output_sample_csv_path}: {e}")

    logging.info(f"✅  Завершено построение признаков для таймфрейма: {tf_arg}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Построение признаков на основе данных свечей из SQLite.")
    parser.add_argument('--tf', type=str, required=True, help='Таймфрейм для обработки (например: 5m, 15m)')
    args = parser.parse_args()

    try:
        main_preprocess(args.tf)
    except KeyboardInterrupt:
        print(f"\n[Preprocess] 🛑 Построение признаков для {args.tf} прервано пользователем (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logging.error(f"[Preprocess] 💥 Непредвиденная ошибка при обработке {args.tf}: {e}", exc_info=True)
        sys.exit(1)