import sqlite3
import pandas as pd
import numpy as np
import ta
import os
import argparse
from tqdm import tqdm
import sys
import logging
from ta.volatility import BollingerBands, KeltnerChannel # Добавлен KeltnerChannel
from ta.momentum import StochasticOscillator, WilliamsRIndicator # Добавлены новые осцилляторы
from ta.trend import ADXIndicator # Добавлен ADX

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [Preprocess] - %(message)s',
                    stream=sys.stdout)

SYMBOL_GROUPS = {
    "top15": [  "ETHUSDT", "SOLUSDT", "BNBUSDT", "BCHUSDT", "LTCUSDT",
    "BTCUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "XRPUSDT",
    "MATICUSDT", "LINKUSDT", "NEARUSDT", "ATOMUSDT", "TRXUSDT"],
    
    "meme": ["DOGEUSDT", "PEPEUSDT", "FLOKIUSDT", "WIFUSDT", "SHIBUSDT"],
    "defi": ["UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "COMPUSDT"],
    "top8": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "XRPUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"
    ],
    "top3": ["BTCUSDT", "ETHUSDT", "SOLUSDT"] # Добавлено
}

DB_PATH = 'database/market_data.db'
# Значения по умолчанию для новых аргументов командной строки
DEFAULT_TARGET_SHIFT = 5
DEFAULT_LONG_THRESHOLD = 0.005
DEFAULT_SHORT_THRESHOLD = 0.005
DEFAULT_TP_THRESHOLD = 0.006


def load_candles_from_db(tf_key):
    logging.info(f"Загрузка свечей из candles_{tf_key}...")
    if not os.path.exists(DB_PATH):
        logging.error(f"Файл базы данных {DB_PATH} не найден!")
        return pd.DataFrame()

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(f"SELECT * FROM candles_{tf_key}", conn)
        if df.empty:
            logging.warning(f"Таблица candles_{tf_key} пуста.")
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
        logging.info(f"Загружено {len(df)} свечей из candles_{tf_key} для {df['symbol'].nunique()} символов.")
        return df
    except Exception as e:
        logging.error(f"Ошибка при загрузке свечей для {tf_key}: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def compute_ta_features(df_group):
    # --- Существующие индикаторы ---
    df_group['rsi'] = ta.momentum.RSIIndicator(close=df_group['close'], window=14).rsi()
    df_group['ema_20'] = ta.trend.EMAIndicator(close=df_group['close'], window=20).ema_indicator()
    df_group['ema_50'] = ta.trend.EMAIndicator(close=df_group['close'], window=50).ema_indicator()
    macd_indicator = ta.trend.MACD(close=df_group['close'], window_slow=26, window_fast=12, window_sign=9)
    df_group['macd'] = macd_indicator.macd()
    df_group['macd_signal'] = macd_indicator.macd_signal()
    df_group['macd_diff'] = macd_indicator.macd_diff()
    df_group['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df_group['close'], volume=df_group['volume']).on_balance_volume()
    df_group['atr'] = ta.volatility.AverageTrueRange(high=df_group['high'], low=df_group['low'], close=df_group['close'], window=14).average_true_range()

    for shift_val in [1, 2, 3]:
        df_group[f'rsi_shift{shift_val}'] = df_group['rsi'].shift(shift_val)
        df_group[f'ema_20_shift{shift_val}'] = df_group['ema_20'].shift(shift_val)
        df_group[f'ema_50_shift{shift_val}'] = df_group['ema_50'].shift(shift_val)

    df_group['volume_z'] = (df_group['volume'] - df_group['volume'].rolling(window=20, min_periods=1).mean()) / \
                           (df_group['volume'].rolling(window=20, min_periods=1).std().replace(0, 1e-9))
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
        df_group['hour_sin'] = np.sin(2 * np.pi * df_group['hour'].astype(float) / 24)
        df_group['dayofweek_sin'] = np.sin(2 * np.pi * df_group['dayofweek'].astype(float) / 7)
    else:
        for col in ['hour', 'minute', 'dayofweek', 'dayofmonth', 'weekofyear', 'hour_sin', 'dayofweek_sin']:
            df_group[col] = np.nan

    df_group['ema_20_slope'] = df_group['ema_20'].diff()
    df_group['ema_50_slope'] = df_group['ema_50'].diff()
    df_group['rsi_cross_50'] = ((df_group['rsi'] > 50) & (df_group['rsi'].shift(1) <= 50)).astype(int) if 'rsi' in df_group.columns and len(df_group['rsi']) > 1 else 0
    
    if all(c in df_group.columns for c in ['high', 'open', 'close', 'low']):
        wick_up = df_group['high'] - df_group[['open', 'close']].max(axis=1)
        wick_down = df_group[['open', 'close']].min(axis=1) - df_group['low']
        body = abs(df_group['close'] - df_group['open'])
        df_group['pin_bar'] = ((wick_up > (body * 2 + 1e-9)) | (wick_down > (body * 2 + 1e-9))).astype(int)
    else: df_group['pin_bar'] = 0
    
    df_group['ema_diff'] = df_group['ema_20'] - df_group['ema_50'] if 'ema_20' in df_group.columns and 'ema_50' in df_group.columns else np.nan
    df_group['rsi_change_3'] = df_group['rsi'].diff(3) if 'rsi' in df_group.columns else np.nan
    
    if 'volume' in df_group.columns:
        df_group['volume_mean'] = df_group['volume'].rolling(window=20, min_periods=1).mean()
        df_group['volume_std'] = df_group['volume'].rolling(window=20, min_periods=1).std().replace(0, 1e-9)
        df_group['volume_spike'] = ((df_group['volume'] - df_group['volume_mean']) > 2 * df_group['volume_std']).astype(int)
    else:
        df_group['volume_mean'], df_group['volume_std'], df_group['volume_spike'] = np.nan, np.nan, 0

    try: # Bollinger Bands
        bb = BollingerBands(close=df_group['close'], window=20, window_dev=2)
        df_group['bb_width'] = bb.bollinger_wband() # Используем wband (width band)
    except Exception: df_group['bb_width'] = np.nan
    
    rolling_windows = [5, 10, 20]
    if 'log_return_1' in df_group.columns:
        for window in rolling_windows:
            df_group[f'close_rolling_mean_{window}'] = df_group['close'].rolling(window, min_periods=1).mean()
            df_group[f'close_rolling_std_{window}'] = df_group['close'].rolling(window, min_periods=1).std().replace(0, 1e-9)
            df_group[f'volume_rolling_mean_{window}'] = df_group['volume'].rolling(window, min_periods=1).mean()
            df_group[f'volume_rolling_std_{window}'] = df_group['volume'].rolling(window, min_periods=1).std().replace(0, 1e-9)
            df_group[f'returns_rolling_std_{window}'] = df_group['log_return_1'].rolling(window, min_periods=1).std().replace(0, 1e-9)
    else:
        for window in rolling_windows: df_group[f'returns_rolling_std_{window}'] = np.nan

    df_group['trend_up'] = (df_group['close'] > df_group['close'].shift(1)).astype(int)
    df_group['trend_down'] = (df_group['close'] < df_group['close'].shift(1)).astype(int)
    df_group['consecutive_up'] = df_group['trend_up'] * (df_group['trend_up'].groupby((df_group['trend_up'] != df_group['trend_up'].shift()).cumsum()).cumcount() + 1)
    df_group['consecutive_down'] = df_group['trend_down'] * (df_group['trend_down'].groupby((df_group['trend_down'] != df_group['trend_down'].shift()).cumsum()).cumcount() + 1)
    df_group.drop(['trend_up', 'trend_down'], axis=1, inplace=True, errors='ignore')

    # --- Новые TA индикаторы ---
    # Stochastic Oscillator
    try:
        stoch = StochasticOscillator(high=df_group['high'], low=df_group['low'], close=df_group['close'], window=14, smooth_window=3)
        df_group['stoch_k'] = stoch.stoch()
        df_group['stoch_d'] = stoch.stoch_signal()
    except Exception: df_group['stoch_k'], df_group['stoch_d'] = np.nan, np.nan

    # Williams %R
    try:
        df_group['williams_r'] = WilliamsRIndicator(high=df_group['high'], low=df_group['low'], close=df_group['close'], lbp=14).williams_r()
    except Exception: df_group['williams_r'] = np.nan

    # ADX
    try:
        adx_indicator = ADXIndicator(high=df_group['high'], low=df_group['low'], close=df_group['close'], window=14)
        df_group['adx'] = adx_indicator.adx()
        df_group['adx_pos'] = adx_indicator.adx_pos() # +DI
        df_group['adx_neg'] = adx_indicator.adx_neg() # -DI
    except Exception: df_group['adx'], df_group['adx_pos'], df_group['adx_neg'] = np.nan, np.nan, np.nan

    # Keltner Channels
    try:
        kc = KeltnerChannel(high=df_group['high'], low=df_group['low'], close=df_group['close'], window=20, window_atr=10)
        df_group['kc_width'] = (kc.keltner_channel_hband() - kc.keltner_channel_lband()) / (kc.keltner_channel_mband() + 1e-9) # Нормализованная ширина
        # Положение цены относительно канала: -1 (ниже нижней), 0 (внутри), 1 (выше верхней)
        df_group['kc_pos'] = np.select(
            [df_group['close'] < kc.keltner_channel_lband(), df_group['close'] > kc.keltner_channel_hband()],
            [-1, 1],
            default=0
        )
    except Exception: df_group['kc_width'], df_group['kc_pos'] = np.nan, np.nan
    
    return df_group


def compute_and_prepare_features(df_input, tf_name, btc_features_df, 
                                 target_shift, long_thresh, short_thresh, tp_thresh): # Новые параметры
    logging.info(f"Начало расчета признаков для {df_input['symbol'].nunique()} символов на ТФ {tf_name}...")
    all_symbols_features = []

    if df_input.empty: return pd.DataFrame()
    unique_symbols = df_input['symbol'].unique()
    if len(unique_symbols) == 0: return pd.DataFrame()

    for symbol_val in tqdm(unique_symbols, desc=f"Обработка {tf_name}", unit="symbol"):
        df_sym = df_input[df_input['symbol'] == symbol_val].copy()
        df_sym = df_sym.set_index('timestamp')

        if len(df_sym) < 100:
            logging.debug(f"Символ {symbol_val} ТФ {tf_name} мало данных ({len(df_sym)}), пропуск.")
            continue

        df_sym['future_close'] = df_sym['close'].shift(-target_shift) # Используем параметр
        df_sym['delta'] = (df_sym['future_close'] / df_sym['close']) - 1

        future_max_high_rolling = df_sym['high'].rolling(window=target_shift, min_periods=1)
        future_min_low_rolling = df_sym['low'].rolling(window=target_shift, min_periods=1)

        df_sym['future_max_high'] = future_max_high_rolling.max().shift(-target_shift + 1)
        df_sym['future_min_low'] = future_min_low_rolling.min().shift(-target_shift + 1)
        df_sym['volatility'] = (df_sym['future_max_high'] - df_sym['future_min_low']) / (df_sym['close'] + 1e-9)


        df_sym['target_long'] = ((df_sym['future_max_high'] / (df_sym['close'] + 1e-9) - 1) >= long_thresh).astype(int)
        df_sym['target_short'] = (((df_sym['close'] + 1e-9) / (df_sym['future_min_low'] + 1e-9) - 1) >= short_thresh).astype(int)
        df_sym.loc[df_sym['future_min_low'] <= 1e-9, 'target_short'] = 0 # Если цена упала до 0, шорт невозможен

        tp_up_level = df_sym['close'] * (1 + tp_thresh)
        tp_down_level = df_sym['close'] * (1 - tp_thresh)
        hit_tp_up = (df_sym['future_max_high'] >= tp_up_level)
        hit_tp_down = (df_sym['future_min_low'] <= tp_down_level) & (df_sym['future_min_low'] > 1e-9) # Условие, что не 0
        df_sym['target_tp_hit'] = (hit_tp_up | hit_tp_down).astype(int)

        df_sym['log_return_1'] = np.log(df_sym['close'] / (df_sym['close'].shift(1).replace(0, 1e-9) ) ) # Защита от деления на ноль

        df_sym = compute_ta_features(df_sym.copy())

        if btc_features_df is not None and not btc_features_df.empty:
            # (логика слияния с BTC признаками остается прежней)
            if isinstance(df_sym.index, pd.DatetimeIndex) and isinstance(btc_features_df.index, pd.DatetimeIndex):
                if df_sym.index.tz != btc_features_df.index.tz:
                    current_sym_tz, btc_tz = df_sym.index.tz, btc_features_df.index.tz
                    if current_sym_tz is not None and btc_tz is not None:
                        df_sym.index, btc_features_df_copy = df_sym.index.tz_convert('UTC'), btc_features_df.copy()
                        btc_features_df_copy.index = btc_features_df_copy.index.tz_convert('UTC')
                        df_sym = df_sym.merge(btc_features_df_copy, left_index=True, right_index=True, how='left', suffixes=('', '_btc'))
                    elif current_sym_tz is not None and btc_tz is None:
                        df_sym.index = df_sym.index.tz_localize(None)
                        df_sym = df_sym.merge(btc_features_df, left_index=True, right_index=True, how='left', suffixes=('', '_btc'))
                    elif current_sym_tz is None and btc_tz is not None:
                        btc_features_df_copy = btc_features_df.copy()
                        btc_features_df_copy.index = btc_features_df_copy.index.tz_localize(None)
                        df_sym = df_sym.merge(btc_features_df_copy, left_index=True, right_index=True, how='left', suffixes=('', '_btc'))
                    else: # Оба наивные
                         df_sym = df_sym.merge(btc_features_df, left_index=True, right_index=True, how='left', suffixes=('', '_btc'))
                else: # Таймзоны совпадают (или обе None)
                    df_sym = df_sym.merge(btc_features_df, left_index=True, right_index=True, how='left', suffixes=('', '_btc'))

            btc_cols_to_ffill = [col for col in df_sym.columns if '_btc' in col]
            if btc_cols_to_ffill: df_sym[btc_cols_to_ffill] = df_sym[btc_cols_to_ffill].ffill()

        # (логика фильтрации флэта остается прежней)
        if 'atr' in df_sym.columns and 'volume' in df_sym.columns:
            min_periods_rolling = min(20, len(df_sym))
            if min_periods_rolling > 0:
                if 'volume_mean' not in df_sym.columns:
                    df_sym['volume_mean'] = df_sym['volume'].rolling(window=20, min_periods=min_periods_rolling).mean()
                condition_volume = df_sym['volume'] > df_sym['volume_mean']
                condition_atr = df_sym['atr'] > df_sym['atr'].rolling(window=20, min_periods=min_periods_rolling).mean()
                df_sym = df_sym[condition_volume.fillna(False) & condition_atr.fillna(False)].copy()

        if not df_sym.empty:
            df_sym['symbol'] = symbol_val
            all_symbols_features.append(df_sym.reset_index())

    if not all_symbols_features: return pd.DataFrame()
    full_features_df = pd.concat(all_symbols_features).reset_index(drop=True)
    if 'target_class' in full_features_df.columns:
        full_features_df = full_features_df.drop(columns=['target_class'], errors='ignore')

    cols_for_dropna = ['target_long', 'target_short', 'delta', 'volatility', 'target_tp_hit', 'rsi'] # Убрал ema_slopes, т.к. они могут быть NaN в начале
    actual_cols_for_dropna = [col for col in cols_for_dropna if col in full_features_df.columns]
    len_before = len(full_features_df)
    full_features_df = full_features_df.dropna(subset=actual_cols_for_dropna)
    if len(full_features_df) < len_before:
        logging.info(f"Удалено {len_before - len(full_features_df)} строк из-за NaN в {actual_cols_for_dropna}.")
    
    logging.info(f"Расчет признаков для {tf_name} завершен. Итого строк: {len(full_features_df)}.")
    return full_features_df


def main_preprocess(args):
    tf_arg = args.tf
    # Используем значения из args для порогов и сдвига
    target_shift_val = args.target_shift
    long_thresh_val = args.long_threshold
    short_thresh_val = args.short_threshold
    tp_thresh_val = args.tp_threshold

    model_name_suffix_for_files = args.symbol_group or args.symbol or "all"

    df_all_candles = load_candles_from_db(tf_arg)
    if df_all_candles.empty: return

    log_processing_details = ""
    symbols_to_process_list = None

    if args.symbol_list:
        symbols_to_process_list = args.symbol_list
        log_processing_details = f" группы '{args.symbol_group}'" if args.symbol_group else f" списка символов"
    elif args.symbol:
        symbols_to_process_list = [args.symbol.upper()]
        log_processing_details = f" символа '{args.symbol.upper()}'"
    
    if symbols_to_process_list:
        df_all_candles = df_all_candles[df_all_candles['symbol'].isin(symbols_to_process_list)]
        if df_all_candles.empty:
            logging.error(f"Нет данных для{log_processing_details} на ТФ {tf_arg} после фильтрации.")
            return
    else:
        log_processing_details = " (все символы)"

    logging.info(f"⚙️ Начало расчета признаков для ТФ: {tf_arg} для{log_processing_details}")
    logging.info(f"Используемые параметры таргетов: SHIFT={target_shift_val}, LONG_THR={long_thresh_val}, SHORT_THR={short_thresh_val}, TP_THR={tp_thresh_val}")
    logging.info(f"Суффикс для файлов: {model_name_suffix_for_files}")

    btc_features_prepared = None
    process_btc_features = ('BTCUSDT' in df_all_candles['symbol'].unique()) if symbols_to_process_list is None else ('BTCUSDT' in symbols_to_process_list)
        
    if process_btc_features:
        df_btc_raw = df_all_candles[df_all_candles['symbol'] == 'BTCUSDT'].copy().set_index('timestamp')
        if not df_btc_raw.empty and len(df_btc_raw) >= 50:
            df_btc_raw['log_return_1_btc'] = np.log(df_btc_raw['close'] / (df_btc_raw['close'].shift(1).replace(0,1e-9)))
            df_btc_raw['rsi_btc'] = ta.momentum.RSIIndicator(close=df_btc_raw['close'], window=14).rsi()
            df_btc_raw['volume_btc'] = df_btc_raw['volume']
            btc_features_prepared = df_btc_raw[['log_return_1_btc', 'rsi_btc', 'volume_btc']].shift(1).dropna()
            logging.info(f"Признаки BTCUSDT подготовлены ({len(btc_features_prepared)} строк).")
        else: logging.warning("Недостаточно данных BTCUSDT или BTCUSDT не в списке.")
    else: logging.info("Признаки BTC использоваться не будут.")

    df_final_features = compute_and_prepare_features(
        df_all_candles, tf_arg, btc_features_prepared,
        target_shift_val, long_thresh_val, short_thresh_val, tp_thresh_val # Передаем параметры
    )

    if df_final_features.empty:
        logging.error(f"Не удалось создать признаки для {tf_arg}{log_processing_details}.")
        return

    os.makedirs('data', exist_ok=True)
    output_pickle_path = f"data/features_{model_name_suffix_for_files}_{tf_arg}.pkl"
    output_sample_csv_path = f"data/sample_{model_name_suffix_for_files}_{tf_arg}.csv"

    try:
        df_final_features.to_pickle(output_pickle_path)
        logging.info(f"💾 Признаки сохранены: {output_pickle_path}, форма: {df_final_features.shape}")
    except Exception as e: logging.error(f"Ошибка сохранения Pickle {output_pickle_path}: {e}")

    try:
        sample_size = min(1000, len(df_final_features))
        if sample_size > 0:
            df_sample = df_final_features.head(sample_size).copy()
            if 'timestamp' in df_sample.columns:
                df_sample['timestamp'] = pd.to_datetime(df_sample['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            df_sample.to_csv(output_sample_csv_path, index=False)
            logging.info(f"📄 Сэмпл сохранен: {output_sample_csv_path} ({sample_size} строк)")
    except Exception as e: logging.error(f"Ошибка сохранения CSV сэмпла {output_sample_csv_path}: {e}")

    logging.info(f"✅ Завершено построение признаков для ТФ: {tf_arg}{log_processing_details}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Построение признаков из свечей SQLite.")
    parser.add_argument('--tf', type=str, required=True, help='Таймфрейм (5m, 15m)')
    parser.add_argument('--symbol', type=str, default=None, help="Символ (BTCUSDT)")
    parser.add_argument('--symbol-list', nargs='+', help="Список символов (BTCUSDT ETHUSDT)")
    parser.add_argument('--symbol-group', type=str, help="Группа монет (top8, meme)")
    
    # --- Новые аргументы для параметризации таргетов ---
    parser.add_argument('--target-shift', type=int, default=DEFAULT_TARGET_SHIFT,
                        help=f"Количество будущих свечей для определения таргета (default: {DEFAULT_TARGET_SHIFT})")
    parser.add_argument('--long-threshold', type=float, default=DEFAULT_LONG_THRESHOLD,
                        help=f"Порог роста для target_long (default: {DEFAULT_LONG_THRESHOLD})")
    parser.add_argument('--short-threshold', type=float, default=DEFAULT_SHORT_THRESHOLD,
                        help=f"Порог падения для target_short (default: {DEFAULT_SHORT_THRESHOLD})")
    parser.add_argument('--tp-threshold', type=float, default=DEFAULT_TP_THRESHOLD,
                        help=f"Порог для target_tp_hit (default: {DEFAULT_TP_THRESHOLD})")
    # ----------------------------------------------------

    args = parser.parse_args()

    if args.symbol_group:
        group_key = args.symbol_group.lower()
        if group_key in SYMBOL_GROUPS:
            args.symbol_list = SYMBOL_GROUPS[group_key]
            logging.info(f"🧩 Группа '{args.symbol_group}': {args.symbol_list}")
        else:
            logging.error(f"Неизвестная группа: {args.symbol_group}. Доступные: {list(SYMBOL_GROUPS.keys())}")
            sys.exit(1)
    elif args.symbol:
        symbol_lower = args.symbol.lower()
        if symbol_lower in SYMBOL_GROUPS:
            args.symbol_group = symbol_lower
            args.symbol_list = SYMBOL_GROUPS[symbol_lower]
            logging.info(f"🧩 --symbol '{args.symbol}' распознан как группа '{args.symbol_group}': {args.symbol_list}")
            args.symbol = None # Очищаем, чтобы суффикс файла был именем группы

    try:
        main_preprocess(args)
    except KeyboardInterrupt:
        print(f"\n[Preprocess] 🛑 Прервано пользователем.")
        sys.exit(130)
    except Exception as e:
        logging.error(f"[Preprocess] 💥 Ошибка: {e}", exc_info=True)
        sys.exit(1)