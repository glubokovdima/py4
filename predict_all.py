import pandas as pd
import joblib
import os
from datetime import datetime
from tabulate import tabulate
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [PredictAll] - %(message)s',
                    stream=sys.stdout)

GROUP_MODELS = {
    "top15": [
        "ETHUSDT", "SOLUSDT", "BNBUSDT", "BCHUSDT", "LTCUSDT",
    "BTCUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "XRPUSDT",
    "MATICUSDT", "LINKUSDT", "NEARUSDT", "ATOMUSDT", "TRXUSDT"
    ],
    "meme": [
        "PEPEUSDT", "DOGEUSDT", "FLOKIUSDT", "WIFUSDT", "SHIBUSDT"
    ],
    "defi": [ # Пример, синхронизируйте с preprocess_features.py
        "UNIUSDT", "AAVEUSDT", "SUSHIUSDT", "COMPUSDT"
    ],
    "top55": ["ETHUSDT", "SOLUSDT", "BNBUSDT", "BCHUSDT", "LTCUSDT"] # Добавим, если используется
}

# Пороги для определения сигнала на основе двух бинарных классификаторов
PROBA_LONG_THRESHOLD = 0.55  # Порог для активации сигнала LONG
PROBA_SHORT_THRESHOLD = 0.55 # Порог для активации сигнала SHORT
PROBA_CONFIRM_OTHER_SIDE_LOW = 0.45 # Если LONG, proba_short[1] должна быть < этого. Если SHORT, proba_long[1] < этого.

TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']
LOG_DIR_PREDICT = 'logs'
os.makedirs(LOG_DIR_PREDICT, exist_ok=True)

# Старая TARGET_CLASS_NAMES больше не используется для основного сигнала
# TARGET_CLASS_NAMES = ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']


def load_model_with_fallback(symbol, tf, model_type, group_suffix=""):
    """
    Пытается загрузить модель.
    model_type может быть 'clf_long', 'clf_short', 'reg_delta', 'reg_vol', 'clf_tp_hit'.
    group_suffix - это files_suffix (имя группы, символ или "all").
    """
    # 1. Персональная модель (если group_suffix - это имя символа)
    # или групповая модель (если group_suffix - это имя группы)
    # или общая модель по ТФ (если group_suffix - "all", но модель названа с "all") - это маловероятно
    
    # Сначала пробуем путь с суффиксом (который может быть символом или группой)
    # models/BTCUSDT_5m_clf_long.pkl или models/top8_5m_clf_long.pkl
    specific_model_path = f"models/{group_suffix}_{tf}_{model_type}.pkl"
    if group_suffix and group_suffix != "all" and os.path.exists(specific_model_path):
        logging.debug(f"Используется модель с суффиксом '{group_suffix}' для {symbol} ({tf}): {model_type}")
        try:
            return joblib.load(specific_model_path)
        except Exception as e:
            logging.error(f"Ошибка загрузки модели {specific_model_path}: {e}")
            # Fallback

    # 2. Если group_suffix был символом и персональная модель не найдена, ищем групповую
    # Это актуально, если group_suffix != symbol, но мы передаем group_suffix = symbol_filter
    # Правильнее будет так: если group_suffix - это символ, то сначала ищем персональную, потом групповую для этого символа.
    # Если group_suffix - это группа, то ищем только эту групповую.
    
    # Путь для персональной модели, если group_suffix это имя группы или "all"
    # (т.е. мы не нашли модель по group_suffix и теперь ищем по конкретному symbol)
    symbol_specific_model_path = f"models/{symbol}_{tf}_{model_type}.pkl"
    if os.path.exists(symbol_specific_model_path):
        logging.debug(f"Используется персональная модель для {symbol} ({tf}): {model_type}")
        try:
            return joblib.load(symbol_specific_model_path)
        except Exception as e:
            logging.error(f"Ошибка загрузки персональной модели {symbol_specific_model_path}: {e}")

    # 3. Групповая модель (если символ входит в какую-либо группу)
    for group_name_iter, symbol_list_iter in GROUP_MODELS.items():
        if symbol in symbol_list_iter:
            group_model_path_iter = f"models/{group_name_iter}_{tf}_{model_type}.pkl"
            if os.path.exists(group_model_path_iter):
                logging.debug(f"Используется групповая модель ({group_name_iter}) для {symbol} ({tf}): {model_type}")
                try:
                    return joblib.load(group_model_path_iter)
                except Exception as e:
                    logging.error(f"Ошибка загрузки групповой модели {group_model_path_iter}: {e}")
                break # Нашли группу, выходим

    # 4. Общая модель по таймфрейму (без суффикса символа/группы)
    default_model_path = f"models/{tf}_{model_type}.pkl"
    if os.path.exists(default_model_path):
        logging.debug(f"Используется общая модель по TF для {symbol} ({tf}): {model_type}")
        try:
            return joblib.load(default_model_path)
        except Exception as e:
            logging.error(f"Ошибка загрузки общей модели {default_model_path}: {e}")
            return None # Явный возврат None при ошибке

    logging.warning(f"Модель {model_type} для {symbol} ({tf}) не найдена (суффикс '{group_suffix}').")
    return None


def load_features_list(tf, model_suffix, features_type):
    """
    Загружает список признаков.
    model_suffix - это files_suffix (имя группы, символ или "all").
    features_type - 'long' или 'short'.
    """
    # Путь к файлу признаков с суффиксом
    # models/BTCUSDT_5m_features_long_selected.txt или models/top8_5m_features_long_selected.txt
    specific_features_path = f"models/{model_suffix}_{tf}_features_{features_type}_selected.txt"
    if model_suffix and model_suffix != "all" and os.path.exists(specific_features_path):
        logging.debug(f"Загрузка списка признаков: {specific_features_path}")
        try:
            with open(specific_features_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Ошибка чтения списка признаков {specific_features_path}: {e}")
            # Fallback

    # Общий файл признаков для данного ТФ (без суффикса группы/символа)
    # models/5m_features_long_selected.txt
    # Это менее вероятно, если train_model всегда сохраняет с суффиксом
    generic_features_path = f"models/{tf}_features_{features_type}_selected.txt"
    if os.path.exists(generic_features_path):
        logging.debug(f"Загрузка общего списка признаков: {generic_features_path}")
        try:
            with open(generic_features_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        except Exception as e:
            logging.error(f"Ошибка чтения списка признаков {generic_features_path}: {e}")
    
    logging.warning(f"Файл списка признаков для {features_type} (суффикс '{model_suffix}', TF {tf}) не найден.")
    return None


def compute_final_delta(delta_model, delta_history, sigma_history):
    if pd.isna(delta_history) or pd.isna(sigma_history) or pd.isna(delta_model): # Добавил проверку delta_model
        return delta_model if pd.notna(delta_model) else np.nan # Возвращаем delta_model если она есть, иначе NaN

    if sigma_history < 1e-9:
         sigma_history = 1e-9

    min_sigma = 0.005
    max_sigma = 0.020
    weight_hist_at_min_sigma = 0.6
    weight_hist_at_max_sigma = 0.2

    if sigma_history <= min_sigma:
        w_hist = weight_hist_at_min_sigma
    elif sigma_history >= max_sigma:
        w_hist = weight_hist_at_max_sigma
    else:
        alpha = (sigma_history - min_sigma) / (max_sigma - min_sigma)
        w_hist = weight_hist_at_min_sigma - alpha * (weight_hist_at_min_sigma - weight_hist_at_max_sigma)
    w_model = 1.0 - w_hist
    return round(w_model * delta_model + w_hist * delta_history, 5)


def get_signal_strength(delta_final, confidence, sigma_history):
    """ Определяет силу сигнала: STRONG, MODERATE, WEAK """
    if pd.isna(delta_final) or pd.isna(confidence):
        return "⚪ Слабый" # или N/A
    if pd.isna(sigma_history):
        sigma_history = float('inf')

    abs_delta = abs(delta_final)
    
    # Пороги (можно настроить)
    delta_strong = 0.015  # Для STRONG сигнала дельта должна быть существенной
    delta_moderate = 0.007

    conf_strong = 0.15    # Уверенность для STRONG
    conf_moderate = 0.05

    sigma_reliable = 0.010 # Надежная история

    if abs_delta >= delta_strong and confidence >= conf_strong and sigma_history <= sigma_reliable:
        return "🟢 Сильный"
    elif abs_delta >= delta_moderate and confidence >= conf_moderate:
        return "🟡 Умеренный"
    else:
        return "⚪ Слабый"


def is_conflict(delta_model, delta_history):
    if pd.isna(delta_history) or pd.isna(delta_model):
        return False
    if abs(delta_history) < 0.005:
        return False
    return (delta_model > 0 and delta_history < 0) or \
           (delta_model < 0 and delta_history > 0)


def get_confidence_hint(score): # Эта функция остается полезной
    if score > 0.20: return "Очень высокая уверенность"
    elif score > 0.10: return "Высокая уверенность"
    elif score > 0.05: return "Умеренная уверенность"
    elif score > 0.02: return "Низкая уверенность - будьте осторожны"
    else: return "Очень низкая уверенность - пропустить"


def calculate_trade_levels(entry, direction, atr_value, rr=2.0):
    if pd.isna(atr_value) or atr_value <= 1e-9 or direction == 'none': # Добавил direction == 'none'
        return np.nan, np.nan
    if direction == 'long':
        sl = entry - atr_value
        tp = entry + atr_value * rr
    elif direction == 'short':
        sl = entry + atr_value
        tp = entry - atr_value * rr
    else: # Should not happen if direction == 'none' is checked
        return np.nan, np.nan
    
    sl = round(max(0, sl), 6) if direction == 'short' else round(sl, 6)
    tp = round(max(0, tp), 6)
    return sl, tp


def similarity_analysis(X_live_df, X_hist_df, deltas_hist_series, top_n=15):
    if X_hist_df.empty or len(X_hist_df) < top_n or X_live_df.empty:
        return np.nan, np.nan, "Недостаточно данных для анализа схожести"
    if not X_live_df.columns.equals(X_hist_df.columns):
        common_cols = X_live_df.columns.intersection(X_hist_df.columns)
        if len(common_cols) == 0: return np.nan, np.nan, "Нет общих признаков"
        X_live_df = X_live_df[common_cols]
        X_hist_df = X_hist_df[common_cols]

    if not deltas_hist_series.index.equals(X_hist_df.index):
        try:
            deltas_hist_series = deltas_hist_series.reindex(X_hist_df.index)
        except Exception: return np.nan, np.nan, "Ошибка выравнивания данных"
    
    valid_indices = X_hist_df.index[~(X_hist_df.isnull().any(axis=1) | deltas_hist_series.isnull())]
    X_hist_df_clean = X_hist_df.loc[valid_indices]
    deltas_hist_clean = deltas_hist_series.loc[valid_indices]

    if len(X_hist_df_clean) < top_n:
        return np.nan, np.nan, f"Мало чистых данных ({len(X_hist_df_clean)} < {top_n})"
    
    scaler = StandardScaler()
    try:
        X_hist_scaled = scaler.fit_transform(X_hist_df_clean)
        x_live_scaled = scaler.transform(X_live_df)
    except Exception: return np.nan, np.nan, "Ошибка масштабирования"

    sims = cosine_similarity(X_hist_scaled, x_live_scaled).flatten()
    actual_top_n = min(top_n, len(sims))
    if actual_top_n <= 0: return np.nan, np.nan, "Нет данных для топ-N"
    
    top_indices_cleaned = sims.argsort()[-actual_top_n:][::-1]
    original_top_indices = X_hist_df_clean.iloc[top_indices_cleaned].index
    if len(original_top_indices) == 0: return np.nan, np.nan, "Нет схожих ситуаций"
    
    similar_deltas = deltas_hist_clean.loc[original_top_indices]
    avg_delta = similar_deltas.mean()
    std_delta = similar_deltas.std() if len(similar_deltas) > 1 else 0.0

    hint_val = "Высокий разброс" if pd.notna(std_delta) and std_delta > 0.02 else \
               "Умеренно стабильно" if pd.notna(std_delta) and std_delta > 0.01 else \
               "Стабильный паттерн" if pd.notna(std_delta) else "N/A разброс"
    if pd.isna(avg_delta) or pd.isna(std_delta): hint_val = "N/A статистика"
    return avg_delta, std_delta, hint_val


def predict_all_tf(save_output_flag, symbol_filter=None, group_filter=None):
    target_syms = None
    files_suffix = "all"
    if symbol_filter:
        target_syms = [symbol_filter.upper()]
        files_suffix = target_syms[0]
    elif group_filter:
        group_key = group_filter.lower()
        if group_key not in GROUP_MODELS:
            logging.error(f"Неизвестная группа: '{group_filter}'. Доступные: {list(GROUP_MODELS.keys())}")
            return
        target_syms = GROUP_MODELS[group_key]
        files_suffix = group_key
    
    LATEST_PREDICTIONS_FILE = os.path.join(LOG_DIR_PREDICT, f'latest_predictions_{files_suffix}.csv')
    TRADE_PLAN_FILE = os.path.join(LOG_DIR_PREDICT, f'trade_plan_{files_suffix}.csv')

    logging.info(f"🚀 Запуск генерации прогнозов (фильтр: {files_suffix})...")
    all_predictions_data = []
    trade_plan_data = []

    for tf in TIMEFRAMES:
        logging.info(f"\n--- Обработка таймфрейма: {tf} ---")

        # Загрузка файла признаков (с учетом files_suffix)
        features_path_specific = f"data/features_{files_suffix}_{tf}.pkl"
        features_path_generic = f"data/features_{tf}.pkl"
        features_path = None

        if files_suffix != "all" and os.path.exists(features_path_specific):
            features_path = features_path_specific
        elif os.path.exists(features_path_generic):
            features_path = features_path_generic
        elif os.path.exists(features_path_specific): # Повторная проверка, если generic не найден
             features_path = features_path_specific
        else:
            logging.warning(f"Файл признаков для {tf} (суффикс '{files_suffix}') не найден. Пропуск ТФ.")
            continue
        
        logging.info(f"Используется файл признаков: {features_path}")
        try:
            df = pd.read_pickle(features_path)
        except Exception as e:
            logging.error(f"Не удалось прочитать {features_path}: {e}. Пропуск ТФ.")
            continue
        if df.empty:
            logging.warning(f"Файл признаков {features_path} пуст. Пропуск ТФ.")
            continue

        # Фильтрация DataFrame если target_syms задан и features_path был generic
        if target_syms and features_path == features_path_generic:
            df = df[df['symbol'].isin(target_syms)].copy()
            if df.empty:
                logging.info(f"После фильтрации по '{files_suffix}' нет данных на {tf}. Пропуск ТФ.")
                continue
        
        # Загрузка списков признаков
        # Признаки для регрессоров и TP-hit будут использовать features_long_cols
        features_long_cols = load_features_list(tf, files_suffix, "long")
        features_short_cols = load_features_list(tf, files_suffix, "short")

        if not features_long_cols or not features_short_cols:
            logging.error(f"Не удалось загрузить списки признаков для {tf} (суффикс '{files_suffix}'). Пропуск ТФ.")
            continue
        
        features_for_others_cols = features_long_cols # Используем признаки long для регрессоров и TP-hit

        symbols_to_process_this_tf = df['symbol'].unique().tolist()
        symbols_to_process_this_tf.sort()

        for symbol in symbols_to_process_this_tf:
            df_sym = df[df['symbol'] == symbol].sort_values('timestamp').copy()
            if df_sym.empty or len(df_sym) < 2:
                logging.debug(f"Недостаточно данных для {symbol} {tf}. Пропуск.")
                continue
            
            logging.info(f"--- Обработка {symbol} на {tf} ---")

            model_long = load_model_with_fallback(symbol, tf, "clf_long", files_suffix)
            model_short = load_model_with_fallback(symbol, tf, "clf_short", files_suffix)
            model_delta = load_model_with_fallback(symbol, tf, "reg_delta", files_suffix)
            model_vol = load_model_with_fallback(symbol, tf, "reg_vol", files_suffix)
            model_tp_hit = load_model_with_fallback(symbol, tf, "clf_tp_hit", files_suffix)

            if not model_long or not model_short or not model_delta or not model_vol:
                logging.warning(f"Основные модели для {symbol} {tf} не загружены. Пропуск.")
                continue

            row_df = df_sym.iloc[-1:].copy()
            hist_df_full = df_sym.iloc[:-1].copy()

            # Проверка наличия всех колонок для каждой модели
            missing_long_cols = [col for col in features_long_cols if col not in row_df.columns]
            missing_short_cols = [col for col in features_short_cols if col not in row_df.columns]
            missing_others_cols = [col for col in features_for_others_cols if col not in row_df.columns]

            if missing_long_cols or missing_short_cols or missing_others_cols:
                logging.error(f"Отсутствуют колонки для {symbol} {tf}: long_missing={missing_long_cols}, short_missing={missing_short_cols}, others_missing={missing_others_cols}. Пропуск.")
                continue

            X_live_long = row_df[features_long_cols]
            X_live_short = row_df[features_short_cols]
            X_live_others = row_df[features_for_others_cols]

            if X_live_long.isnull().values.any() or X_live_short.isnull().values.any() or X_live_others.isnull().values.any():
                logging.warning(f"NaN в признаках для {symbol} {tf}. Пропуск.")
                continue
            
            # Анализ схожести (использует признаки "others", т.е. от long)
            hist_for_sim_features = hist_df_full[features_for_others_cols] if features_for_others_cols and not hist_df_full.empty else pd.DataFrame()
            hist_for_sim_deltas = hist_df_full['delta'] if 'delta' in hist_df_full.columns and not hist_df_full.empty else pd.Series(dtype=float)
            avg_delta_similar, std_delta_similar, similarity_hint = similarity_analysis(
                X_live_others, hist_for_sim_features, hist_for_sim_deltas
            )

            # Прогнозы
            try:
                proba_long_raw = model_long.predict_proba(X_live_long)[0]
                proba_short_raw = model_short.predict_proba(X_live_short)[0]
                
                # Вероятность класса 1 (Long=1 или Short=1)
                proba_long_1 = proba_long_raw[list(model_long.classes_).index(1)] if 1 in model_long.classes_ else 0.0
                proba_short_1 = proba_short_raw[list(model_short.classes_).index(1)] if 1 in model_short.classes_ else 0.0

                predicted_delta = model_delta.predict(X_live_others)[0]
                predicted_volatility = model_vol.predict(X_live_others)[0]
                if pd.notna(predicted_volatility) and predicted_volatility < 0: predicted_volatility = 0.0

                tp_hit_proba = np.nan
                if model_tp_hit:
                    tp_hit_all_classes = model_tp_hit.predict_proba(X_live_others)[0]
                    if 1 in model_tp_hit.classes_:
                        tp_hit_proba = tp_hit_all_classes[list(model_tp_hit.classes_).index(1)]
                    elif len(tp_hit_all_classes) > 1 : # Fallback
                        tp_hit_proba = tp_hit_all_classes[1]


            except Exception as e:
                logging.error(f"Ошибка прогнозирования для {symbol} {tf}: {e}", exc_info=True)
                continue

            # Определение сигнала и уверенности
            signal = "NEUTRAL"
            confidence = 0.0
            direction = 'none'

            if proba_long_1 > PROBA_LONG_THRESHOLD and proba_short_1 < PROBA_CONFIRM_OTHER_SIDE_LOW:
                signal = "UP"
                confidence = proba_long_1
                direction = 'long'
            elif proba_short_1 > PROBA_SHORT_THRESHOLD and proba_long_1 < PROBA_CONFIRM_OTHER_SIDE_LOW:
                signal = "DOWN"
                confidence = proba_short_1
                direction = 'short'
            
            hint = get_confidence_hint(confidence)
            ts_obj = pd.to_datetime(row_df['timestamp'].values[0])
            entry_price = float(row_df['close'].values[0])
            sl, tp = calculate_trade_levels(entry_price, direction, predicted_volatility)
            delta_final = compute_final_delta(predicted_delta, avg_delta_similar, std_delta_similar)
            signal_strength_val = get_signal_strength(delta_final, confidence, std_delta_similar)
            conflict_flag = is_conflict(predicted_delta, avg_delta_similar)

            prediction_entry = {
                'symbol': symbol, 'tf': tf, 'timestamp_obj': ts_obj,
                'timestamp_str_log': ts_obj.strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp_str_display': ts_obj.strftime('%Y-%m-%d %H:%M'),
                'signal': signal, 'confidence_score': confidence, 'confidence_hint': hint,
                'proba_long_1': proba_long_1, 'proba_short_1': proba_short_1,
                'predicted_delta': predicted_delta,
                'predicted_volatility': predicted_volatility, 'entry': entry_price,
                'sl': sl, 'tp': tp, 'direction': direction,
                'avg_delta_similar': avg_delta_similar, 'std_delta_similar': std_delta_similar,
                'similarity_hint': similarity_hint,
                'delta_final': delta_final, 'signal_strength': signal_strength_val, 'conflict': conflict_flag,
                'tp_hit_proba': tp_hit_proba
            }
            all_predictions_data.append(prediction_entry)

            TRADE_PLAN_CONFIDENCE_THRESHOLD = 0.08
            TRADE_PLAN_TP_HIT_THRESHOLD = 0.55
            if direction != 'none' and confidence >= TRADE_PLAN_CONFIDENCE_THRESHOLD:
                if pd.isna(tp_hit_proba) or tp_hit_proba >= TRADE_PLAN_TP_HIT_THRESHOLD:
                    rr_value = np.nan
                    if pd.notna(sl) and pd.notna(tp) and abs(entry_price - sl) > 1e-9:
                        try: rr_value = round(abs(tp - entry_price) / abs(entry_price - sl), 2)
                        except ZeroDivisionError: rr_value = np.nan
                    
                    trade_plan_data.append({
                        'symbol': symbol, 'tf': tf, 'entry': entry_price, 'direction': direction,
                        'sl': sl, 'tp': tp, 'rr': rr_value,
                        'confidence': confidence, 'signal': signal, 
                        'timestamp': prediction_entry['timestamp_str_log'], 'hint': hint,
                        'tp_hit_proba': tp_hit_proba,
                        'proba_long': proba_long_1, 'proba_short': proba_short_1 # Добавим для анализа
                    })

    # Сохранение и вывод
    if save_output_flag and all_predictions_data:
        logging.info("💾 Сохранение результатов...")
        df_out = pd.DataFrame(all_predictions_data)
        # Удаляем proba_dict если он там случайно оказался, и timestamp_obj
        df_out_save = df_out.drop(columns=['proba_dict', 'timestamp_obj'], errors='ignore')
        
        csv_columns_order = [
            'symbol', 'tf', 'timestamp_str_log', 'signal', 'confidence_score', 'tp_hit_proba',
            'proba_long_1', 'proba_short_1', # Новые вероятности
            'predicted_delta', 'avg_delta_similar', 'std_delta_similar', 'delta_final',
            'predicted_volatility', 'entry', 'sl', 'tp', 'direction',
            'signal_strength', 'conflict', 'confidence_hint', 'similarity_hint'
        ]
        # Убедимся, что все колонки из df_out_save включены
        final_csv_columns = [col for col in csv_columns_order if col in df_out_save.columns]
        for col in df_out_save.columns:
            if col not in final_csv_columns:
                final_csv_columns.append(col)
        
        if not df_out_save.empty:
            df_out_save = df_out_save[final_csv_columns] # Переупорядочиваем
            df_out_save.rename(columns={'timestamp_str_log': 'timestamp'}, inplace=True)
            df_out_save.to_csv(LATEST_PREDICTIONS_FILE, index=False, float_format='%.6f')
            logging.info(f"📄 Сигналы сохранены: {LATEST_PREDICTIONS_FILE}")

        if trade_plan_data:
            df_trade_plan = pd.DataFrame(trade_plan_data)
            trade_plan_cols_order = [
                'symbol', 'tf', 'timestamp', 'direction', 'signal', 'confidence', 'tp_hit_proba',
                'proba_long', 'proba_short', # Новые вероятности
                'entry', 'sl', 'tp', 'rr', 'hint'
            ]
            final_trade_plan_cols = [col for col in trade_plan_cols_order if col in df_trade_plan.columns]
            for col in df_trade_plan.columns:
                if col not in final_trade_plan_cols:
                    final_trade_plan_cols.append(col)

            if not df_trade_plan.empty:
                df_trade_plan = df_trade_plan[final_trade_plan_cols] # Переупорядочиваем
                df_trade_plan.to_csv(TRADE_PLAN_FILE, index=False, float_format='%.6f')
                logging.info(f"📈 Торговый план сохранён: {TRADE_PLAN_FILE}")
        else:
            logging.info("🤷 Нет данных для торгового плана.")


    if all_predictions_data:
        headers_tabulate = ["TF", "Timestamp", "Сигнал", "Conf.", "P(L)", "P(S)", "ΔModel", "ΔHist", "σHist", "ΔFinal", "Конф?", "Сила", "TP Hit%"]
        grouped_by_symbol = {}
        for item in all_predictions_data: # Используем all_predictions_data
            grouped_by_symbol.setdefault(item['symbol'], []).append(item)
        
        sorted_symbols_for_display = sorted(grouped_by_symbol.keys())
        for symbol_key in sorted_symbols_for_display:
            rows_list = grouped_by_symbol[symbol_key]
            try:
                sorted_rows = sorted(rows_list, key=lambda r: (TIMEFRAMES.index(r['tf']) if r['tf'] in TIMEFRAMES else float('inf'), -r['confidence_score']))
            except ValueError:
                sorted_rows = sorted(rows_list, key=lambda r: -r['confidence_score'])

            table_data_tabulate = []
            for r_tab in sorted_rows:
                table_data_tabulate.append([
                    r_tab['tf'], r_tab['timestamp_str_display'], r_tab['signal'],
                    f"{r_tab['confidence_score']:.3f}",
                    f"{r_tab['proba_long_1']:.2f}", f"{r_tab['proba_short_1']:.2f}", # Новые вероятности
                    f"{r_tab['predicted_delta']:.2%}" if pd.notna(r_tab['predicted_delta']) else "N/A",
                    f"{r_tab['avg_delta_similar']:.2%}" if pd.notna(r_tab['avg_delta_similar']) else "N/A",
                    f"{r_tab['std_delta_similar']:.2%}" if pd.notna(r_tab['std_delta_similar']) else "N/A",
                    f"{r_tab['delta_final']:.2%}" if pd.notna(r_tab['delta_final']) else "N/A",
                    "❗" if r_tab['conflict'] else " ",
                    r_tab['signal_strength'].replace(" Сильный", "🟢").replace(" Умеренный", "🟡").replace(" Слабый", "⚪"),
                    f"{r_tab['tp_hit_proba']:.1%}" if pd.notna(r_tab['tp_hit_proba']) else "N/A"
                ])
            print(f"\n📊 Прогноз по символу: {symbol_key}")
            try:
                print(tabulate(table_data_tabulate, headers=headers_tabulate, tablefmt="pretty", stralign="right", numalign="right"))
            except Exception as e:
                logging.error(f"Ошибка вывода таблицы для {symbol_key}: {e}")
                for row in table_data_tabulate: print(row)
    
    logging.info("✅ Процесс генерации прогнозов завершён.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Генерация прогнозов на основе обученных моделей.")
    parser.add_argument('--save', action='store_true', help="Сохранять результаты прогнозов в CSV файлы.")
    parser.add_argument('--symbol', type=str, help="Один символ для анализа (или имя группы).")
    parser.add_argument('--symbol-group', type=str, help="Группа символов (игнорируется, если --symbol это имя группы).")
    args = parser.parse_args()

    symbol_filter_arg = None
    group_filter_arg = None

    if args.symbol_group: # Если явно указана группа
        group_filter_arg = args.symbol_group.lower()
        if group_filter_arg not in GROUP_MODELS:
            logging.error(f"Неизвестная группа: '{args.symbol_group}'. Доступные: {list(GROUP_MODELS.keys())}")
            sys.exit(1)
    elif args.symbol: # Если указан символ (который может быть именем группы)
        symbol_lower_arg = args.symbol.lower()
        if symbol_lower_arg in GROUP_MODELS:
            group_filter_arg = symbol_lower_arg
            logging.info(f"Аргумент --symbol '{args.symbol}' распознан как группа '{group_filter_arg}'.")
        else:
            symbol_filter_arg = args.symbol.upper()
    
    try:
        predict_all_tf(
            args.save,
            symbol_filter=symbol_filter_arg,
            group_filter=group_filter_arg
        )
    except KeyboardInterrupt:
        print("\n[PredictAll] 🛑 Генерация прогнозов прервана (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logging.error(f"[PredictAll] 💥 Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)