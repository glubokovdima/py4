import pandas as pd
import joblib
import os
from datetime import datetime
from tabulate import tabulate
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys  # Для Ctrl+C
import logging  # Для более детального логгирования


# --- Конфигурация ---
# Настройка логирования для этого скрипта (можно выводить в файл или только в консоль)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [PredictAll] - %(message)s',
                    stream=sys.stdout)

# Словарь групп: символы сгруппированы под именем (ключом)
# >>> ДОБАВЛЕНО/ОБНОВЛЕНО согласно патчу
GROUP_MODELS = {
    "top8": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "XRPUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"
    ],
    "meme": [
        "PEPEUSDT", "DOGEUSDT", "FLOKIUSDT", "WIFUSDT", "SHIBUSDT"
    ],
    # Добавьте сюда другие группы по необходимости
    "defi": [
        # Пример: "UNIUSDT", "AAVEUSDT", ...
    ]
}

def load_model_with_fallback(symbol, tf, model_type):
    """
    Пытается загрузить:
    1. Индивидуальную модель: models/BTCUSDT_15m_clf_class.pkl
    2. Групповую модель: models/top8_15m_clf_class.pkl (если symbol входит в группу)
    3. Общую по таймфрейму: models/15m_clf_class.pkl
    """
    # 1. Персональная модель
    symbol_model_path = f"models/{symbol}_{tf}_{model_type}.pkl"
    if os.path.exists(symbol_model_path):
        # print(f"✅ Используется персональная модель для {symbol}: {model_type}") # Использовал print, переключу на logging
        logging.debug(f"✅ Используется персональная модель для {symbol}: {model_type}")
        try:
            return joblib.load(symbol_model_path)
        except Exception as e:
            logging.error(f"Ошибка загрузки персональной модели {symbol_model_path}: {e}")
            # Fallback to next option

    # 2. Групповая модель
    for group_name, symbol_list in GROUP_MODELS.items():
        if symbol in symbol_list:
            group_model_path = f"models/{group_name}_{tf}_{model_type}.pkl"
            if os.path.exists(group_model_path):
                # print(f"✅ Используется групповая модель ({group_name}) для {symbol}: {model_type}") # Использовал print, переключу на logging
                logging.debug(f"✅ Используется групповая модель ({group_name}) для {symbol}: {model_type}")
                try:
                    return joblib.load(group_model_path)
                except Exception as e:
                    logging.error(f"Ошибка загрузки групповой модели {group_model_path}: {e}")
                    # Fallback to next option
                # Found group model, no need to check other groups for this symbol
                break # Exit the group loop

    # 3. Общая модель
    default_model_path = f"models/{tf}_{model_type}.pkl"
    if os.path.exists(default_model_path):
        # print(f"⚠️ Используется общая модель по TF: {model_type} → {default_model_path}") # Использовал print, переключу на logging
        logging.debug(f"⚠️ Используется общая модель по TF: {model_type} → {default_model_path}")
        try:
            return joblib.load(default_model_path)
        except Exception as e:
            logging.error(f"Ошибка загрузки общей модели {default_model_path}: {e}")
            return None

    # print(f"❌ Модель не найдена ни для символа {symbol}, ни для группы, ни общая: {model_type}") # Использовал print, переключу на logging
    logging.warning(f"❌ Модель не найдена ни для символа {symbol}, ни для группы, ни общая: {model_type} на {tf}")
    return None

# Удалил старую дублирующуюся функцию load_model (которая load_model(tf, model_type)),
# так как load_model_with_fallback теперь основной метод загрузки и включает логику fallback.
# Если эта старая функция использовалась где-то еще с другими путями, возможно, ее нужно
# оставить и переименовать, но для текущей задачи predict_all_tf достаточно load_model_with_fallback.
# Судя по предоставленному коду, старая load_model используется только внутри predict_all_tf
# и ее функциональность теперь полностью покрыта load_model_with_fallback.

TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']  # Основные ТФ для прогноза
# FEATURES_PATH_TEMPLATE = 'data/features_{tf}.pkl' # Старый шаблон
# Модифицируем шаблон пути, чтобы он учитывал суффикс (all, group, symbol)
# Новая логика загрузки features_path будет внутри predict_all_tf
# MODEL_PATH_TEMPLATE = 'models/{tf}_{model_type}.pkl' # Удалил, так как теперь используется load_model_with_fallback путями

LOG_DIR_PREDICT = 'logs'
# Модифицируем имена файлов для сохранения, чтобы они включали суффикс фильтра
# Это нужно, чтобы результаты для разных фильтров не перезаписывали друг друга
# FILES_SUFFIX will be determined inside predict_all_tf based on the filter
# LATEST_PREDICTIONS_FILE = os.path.join(LOG_DIR_PREDICT, 'latest_predictions.csv') # Старый путь
# TRADE_PLAN_FILE = os.path.join(LOG_DIR_PREDICT, 'trade_plan.csv') # Старый путь


# Убедимся, что директория для логов/результатов существует
os.makedirs(LOG_DIR_PREDICT, exist_ok=True)

TARGET_CLASS_NAMES = ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']


def compute_final_delta(delta_model, delta_history, sigma_history):
    if pd.isna(delta_history) or pd.isna(sigma_history):
        return round(delta_model, 5)
    # Убедимся, что sigma_history не слишком мала во избежание деления на ноль или нестабильных весов
    if sigma_history < 1e-9: # Очень маленькое значение, считаем его практически нулем
         sigma_history = 1e-9 # Заменяем на очень маленькое положительное число
         # Можно также решить полностью игнорировать history, если sigma_history близка к нулю, т.к. это идеальная история
         # return round(delta_model, 5) # Альтернатива: если sigma_history ~ 0, то history идеальна, игнорируем model

    # Старая логика весов - можно настроить
    # if sigma_history < 0.005:
    #     w1, w2 = 0.4, 0.6
    # elif sigma_history > 0.02:
    #     w1, w2 = 0.8, 0.2
    # else:
    #     alpha = (sigma_history - 0.005) / (0.02 - 0.005)
    #     w1 = 0.4 + alpha * (0.8 - 0.4)
    #     w2 = 1.0 - w1

    # Альтернативная логика весов, основанная на обратной зависимости от сигмы истории
    # Чем меньше сигма истории, тем больший вес у истории
    # Можно использовать сигму как показатель "шума" или "надежности" исторического паттерна
    # Пример: вес истории = 1 / sigma_history, вес модели = константа
    # Нормализация весов: w_hist = (1/sigma_history) / ((1/sigma_history) + C), w_model = C / ((1/sigma_history) + C)
    # Где C - константа, регулирующая базовый "доверие" к модели
    # Или просто линейно: w_hist = max_weight - (sigma_history - min_sigma) / (max_sigma - min_sigma) * (max_weight - min_weight)
    # Где min_sigma, max_sigma, min_weight, max_weight - настраиваемые параметры.

    # Вернемся к простой линейной интерполяции весов из примера
    min_sigma = 0.005
    max_sigma = 0.020
    weight_hist_at_min_sigma = 0.6 # Вес истории при очень низкой сигме
    weight_hist_at_max_sigma = 0.2 # Вес истории при высокой сигме

    if sigma_history <= min_sigma:
        w_hist = weight_hist_at_min_sigma
    elif sigma_history >= max_sigma:
        w_hist = weight_hist_at_max_sigma
    else:
        # Линейная интерполяция веса истории между weight_hist_at_min_sigma и weight_hist_at_max_sigma
        # по мере роста sigma_history от min_sigma до max_sigma
        alpha = (sigma_history - min_sigma) / (max_sigma - min_sigma)
        w_hist = weight_hist_at_min_sigma - alpha * (weight_hist_at_min_sigma - weight_hist_at_max_sigma)

    w_model = 1.0 - w_hist

    return round(w_model * delta_model + w_hist * delta_history, 5)


def get_signal_strength(delta_final, confidence, sigma_history):
    if pd.isna(sigma_history):
        sigma_history = float('inf')

    # Немного скорректировал логику силы сигнала, чтобы она лучше соответствовала диапазонам confidence и sigma
    # Это мое предположение, можно настроить по желанию
    # Пороги для delta_final и confidence, а также зависимость от надежности истории (обратная к sigma)
    # Сильный сигнал: большая delta_final, высокая confidence, низкая sigma_history (надежная история)
    # Умеренный сигнал: средняя delta_final, средняя confidence, средняя sigma_history
    # Слабый сигнал: маленькая delta_final, низкая confidence, высокая sigma_history (ненадежная история)

    delta_threshold_strong = 0.025 # 2.5%
    delta_threshold_moderate = 0.010 # 1.0%

    conf_threshold_strong = 0.15 # > conf_score - next_conf_score
    conf_threshold_moderate = 0.05

    sigma_threshold_reliable = 0.010 # История надежна, если сигма < 1%
    sigma_threshold_unreliable = 0.020 # История ненадежна, если сигма > 2%


    abs_delta = abs(delta_final)

    if abs_delta > delta_threshold_strong and confidence > conf_threshold_strong:
        # Сильный кандидат, теперь уточним по истории
        if sigma_history < sigma_threshold_reliable:
            return "🟢 Сильный"
        elif sigma_history < sigma_threshold_unreliable:
             # Умеренно надежная история снижает уверенность
             return "🟡 Умеренный"
        else:
             # Ненадежная история сильно снижает уверенность
             return "⚪ Слабый" # Или даже "Неопределенный"?

    elif abs_delta > delta_threshold_moderate and confidence > conf_threshold_moderate:
        # Умеренный кандидат, уточним по истории
        if sigma_history < sigma_threshold_reliable:
             # Умеренный сигнал + надежная история = может быть повышен до Умеренного
             return "🟡 Умеренный" # Или даже "Сильный"? Решим оставить Умеренным
        elif sigma_history < sigma_threshold_unreliable:
             # Умеренный сигнал + умеренно надежная история = остается Умеренным
             return "🟡 Умеренный"
        else:
             # Умеренный сигнал + ненадежная история = понижается до Слабого
             return "⚪ Слабый"

    else:
        # Слабый сигнал по дельте или уверенности
        return "⚪ Слабый"

    # Простая логика из примера:
    # if abs(delta_final) > 0.02 and confidence > 0.5 and sigma_history < 0.015: # Более строгие условия для Сильного
    #     return "🟢 Сильный"
    # elif abs(delta_final) > 0.01 and confidence > 0.2 and sigma_history < 0.025: # Умеренный
    #     return "🟡 Умеренный"
    # else: # Слабый или неопределенный
    #     return "⚪ Слабый"


def is_conflict(delta_model, delta_history):
    if pd.isna(delta_history) or pd.isna(delta_model):
        return False
    # Проверка на конфликт только если дельта истории не близка к нулю
    if abs(delta_history) < 0.005: # Порог для "близко к нулю", можно настроить
        return False
    return (delta_model > 0 and delta_history < 0) or \
        (delta_model < 0 and delta_history > 0)


# Удалил старую функцию load_model, ее функциональность теперь в load_model_with_fallback.


def get_confidence_hint(score):
    if score > 0.20:
        return "Очень высокая уверенность"
    elif score > 0.10:
        return "Высокая уверенность"
    elif score > 0.05:
        return "Умеренная уверенность"
    elif score > 0.02: # Можно установить порог, ниже которого сигналы не торгуются
         return "Низкая уверенность - будьте осторожны"
    else:
        return "Очень низкая уверенность - пропустить"


def calculate_trade_levels(entry, direction, atr_value, rr=2.0):
    # atr_value здесь интерпретируется как базовый размер движения (например, предсказанная волатильность или ATR)
    # SL ставим на расстоянии atr_value, TP на расстоянии atr_value * rr
    if pd.isna(atr_value) or atr_value <= 1e-9:
        return np.nan, np.nan
    if direction == 'long':
        sl = entry - atr_value
        tp = entry + atr_value * rr
    elif direction == 'short':
        sl = entry + atr_value
        tp = entry - atr_value * rr
    else:
        return np.nan, np.nan
    # Округляем до разумного количества знаков после запятой, зависит от цены актива
    # Для крипты 6 знаков обычно достаточно, но может потребоваться адаптация для очень дешевых монет
    # Также нужно убедиться, что SL/TP не становятся отрицательными для очень дешевых монет при шort
    sl = round(max(0, sl), 6) if direction == 'short' else round(sl, 6)
    tp = round(max(0, tp), 6) if direction == 'short' else round(tp, 6) # TP тоже не может быть отрицательным

    return sl, tp


def similarity_analysis(X_live_df, X_hist_df, deltas_hist_series, top_n=15):
    # X_live_df - одна строка DataFrame, X_hist_df - исторический DataFrame признаков
    # deltas_hist_series - серия дельт для соответствующего исторического DataFrame

    if X_hist_df.empty or len(X_hist_df) < top_n:
        return np.nan, np.nan, "Недостаточно исторических данных для анализа схожести"

    # Важно: использовать только те колонки, которые есть и в X_live, и в X_hist
    # X_live_df приходит сюда уже с отобранными признаками feature_cols_from_file
    # X_hist_df - это hist_for_sim_features, которая тоже отобрана по feature_cols_from_file
    # Так что колонки должны совпадать. Проверим на всякий случай.
    if not X_live_df.columns.equals(X_hist_df.columns):
         logging.error("Несоответствие колонок между X_live_df и X_hist_df перед анализом схожести!")
         # Можно попытаться найти общие, но лучше, чтобы они совпадали на этапе подготовки
         common_cols = X_live_df.columns.intersection(X_hist_df.columns)
         if len(common_cols) == 0:
              return np.nan, np.nan, "Нет общих признаков для анализа схожести"
         logging.warning(f"Используются только общие признаки ({len(common_cols)}) для анализа схожести.")
         X_live_df_common = X_live_df[common_cols]
         X_hist_df_common = X_hist_df[common_cols]
    else:
         X_live_df_common = X_live_df
         X_hist_df_common = X_hist_df


    # Убедимся, что X_hist_df_common и deltas_hist_series имеют совпадающие индексы
    # и удалим строки с NaN из признаков перед масштабированием и анализом схожести
    hist_df_aligned = X_hist_df_common.copy()
    # Проверим, что deltas_hist_series имеет тот же индекс, что и X_hist_df_common
    if not deltas_hist_series.index.equals(hist_df_aligned.index):
         # Если индексы не совпадают, попробуем выровнять по индексу признаков
         try:
              hist_deltas_aligned = deltas_hist_series.reindex(hist_df_aligned.index).copy()
              logging.debug("Индексы deltas_hist_series выровнены по X_hist_df_common.")
         except Exception as e:
              logging.error(f"Ошибка при выравнивании индексов deltas_hist_series по X_hist_df_common: {e}")
              return np.nan, np.nan, "Ошибка выравнивания данных для схожести"
    else:
        hist_deltas_aligned = deltas_hist_series.copy()


    # Удаляем строки, где есть NaN в признаках ИЛИ где есть NaN в дельтах истории
    # Создаем булевы маски для NaN в признаках и дельтах
    nan_in_features = hist_df_aligned.isnull().any(axis=1)
    nan_in_deltas = hist_deltas_aligned.isnull()

    # Индексы строк, которые НЕ содержат NaN ни в признаках, ни в дельтах
    valid_indices = hist_df_aligned.index[~(nan_in_features | nan_in_deltas)]

    hist_df_aligned_clean = hist_df_aligned.loc[valid_indices]
    hist_deltas_aligned_clean = hist_deltas_aligned.loc[valid_indices]

    if len(hist_df_aligned_clean) < top_n:
         return np.nan, np.nan, f"Недостаточно чистых исторических данных ({len(hist_df_aligned_clean)} < {top_n}) для анализа схожести"

    scaler = StandardScaler()
    try:
        X_hist_scaled = scaler.fit_transform(hist_df_aligned_clean)
        # X_live_df_common - это одна строка, масштабируем ее тем же скейлером
        x_live_scaled = scaler.transform(X_live_df_common)

    except ValueError as e:
        logging.error(f"Ошибка при масштабировании в similarity_analysis: {e}")
        return np.nan, np.nan, "Ошибка масштабирования признаков"
    except Exception as e:
         logging.error(f"Непредвиденная ошибка при масштабировании в similarity_analysis: {e}", exc_info=True)
         return np.nan, np.nan, "Неизвестная ошибка масштабирования признаков"


    sims = cosine_similarity(X_hist_scaled, x_live_scaled).flatten()

    # Indices of the top N similar historical points in the *cleaned and scaled* history data
    # Handle case where top_n might be larger than available data after cleaning
    actual_top_n = min(top_n, len(sims))
    if actual_top_n <= 0:
         return np.nan, np.nan, "Нет данных для топ-N после чистки и масштабирования"

    top_indices_in_cleaned_hist = sims.argsort()[-actual_top_n:][::-1]

    # Get the original indices from the cleaned data index
    original_top_indices = hist_df_aligned_clean.iloc[top_indices_in_cleaned_hist].index

    if len(original_top_indices) == 0:
        # Это условие, по идее, не должно выполняться, если actual_top_n > 0, но на всякий случай
        return np.nan, np.nan, "Не найдено достаточно схожих ситуаций после фильтрации"

    # Get the deltas for these original indices from the cleaned deltas series
    similar_deltas = hist_deltas_aligned_clean.loc[original_top_indices]

    # Calculate mean and std of these deltas
    avg_delta = similar_deltas.mean()
    std_delta = similar_deltas.std()

    # Ensure std_delta is not NaN if there's only one point (std is 0)
    if len(similar_deltas) == 1:
        std_delta = 0.0

    hint = (
        "Высокий разброс" if pd.notna(std_delta) and std_delta > 0.02 else
        "Умеренно стабильно" if pd.notna(std_delta) and std_delta > 0.01 else
        "Стабильный паттерн" if pd.notna(std_delta) else
        "Не удалось определить разброс"
    )
    if pd.isna(avg_delta) or pd.isna(std_delta):
        hint = "Не удалось рассчитать статистику схожести"

    return avg_delta, std_delta, hint


# >>> ИЗМЕНЕНО: Добавлены параметры symbol_filter и group_filter
def predict_all_tf(save_output_flag, symbol_filter=None, group_filter=None):
    # ----- Новая фильтрация по символу или группе -----
    # >>> ДОБАВЛЕНО согласно патчу
    target_syms = None
    files_suffix = "all" # Суффикс для файлов результатов по умолчанию
    if symbol_filter:
        target_syms = [symbol_filter.upper()] # Приводим к верхнему регистру для безопасности
        logging.info(f"🛠️ Фильтр по символу: {target_syms[0]}")
        files_suffix = target_syms[0]
    elif group_filter:
        group_key = group_filter.lower() # Приводим ключ группы к нижнему регистру
        if group_key not in GROUP_MODELS:
            # Этот случай должен быть обработан в __main__ перед вызывом, но оставим проверку
            logging.error(f"❌ Неизвестная группа символов: '{group_filter}'. Доступные группы: {list(GROUP_MODELS.keys())}")
            return # Выходим из функции, если группа не найдена
        target_syms = GROUP_MODELS[group_key]
        logging.info(f"🛠️ Фильтр по группе: '{group_filter}' ({len(target_syms)} символов)")
        files_suffix = group_key
    else:
        logging.info("🛠️ Без фильтров по символу/группе (обработка всех доступных).")
        files_suffix = "all" # Явно устанавливаем суффикс "all"

    # Определяем имена файлов для сохранения с учётом суффикса
    LATEST_PREDICTIONS_FILE = os.path.join(LOG_DIR_PREDICT, f'latest_predictions_{files_suffix}.csv')
    TRADE_PLAN_FILE = os.path.join(LOG_DIR_PREDICT, f'trade_plan_{files_suffix}.csv')


    logging.info(f"🚀  Запуск генерации прогнозов (фильтр: {files_suffix})...")
    all_predictions_data = []
    trade_plan = []

    for tf in TIMEFRAMES:
        logging.info(f"\n--- Обработка таймфрейма: {tf} ---")

        # >>> ИЗМЕНЕНО согласно патчу: Проверка наличия файлов признаков с fallback
        # 1) сначала пробуем групповой/символьный файл
        # Используем files_suffix, который был определен выше
        group_path   = f"data/features_{files_suffix}_{tf}.pkl"
        # 2) затем пробуем общий файл
        generic_path = f"data/features_{tf}.pkl"

        features_path = None
        if os.path.exists(group_path):
            features_path = group_path
            logging.debug(f"Использую фильтрованный файл признаков: {group_path}")
        elif os.path.exists(generic_path):
            features_path = generic_path
            logging.info(f"🛠️ Использую общий файл признаков: {generic_path}")
        else:
            logging.warning(f"⚠️ Ни фильтрованная ({group_path}), ни общая ({generic_path}) версия файла признаков не найдены. Пропуск таймфрейма {tf}.")
            continue

        try:
            # >>> ИЗМЕНЕНО согласно патчу: убираем engine=...
            # Читаем pickle файл
            df = pd.read_pickle(features_path)
        except Exception as e:
            logging.error(f"❌ Не удалось прочитать файл признаков {features_path}: {e}. Пропуск таймфрейма {tf}.")
            continue

        if df.empty:
            logging.warning(f"⚠️ Файл признаков пуст: {features_path}. Пропуск таймфрейма {tf}.")
            continue

        # Символы в df уже отфильтрованы preprocess_features, если файл с суффиксом был найден.
        # Если загружен generic_path, то df содержит все символы, и нам нужно отфильтровать его.
        # Проверка, что символы в загруженном df соответствуют target_syms (если он задан)
        available_symbols_in_data_before_filter = df['symbol'].unique().tolist()

        if target_syms is not None:
            # Если мы загрузили общий файл (generic_path), то df еще не отфильтрован.
            # Если мы загрузили фильтрованный файл (group_path), то df уже отфильтрован preprocess_features.
            # В любом случае, применяем фильтр на df, чтобы быть уверенными.
            before_filter_count = len(df)
            # Фильтруем только если загрузили generic_path ИЛИ если загруженный файл содержит символы,
            # которых нет в target_syms (проверка консистентности).
            # Самый простой способ - просто всегда фильтровать df по target_syms, если target_syms не None.
            # Это безопасно, даже если df уже отфильтрован.
            df = df[df['symbol'].isin(target_syms)].copy() # Используем .copy() для избежания SettingWithCopyWarning
            after_filter_count = len(df)

            symbols_in_file_but_not_in_filter = [sym for sym in available_symbols_in_data_before_filter if sym not in target_syms]
            symbols_in_filter_but_not_in_file_after_load = [sym for sym in target_syms if sym not in available_symbols_in_data_before_filter]


            if symbols_in_file_but_not_in_filter:
                 logging.warning(f"⚠️ Отфильтровано {len(symbols_in_file_but_not_in_filter)} неожиданных символов из файла признаков {features_path}.")

            if symbols_in_filter_but_not_in_file_after_load:
                 logging.warning(f"⚠️ Символы из фильтра отсутствуют в загруженных данных ({features_path}): {symbols_in_filter_but_not_in_file_after_load}. Пропускаем их.")


            if df.empty:
                 logging.info(f"🤷 После фильтрации по символам/группе нет данных на {tf}. Пропуск таймфрейма {tf}.")
                 continue # Переходим к следующему таймфрейму

            logging.info(f"Фильтрация DataFrame: {before_filter_count} → {after_filter_count} строк для TF {tf}.")


        # >>> КОНЕЦ БЛОКА ФИЛЬТРАЦИИ DATAFRAME (модифицированного)

        features_list_path = f"models/{files_suffix}_{tf}_features_selected.txt"
        if not os.path.exists(features_list_path):
            logging.error(
                f"❌ Файл со списком признаков '{features_list_path}' не найден. Невозможно безопасно продолжить для {tf}. Пропуск.")
            continue
        try:
            with open(features_list_path, "r", encoding="utf-8") as f:
                feature_cols_from_file = [line.strip() for line in f if line.strip()]
            if not feature_cols_from_file:
                logging.error(f"❌ Файл со списком признаков '{features_list_path}' пуст. Пропуск {tf}.")
                continue
        except Exception as e:
            logging.error(f"❌ Ошибка чтения файла со списком признаков '{features_list_path}': {e}. Пропуск {tf}.")
            continue

        # Model loading is now per-symbol, so it's moved inside the symbol loop.
        # The old loading location and logging for TP-hit classes per TF is removed from here.

        # Получаем список уникальных символов из *отфильтрованных* или *загруженных* данных
        # Теперь df содержит только те символы, которые нужны для этого фильтра.
        available_symbols_in_data = df['symbol'].unique()

        # Определяем, какие символы мы будем обрабатывать в этом TF
        # Этот список уже фактически определен загруженным файлом и предыдущей (опциональной) фильтрацией
        symbols_to_process_this_tf = available_symbols_in_data.tolist()

        # Проверяем, что в списке symbol_list (если он был задан) есть хотя бы один символ,
        # который присутствует в текущем df.
        # Это уже не строгая необходимость, т.к. мы итерируемся по symbols_to_process_this_tf,
        # который берется из загруженного df, который уже отфильтрован (или не отфильтрован).
        # Оставим только проверку на пустой список для обработки
        if not symbols_to_process_this_tf:
             # Этот лог уже должен быть покрыт фильтрацией выше, но на всякий случай
             logging.info(f"🤷 Нет символов для обработки на {tf} из загруженного DataFrame после потенциальной фильтрации.")
             continue # Переходим к следующему таймфрейму


        # Сортируем символы для предсказуемого вывода
        symbols_to_process_this_tf.sort()

        for symbol in symbols_to_process_this_tf:

            df_sym = df[df['symbol'] == symbol].sort_values('timestamp').copy()
            # Пустой df_sym не должен возникать тут, т.к. символ взят из available_symbols_in_data,
            # но проверка не повредит.
            if df_sym.empty:
                logging.warning(f"⚠️ Неожиданно пустой DataFrame для {symbol} на {tf}. Пропуск.")
                continue

            logging.info(f"--- Обработка {symbol} на {tf} ---")

            # Загрузка моделей для конкретного символа и ТФ
            # load_model_with_fallback уже реализует логику загрузки групповых/общих моделей
            model_class = load_model_with_fallback(symbol, tf, "clf_class")
            model_delta = load_model_with_fallback(symbol, tf, "reg_delta")
            model_vol = load_model_with_fallback(symbol, tf, "reg_vol")
            model_tp_hit = load_model_with_fallback(symbol, tf, "clf_tp_hit")

            # Логирование классов TP-hit модели (перемещено сюда)
            if model_tp_hit:
                logging.debug(f"Классы TP-hit модели ({symbol}, {tf}): {getattr(model_tp_hit, 'classes_', 'N/A')}")

            if not all([model_class, model_delta, model_vol]):  # model_tp_hit is optional
                logging.warning(
                    f"⚠️ Одна или несколько основных моделей для {symbol} на {tf} не загружены. Пропуск символа {symbol} на этом таймфрейме.")
                continue  # Skip this symbol for this tf

            # Получаем последнюю строку для прогноза и остальное для истории
            # Убедимся, что в df_sym достаточно строк (минимум 1 для прогноза, плюс история для схожести)
            if len(df_sym) < 2: # Минимум 2 строки нужно: 1 для X_live, 1 для истории (хотя бы одна)
                 logging.warning(f"⚠️ Недостаточно данных ({len(df_sym)} строк) для {symbol} {tf}. Пропуск.")
                 continue

            row_df = df_sym.iloc[-1:].copy()
            hist_df_full = df_sym.iloc[:-1].copy()

            if 'close' not in row_df.columns or 'timestamp' not in row_df.columns:
                logging.warning(f"⚠️ В последних данных для {symbol} {tf} отсутствует 'close' или 'timestamp'. Пропуск.")
                continue

            # Проверка, что feature_cols_from_file не отсутствуют в row_df
            missing_cols_in_df_for_symbol = [col for col in feature_cols_from_file if col not in row_df.columns]
            if missing_cols_in_df_for_symbol:
                logging.error(
                    f"❌ В DataFrame для {symbol} на {tf} из {features_path} отсутствуют столбцы {missing_cols_in_df_for_symbol}, "
                    f"необходимые для модели (согласно {features_list_path}). Пропуск символа {symbol} на этом таймфрейме."
                )
                continue

            X_live = row_df[feature_cols_from_file]
            # Проверка на NaN в X_live уже делается ниже, но можно усилить тут
            # if X_live.isnull().values.any():
            #     nan_features = X_live.columns[X_live.isnull().any()].tolist()
            #     logging.warning(f"⚠️ В X_live для {symbol} {tf} есть NaN в признаках: {nan_features}. Пропуск символа {symbol} на этом таймфрейме.")
            #     continue


            # Анализ схожести с историей
            avg_delta_similar, std_delta_similar, similarity_hint = np.nan, np.nan, "Нет данных для анализа схожести"
            if 'delta' not in hist_df_full.columns:
                logging.debug(f"Столбец 'delta' отсутствует в исторических данных hist_df_full для {symbol} {tf}. Анализ схожести невозможен.")
                similarity_hint = "Отсутствует столбец 'delta' в истории"
            else:
                 # Проверяем наличие всех необходимых признаков И столбца 'delta' в исторических данных
                required_hist_cols = feature_cols_from_file + ['delta']
                missing_hist_features_and_delta = [col for col in required_hist_cols if col not in hist_df_full.columns]

                if missing_hist_features_and_delta:
                    logging.debug(f"В hist_df_full для {symbol} {tf} отсутствуют столбцы {missing_hist_features_and_delta} для анализа схожести.")
                    similarity_hint = "Отсутствуют необходимые исторические признаки"
                else:
                    hist_for_sim_features = hist_df_full[feature_cols_from_file].copy()
                    hist_for_sim_deltas = hist_df_full['delta'].copy()

                    # Чистка от NaN и проверка на sufficient data происходит внутри similarity_analysis
                    avg_delta_similar, std_delta_similar, similarity_hint = similarity_analysis(
                        X_live, hist_for_sim_features, hist_for_sim_deltas, top_n=15) # top_n=15


            # Прогнозирование
            # Проверяем на NaN в X_live еще раз перед подачей в модель
            if X_live.isnull().values.any():
                 nan_features = X_live.columns[X_live.isnull().any()].tolist()
                 logging.warning(f"⚠️ В X_live для {symbol} {tf} есть NaN в признаках: {nan_features}. Пропуск символа {symbol} на этом таймфрейме.")
                 continue # Пропустить этот символ/ТФ

            try:
                proba_raw = model_class.predict_proba(X_live)[0]
            except Exception as e:
                logging.error(
                    f"❌ Ошибка при model_class.predict_proba для {symbol} {tf} с X_live shape {X_live.shape}: {e}. Пропуск символа.")
                continue

            # Убедимся, что proba_raw соответствует количеству классов модели
            model_classes = getattr(model_class, 'classes_', None)
            if model_classes is None or len(proba_raw) != len(model_classes):
                 logging.error(f"❌ Количество вероятностей ({len(proba_raw)}) не соответствует классам модели ({len(model_classes) if model_classes is not None else 'N/A'}) для {symbol} {tf}. Пропуск.")
                 continue

            # Сопоставляем вероятности с именами классов модели
            proba_dict_model_order = {model_classes[i]: proba_raw[i] for i in range(len(proba_raw))}

            # Находим предсказанный класс (используем классы из модели)
            pred_class_label = model_classes[proba_raw.argmax()]
            signal = pred_class_label # Сигнал теперь напрямую из класса модели

            # Расчет уверенности: разница между лучшей и второй лучшей вероятностью
            if len(proba_raw) < 2:
                confidence = float(proba_raw.max()) if len(proba_raw) == 1 else 0.0
            else:
                sorted_probas = np.sort(proba_raw)
                confidence = float(sorted_probas[-1] - sorted_probas[-2])

            hint = get_confidence_hint(confidence)

            try:
                predicted_delta = model_delta.predict(X_live)[0]
            except Exception as e:
                 logging.error(f"❌ Ошибка при model_delta.predict для {symbol} {tf}: {e}. Устанавливаю predicted_delta в NaN.")
                 predicted_delta = np.nan

            try:
                predicted_volatility = model_vol.predict(X_live)[0]
                # Проверяем, что предсказанная волатильность положительна и не близка к нулю
                if pd.notna(predicted_volatility) and predicted_volatility < 1e-9:
                    logging.warning(f"⚠️ Предсказанная волатильность очень близка к нулю ({predicted_volatility:.6f}) для {symbol} {tf}. Использовано 0.")
                    predicted_volatility = 0.0
                elif pd.notna(predicted_volatility) and predicted_volatility < 0:
                     logging.warning(f"⚠️ Предсказанная волатильность отрицательна ({predicted_volatility:.6f}) для {symbol} {tf}. Использовано 0.")
                     predicted_volatility = 0.0

            except Exception as e:
                 logging.error(f"❌ Ошибка при model_vol.predict для {symbol} {tf}: {e}. Устанавливаю predicted_volatility в NaN.")
                 predicted_volatility = np.nan


            # TP-hit probability
            tp_hit_proba = np.nan
            if model_tp_hit:
                try:
                    tp_hit_proba_all_classes = model_tp_hit.predict_proba(X_live)[0]
                    model_tp_classes = getattr(model_tp_hit, 'classes_', None)

                    # Ищем вероятность класса '1' (успех TP)
                    if model_tp_classes is not None and 1 in model_tp_classes:
                        try:
                            class_1_idx = list(model_tp_classes).index(1)
                            tp_hit_proba = tp_hit_proba_all_classes[class_1_idx]
                        except ValueError:
                            # Этого не должно произойти, если 1 в model_tp_classes, но на всякий случай
                            logging.error(f"❌ Внутренняя ошибка: класс '1' не найден по индексу, хотя присутствует в model_tp_hit.classes_ ({model_tp_classes}) для {symbol} {tf}. TP Hit% будет NaN.")
                            tp_hit_proba = np.nan # Остается NaN
                    elif len(tp_hit_proba_all_classes) > 1:
                         # Fallback: Если classes_ нет или не содержит 1, но predict_proba вернул >1 значение,
                         # предполагаем бинарную классификацию и класс 1 находится по индексу 1
                         logging.warning(f"⚠️ Класс '1' не найден в model_tp_hit.classes_ ({model_tp_classes if model_tp_classes is not None else 'N/A'}) для {symbol} {tf}. Использую вероятность по индексу 1.")
                         if len(tp_hit_proba_all_classes) > 1:
                            tp_hit_proba = tp_hit_proba_all_classes[1] # Предполагаем, что индекс 1 соответствует классу 1
                         else:
                             logging.warning(f"⚠️ Недостаточно вероятностей в predict_proba TP-hit модели ({len(tp_hit_proba_all_classes)}) для {symbol} {tf}. TP Hit% будет NaN.")
                             tp_hit_proba = np.nan
                    else:
                         logging.warning(f"⚠️ Модель TP-hit для {symbol} {tf} не имеет класса '1' и predict_proba вернула {len(tp_hit_proba_all_classes)} значений. Невозможно определить tp_hit_proba.")
                         # tp_hit_proba остается np.nan

                except Exception as e:
                    logging.error(f"❌ Ошибка при model_tp_hit.predict_proba для {symbol} {tf}: {e}. TP Hit% будет NaN.")
                    # tp_hit_proba остается np.nan


            # Определение направления по предсказанному сигналу
            direction = 'long' if signal in ['UP', 'STRONG UP'] else 'short' if signal in ['DOWN',
                                                                                           'STRONG DOWN'] else 'none'
            ts_obj = pd.to_datetime(row_df['timestamp'].values[0])
            ts_str_log = ts_obj.strftime('%Y-%m-%d %H:%M:%S')
            ts_str_display = ts_obj.strftime('%Y-%m-%d %H:%M')

            entry_price = float(row_df['close'].values[0])

            # Расчет SL/TP только если есть положительная волатильность
            sl, tp = calculate_trade_levels(entry_price, direction, predicted_volatility)

            # Расчет финальной дельты и силы сигнала
            delta_final = compute_final_delta(predicted_delta, avg_delta_similar, std_delta_similar)
            signal_strength_val = get_signal_strength(delta_final, confidence, std_delta_similar)
            conflict_flag = is_conflict(predicted_delta, avg_delta_similar)

            # Собираем вероятности в порядке TARGET_CLASS_NAMES для консистентности вывода и сохранения
            # Если какие-то классы отсутствуют в модели, их вероятность будет 0
            proba_dict_ordered = {cls_name: proba_dict_model_order.get(cls_name, 0.0) for cls_name in TARGET_CLASS_NAMES}


            prediction_entry = {
                'symbol': symbol, 'tf': tf, 'timestamp_obj': ts_obj,
                'timestamp_str_log': ts_str_log, 'timestamp_str_display': ts_str_display,
                'signal': signal, 'confidence_score': confidence, 'confidence_hint': hint,
                # 'proba_dict': proba_dict_model_order, # Используем proba_dict_ordered для сохранения и вывода
                'proba_dict': proba_dict_ordered,
                'predicted_delta': predicted_delta,
                'predicted_volatility': predicted_volatility, 'entry': entry_price,
                'sl': sl, 'tp': tp, 'direction': direction,
                'avg_delta_similar': avg_delta_similar, 'std_delta_similar': std_delta_similar,
                'similarity_hint': similarity_hint,
                'delta_final': delta_final, 'signal_strength': signal_strength_val, 'conflict': conflict_flag,
                'tp_hit_proba': tp_hit_proba,
                'error': None # Поле для будущих ошибок по символу/ТФ
            }
            all_predictions_data.append(prediction_entry)

            # Формирование торгового плана
            # Условие для добавления в торговый план: не нейтральный сигнал, уверенность выше порога,
            # и TP-hit вероятность либо не определена, либо выше порога.
            TRADE_PLAN_CONFIDENCE_THRESHOLD = 0.08 # Минимальная уверенность для плана
            TRADE_PLAN_TP_HIT_THRESHOLD = 0.55 # Минимальная TP Hit% для плана (если определена)

            if direction != 'none' and confidence >= TRADE_PLAN_CONFIDENCE_THRESHOLD:
                 if pd.isna(prediction_entry['tp_hit_proba']) or prediction_entry['tp_hit_proba'] >= TRADE_PLAN_TP_HIT_THRESHOLD:
                     rr_value = np.nan # Инициализируем как NaN
                     if pd.notna(sl) and pd.notna(tp):
                         # Убедимся, что SL не равен Entry (с учетом округления или очень маленькой разницы)
                         if abs(entry_price - sl) > 1e-9:
                             try:
                                rr_value = round(abs(tp - entry_price) / abs(entry_price - sl), 2)
                             except ZeroDivisionError:
                                 logging.warning(f"⚠️ Деление на ноль при расчете RR для {symbol} {tf} (Entry={entry_price}, SL={sl}).")
                                 rr_value = np.nan
                         else:
                             logging.warning(f"⚠️ SL очень близко к цене входа для {symbol} {tf} (Entry={entry_price}, SL={sl}). Невозможно рассчитать RR.")


                     trade_plan.append({
                         'symbol': symbol, 'tf': tf, 'entry': entry_price, 'direction': direction,
                         'sl': sl, 'tp': tp, 'rr': rr_value,
                         'confidence': confidence, 'signal': signal, 'timestamp': ts_str_log, 'hint': hint,
                         'tp_hit_proba': tp_hit_proba
                     })
                 else:
                     tp_hit_display = f"{prediction_entry['tp_hit_proba']:.1%}" if pd.notna(prediction_entry['tp_hit_proba']) else "N/A"
                     logging.debug(f"Пропуск {symbol} {tf} для торгового плана: TP Hit ({tp_hit_display}) ниже порога {TRADE_PLAN_TP_HIT_THRESHOLD:.1%}.")
            else:
                 logging.debug(f"Пропуск {symbol} {tf} для торгового плана: Направление {direction} или уверенность ({confidence:.3f}) ниже порога {TRADE_PLAN_CONFIDENCE_THRESHOLD:.3f}.")


    # Сохранение и вывод результатов
    if save_output_flag and all_predictions_data:
        logging.info("💾  Сохранение результатов...")
        df_out_list = []
        # Используем TARGET_CLASS_NAMES как базовый порядок для колонок вероятностей
        proba_dict_keys_order = TARGET_CLASS_NAMES

        for r_item in all_predictions_data:
            item = {
                'symbol': r_item['symbol'], 'tf': r_item['tf'],
                'timestamp': r_item['timestamp_obj'].strftime('%Y-%m-%d %H:%M:%S'),
                'signal': r_item['signal'], 'confidence_score': r_item['confidence_score'],
                'tp_hit_proba': r_item['tp_hit_proba'],
                'predicted_delta': r_item['predicted_delta'],
                'predicted_volatility': r_item['predicted_volatility'],
                'avg_delta_similar': r_item['avg_delta_similar'],
                'std_delta_similar': r_item['std_delta_similar'],
                'delta_final': r_item['delta_final'],
                'entry': r_item['entry'], 'sl': r_item['sl'], 'tp': r_item['tp'],
                'direction': r_item['direction'],
                'signal_strength': r_item['signal_strength'], 'conflict': r_item['conflict'],
                'confidence_hint': r_item['confidence_hint'],
                'similarity_hint': r_item['similarity_hint'],
            }
            # Добавляем вероятности классов. Используем ключи из proba_dict_ordered,
            # которые должны соответствовать TARGET_CLASS_NAMES
            item.update(r_item['proba_dict'])

            df_out_list.append(item)

        df_out = pd.DataFrame(df_out_list)

        # Формируем список колонок для сохранения
        # Основные колонки, затем вероятности классов, затем любые другие, которые могли появиться
        csv_columns_order = [
            'symbol', 'tf', 'timestamp', 'signal', 'confidence_score', 'tp_hit_proba',
            'predicted_delta', 'avg_delta_similar', 'std_delta_similar', 'delta_final',
            'predicted_volatility', 'entry', 'sl', 'tp', 'direction',
            'signal_strength', 'conflict', 'confidence_hint', 'similarity_hint'
        ]
        csv_columns_order.extend(proba_dict_keys_order) # Добавляем ключи типа 'STRONG DOWN', 'DOWN' и т.д.

        # Убедимся, что все колонки из df_out включены, даже если их нет в csv_columns_order
        final_csv_columns = [col for col in csv_columns_order if col in df_out.columns]
        for col in df_out.columns:
            if col not in final_csv_columns:
                final_csv_columns.append(col)

        if not df_out.empty:
            try:
                # Переупорядочиваем колонки перед сохранением
                df_out = df_out[final_csv_columns]
                df_out.to_csv(LATEST_PREDICTIONS_FILE, index=False, float_format='%.6f')
                logging.info(f"📄  Сигналы сохранены в: {LATEST_PREDICTIONS_FILE}")
            except Exception as e:
                logging.error(f"❌ Не удалось сохранить {LATEST_PREDICTIONS_FILE}: {e}")
        else:
            logging.info(f"🤷 Нет данных для сохранения в {LATEST_PREDICTIONS_FILE}.")

        if trade_plan:
            df_trade_plan = pd.DataFrame(trade_plan)
            trade_plan_cols_order = [
                'symbol', 'tf', 'timestamp', 'direction', 'signal', 'confidence', 'tp_hit_proba',
                'entry', 'sl', 'tp', 'rr', 'hint'
            ]
            # Убедимся, что все колонки из df_trade_plan включены
            final_trade_plan_cols = [col for col in trade_plan_cols_order if col in df_trade_plan.columns]
            for col in df_trade_plan.columns:
                if col not in final_trade_plan_cols:
                    final_trade_plan_cols.append(col)

            if not df_trade_plan.empty:
                try:
                    # Переупорядочиваем колонки перед сохранением
                    df_trade_plan = df_trade_plan[final_trade_plan_cols]
                    df_trade_plan.to_csv(TRADE_PLAN_FILE, index=False, float_format='%.6f')
                    logging.info(f"📈  Торговый план сохранён в: {TRADE_PLAN_FILE}")
                except Exception as e:
                    logging.error(f"❌ Не удалось сохранить {TRADE_PLAN_FILE}: {e}")
            else:
                logging.info(f"🤷 Торговый план пуст, файл {TRADE_PLAN_FILE} не создан.")
        else:
            logging.info("🤷 Нет данных для торгового плана.")

    # Вывод в консоль
    if all_predictions_data:
        headers_tabulate = ["TF", "Timestamp", "Сигнал", "Conf.", "ΔModel", "ΔHist", "σHist", "ΔFinal", "Конф?", "Сила",
                            "TP Hit%"]
        grouped_by_symbol = {}
        for row_data_item in all_predictions_data:
            grouped_by_symbol.setdefault(row_data_item['symbol'], []).append(row_data_item)

        # Сортируем символы для вывода в алфавитном порядке
        sorted_symbols_for_display = sorted(grouped_by_symbol.keys())

        for symbol_key in sorted_symbols_for_display:
            rows_list = grouped_by_symbol[symbol_key]
            try:
                # Sort by TIMEFRAMES order, then by confidence score descending
                sorted_rows = sorted(rows_list,
                                     key=lambda r_item_sort: (TIMEFRAMES.index(r_item_sort['tf'])
                                                              if r_item_sort['tf'] in TIMEFRAMES else float('inf'),
                                                              # Handle TFs not in TIMEFRAMES gracefully
                                                              -r_item_sort['confidence_score']))
            except ValueError:
                logging.warning(
                    f"⚠️ Ошибка сортировки для {symbol_key}, возможно TF не в списке TIMEFRAMES. Сортировка только по уверенности.")
                sorted_rows = sorted(rows_list, key=lambda r_item_sort: (-r_item_sort['confidence_score']))

            table_data_tabulate = []
            for r_tab in sorted_rows:
                table_data_tabulate.append([
                    r_tab['tf'],
                    r_tab['timestamp_str_display'],
                    r_tab['signal'],
                    f"{r_tab['confidence_score']:.3f}",
                    f"{r_tab['predicted_delta']:.2%}" if pd.notna(r_tab['predicted_delta']) else "N/A",
                    f"{r_tab['avg_delta_similar']:.2%}" if pd.notna(r_tab['avg_delta_similar']) else "N/A",
                    f"{r_tab['std_delta_similar']:.2%}" if pd.notna(r_tab['std_delta_similar']) else "N/A",
                    f"{r_tab['delta_final']:.2%}" if pd.notna(r_tab['delta_final']) else "N/A",
                    "❗" if r_tab['conflict'] else " ",
                    r_tab['signal_strength'].replace(" Сильный", "🟢").replace(" Умеренный", "🟡").replace(" Слабый",
                                                                                                         "⚪"),
                    f"{r_tab['tp_hit_proba']:.1%}" if pd.notna(r_tab['tp_hit_proba']) else "N/A"
                ])
            print(f"\n📊  Прогноз по символу: {symbol_key}")
            try:
                print(tabulate(table_data_tabulate, headers=headers_tabulate, tablefmt="pretty", stralign="right",
                               numalign="right"))
            except Exception as e:
                logging.error(f"❌ Ошибка при выводе таблицы для {symbol_key}: {e}")
                # Попробуем вывести просто данные, если tabulate не работает
                print("Не удалось отформатировать таблицу. Данные:")
                for row in table_data_tabulate:
                    print(row)


    logging.info("✅  Процесс генерации прогнозов завершён.")


if __name__ == '__main__':
    # >>> ИЗМЕНЕНО: Новый парсер и логика определения фильтров из патча
    parser = argparse.ArgumentParser(description="Генерация прогнозов на основе обученных моделей.")
    parser.add_argument('--save',         action='store_true', help="Сохранять результаты прогнозов в CSV файлы.")
    # Обновляем help текст согласно патчу
    parser.add_argument('--symbol',       type=str, help="Один символ для анализа, напр. BTCUSDT (или группа, если совпадает с top8/meme/defi)")
    parser.add_argument('--symbol-group', type=str, help="Группа символов, напр. top8 или meme (игнорируется, если указан --symbol, совпадающий с группой)") # Обновляем help для ясности
    args = parser.parse_args()

    # Разводим в symbol_filter vs group_filter согласно логике патча
    symbol_filter = None
    group_filter  = None

    if args.symbol and args.symbol_group:
        logging.error("❌ Нельзя указывать одновременно --symbol и --symbol-group.")
        sys.exit(1)
    elif args.symbol_group:
        group_filter = args.symbol_group.lower() # Всегда приводим группу к нижнему регистру
        # Проверка существования группы теперь здесь, перед вызовом predict_all_tf
        if group_filter not in GROUP_MODELS:
             logging.error(f"❌ Неизвестная группа символов: '{args.symbol_group}'. Доступные группы: {list(GROUP_MODELS.keys())}")
             sys.exit(1)

    elif args.symbol:
        # если в --symbol передано имя известной группы (независимо от регистра) — считаем это group_filter
        symbol_lower = args.symbol.lower()
        if symbol_lower in GROUP_MODELS:
            group_filter = symbol_lower
            logging.info(f"Interpreting --symbol '{args.symbol}' as group '{group_filter}'.")
        else:
            symbol_filter = args.symbol.upper() # Всегда приводим символ к верхнему регистру


    try:
        predict_all_tf(
            args.save,
            symbol_filter=symbol_filter,
            group_filter=group_filter
        )
    except KeyboardInterrupt:
        print("\n[PredictAll] 🛑 Генерация прогнозов прервана пользователем (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logging.error(f"[PredictAll] 💥 Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)