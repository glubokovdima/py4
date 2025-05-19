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
FEATURES_PATH_TEMPLATE = 'data/features_{tf}.pkl'
# MODEL_PATH_TEMPLATE = 'models/{tf}_{model_type}.pkl' # Удалил, так как теперь используется load_model_with_fallback путями

LOG_DIR_PREDICT = 'logs'
LATEST_PREDICTIONS_FILE = os.path.join(LOG_DIR_PREDICT, 'latest_predictions.csv')
TRADE_PLAN_FILE = os.path.join(LOG_DIR_PREDICT, 'trade_plan.csv')

# Убедимся, что директория для логов/результатов существует
os.makedirs(LOG_DIR_PREDICT, exist_ok=True)

TARGET_CLASS_NAMES = ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']


def compute_final_delta(delta_model, delta_history, sigma_history):
    if pd.isna(delta_history) or pd.isna(sigma_history):
        return round(delta_model, 5)
    if sigma_history < 0.005:
        w1, w2 = 0.4, 0.6
    elif sigma_history > 0.02:
        w1, w2 = 0.8, 0.2
    else:
        alpha = (sigma_history - 0.005) / (0.02 - 0.005)
        w1 = 0.4 + alpha * (0.8 - 0.4)
        w2 = 1.0 - w1
    return round(w1 * delta_model + w2 * delta_history, 5)


def get_signal_strength(delta_final, confidence, sigma_history):
    if pd.isna(sigma_history):
        sigma_history = float('inf')

    # Немного скорректировал логику силы сигнала, чтобы она лучше соответствовала диапазонам confidence и sigma
    # Это мое предположение, можно настроить по желанию
    if abs(delta_final) > 0.02 and confidence > 0.5 and sigma_history < 0.015: # Более строгие условия для Сильного
        return "🟢 Сильный"
    elif abs(delta_final) > 0.01 and confidence > 0.2 and sigma_history < 0.025: # Умеренный
        return "🟡 Умеренный"
    else: # Слабый или неопределенный
        return "⚪ Слабый"

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
    if score > 0.2:
        return "Очень высокая уверенность"
    elif score > 0.1:
        return "Хорошая уверенность"
    elif score > 0.05:
        return "Слабый сигнал" # Это может быть границей, после которой сигнал не торгуем
    else:
        return "Низкая уверенность — лучше пропустить" # Ниже 0.05 совсем низко


def calculate_trade_levels(entry, direction, atr_value, rr=2.0):
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
    return round(sl, 6), round(tp, 6)


def similarity_analysis(X_live_df, X_hist_df, deltas_hist_series, top_n=15):
    if X_hist_df.empty or len(X_hist_df) < top_n:
        return np.nan, np.nan, "Недостаточно исторических данных для анализа схожести"

    common_cols = X_live_df.columns.intersection(X_hist_df.columns)
    # Проверим, есть ли признаки в X_live, которых нет в X_hist_df
    missing_in_hist = list(set(X_live_df.columns) - set(common_cols))
    if missing_in_hist:
         logging.warning(f"Признаки {missing_in_hist} из X_live отсутствуют в X_hist_df. Анализ будет по общим признакам.")

    # Проверим, есть ли признаки в X_hist_df, которых нет в X_live
    missing_in_live = list(set(X_hist_df.columns) - set(common_cols))
    if missing_in_live:
         logging.warning(f"Признаки {missing_in_live} из X_hist_df отсутствуют в X_live. Анализ будет по общим признакам.")


    X_live_df_common = X_live_df[common_cols]
    X_hist_df_common = X_hist_df[common_cols]

    if X_live_df_common.empty or X_hist_df_common.empty:
        return np.nan, np.nan, "Нет общих признаков для анализа схожести"

    # Убедимся, что X_hist_df_common и deltas_hist_series имеют совпадающие индексы
    # и удалим строки с NaN из признаков перед масштабированием и анализом схожести
    hist_df_aligned = X_hist_df_common.copy()
    hist_deltas_aligned = deltas_hist_series.loc[hist_df_aligned.index].copy() # Привязываем дельты к текущему индексу признаков

    # Удаляем строки, где есть NaN в признаках ИЛИ где есть NaN в дельтах истории
    valid_indices = hist_df_aligned.dropna(how='any').index.intersection(hist_deltas_aligned.dropna().index)

    hist_df_aligned_clean = hist_df_aligned.loc[valid_indices]
    hist_deltas_aligned_clean = hist_deltas_aligned.loc[valid_indices]


    if len(hist_df_aligned_clean) < top_n:
         return np.nan, np.nan, f"Недостаточно чистых исторических данных ({len(hist_df_aligned_clean)} < {top_n}) для анализа схожести"

    scaler = StandardScaler()
    try:
        X_hist_scaled = scaler.fit_transform(hist_df_aligned_clean)
        # Убедимся, что X_live_df_common имеет ту же форму и колонки, что и hist_df_aligned_clean
        # перед масштабированием. Если нет, используем только общие колонки еще раз или выведем ошибку.
        # Предполагаем, что common_cols уже обеспечил одинаковые колонки, но на всякий случай...
        if not X_live_df_common.columns.equals(hist_df_aligned_clean.columns):
             logging.error("Несоответствие колонок между X_live_df_common и hist_df_aligned_clean перед масштабированием схожести!")
             return np.nan, np.nan, "Ошибка сопоставления признаков для схожести"

        x_live_scaled = scaler.transform(X_live_df_common) # X_live_df_common - это одна строка

    except ValueError as e:
        logging.error(f"Ошибка при масштабировании в similarity_analysis: {e}")
        return np.nan, np.nan, "Ошибка масштабирования признаков"
    except Exception as e:
         logging.error(f"Непредвиденная ошибка при масштабировании в similarity_analysis: {e}", exc_info=True)
         return np.nan, np.nan, "Неизвестная ошибка масштабирования признаков"


    sims = cosine_similarity(X_hist_scaled, x_live_scaled).flatten()

    # Indices of the top N similar historical points in the *cleaned and scaled* history data
    top_indices_in_cleaned_hist = sims.argsort()[-top_n:][::-1]

    # Get the original indices from the full historical data (or from the cleaned data index)
    # Use the indices from the cleaned data dataframe
    original_top_indices = hist_df_aligned_clean.iloc[top_indices_in_cleaned_hist].index

    if len(original_top_indices) == 0:
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
        "Высокий разброс" if std_delta > 0.02 else
        "Умеренно стабильно" if std_delta > 0.01 else
        "Стабильный паттерн"
    )
    if pd.isna(avg_delta) or pd.isna(std_delta):
        hint = "Не удалось рассчитать статистику схожести"

    return avg_delta, std_delta, hint


# >>> ИЗМЕНЕНО: Добавлены параметры symbol_filter и group_filter
def predict_all_tf(save_output_flag, symbol_filter=None, group_filter=None):
    # ----- Новая фильтрация по символу или группе -----
    # >>> ДОБАВЛЕНО согласно патчу
    target_syms = None
    if symbol_filter:
        target_syms = [symbol_filter.upper()] # Приводим к верхнему регистру для безопасности
        logging.info(f"🛠️ Фильтр по символу: {target_syms[0]}")
    elif group_filter:
        group_key = group_filter.lower() # Приводим ключ группы к нижнему регистру
        if group_key not in GROUP_MODELS:
            logging.error(f"❌ Неизвестная группа символов: '{group_filter}'. Доступные группы: {list(GROUP_MODELS.keys())}")
            return # Выходим из функции, если группа не найдена
        target_syms = GROUP_MODELS[group_key]
        logging.info(f"🛠️ Фильтр по группе: '{group_filter}' ({len(target_syms)} символов)")
    else:
        logging.info("🛠️ Без фильтров по символу/группе (обработка всех доступных).")


    logging.info("🚀  Запуск генерации прогнозов...")
    all_predictions_data = []
    trade_plan = []

    for tf in TIMEFRAMES:
        logging.info(f"\n--- Обработка таймфрейма: {tf} ---")
        features_path = FEATURES_PATH_TEMPLATE.format(tf=tf)
        if not os.path.exists(features_path):
            logging.warning(f"⚠️ Файл признаков не найден: {features_path}. Пропуск таймфрейма {tf}.")
            continue

        try:
            df = pd.read_pickle(features_path)
        except Exception as e:
            logging.error(f"❌ Не удалось прочитать файл признаков {features_path}: {e}. Пропуск таймфрейма {tf}.")
            continue

        if df.empty:
            logging.warning(f"⚠️ Файл признаков пуст: {features_path}. Пропуск таймфрейма {tf}.")
            continue

        features_list_path = f"models/{tf}_features_selected.txt"
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

        # Получаем список уникальных символов из данных
        available_symbols_in_data = df['symbol'].unique()

        # Определяем, какие символы мы будем обрабатывать в этом TF
        symbols_to_process_this_tf = []
        if target_syms is None:
             # Нет фильтра, обрабатываем все символы, для которых есть данные
             symbols_to_process_this_tf = available_symbols_in_data
        else:
             # Есть фильтр, обрабатываем только символы из target_syms, для которых есть данные
             symbols_to_process_this_tf = [sym for sym in target_syms if sym in available_symbols_in_data]
             missing_filtered_symbols = [sym for sym in target_syms if sym not in available_symbols_in_data]
             if missing_filtered_symbols:
                 logging.warning(f"⚠️ Данные для символов из фильтра отсутствуют на {tf}: {missing_filtered_symbols}. Пропускаем их.")


        # Сортируем символы для предсказуемого вывода
        symbols_to_process_this_tf.sort()

        if not symbols_to_process_this_tf:
             logging.info(f"🤷 Нет символов для обработки на {tf} после применения фильтра или из-за отсутствия данных.")
             continue # Переходим к следующему таймфрейму

        for symbol in symbols_to_process_this_tf:
            # Проверка фильтра больше не нужна здесь, так как список symbols_to_process_this_tf уже отфильтрован

            df_sym = df[df['symbol'] == symbol].sort_values('timestamp').copy()
            # Пустой df_sym не должен возникать тут, т.к. символ взят из available_symbols_in_data,
            # но проверка не повредит.
            if df_sym.empty:
                logging.warning(f"⚠️ Неожиданно пустой DataFrame для {symbol} на {tf}. Пропуск.")
                continue

            logging.info(f"--- Обработка {symbol} на {tf} ---")

            # Загрузка моделей для конкретного символа и ТФ
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
            if X_live.isnull().values.any():
                nan_features = X_live.columns[X_live.isnull().any()].tolist()
                logging.warning(f"⚠️ В X_live для {symbol} {tf} есть NaN в признаках: {nan_features}. Пропуск символа {symbol} на этом таймфрейме.")
                continue

            # Анализ схожести с историей
            avg_delta_similar, std_delta_similar, similarity_hint = np.nan, np.nan, "Нет данных для анализа схожести"
            if 'delta' not in hist_df_full.columns:
                logging.debug(f"Столбец 'delta' отсутствует в исторических данных hist_df_full для {symbol} {tf}.")
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

                    # Чистка от NaN производится внутри similarity_analysis

                    # Ensure enough data for similarity analysis - moved check inside similarity_analysis
                    avg_delta_similar, std_delta_similar, similarity_hint = similarity_analysis(
                        X_live, hist_for_sim_features, hist_for_sim_deltas, top_n=min(15, len(hist_for_sim_features)-1)) # top_n не больше, чем строк-1


            # Прогнозирование
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
                # Проверяем, что предсказанная волатильность положительна
                if predicted_volatility < 0:
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
                         tp_hit_proba = tp_hit_proba_all_classes[1] # Предполагаем, что индекс 1 соответствует классу 1
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
                     if not pd.isna(sl) and not pd.isna(tp):
                         # Убедимся, что SL не равен Entry, чтобы избежать деления на ноль
                         if abs(entry_price - sl) > 1e-9:
                             rr_value = round(abs(tp - entry_price) / abs(entry_price - sl), 2)
                         else:
                             logging.warning(f"⚠️ SL равен цене входа для {symbol} {tf}. Невозможно рассчитать RR.")


                     trade_plan.append({
                         'symbol': symbol, 'tf': tf, 'entry': entry_price, 'direction': direction,
                         'sl': sl, 'tp': tp, 'rr': rr_value,
                         'confidence': confidence, 'signal': signal, 'timestamp': ts_str_log, 'hint': hint,
                         'tp_hit_proba': tp_hit_proba
                     })
                 else:
                     logging.debug(f"Пропуск {symbol} {tf} для торгового плана: TP Hit ({prediction_entry['tp_hit_proba']:.1%}) ниже порога {TRADE_PLAN_TP_HIT_THRESHOLD:.1%}.")
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
            logging.info("🤷 Нет данных для сохранения в LATEST_PREDICTIONS_FILE.")

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
                logging.info("🤷 Торговый план пуст, файл не создан.")
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
    parser = argparse.ArgumentParser(description="Генерация прогнозов на основе обученных моделей.")
    parser.add_argument('--save', action='store_true', help="Сохранять результаты прогнозов в CSV файлы.")
    # >>> ДОБАВЛЕНО согласно патчу
    parser.add_argument('--symbol',       type=str, help="Один символ для анализа, напр. BTCUSDT")
    parser.add_argument('--symbol-group', type=str, help="Группа символов, напр. top8 или meme")
    args = parser.parse_args()

    # >>> ИЗМЕНЕНО: Передача аргументов в predict_all_tf
    # Вместо глобальных переменных, передаем фильтры явно в функцию
    try:
        # Проверяем, чтобы не были указаны оба фильтра одновременно
        if args.symbol and args.symbol_group:
            logging.error("❌ Нельзя указывать одновременно --symbol и --symbol-group.")
            sys.exit(1)

        predict_all_tf(args.save, symbol_filter=args.symbol, group_filter=args.symbol_group)

    except KeyboardInterrupt:
        print("\n[PredictAll] 🛑 Генерация прогнозов прервана пользователем (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logging.error(f"[PredictAll] 💥 Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)