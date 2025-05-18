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

TIMEFRAMES = ['5m', '15m', '30m', '1h', '4h', '1d']  # Основные ТФ для прогноза
FEATURES_PATH_TEMPLATE = 'data/features_{tf}.pkl'
MODEL_PATH_TEMPLATE = 'models/{tf}_{model_type}.pkl'
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

    if abs(delta_final) > 0.02 and sigma_history < 0.01 and confidence > 0.5:
        return "🟢 Сильный"
    elif abs(delta_final) > 0.01 and sigma_history < 0.02 and confidence > 0.2:
        return "🟡 Умеренный"
    else:
        return "⚪ Слабый"


def is_conflict(delta_model, delta_history):
    if pd.isna(delta_history) or pd.isna(delta_model):
        return False
    return (delta_model > 0 and delta_history < 0) or \
        (delta_model < 0 and delta_history > 0)


def load_model(tf, model_type):
    path = MODEL_PATH_TEMPLATE.format(tf=tf, model_type=model_type)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logging.error(f"Ошибка загрузки модели {path}: {e}")
            return None
    logging.warning(f"Модель не найдена: {path}")
    return None


def get_confidence_hint(score):
    if score > 0.2:
        return "Очень высокая уверенность"
    elif score > 0.1:
        return "Хорошая уверенность"
    elif score > 0.05:
        return "Слабый сигнал"
    else:
        return "Низкая уверенность — лучше пропустить"


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
    return round(sl, 6), round(tp, 6)


def similarity_analysis(X_live_df, X_hist_df, deltas_hist_series, top_n=15):
    if X_hist_df.empty or len(X_hist_df) < top_n:
        return np.nan, np.nan, "Недостаточно исторических данных для анализа схожести"

    common_cols = X_live_df.columns.intersection(X_hist_df.columns)
    if len(common_cols) != len(X_live_df.columns):
        missing_in_hist = list(set(X_live_df.columns) - set(common_cols))

    X_live_df_common = X_live_df[common_cols]
    X_hist_df_common = X_hist_df[common_cols]

    if X_live_df_common.empty or X_hist_df_common.empty:
        return np.nan, np.nan, "Нет общих признаков для анализа схожести"

    scaler = StandardScaler()
    try:
        X_hist_scaled = scaler.fit_transform(X_hist_df_common)
        x_live_scaled = scaler.transform(X_live_df_common)
    except ValueError as e:
        logging.error(f"Ошибка при масштабировании в similarity_analysis: {e}")
        return np.nan, np.nan, "Ошибка масштабирования признаков"

    sims = cosine_similarity(X_hist_scaled, x_live_scaled).flatten()

    if len(sims) != len(deltas_hist_series.loc[X_hist_df_common.index]):
        aligned_deltas_hist_series = deltas_hist_series.loc[X_hist_df_common.index]
        if len(sims) != len(aligned_deltas_hist_series):
            logging.error(
                f"Окончательное несоответствие длин sims ({len(sims)}) и deltas_hist_series ({len(aligned_deltas_hist_series)})")
            return np.nan, np.nan, "Ошибка сопоставления схожести с дельтами"
    else:
        aligned_deltas_hist_series = deltas_hist_series.loc[X_hist_df_common.index]

    top_indices_in_sims = sims.argsort()[-top_n:][::-1]
    original_top_indices = X_hist_df_common.index[top_indices_in_sims]

    if len(original_top_indices) == 0:
        return np.nan, np.nan, "Не найдено схожих ситуаций после фильтрации"

    similar_deltas = aligned_deltas_hist_series.loc[original_top_indices]
    avg_delta = similar_deltas.mean()
    std_delta = similar_deltas.std()

    hint = (
        "Высокий разброс" if std_delta > 0.02 else
        "Умеренно стабильно" if std_delta > 0.01 else
        "Стабильный паттерн"
    )
    if pd.isna(avg_delta) or pd.isna(std_delta):
        hint = "Не удалось рассчитать статистику схожести"

    return avg_delta, std_delta, hint


def predict_all_tf(save_output_flag):
    logging.info("🚀  Запуск генерации прогнозов...")
    all_predictions_data = []
    trade_plan = []

    for tf in TIMEFRAMES:
        logging.info(f"\n--- Обработка таймфрейма: {tf} ---")
        features_path = FEATURES_PATH_TEMPLATE.format(tf=tf)
        if not os.path.exists(features_path):
            logging.warning(f"Файл признаков не найден: {features_path}. Пропуск таймфрейма {tf}.")
            continue

        try:
            df = pd.read_pickle(features_path)
        except Exception as e:
            logging.error(f"Не удалось прочитать файл признаков {features_path}: {e}. Пропуск таймфрейма {tf}.")
            continue

        if df.empty:
            logging.warning(f"Файл признаков пуст: {features_path}. Пропуск таймфрейма {tf}.")
            continue

        # ИЗМЕНЕНИЕ 2: Фиксация пути features_list_path
        features_list_path = f"models/{tf}_features_selected.txt"
        if not os.path.exists(features_list_path):
            logging.error(
                f"Файл со списком признаков '{features_list_path}' не найден. Невозможно безопасно продолжить для {tf}. Пропуск.")
            continue
        try:
            with open(features_list_path, "r", encoding="utf-8") as f:
                feature_cols_from_file = [line.strip() for line in f if line.strip()]
            if not feature_cols_from_file:
                logging.error(f"Файл со списком признаков '{features_list_path}' пуст. Пропуск {tf}.")
                continue
        except Exception as e:
            logging.error(f"Ошибка чтения файла со списком признаков '{features_list_path}': {e}. Пропуск {tf}.")
            continue

        model_class = load_model(tf, 'clf_class')
        model_delta = load_model(tf, 'reg_delta')
        model_vol = load_model(tf, 'reg_vol')
        model_tp_hit = load_model(tf, 'clf_tp_hit')

        # ИЗМЕНЕНИЕ 1: Логирование классов TP-hit модели
        if model_tp_hit:
            logging.info(f"Классы TP-hit модели ({tf}): {getattr(model_tp_hit, 'classes_', 'N/A')}")


        if not all([model_class, model_delta, model_vol]):
            logging.warning(f"Одна или несколько основных моделей для {tf} не загружены. Пропуск таймфрейма {tf}.")
            continue

        missing_cols_in_df = [col for col in feature_cols_from_file if col not in df.columns]
        if missing_cols_in_df:
            logging.error(
                f"В DataFrame из {features_path} отсутствуют столбцы {missing_cols_in_df}, "
                f"необходимые для модели {tf} (согласно {features_list_path}). Пропуск таймфрейма."
            )
            continue

        for symbol in df['symbol'].unique():
            df_sym = df[df['symbol'] == symbol].sort_values('timestamp').copy()
            if df_sym.empty:
                continue

            row_df = df_sym.iloc[-1:].copy()
            hist_df_full = df_sym.iloc[:-1].copy()

            if 'close' not in row_df.columns or 'timestamp' not in row_df.columns:
                logging.warning(f"В последних данных для {symbol} {tf} отсутствует 'close' или 'timestamp'. Пропуск.")
                continue

            X_live = row_df[feature_cols_from_file]
            if X_live.isnull().values.any():
                nan_features = X_live.columns[X_live.isnull().any()].tolist()
                logging.warning(f"В X_live для {symbol} {tf} есть NaN в признаках: {nan_features}. Пропуск символа.")
                continue

            avg_delta_similar, std_delta_similar, similarity_hint = np.nan, np.nan, "Нет данных для анализа схожести"
            if 'delta' not in hist_df_full.columns:
                logging.debug(f"Столбец 'delta' отсутствует в исторических данных hist_df_full для {symbol} {tf}.")
            else:
                missing_hist_features = [col for col in feature_cols_from_file if col not in hist_df_full.columns]
                if missing_hist_features:
                    logging.debug(f"В hist_df_full отсутствуют столбцы {missing_hist_features} для {symbol} {tf}.")
                else:
                    hist_for_sim_features = hist_df_full[feature_cols_from_file].copy()
                    hist_for_sim_deltas = hist_df_full['delta'].copy()

                    valid_indices_features = hist_for_sim_features.dropna().index
                    valid_indices_deltas = hist_for_sim_deltas.dropna().index
                    common_valid_indices = valid_indices_features.intersection(valid_indices_deltas)

                    hist_for_sim_features_clean = hist_for_sim_features.loc[common_valid_indices]
                    hist_for_sim_deltas_clean = hist_for_sim_deltas.loc[common_valid_indices]

                    if len(hist_for_sim_features_clean) >= 15:
                        avg_delta_similar, std_delta_similar, similarity_hint = similarity_analysis(
                            X_live, hist_for_sim_features_clean, hist_for_sim_deltas_clean)
                    else:
                        similarity_hint = f"Мало данных ({len(hist_for_sim_features_clean)}) для схожести"

            try:
                proba_raw = model_class.predict_proba(X_live)[0]
            except Exception as e:
                logging.error(
                    f"Ошибка при model_class.predict_proba для {symbol} {tf} с X_live shape {X_live.shape}: {e}")
                continue

            pred_class_idx = proba_raw.argmax()
            if pred_class_idx >= len(TARGET_CLASS_NAMES):
                logging.error(
                    f"Индекс предсказанного класса ({pred_class_idx}) выходит за пределы TARGET_CLASS_NAMES для {symbol} {tf}. Пропуск.")
                continue
            signal = TARGET_CLASS_NAMES[pred_class_idx]

            if len(proba_raw) < 2:
                confidence = float(proba_raw.max()) if len(proba_raw) == 1 else 0.0
            else:
                sorted_probas = np.sort(proba_raw)
                confidence = float(sorted_probas[-1] - sorted_probas[-2])

            hint = get_confidence_hint(confidence)
            predicted_delta = model_delta.predict(X_live)[0]
            predicted_volatility = model_vol.predict(X_live)[0]

            tp_hit_proba = np.nan
            if model_tp_hit:
                try:
                    tp_hit_proba_all_classes = model_tp_hit.predict_proba(X_live)[0]
                    # Проверяем, соответствуют ли классы модели ожидаемым 0 и 1
                    model_tp_classes = getattr(model_tp_hit, 'classes_', None)
                    if model_tp_classes is not None and len(model_tp_classes) == 2:
                        # Ищем индекс класса 1 (положительный исход)
                        try:
                            class_1_idx = list(model_tp_classes).index(1) # или float(1), bool(True) в зависимости от того, как обучалось
                            tp_hit_proba = tp_hit_proba_all_classes[class_1_idx]
                        except ValueError:
                            logging.warning(f"Класс '1' не найден в model_tp_hit.classes_ ({model_tp_classes}) для {symbol} {tf}. Использую индекс 1 по умолчанию.")
                            if len(tp_hit_proba_all_classes) > 1:
                                tp_hit_proba = tp_hit_proba_all_classes[1]
                            else: # Если только один класс, и он не 1
                                tp_hit_proba = 0.0 # Или другое значение по умолчанию
                    elif len(tp_hit_proba_all_classes) > 1 : # Если классов нет, но вероятностей две
                         tp_hit_proba = tp_hit_proba_all_classes[1] # По умолчанию берем вторую
                         logging.debug(f"Атрибут classes_ отсутствует у model_tp_hit или содержит не 2 класса. Использую индекс 1 для tp_hit_proba для {symbol} {tf}.")
                    elif len(tp_hit_proba_all_classes) == 1:
                        logging.warning(f"Модель TP-hit для {symbol} {tf} вернула только одну вероятность: {tp_hit_proba_all_classes[0]}. Невозможно определить tp_hit_proba.")
                        tp_hit_proba = np.nan # или 0.0, или логика на основе pred_class_idx
                    else:
                        logging.warning(f"Неожиданный формат вывода predict_proba от model_tp_hit для {symbol} {tf}: {tp_hit_proba_all_classes}")

                except Exception as e:
                    logging.error(f"Ошибка в модели TP-hit для {symbol} {tf}: {e}")


            direction = 'long' if signal in ['UP', 'STRONG UP'] else 'short' if signal in ['DOWN',
                                                                                           'STRONG DOWN'] else 'none'
            ts_obj = pd.to_datetime(row_df['timestamp'].values[0])
            ts_str_log = ts_obj.strftime('%Y-%m-%d %H:%M:%S')
            ts_str_display = ts_obj.strftime('%Y-%m-%d %H:%M')

            proba_dict = {TARGET_CLASS_NAMES[i]: proba_raw[i] for i in range(len(proba_raw))}
            entry_price = float(row_df['close'].values[0])

            sl, tp = calculate_trade_levels(entry_price, direction, predicted_volatility)
            delta_final = compute_final_delta(predicted_delta, avg_delta_similar, std_delta_similar)
            signal_strength_val = get_signal_strength(delta_final, confidence, std_delta_similar)
            conflict_flag = is_conflict(predicted_delta, avg_delta_similar)

            prediction_entry = {
                'symbol': symbol, 'tf': tf, 'timestamp_obj': ts_obj,
                'timestamp_str_log': ts_str_log, 'timestamp_str_display': ts_str_display,
                'signal': signal, 'confidence_score': confidence, 'confidence_hint': hint,
                'proba_dict': proba_dict, 'predicted_delta': predicted_delta,
                'predicted_volatility': predicted_volatility, 'entry': entry_price,
                'sl': sl, 'tp': tp, 'direction': direction,
                'avg_delta_similar': avg_delta_similar, 'std_delta_similar': std_delta_similar,
                'similarity_hint': similarity_hint,
                'delta_final': delta_final, 'signal_strength': signal_strength_val, 'conflict': conflict_flag,
                'tp_hit_proba': tp_hit_proba,
                'error': None
            }
            all_predictions_data.append(prediction_entry)

            if direction != 'none' and confidence > 0.05 and (pd.isna(tp_hit_proba) or tp_hit_proba > 0.6):
                rr_value = 0
                if not pd.isna(sl) and not pd.isna(tp) and abs(entry_price - sl) > 1e-9:
                    rr_value = round(abs(tp - entry_price) / abs(entry_price - sl), 2)

                trade_plan.append({
                    'symbol': symbol, 'tf': tf, 'entry': entry_price, 'direction': direction,
                    'sl': sl, 'tp': tp, 'rr': rr_value,
                    'confidence': confidence, 'signal': signal, 'timestamp': ts_str_log, 'hint': hint,
                    'tp_hit_proba': tp_hit_proba
                })

    if save_output_flag and all_predictions_data:
        logging.info("Сохранение результатов...")
        df_out_list = []
        proba_dict_keys_example = TARGET_CLASS_NAMES

        for r_item in all_predictions_data:
            item = {
                'symbol': r_item['symbol'], 'tf': r_item['tf'],
                'timestamp': r_item['timestamp_obj'].strftime('%Y-%m-%d %H:%M:%S'),
                'signal': r_item['signal'], 'confidence_score': r_item['confidence_score'],
                'predicted_delta': r_item['predicted_delta'],
                'predicted_volatility': r_item['predicted_volatility'],
                'avg_delta_similar': r_item['avg_delta_similar'],
                'std_delta_similar': r_item['std_delta_similar'],
                'delta_final': r_item['delta_final'],
                'entry': r_item['entry'], 'sl': r_item['sl'], 'tp': r_item['tp'],
                'direction': r_item['direction'],
                'signal_strength': r_item['signal_strength'], 'conflict': r_item['conflict'],
                'tp_hit_proba': r_item['tp_hit_proba'],
                'confidence_hint': r_item['confidence_hint'],
                'similarity_hint': r_item['similarity_hint'],
            }
            if r_item['proba_dict']:
                item.update(r_item['proba_dict'])
                proba_dict_keys_example = list(r_item['proba_dict'].keys())

            df_out_list.append(item)

        df_out = pd.DataFrame(df_out_list)
        csv_columns_order = [
            'symbol', 'tf', 'timestamp', 'signal', 'confidence_score', 'tp_hit_proba',
            'predicted_delta', 'avg_delta_similar', 'std_delta_similar', 'delta_final',
            'predicted_volatility', 'entry', 'sl', 'tp', 'direction',
            'signal_strength', 'conflict', 'confidence_hint', 'similarity_hint'
        ]
        csv_columns_order.extend(proba_dict_keys_example)
        final_csv_columns = [col for col in csv_columns_order if col in df_out.columns]
        for col in df_out.columns:
            if col not in final_csv_columns:
                final_csv_columns.append(col)

        if not df_out.empty:
            df_out = df_out[final_csv_columns]
            try:
                df_out.to_csv(LATEST_PREDICTIONS_FILE, index=False, float_format='%.6f')
                logging.info(f"📄  Сигналы сохранены в: {LATEST_PREDICTIONS_FILE}")
            except Exception as e:
                logging.error(f"Не удалось сохранить {LATEST_PREDICTIONS_FILE}: {e}")

        if trade_plan:
            df_trade_plan = pd.DataFrame(trade_plan)
            trade_plan_cols_order = [
                'symbol', 'tf', 'timestamp', 'direction', 'signal', 'confidence', 'tp_hit_proba',
                'entry', 'sl', 'tp', 'rr', 'hint'
            ]
            final_trade_plan_cols = [col for col in trade_plan_cols_order if col in df_trade_plan.columns]
            for col in df_trade_plan.columns:
                if col not in final_trade_plan_cols:
                    final_trade_plan_cols.append(col)

            if not df_trade_plan.empty:
                df_trade_plan = df_trade_plan[final_trade_plan_cols]
                try:
                    df_trade_plan.to_csv(TRADE_PLAN_FILE, index=False, float_format='%.6f')
                    logging.info(f"📈  Торговый план сохранён в: {TRADE_PLAN_FILE}")
                except Exception as e:
                    logging.error(f"Не удалось сохранить {TRADE_PLAN_FILE}: {e}")
            else:
                logging.info("Торговый план пуст, файл не создан.")
        else:
            logging.info("Нет данных для торгового плана.")

    if all_predictions_data:
        headers_tabulate = ["TF", "Timestamp", "Сигнал", "Conf.", "ΔModel", "ΔHist", "σHist", "ΔFinal", "Конф?", "Сила",
                            "TP Hit%"]
        grouped_by_symbol = {}
        for row_data_item in all_predictions_data:
            grouped_by_symbol.setdefault(row_data_item['symbol'], []).append(row_data_item)

        for symbol_key, rows_list in grouped_by_symbol.items():
            try:
                sorted_rows = sorted(rows_list,
                                     key=lambda r_item_sort: (TIMEFRAMES.index(r_item_sort['tf']),
                                                              -r_item_sort['confidence_score']))
            except ValueError:
                sorted_rows = sorted(rows_list, key=lambda r_item_sort: (-r_item_sort['confidence_score']))

            table_data_tabulate = []
            for r_tab in sorted_rows:
                table_data_tabulate.append([
                    r_tab['tf'],
                    r_tab['timestamp_str_display'],
                    r_tab['signal'],
                    f"{r_tab['confidence_score']:.3f}",
                    f"{r_tab['predicted_delta']:.2%}" if not pd.isna(r_tab['predicted_delta']) else "N/A",
                    f"{r_tab['avg_delta_similar']:.2%}" if not pd.isna(r_tab['avg_delta_similar']) else "N/A",
                    f"{r_tab['std_delta_similar']:.2%}" if not pd.isna(r_tab['std_delta_similar']) else "N/A",
                    f"{r_tab['delta_final']:.2%}" if not pd.isna(r_tab['delta_final']) else "N/A",
                    "❗" if r_tab['conflict'] else " ",
                    r_tab['signal_strength'].replace(" Сильный", "🟢").replace(" Умеренный", "🟡").replace(" Слабый", "⚪"),
                    f"{r_tab['tp_hit_proba']:.1%}" if not pd.isna(r_tab['tp_hit_proba']) else "N/A"
                ])
            print(f"\n📊  Прогноз по символу: {symbol_key}")
            try:
                print(tabulate(table_data_tabulate, headers=headers_tabulate, tablefmt="pretty", stralign="right",
                               numalign="right"))
            except Exception as e:
                logging.error(f"Ошибка при выводе таблицы для {symbol_key}: {e}")

    logging.info("✅  Процесс генерации прогнозов завершён.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Генерация прогнозов на основе обученных моделей.")
    parser.add_argument('--save', action='store_true', help="Сохранять результаты прогнозов в CSV файлы.")
    args = parser.parse_args()

    try:
        predict_all_tf(args.save)
    except KeyboardInterrupt:
        print("\n[PredictAll] 🛑 Генерация прогнозов прервана пользователем (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logging.error(f"[PredictAll] 💥 Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)