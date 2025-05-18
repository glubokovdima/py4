import pandas as pd
import joblib
import argparse
import os
# from features import compute_features  # Этот импорт не работает без файла features.py
# from utils import load_latest_data    # Этот импорт не работает без файла utils.py
from datetime import datetime
import sys  # Для Ctrl+C
import logging

# Предположим, что compute_features есть в preprocess_features.py
# Но его нужно адаптировать для работы с одним DataFrame на входе
# Для простоты, этот скрипт остается как есть, с пониманием, что он может не работать "из коробки"
# без адаптации compute_features или предоставления этих недостающих модулей.

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [MultiFrame] - %(message)s',
                    stream=sys.stdout)

# === Конфигурация ===
TIMEFRAMES_MULTI = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']  # Свой список ТФ
MODEL_DIR_MULTI = 'models'  # Использует ту же папку моделей


# DATA_DIR_MULTI = 'data_binance' # Этот скрипт ожидает CSV в этой папке
# Это отличается от основного пайплайна, который использует SQLite и data/features_*.pkl
# Для работы этого скрипта нужно, чтобы data_binance/{symbol}/{tf}.csv существовали.
# И compute_features был доступен и адаптирован.

# Заглушка для compute_features, так как он не предоставлен отдельно
def compute_features_stub(df_input):
    logging.warning("Используется ЗАГЛУШКА для compute_features. Результаты могут быть некорректны!")
    # Просто добавим несколько колонок, чтобы код не падал
    df_input['rsi_stub'] = 50
    df_input['delta_stub'] = 0.01
    df_input['vol_stub'] = 0.02
    # Важно: модели обучались на определенном наборе признаков.
    # Эта заглушка не создаст их, поэтому predict упадет, если модели ожидают другие признаки.
    # Чтобы это работало, нужно либо передавать X из features_*.pkl, либо иметь реальный compute_features.
    return df_input.iloc[-1:]  # Возвращаем только последнюю строку для предсказания


def predict_one_tf(symbol_arg, tf_arg):  # Переименованы аргументы
    logging.info(f"Прогноз для {symbol_arg} на {tf_arg}")

    # Проверяем наличие моделей
    model_clf_path = f"{MODEL_DIR_MULTI}/{tf_arg}_clf_class.pkl"  # Используем _clf_class
    model_delta_path = f"{MODEL_DIR_MULTI}/{tf_arg}_reg_delta.pkl"
    model_vol_path = f"{MODEL_DIR_MULTI}/{tf_arg}_reg_vol.pkl"

    if not all(os.path.exists(p) for p in [model_clf_path, model_delta_path, model_vol_path]):
        logging.error(f"Одна или несколько моделей для {tf_arg} не найдены. Пропуск.")
        return None

    model_clf = joblib.load(model_clf_path)
    model_delta = joblib.load(model_delta_path)
    model_vol = joblib.load(model_vol_path)

    # Загрузка данных (этот путь отличается от основного пайплайна)
    # path_csv = f"{DATA_DIR_MULTI}/{symbol_arg}/{tf_arg}.csv"
    # if not os.path.exists(path_csv):
    #     logging.error(f"CSV файл не найден: {path_csv}. Пропуск {tf_arg} для {symbol_arg}.")
    #     return None
    # df_raw = pd.read_csv(path_csv)
    # df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms') # Предполагаем такой формат
    # df_raw.set_index('timestamp', inplace=True)

    # Вместо чтения CSV и compute_features, попытаемся загрузить готовые признаки из data/features_TF.pkl
    features_pkl_path = f"data/features_{tf_arg}.pkl"
    if not os.path.exists(features_pkl_path):
        logging.error(f"Файл признаков {features_pkl_path} не найден. Пропуск {tf_arg} для {symbol_arg}.")
        return None

    try:
        df_all_features = pd.read_pickle(features_pkl_path)
    except Exception as e:
        logging.error(f"Ошибка чтения {features_pkl_path}: {e}")
        return None

    df_sym_features = df_all_features[df_all_features['symbol'] == symbol_arg].sort_values('timestamp')
    if df_sym_features.empty:
        logging.warning(f"Нет данных/признаков для {symbol_arg} в {features_pkl_path}. Пропуск {tf_arg}.")
        return None

    last_row_features = df_sym_features.iloc[-1:].copy()

    # Загрузка списка признаков, на которых обучалась модель
    features_list_path = f"models/{tf_arg}_features.txt"
    if not os.path.exists(features_list_path):
        logging.error(f"Файл со списком признаков 'models/{tf_arg}_features.txt' не найден. Пропуск.")
        return None
    try:
        with open(features_list_path, "r", encoding="utf-8") as f:
            feature_cols_model = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Ошибка чтения файла со списком признаков: {e}")
        return None

    if not feature_cols_model:
        logging.error("Список признаков из файла пуст.")
        return None

    missing_cols = [col for col in feature_cols_model if col not in last_row_features.columns]
    if missing_cols:
        logging.error(f"В данных для {symbol_arg} {tf_arg} отсутствуют признаки: {missing_cols}")
        return None

    X_predict = last_row_features[feature_cols_model]
    if X_predict.isnull().values.any():
        logging.warning(f"NaN в признаках для {symbol_arg} {tf_arg}. Результат может быть неточным.")
        # X_predict = X_predict.fillna(0) # Простая обработка

    # df_with_features = compute_features_stub(df_raw) # Используем заглушку или реальную функцию
    # df_with_features.dropna(inplace=True) # Удаление NaN после вычисления признаков
    # if df_with_features.empty:
    #     logging.warning(f"Нет данных после compute_features для {symbol_arg} {tf_arg}.")
    #     return None
    # last_features = df_with_features.iloc[[-1]] # Берем последнюю строку с признаками

    # Предполагаем, что модели ожидают признаки, совместимые с compute_features_stub или реальной
    # Важно: если модели обучались на признаках из preprocess_features.py, то X_predict должен их содержать.
    try:
        proba_all = model_clf.predict_proba(X_predict)[0]
        # TARGET_CLASS_NAMES_MULTI = ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']
        # prob_up = proba_all[TARGET_CLASS_NAMES_MULTI.index('UP')] + proba_all[TARGET_CLASS_NAMES_MULTI.index('STRONG UP')]
        # Для простоты, как в оригинале, но это может быть неверно для clf_class
        prob_up = proba_all[3] + proba_all[4] if len(proba_all) == 5 else proba_all[1]

        delta = model_delta.predict(X_predict)[0]
        vol = model_vol.predict(X_predict)[0]
    except Exception as e:
        logging.error(f"Ошибка предсказания для {symbol_arg} {tf_arg}: {e}")
        # logging.debug(f"Признаки для предсказания: {X_predict.columns.tolist()}")
        return None

    momentum = delta / vol if vol > 1e-9 else 0  # Защита от деления на ноль

    # Логика сигнала из оригинала, может потребовать адаптации
    signal_text = "LONG" if prob_up > 0.7 and delta > 0.002 else \
        "SHORT" if prob_up < 0.3 and delta < -0.002 else "NEUTRAL"
    # Пороги (0.7, 0.3, 0.002) стоит сделать настраиваемыми или обосновать

    return {
        'timeframe': tf_arg,
        'prob_up_pct': round(prob_up * 100, 2),  # prob_up это уже вероятность
        'delta_pct': round(delta * 100, 2),  # delta это уже изменение в долях
        'volatility_pct': round(vol * 100, 2),  # vol это уже изменение в долях
        'momentum': round(momentum, 2),
        'signal': signal_text
    }


def summarize_results(results_list):  # Переименовано
    if not results_list:
        return "Нет результатов для суммирования."

    long_count = sum(1 for r in results_list if r and r['signal'].startswith('LONG'))
    short_count = sum(1 for r in results_list if r and r['signal'].startswith('SHORT'))

    if long_count >= 3:  # Порог для общего сигнала
        return '📈  Итоговый тренд: ВВЕРХ'
    elif short_count >= 3:
        return '📉  Итоговый тренд: ВНИЗ'
    else:
        return '⚖️  Итоговый тренд: НЕЙТРАЛЬНО'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Мультифреймовый прогноз для одного символа.")
    parser.add_argument("--symbol", type=str, required=True, help="Символ для прогноза (например, BTCUSDT)")
    args = parser.parse_args()
    symbol_main = args.symbol.upper()

    print(f"\n--- Мультифрейм-прогноз для {symbol_main} ---")
    all_results = []

    try:
        for tf_main in TIMEFRAMES_MULTI:
            try:
                result_one = predict_one_tf(symbol_main, tf_main)
                if result_one:
                    all_results.append(result_one)
                    print(f"  {result_one['timeframe']:>4s} | {result_one['signal']:<7s} | "
                          f"P(Up): {result_one['prob_up_pct']:>6.2f}% | "
                          f"Δ: {result_one['delta_pct']:>6.2f}% | "
                          f"σ: {result_one['volatility_pct']:>6.2f}% | "
                          f"Mom: {result_one['momentum']:>5.2f}")
            except Exception as e_inner:  # Ловим ошибки для одного ТФ, чтобы не прерывать весь цикл
                logging.error(f"Критическая ошибка при обработке {tf_main} для {symbol_main}: {e_inner}")

        print("\n" + summarize_results(all_results))

    except KeyboardInterrupt:
        print("\n[MultiFrame] 🛑 Прогноз прерван пользователем (Ctrl+C).")
        sys.exit(130)
    except Exception as e_outer:
        logging.error(f"[MultiFrame] 💥 Непредвиденная ошибка: {e_outer}", exc_info=True)
        sys.exit(1)

    print("--- Мультифрейм-прогноз завершён ---")