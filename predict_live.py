import pandas as pd
import argparse
from datetime import datetime
import os
import joblib
import sys  # Для Ctrl+C
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [PredictLive] - %(message)s',
                    stream=sys.stdout)

FEATURES_PATH_CONFIG = {  # Переименовано, чтобы не конфликтовать с FEATURES_PATH_TEMPLATE
    '5m': 'data/features_5m.pkl',
    '15m': 'data/features_15m.pkl',
    '30m': 'data/features_30m.pkl'
    # Можно добавить другие ТФ, если для них есть отдельные features файлы и модели
}


def load_latest_features(tf_key):
    path = FEATURES_PATH_CONFIG.get(tf_key)
    if not path:
        logging.error(f"Путь к файлу признаков для {tf_key} не определен в FEATURES_PATH_CONFIG.")
        return None
    if not os.path.exists(path):
        logging.error(f"Файл признаков не найден: {path}")
        return None
    try:
        df = pd.read_pickle(path)
        if df.empty:
            logging.warning(f"Файл признаков {path} пуст.")
            return None
        # Предполагаем, что для predict_live нужен только последний прогноз по одному символу (например, BTCUSDT)
        # или нужно будет передавать символ
        # Для примера, возьмем последнюю запись любого символа
        return df.sort_values('timestamp').iloc[-1:]
    except Exception as e:
        logging.error(f"Ошибка чтения файла признаков {path}: {e}")
        return None


def load_model_live(model_path_live):  # Переименовано
    if not os.path.exists(model_path_live):
        logging.error(f"Модель не найдена: {model_path_live}")
        return None
    try:
        return joblib.load(model_path_live)
    except Exception as e:
        logging.error(f"Ошибка загрузки модели {model_path_live}: {e}")
        return None


def predict_single_tf(tf_to_predict):  # Переименовано
    logging.info(f"🚀 Запуск live-прогноза для таймфрейма: {tf_to_predict}")

    # Пути к моделям (эти имена могут отличаться от тех, что в train_model.py)
    # train_model.py сохраняет как _clf_class.pkl, _reg_delta.pkl, _reg_vol.pkl
    # Этот скрипт ожидает _clf_up.pkl. Нужно будет или переименовать модели,
    # или изменить ожидаемые имена здесь, или этот скрипт устарел.
    # Для примера, будем использовать имена из train_model.py
    path_class = f'models/{tf_to_predict}_clf_class.pkl'  # ИЗМЕНЕНО
    path_delta = f'models/{tf_to_predict}_reg_delta.pkl'
    path_vol = f'models/{tf_to_predict}_reg_vol.pkl'

    models_ok = True
    for p in [path_class, path_delta, path_vol]:
        if not os.path.exists(p):
            logging.error(f"Модель не найдена: {p}")
            models_ok = False
    if not models_ok:
        logging.error(f"ℹ Обучите модели: python train_model.py --tf {tf_to_predict}")
        return

    df_features = load_latest_features(tf_to_predict)
    if df_features is None or df_features.empty:
        logging.error(f"Нет данных для прогноза ({tf_to_predict})")
        return

    # Загрузка списка признаков, на которых обучалась модель
    features_list_path = f"models/{tf_to_predict}_features.txt"
    if not os.path.exists(features_list_path):
        logging.error(f"Файл со списком признаков 'models/{tf_to_predict}_features.txt' не найден. Прогноз невозможен.")
        return
    try:
        with open(features_list_path, "r", encoding="utf-8") as f:
            feature_cols_from_file = [line.strip() for line in f if line.strip()]
    except Exception as e:
        logging.error(f"Ошибка чтения файла со списком признаков: {e}")
        return

    if not feature_cols_from_file:
        logging.error("Список признаков из файла пуст.")
        return

    # Убедимся, что все нужные колонки есть в df_features
    missing_cols = [col for col in feature_cols_from_file if col not in df_features.columns]
    if missing_cols:
        logging.error(f"В загруженных признаках отсутствуют колонки: {missing_cols}")
        return

    X_live = df_features[feature_cols_from_file]
    if X_live.isnull().values.any():
        logging.warning(f"В данных для прогноза {tf_to_predict} есть NaN. Результат может быть неточным.")
        # X_live = X_live.fillna(0) # Простая обработка NaN, лучше делать это на этапе подготовки признаков

    # Прогнозы
    model_class_live = load_model_live(path_class)
    model_delta_live = load_model_live(path_delta)
    model_vol_live = load_model_live(path_vol)

    if not all([model_class_live, model_delta_live, model_vol_live]):
        logging.error("Не удалось загрузить одну или несколько моделей.")
        return

    try:
        proba_all_classes = model_class_live.predict_proba(X_live)[0]
        # proba_up - это вероятность класса 'UP' или 'STRONG UP'.
        # Индексы классов: ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']
        # UP это индекс 3, STRONG UP это индекс 4
        # Если модель была _clf_up (бинарная), то proba_all_classes[1]
        # Для _clf_class (мультиклассовой):
        proba_up_combined = proba_all_classes[3] + proba_all_classes[4] if len(proba_all_classes) == 5 else \
        proba_all_classes[1]  # Пример

        pred_class_idx = proba_all_classes.argmax()
        # TARGET_CLASS_NAMES из predict_all.py или определить здесь
        # TARGET_CLASS_NAMES_LIVE = ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']
        # signal = TARGET_CLASS_NAMES_LIVE[pred_class_idx]
        # Для простоты, как в оригинале:
        signal = 'UP' if proba_up_combined > 0.55 else 'DOWN' if proba_up_combined < 0.45 else 'NEUTRAL'

        delta = model_delta_live.predict(X_live)[0]
        volat = model_vol_live.predict(X_live)[0]
    except Exception as e:
        logging.error(f"Ошибка во время предсказания: {e}")
        return

    ts_val = pd.to_datetime(df_features['timestamp'].values[0])
    symbol_val = df_features['symbol'].values[0]

    print(f"\n--- Прогноз для {symbol_val} [{tf_to_predict}] на {ts_val.strftime('%Y-%m-%d %H:%M')} ---")
    print(f"  Вероятность UP/STRONG_UP: {proba_up_combined:.2%}")
    # print(f"  Все вероятности классов: {proba_all_classes}") # Для отладки
    print(f"  ↗️  Ожидаемое изменение (delta): {delta:.2%}")
    print(f"  ⚡ Ожидаемая волатильность: {volat:.2%}")
    print(f"  🚦 Сигнал (упрощенный): {signal}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live-прогноз для одного таймфрейма.")
    parser.add_argument('--tf', type=str, default='15m', choices=list(FEATURES_PATH_CONFIG.keys()),
                        help=f"Таймфрейм. Доступные: {', '.join(FEATURES_PATH_CONFIG.keys())}")
    args = parser.parse_args()

    try:
        predict_single_tf(args.tf)
    except KeyboardInterrupt:
        print("\n[PredictLive] 🛑 Прогноз прерван пользователем (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logging.error(f"[PredictLive] 💥 Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)