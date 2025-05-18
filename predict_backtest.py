import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys  # Для Ctrl+C
import logging  # Для более детального логгирования

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [Backtest] - %(message)s',
                    stream=sys.stdout)

TIMEFRAMES_CHOICES = ['5m', '15m', '30m', '1h', '4h', '1d']  # Доступные ТФ для бэктеста
FEATURES_PATH_TEMPLATE = 'data/features_{tf}.pkl'
MODEL_PATH_TEMPLATE = 'models/{tf}_{model_type}.pkl'
OUTPUT_DIR_BACKTEST = 'logs'  # Используем LOGS_DIR или свою константу
OUTPUT_CSV_PATH_TEMPLATE = os.path.join(OUTPUT_DIR_BACKTEST, 'backtest_results_{tf}.csv')
OUTPUT_PLOT_PATH_TEMPLATE = os.path.join(OUTPUT_DIR_BACKTEST, 'backtest_accuracy_{tf}.png')
OUTPUT_CONF_MATRIX_PATH_TEMPLATE = os.path.join(OUTPUT_DIR_BACKTEST, 'backtest_conf_matrix_{tf}.png')

TARGET_CLASS_NAMES = ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']

os.makedirs(OUTPUT_DIR_BACKTEST, exist_ok=True)


def load_model_backtest(tf, model_type):  # Переименовано
    path = MODEL_PATH_TEMPLATE.format(tf=tf, model_type=model_type)
    if os.path.exists(path):
        try:
            # logging.debug(f"Загрузка модели: {path}")
            return joblib.load(path)
        except Exception as e:
            logging.error(f"Ошибка загрузки модели {path}: {e}")
            return None
    logging.warning(f"Модель не найдена: {path}")
    return None


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, tf_suffix=''):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # logging.info("Нормализованная матрица ошибок")
    # else:
    # logging.info('Матрица ошибок, без нормализации')
    # print(cm) # Вывод матрицы в консоль

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + f' ({tf_suffix})')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plot_path = OUTPUT_CONF_MATRIX_PATH_TEMPLATE.format(tf=tf_suffix)
    try:
        plt.savefig(plot_path)
        logging.info(f"Матрица ошибок сохранена: {plot_path}")
    except Exception as e:
        logging.error(f"Не удалось сохранить матрицу ошибок {plot_path}: {e}")
    plt.close()


def plot_daily_accuracy(df_result_plot, tf_plot):  # Переименованы аргументы
    if df_result_plot.empty or 'timestamp' not in df_result_plot.columns:
        logging.warning(f"Нет данных или столбца 'timestamp' для построения графика Accuracy по дням для {tf_plot}.")
        return

    df_result_plot['timestamp'] = pd.to_datetime(df_result_plot['timestamp'])
    df_result_plot['date'] = df_result_plot['timestamp'].dt.date

    # Группируем по дате и считаем accuracy для каждого дня
    daily_accuracy_data = []
    for date_val, group in df_result_plot.groupby('date'):
        if not group.empty:
            acc = (group['true_label_idx'] == group['pred_label_idx']).sum() / len(group)
            daily_accuracy_data.append({'date': date_val, 'accuracy': acc, 'samples': len(group)})

    if not daily_accuracy_data:
        logging.warning(f"Нет данных после группировки по дням для {tf_plot}.")
        return

    accuracy_by_day_df = pd.DataFrame(daily_accuracy_data)
    accuracy_by_day_df = accuracy_by_day_df.sort_values('date')

    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_by_day_df['date'], accuracy_by_day_df['accuracy'], marker='o', linestyle='-',
             label='Daily Accuracy')

    # Скользящее среднее для сглаживания
    if len(accuracy_by_day_df) >= 7:
        accuracy_by_day_df['accuracy_ma_7'] = accuracy_by_day_df['accuracy'].rolling(window=7, min_periods=1).mean()
        plt.plot(accuracy_by_day_df['date'], accuracy_by_day_df['accuracy_ma_7'], linestyle='--',
                 label='7-day MA Accuracy')

    plt.title(f'Accuracy по дням — {tf_plot}')
    plt.xlabel('Дата')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_path_val = OUTPUT_PLOT_PATH_TEMPLATE.format(tf=tf_plot)
    try:
        plt.savefig(plot_path_val)
        logging.info(f"График Accuracy по дням сохранён: {plot_path_val}")
    except Exception as e:
        logging.error(f"Не удалось сохранить график Accuracy {plot_path_val}: {e}")
    plt.close()


def run_backtest(tf_backtest):  # Переименовано
    logging.info(f"🚀  Запуск бэктеста для таймфрейма: {tf_backtest}")
    features_file_path = FEATURES_PATH_TEMPLATE.format(tf=tf_backtest)
    if not os.path.exists(features_file_path):
        logging.error(f"Нет файла признаков {features_file_path} для {tf_backtest}. Бэктест невозможен.")
        return pd.DataFrame()

    try:
        df = pd.read_pickle(features_file_path)
    except Exception as e:
        logging.error(f"Не удалось прочитать файл признаков {features_file_path}: {e}. Пропуск {tf_backtest}.")
        return pd.DataFrame()

    if df.empty:
        logging.warning(f"Пустой датафрейм признаков для {tf_backtest}. Бэктест невозможен.")
        return pd.DataFrame()

    model_class = load_model_backtest(tf_backtest, 'clf_class')
    if not model_class:
        logging.error(f"Не найдена модель clf_class для {tf_backtest}. Бэктест невозможен.")
        return pd.DataFrame()

    # Загрузка списка признаков, на которых обучалась модель
    features_list_path = f"models/{tf_backtest}_features.txt"
    if not os.path.exists(features_list_path):
        logging.error(
            f"Файл со списком признаков 'models/{tf_backtest}_features.txt' не найден. Бэктест невозможен для {tf_backtest}.")
        return pd.DataFrame()
    try:
        with open(features_list_path, "r", encoding="utf-8") as f:
            feature_cols_from_file = [line.strip() for line in f if line.strip()]
        if not feature_cols_from_file:
            logging.error(f"Файл со списком признаков 'models/{tf_backtest}_features.txt' пуст. Бэктест невозможен.")
            return pd.DataFrame()
    except Exception as e:
        logging.error(
            f"Ошибка чтения файла со списком признаков 'models/{tf_backtest}_features.txt': {e}. Бэктест невозможен.")
        return pd.DataFrame()

    # logging.debug(f"Для бэктеста {tf_backtest} будут использованы признаки: {feature_cols_from_file}")

    missing_cols_in_df = [col for col in feature_cols_from_file if col not in df.columns]
    if missing_cols_in_df:
        logging.error(
            f"В DataFrame из {features_file_path} отсутствуют столбцы {missing_cols_in_df}, "
            f"необходимые для модели {tf_backtest} (согласно {features_list_path}). Бэктест невозможен."
        )
        return pd.DataFrame()

    # Убедимся, что целевая колонка есть
    if 'target_class' not in df.columns:
        logging.error(f"Целевая колонка 'target_class' отсутствует в {features_file_path}. Бэктест невозможен.")
        return pd.DataFrame()

    # Удаляем строки с NaN в признаках или целевой переменной
    df_cleaned = df.dropna(subset=feature_cols_from_file + ['target_class']).copy()
    if df_cleaned.empty:
        logging.warning(f"Нет данных после очистки NaN для {tf_backtest}. Бэктест невозможен.")
        return pd.DataFrame()

    X_test_data = df_cleaned[feature_cols_from_file]
    # y_true_named - это имена классов ('UP', 'NEUTRAL', etc.)
    y_true_named = df_cleaned['target_class']

    # Преобразуем имена классов в индексы для метрик и сравнения
    # Создаем словарь для маппинга имен классов в индексы
    class_to_idx = {name: i for i, name in enumerate(TARGET_CLASS_NAMES)}
    # Убедимся, что все y_true_named есть в TARGET_CLASS_NAMES
    y_true_named = y_true_named[y_true_named.isin(TARGET_CLASS_NAMES)]
    if y_true_named.empty:
        logging.error(f"После фильтрации по TARGET_CLASS_NAMES в y_true не осталось значений для {tf_backtest}.")
        return pd.DataFrame()

    # Обновляем X_test_data, чтобы соответствовать отфильтрованным y_true_named
    X_test_data = X_test_data.loc[y_true_named.index]
    if X_test_data.empty:
        logging.error(f"X_test_data пуст после выравнивания с y_true_named для {tf_backtest}.")
        return pd.DataFrame()

    y_true_indices = y_true_named.map(class_to_idx)

    try:
        probas = model_class.predict_proba(X_test_data)
        y_pred_indices = probas.argmax(axis=1)  # Индексы предсказанных классов
    except Exception as e:
        logging.error(f"Ошибка при predict_proba или argmax для {tf_backtest}: {e}")
        return pd.DataFrame()

    # Расчет confidence (разница между топ-1 и топ-2 вероятностями)
    if probas.shape[1] < 2:  # Если модель предсказывает только один класс
        confidence_scores = probas.max(axis=1)
    else:
        # Сортируем вероятности по убыванию для каждой строки
        sorted_probas_desc = -np.sort(-probas, axis=1)  # Сортировка по убыванию
        confidence_scores = sorted_probas_desc[:, 0] - sorted_probas_desc[:, 1]

    df_result_out = pd.DataFrame({
        'timestamp': df_cleaned.loc[y_true_named.index, 'timestamp'],  # Берем timestamp от выровненных данных
        'symbol': df_cleaned.loc[y_true_named.index, 'symbol'],
        'true_label_idx': y_true_indices,
        'pred_label_idx': y_pred_indices,
        'confidence_score': confidence_scores
    })
    # Добавляем текстовые метки для читаемости
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    df_result_out['true_label_text'] = df_result_out['true_label_idx'].map(idx_to_class)
    df_result_out['pred_label_text'] = df_result_out['pred_label_idx'].map(idx_to_class)

    logging.info(f"\n--- Классификационный отчёт ({tf_backtest}) ---")
    # Используем y_true_indices и y_pred_indices для classification_report
    # target_names должны соответствовать порядку индексов 0, 1, 2...
    report = classification_report(y_true_indices, y_pred_indices, target_names=TARGET_CLASS_NAMES, zero_division=0)
    print(report)

    # Матрица ошибок
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=list(range(len(TARGET_CLASS_NAMES))))
    plot_confusion_matrix(cm, classes=TARGET_CLASS_NAMES, title='Матрица ошибок', tf_suffix=tf_backtest)
    plot_confusion_matrix(cm, classes=TARGET_CLASS_NAMES, normalize=True, title='Нормализованная матрица ошибок',
                          tf_suffix=tf_backtest)

    return df_result_out


def analyze_confidence_backtest(df_result_conf, tf_conf):  # Переименованы аргументы
    if df_result_conf.empty:
        logging.warning(f"Нет данных для анализа уверенности для {tf_conf}.")
        return

    logging.info(f"\n--- Анализ по confidence_score ({tf_conf}) ---")
    bins = [0, 0.05, 0.1, 0.2, 0.4, 1.01]  # Добавлен бин 0.4-1.0, 1.01 для включения 1.0
    labels = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.4", ">0.4"]
    df_result_conf['conf_group'] = pd.cut(df_result_conf['confidence_score'], bins=bins, labels=labels,
                                          right=False)  # right=False: [0, 0.05)

    results_table = []
    for label_val in labels:
        group_df = df_result_conf[df_result_conf['conf_group'] == label_val]
        if group_df.empty:
            results_table.append((label_val, np.nan, 0, 0))  # Добавляем count и correct_preds
            continue

        correct_preds = (group_df['true_label_idx'] == group_df['pred_label_idx']).sum()
        total_in_group = len(group_df)
        acc = correct_preds / total_in_group if total_in_group > 0 else 0
        results_table.append((label_val, acc, total_in_group, correct_preds))

    print(f"{'Диапазон Conf.':<15} | {'Accuracy':<9} | {'Samples':<9} | {'Correct':<9}")
    print("-" * 50)
    for label_print, acc_print, n_print, correct_print in results_table:
        acc_str = f"{acc_print:.4f}" if not pd.isna(acc_print) else "N/A"
        print(f"{label_print:<15} | {acc_str:<9} | {n_print:<9} | {correct_print:<9}")


def main_backtest():
    parser = argparse.ArgumentParser(description="Бэктестирование классификационной модели.")
    parser.add_argument('--tf', type=str, choices=TIMEFRAMES_CHOICES, required=True,
                        help=f"Таймфрейм для бэктеста. Доступные: {', '.join(TIMEFRAMES_CHOICES)}")
    args = parser.parse_args()

    df_result_main = run_backtest(args.tf)

    if not df_result_main.empty:
        output_csv_file = OUTPUT_CSV_PATH_TEMPLATE.format(tf=args.tf)
        try:
            df_result_main.to_csv(output_csv_file, index=False)
            logging.info(f"Результаты бэктеста сохранены в {output_csv_file}")
        except Exception as e:
            logging.error(f"Не удалось сохранить результаты бэктеста в {output_csv_file}: {e}")

        plot_daily_accuracy(df_result_main, args.tf)
        analyze_confidence_backtest(df_result_main, args.tf)
    else:
        logging.warning(f"Бэктест для {args.tf} не дал результатов. Файлы и графики не созданы.")

    logging.info(f"✅  Бэктест для {args.tf} завершён.")


if __name__ == '__main__':
    try:
        main_backtest()
    except KeyboardInterrupt:
        print("\n[Backtest] 🛑 Бэктест прерван пользователем (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logging.error(f"[Backtest] 💥 Непредвиденная ошибка: {e}", exc_info=True)
        sys.exit(1)