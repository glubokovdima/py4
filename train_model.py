import pandas as pd
import numpy as np
import argparse
import os
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score
from sklearn.metrics import f1_score, precision_score, confusion_matrix
# ✅ Импорт для ROC AUC
from sklearn.metrics import roc_auc_score
import joblib
import sys
import logging
from sklearn.metrics import average_precision_score
import random
import json  # Added for Change 3

SYMBOL_GROUPS = {
    "top8": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "XRPUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"
    ],
    "meme": [
        "DOGEUSDT", "PEPEUSDT", "FLOKIUSDT", "WIFUSDT", "SHIBUSDT"
    ]
}

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [TrainModel] - %(message)s',
                    stream=sys.stdout)

# Убедимся, что директории существуют
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

TARGET_CLASS_NAMES = ['DOWN', 'UP']  # For display purposes in reports
PREDICT_PROBA_THRESHOLD_CLASS = 0.55


def train_all_models(tf_train, symbol=None):  # Added symbol argument
    """
    Обучает модели CatBoost для заданного таймфрейма и/или символа.

    Args:
        tf_train (str): Таймфрейм данных (например, '1m', '5m').
        symbol (str, optional): Символ, для которого обучаются модели.
                                 Если None, предполагается групповая модель.
                                 Defaults to None.
    """
    # Define a prefix for filenames based on symbol and timeframe
    file_prefix = f"{symbol}_{tf_train}" if symbol else tf_train

    # Define a string for logging context
    log_context_str = f"таймфрейма: {tf_train}"
    if symbol:
        log_context_str += f", символа: {symbol}"

    logging.info(f"🚀  Начало обучения моделей для {log_context_str}")

    # Load features based on symbol
    if symbol:
        features_path = f"data/features_{symbol}_{tf_train}.pkl"
    else:
        features_path = f"data/features_{tf_train}.pkl"

    if not os.path.exists(features_path):
        logging.error(f"Файл признаков {features_path} не найден. Обучение для {log_context_str} невозможно.")
        return

    try:
        df = pd.read_pickle(features_path)
    except Exception as e:
        logging.error(f"Ошибка чтения файла признаков {features_path}: {e}")
        return

    if df.empty:
        logging.error(f"Файл признаков {features_path} пуст. Обучение для {log_context_str} невозможно.")
        return

    required_cols_for_dropna = ['target_class', 'delta', 'volatility']
    target_tp_hit_col = 'target_tp_hit'

    train_tp_hit_model_flag = False
    use_stratified_cv_for_tp_hit = False

    if target_tp_hit_col in df.columns:
        logging.info(f"Обнаружена колонка '{target_tp_hit_col}'. Попытка обучения модели TP-hit для {log_context_str}.")
        if df[target_tp_hit_col].isnull().all() or df[target_tp_hit_col].nunique() < 2:
            logging.warning(
                f"Колонка '{target_tp_hit_col}' содержит все NaN или только одно уникальное значение. Модель TP-hit НЕ будет обучена для {log_context_str}.")
        elif df[target_tp_hit_col].value_counts().min() < 10:
            logging.warning(
                f"Недостаточно примеров в одном из классов '{target_tp_hit_col}' ({df[target_tp_hit_col].value_counts().to_dict()}) для {log_context_str}. "
                f"Stratified CV для TP-hit может быть пропущено или обучение будет стандартным."
            )
            train_tp_hit_model_flag = True
            use_stratified_cv_for_tp_hit = False
        else:
            train_tp_hit_model_flag = True
            use_stratified_cv_for_tp_hit = True
            logging.info(
                f"Распределение классов в '{target_tp_hit_col}' (до очистки NaN) для {log_context_str}: {df[target_tp_hit_col].value_counts().to_dict()}")
        required_cols_for_dropna.append(target_tp_hit_col)
    else:
        logging.warning(
            f"Колонка '{target_tp_hit_col}' не найдена в {features_path}. "
            f"Модель TP-hit (clf_tp_hit) НЕ будет обучена для {log_context_str}."
        )

    df_cleaned = df.dropna(subset=required_cols_for_dropna).copy()
    if df_cleaned.empty:
        logging.error(
            f"DataFrame пуст после удаления NaN по колонкам {required_cols_for_dropna} для {log_context_str}. Проверьте данные.")
        return

    logging.info(f"Размер DataFrame после очистки NaN для {log_context_str}: {df_cleaned.shape}")
    if train_tp_hit_model_flag and target_tp_hit_col in df_cleaned.columns:
        logging.info(
            f"Распределение классов в '{target_tp_hit_col}' (после очистки NaN) для {log_context_str}: {df_cleaned[target_tp_hit_col].value_counts().to_dict()}")
        if df_cleaned[target_tp_hit_col].nunique() < 2 or df_cleaned[
            target_tp_hit_col].value_counts().min() < 5:
            logging.warning(
                f"После очистки NaN, недостаточно примеров или классов в '{target_tp_hit_col}' ({df_cleaned[target_tp_hit_col].value_counts().to_dict()}) для {log_context_str}. "
                f"Stratified CV для TP-hit будет пропущено."
            )
            use_stratified_cv_for_tp_hit = False

    excluded_cols_for_features = ['timestamp', 'symbol', 'target_class', 'delta', 'volatility', 'future_close']
    # Ensure 'symbol' column is excluded if it exists, especially if symbol is part of features_path name
    # but the actual features might still contain a 'symbol' column from data generation.
    if 'symbol' not in excluded_cols_for_features:
        excluded_cols_for_features.append('symbol')

    if train_tp_hit_model_flag and target_tp_hit_col in df_cleaned.columns:
        excluded_cols_for_features.append(target_tp_hit_col)

    potential_leak_cols = ['target_up']
    for col in potential_leak_cols:
        if col not in excluded_cols_for_features:
            excluded_cols_for_features.append(col)

    feature_cols_initial = [col for col in df_cleaned.columns if
                            col not in excluded_cols_for_features and df_cleaned[col].dtype in [np.int64, np.float64,
                                                                                                np.int32, np.float32,
                                                                                                bool]]
    feature_cols_initial = [col for col in feature_cols_initial if not pd.api.types.is_datetime64_any_dtype(
        df_cleaned[col]) and not pd.api.types.is_string_dtype(df_cleaned[col])]

    if not feature_cols_initial:
        logging.error(
            f"После удаления целевых и служебных колонок не осталось признаков для обучения для {log_context_str}.")
        return

    logging.info(
        f"Начальное количество признаков для обучения ({log_context_str}): {len(feature_cols_initial)}. Первые 5: {feature_cols_initial[:5]}")

    X_all_data = df_cleaned[feature_cols_initial]
    y_class_all = df_cleaned['target_class']
    y_delta_all = df_cleaned['delta']
    y_vol_all = df_cleaned['volatility']
    y_tp_hit_all = df_cleaned[
        target_tp_hit_col] if train_tp_hit_model_flag and target_tp_hit_col in df_cleaned.columns else None

    if X_all_data.empty:
        logging.error(
            f"DataFrame X_all_data (признаки) пуст для {log_context_str}. Проверьте feature_cols_initial и исходные данные.")
        return

    if len(X_all_data) < 20:
        logging.error(
            f"Слишком мало данных ({len(X_all_data)}) для разделения на train/test для {log_context_str}. Обучение невозможно.")
        return

    try:
        # ✅ 4. train_test_split со стратификацией и перемешиванием
        # Split data for classification, delta, and volatility models
        X_train_cv_pool, X_test_full, y_class_train_cv_pool, y_test_class, \
            y_delta_train_cv_pool, y_test_delta, y_vol_train_cv_pool, y_test_vol = train_test_split(
            X_all_data, y_class_all, y_delta_all, y_vol_all,
            test_size=0.15, random_state=42, shuffle=True, stratify=y_class_all
        )

        X_cv_pool = X_train_cv_pool
        y_class_cv_pool = y_class_train_cv_pool  # At this point, these are string labels

        X_train_full = X_train_cv_pool
        y_train_class = y_class_train_cv_pool  # At this point, these are string labels
        y_train_delta = y_delta_train_cv_pool
        y_train_vol = y_vol_train_cv_pool

        # Split data for TP-hit model if applicable
        if train_tp_hit_model_flag and y_tp_hit_all is not None:
            try:
                # Stratify TP-hit split if possible, else use the same split indices
                if use_stratified_cv_for_tp_hit and y_tp_hit_all.nunique() >= 2 and y_tp_hit_all.value_counts().min() >= 5:
                    # Perform a separate split for TP-hit if stratification is viable
                    # Note: This means TP-hit test set might be slightly different rows
                    # from class/delta/vol test sets, or we re-split the train/CV pool
                    # Let's stick to using the same train/test split indices for consistency
                    # and just locate the TP-hit targets based on those indices.
                    y_tp_hit_train_cv_pool = y_tp_hit_all.loc[X_train_cv_pool.index]
                    y_test_tp_hit = y_tp_hit_all.loc[X_test_full.index]
                    y_tp_hit_cv_pool = y_tp_hit_train_cv_pool # Use the same split for TP-hit CV pool
                    y_train_tp_hit = y_tp_hit_train_cv_pool # Use the same split for TP-hit train

                else:
                    # If stratification is not viable for TP-hit, just use indices from the main split
                    y_tp_hit_train_cv_pool = y_tp_hit_all.loc[X_train_cv_pool.index]
                    y_test_tp_hit = y_tp_hit_all.loc[X_test_full.index]
                    y_tp_hit_cv_pool = y_tp_hit_train_cv_pool
                    y_train_tp_hit = y_tp_hit_train_cv_pool
                    if use_stratified_cv_for_tp_hit: # Log if we intended to stratify but couldn't
                         logging.warning(f"Stratification for TP-hit split requested but not possible ({log_context_str}). Using non-stratified split indices.")
                         use_stratified_cv_for_tp_hit = False # Update flag

                if not y_train_tp_hit.empty:
                    logging.info(
                        f"Распределение классов в y_train_tp_hit ({log_context_str}): {y_train_tp_hit.value_counts().to_dict()}")
                if not y_test_tp_hit.empty:
                    logging.info(
                        f"Распределение классов в y_test_tp_hit ({log_context_str}): {y_test_tp_hit.value_counts().to_dict()}")
                if not y_tp_hit_cv_pool.empty:
                     logging.info(
                        f"Распределение классов в y_tp_hit_cv_pool ({log_context_str}): {y_tp_hit_cv_pool.value_counts().to_dict()}")


            except Exception as e:
                 logging.error(f"Ошибка при выделении y_tp_hit выборок для {log_context_str}: {e}", exc_info=True)
                 y_train_tp_hit, y_test_tp_hit, y_tp_hit_cv_pool = None, None, None
                 train_tp_hit_model_flag = False # Disable TP-hit training if data extraction fails
        else:
            y_train_tp_hit, y_test_tp_hit, y_tp_hit_cv_pool = None, None, None
            train_tp_hit_model_flag = False


    except ValueError as e:
        logging.error(
            f"Ошибка ValueError при разделении данных для {log_context_str}: {e}. Возможно, слишком мало данных или проблемы с индексами.")
        return
    except Exception as e:
        logging.error(f"Непредвиденная ошибка при разделении данных для {log_context_str}: {e}", exc_info=True)
        return

    logging.info(f"Преобразование target_class в 0/1 (DOWN=0, UP=1) для {log_context_str}...")
    label_mapping = {'DOWN': 0, 'UP': 1}

    # Apply mapping carefully, checking for empty series
    if not df_cleaned['target_class'].empty:
        y_class_all_mapped = df_cleaned['target_class'].map(label_mapping)
    else:
        y_class_all_mapped = pd.Series([], dtype=int) # Empty series if source is empty

    if not y_class_cv_pool.empty:
        y_class_cv_pool = y_class_cv_pool.map(label_mapping)
    else:
        y_class_cv_pool = pd.Series([], dtype=int)

    if not y_train_class.empty:
        y_train_class = y_train_class.map(label_mapping)
    else:
        y_train_class = pd.Series([], dtype=int)

    if not y_test_class.empty:
        y_test_class = y_test_class.map(label_mapping)
    else:
        y_test_class = pd.Series([], dtype=int)


    logging.info(
        f"Пример y_train_class после преобразования ({log_context_str}): {y_train_class.head().tolist() if not y_train_class.empty else 'пусто'}")
    logging.info(
        f"Распределение классов в y_train_class после преобразования ({log_context_str}): {y_train_class.value_counts().to_dict() if not y_train_class.empty else 'пусто'}")
    logging.info(
        f"Распределение классов в y_test_class после преобразования ({log_context_str}): {y_test_class.value_counts().to_dict() if not y_test_class.empty else 'пусто'}")
    logging.info(
        f"Распределение классов в y_class_cv_pool после преобразования ({log_context_str}): {y_class_cv_pool.value_counts().to_dict() if not y_class_cv_pool.empty else 'пусто'}")


    logging.info(f"Размеры выборок ({log_context_str}): Train/CV Pool: {len(X_cv_pool)}, Test: {len(X_test_full)}")
    if X_cv_pool.empty or X_test_full.empty:
        logging.error(f"X_cv_pool (Train/CV) или X_test_full пусты после разделения для {log_context_str}.")
        return

    logging.info(
        f"🚀 Начало отбора признаков (Top 20) для clf_class ({log_context_str}) на основе {len(X_cv_pool)} образцов...")
    top20_features = []
    if X_cv_pool.empty or y_class_cv_pool.empty:
        logging.warning(
            f"X_cv_pool или y_class_cv_pool пусты перед отбором признаков для {log_context_str}. Отбор пропускается, используются все признаки.")
        top20_features = X_cv_pool.columns.tolist()
    elif len(X_cv_pool.columns) <= 20:
        logging.info(
            f"Количество признаков ({len(X_cv_pool.columns)}) меньше или равно 20 для {log_context_str}. Отбор не требуется.")
        top20_features = X_cv_pool.columns.tolist()
    else:
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        min_samples_in_class = y_class_cv_pool.value_counts().min() if not y_class_cv_pool.empty else 0

        # Ensure enough samples for StratifiedKFold
        if min_samples_in_class < kf.get_n_splits() or y_class_cv_pool.nunique() < 2:
            logging.warning(
                f"Недостаточно примеров в классе y_class_cv_pool ({y_class_cv_pool.value_counts().to_dict() if not y_class_cv_pool.empty else 'пусто'}) "
                f"или только один класс ({y_class_cv_pool.nunique() if not y_class_cv_pool.empty else 'N/A'}) "
                f"для StratifiedKFold ({log_context_str}). Отбор признаков пропускается."
            )
            top20_features = X_cv_pool.columns.tolist()
        else:
            feature_importances_fi = np.zeros(X_cv_pool.shape[1])
            num_successful_folds = 0
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_pool, y_class_cv_pool)):
                logging.info(f"Отбор признаков ({log_context_str}): фолд {fold + 1}/{kf.get_n_splits()}")
                X_fold_train, X_fold_val = X_cv_pool.iloc[train_idx], X_cv_pool.iloc[val_idx]
                y_fold_train, y_fold_val = y_class_cv_pool.iloc[train_idx], y_class_cv_pool.iloc[val_idx]

                if X_fold_val.empty or y_fold_val.empty:
                    logging.warning(
                        f"Валидационный набор пуст в фолде {fold + 1} ({log_context_str}). Пропускаем.")
                    continue
                # Check if validation set has at least two classes for Logloss/Accuracy
                if y_fold_val.nunique() < 2:
                     logging.warning(
                        f"Валидационный набор в фолде {fold + 1} имеет только один класс ({y_fold_val.nunique()}) для {log_context_str}. Пропускаем.")
                     continue

                clf_fi = CatBoostClassifier(
                    iterations=200, learning_rate=0.05,
                    early_stopping_rounds=25, verbose=False,
                    loss_function='Logloss', # Use Logloss for binary classification
                    eval_metric='Accuracy', # Or F1, depending on preference
                    task_type="GPU", devices='0', random_seed=42
                )
                try:
                     clf_fi.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val))
                     feature_importances_fi += clf_fi.get_feature_importance(type='FeatureImportance')
                     num_successful_folds += 1
                except Exception as e:
                     logging.warning(f"Ошибка при обучении в фолде {fold + 1} для отбора признаков ({log_context_str}): {e}")


            if num_successful_folds > 0:
                fi_mean = pd.Series(feature_importances_fi / num_successful_folds,
                                    index=X_cv_pool.columns).sort_values(ascending=False)
                top20_features = fi_mean.head(20).index.tolist()
                logging.info(f"Топ-20 признаков для {log_context_str}: {top20_features}")
            else:
                logging.warning(
                    f"Не удалось успешно завершить ни одного фолда для отбора признаков ({log_context_str}). Используются все.")
                top20_features = X_cv_pool.columns.tolist()

    # Ensure selected features are actually present in the dataframes
    feature_cols_final = [col for col in top20_features if col in X_train_full.columns]
    if not feature_cols_final:
        logging.error(f"После отбора признаков не осталось ни одного признака, присутствующего в данных ({log_context_str}). Обучение невозможно.")
        return

    X_train_sel = X_train_full[feature_cols_final]
    X_test_sel = X_test_full[feature_cols_final]
    X_cv_pool_sel = X_cv_pool[feature_cols_final]


    logging.info(f"Количество признаков после отбора ({log_context_str}): {len(feature_cols_final)}")

    try:
        # Use file_prefix for saving selected features
        features_list_path = f"models/{file_prefix}_features_selected.txt"
        with open(features_list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(feature_cols_final))
        logging.info(f"Список отобранных признаков сохранен в {features_list_path}")

        features_json_path = f"models/{file_prefix}_features_selected.json"
        try:
            with open(features_json_path, "w", encoding="utf-8") as f_json:
                json.dump(feature_cols_final, f_json, ensure_ascii=False, indent=2)
            logging.info(f"Список отобранных признаков также сохранен в {features_json_path}")
        except Exception as e_json:
            logging.error(f"Ошибка при сохранении признаков в JSON ({log_context_str}): {e_json}")

    except Exception as e:
        logging.error(f"Ошибка при сохранении списка признаков ({log_context_str}): {e}")


    logging.info(f"Расчет весов классов для clf_class ({log_context_str}) на основе y_train_class (0/1)...")
    class_weights_dict_for_fit = {}
    if not y_train_class.empty:
        counts_class_in_train = y_train_class.value_counts().to_dict()
        total_class_in_train = sum(counts_class_in_train.values())
        # Calculate inverse frequency weights
        count_0 = counts_class_in_train.get(0, 0)
        if count_0 > 0:
            class_weights_dict_for_fit[0] = total_class_in_train / (2 * count_0) # /2 for balanced
        else:
            class_weights_dict_for_fit[0] = 1.0
            logging.warning(f"Класс 0 ('DOWN') отсутствует в y_train_class ({log_context_str}). Вес установлен в 1.0.")
        count_1 = counts_class_in_train.get(1, 0)
        if count_1 > 0:
            class_weights_dict_for_fit[1] = total_class_in_train / (2 * count_1) # /2 for balanced
        else:
            class_weights_dict_for_fit[1] = 1.0
            logging.warning(f"Класс 1 ('UP') отсутствует в y_train_class ({log_context_str}). Вес установлен в 1.0.")
        logging.info(f"Распределение классов в y_train_class (0/1) ({log_context_str}): {counts_class_in_train}")
    else:
        logging.warning(
            f"y_train_class пуст, веса классов не могут быть рассчитаны ({log_context_str}). Используются веса по умолчанию (1.0).")
        class_weights_dict_for_fit = {0: 1.0, 1: 1.0}

    logging.info(
        f"Веса для clf_class (CatBoost, ключи 0/1, для .fit()) ({log_context_str}): {class_weights_dict_for_fit}")

    # CatBoost CV expects class weights as a list [weight_for_class_0, weight_for_class_1]
    class_weights_list_for_cv = [
        class_weights_dict_for_fit.get(0, 1.0),
        class_weights_dict_for_fit.get(1, 1.0)
    ]
    logging.info(
        f"Веса для CV (список [вес_для_класса_0, вес_для_класса_1]) ({log_context_str}): {class_weights_list_for_cv}")


    clf_class_model = None
    num_unique_classes_clf = y_class_cv_pool.nunique() if not y_class_cv_pool.empty else 0
    if num_unique_classes_clf == 2:
        loss_func_clf = 'Logloss'
        eval_metric_clf = 'F1' # Using F1 as primary metric for imbalanced data
    elif num_unique_classes_clf == 1:
        logging.warning(
            f"В y_class_cv_pool только {num_unique_classes_clf} уникальный класс ({log_context_str}). "
            f"Обучение классификатора может быть неэффективным или невозможным. Используется Logloss/Accuracy."
        )
        loss_func_clf = 'Logloss'
        eval_metric_clf = 'Accuracy' # Fallback to Accuracy
    else:
        logging.error(
            f"В y_class_cv_pool {num_unique_classes_clf} уникальных классов ({log_context_str}). "
            f"Проверьте данные. Используется Logloss/Accuracy по умолчанию."
        )
        loss_func_clf = 'Logloss'
        eval_metric_clf = 'Accuracy' # Fallback to Accuracy


    logging.info(f"\n🔄 Запуск RandomSearch для clf_class ({log_context_str})...")
    best_score_rs = -1 # Maximize F1 or Accuracy
    best_params_rs = None
    num_rs_trials = 10 # Number of Random Search trials

    default_cv_params_class = {
        'iterations': 700, 'learning_rate': 0.03, 'depth': 6,
        'loss_function': loss_func_clf, 'eval_metric': eval_metric_clf,
        'early_stopping_rounds': 50, 'random_seed': 42,
        'task_type': 'GPU', 'devices': '0', 'verbose': 0,
        # Use list for CV parameters
        'class_weights': class_weights_list_for_cv,
    }
    cv_params_class = default_cv_params_class.copy()
    best_iter_class_cv = cv_params_class['iterations']

    min_samples_for_cv = 5 # Minimum samples per class for Stratified KFold
    if not X_cv_pool_sel.empty and not y_class_cv_pool.empty and \
            y_class_cv_pool.nunique() >= 2 and \
            (y_class_cv_pool.value_counts().min() if not y_class_cv_pool.empty else 0) >= min_samples_for_cv:

        logging.info(f"Проведение RandomSearch ({num_rs_trials} попыток) для clf_class ({log_context_str})...")
        # Define parameter grid for Random Search
        param_distributions = {
            'iterations': [300, 500, 700, 1000, 1500],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
            'bagging_temperature': [random.uniform(0.1, 1.0) for _ in range(num_rs_trials * 2)], # More options
            'depth': [4, 5, 6, 7, 8, 9, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9, 11, 15, 20],
        }

        for i in range(num_rs_trials):
            logging.info(f"RandomSearch для clf_class ({log_context_str}): попытка {i + 1}/{num_rs_trials}")
            trial_params = {
                'iterations': random.choice(param_distributions['iterations']),
                'learning_rate': random.choice(param_distributions['learning_rate']),
                'bagging_temperature': random.choice(param_distributions['bagging_temperature']),
                'depth': random.choice(param_distributions['depth']),
                'l2_leaf_reg': random.choice(param_distributions['l2_leaf_reg']),
                'loss_function': loss_func_clf,
                'eval_metric': eval_metric_clf,
                'early_stopping_rounds': 50,
                'random_seed': 42,
                'task_type': 'GPU', 'devices': '0',
                'verbose': 0, # Keep verbose low during search
                'class_weights': class_weights_list_for_cv, # Use list for CV
            }
            try:
                current_pool = Pool(
                    data=X_cv_pool_sel,
                    label=y_class_cv_pool.astype(np.int32).values,
                    feature_names=list(X_cv_pool_sel.columns),
                    cat_features=[] # Assuming no categorical features
                )
                # Use StratifiedKFold directly for CV
                stratified_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                cv_data = cv(current_pool, params=trial_params, folds=stratified_kf, plot=False)

                metric_key_rs = f'test-{eval_metric_clf}-mean'
                if metric_key_rs not in cv_data.columns:
                     logging.warning(f"Метрика '{eval_metric_clf}' не найдена в результатах CV ({log_context_str}). Доступные метрики: {list(cv_data.columns)}. Пропускаем попытку.")
                     continue

                # For metrics to maximize (Accuracy, F1, AUC), find max; for loss (Logloss), find min
                if eval_metric_clf in ['Logloss', 'RMSE', 'MultiClass']: # Add other loss functions if needed
                    current_score_rs = -cv_data[metric_key_rs].min() # Negate min for maximization
                    current_best_iter_rs = cv_data[metric_key_rs].idxmin() + 1
                else: # Assume metric to maximize (Accuracy, F1, AUC, etc.)
                    current_score_rs = cv_data[metric_key_rs].max()
                    current_best_iter_rs = cv_data[metric_key_rs].idxmax() + 1

                # Adjust iterations to be at least early_stopping_rounds + buffer
                min_iterations_needed = trial_params['early_stopping_rounds'] * 2 # Heuristic
                trial_params['iterations'] = max(int(current_best_iter_rs * 1.2), min_iterations_needed) # Add a buffer

                params_for_best_rs = trial_params.copy()
                # Use dict for final model fit
                params_for_best_rs['class_weights'] = class_weights_dict_for_fit

                logging.info(
                    f"  RandomSearch Trial {i + 1} ({log_context_str}): params={ {k: v for k, v in trial_params.items() if k in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'bagging_temperature']} }, score ({eval_metric_clf})={current_score_rs:.4f} (raw_cv_score: {cv_data[metric_key_rs].iloc[-1]:.4f})")
                if current_score_rs > best_score_rs:
                    best_score_rs = current_score_rs
                    best_params_rs = params_for_best_rs
                    logging.info(
                        f"  🎉 New best RandomSearch score ({log_context_str}): {best_score_rs:.4f} with params: { {k: v for k, v in best_params_rs.items() if k in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'bagging_temperature']} }")
            except Exception as e:
                params_causing_error = {k: v for k, v in trial_params.items()}
                logging.warning(
                    f"  Ошибка в RandomSearch trial {i + 1} ({log_context_str}) ({params_causing_error}): {e}")
                continue # Continue to the next trial

        if best_params_rs:
            cv_params_class = best_params_rs
            best_iter_class_cv = best_params_rs['iterations']
            logging.info(
                f"🎯 Лучшие параметры для clf_class от RandomSearch ({log_context_str}): { {k: v for k, v in cv_params_class.items() if k in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'bagging_temperature']} } с CV-score ({eval_metric_clf}): {best_score_rs:.4f}")
        else:
            logging.warning(f"RandomSearch не нашел лучших параметров для {log_context_str}, используются дефолтные.")
            # Ensure default params use the dict for fit
            default_cv_params_class['class_weights'] = class_weights_dict_for_fit
            cv_params_class = default_cv_params_class.copy()

    else:
        logging.warning(
            f"RandomSearch для clf_class пропущен для {log_context_str}. Недостаточно данных или классов. X_cv_pool_sel empty: {X_cv_pool_sel.empty}, y_class_cv_pool empty: {y_class_cv_pool.empty}, unique classes: {y_class_cv_pool.nunique() if not y_class_cv_pool.empty else 'N/A'}, min class count: {(y_class_cv_pool.value_counts().min() if not y_class_cv_pool.empty and y_class_cv_pool.nunique() > 0 else 'N/A')}. Используются дефолтные параметры.")
        # Ensure default params use the dict for fit
        default_cv_params_class['class_weights'] = class_weights_dict_for_fit
        cv_params_class = default_cv_params_class.copy()


    logging.info(
        f"\n🚀 Обучение финальной модели clf_class для {log_context_str} с параметрами: { {k: v for k, v in cv_params_class.items() if k in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'bagging_temperature']} }")
    # Use parameters found by RS or default parameters
    clf_class_model = CatBoostClassifier(**cv_params_class)
    # Set verbose for the final training run
    clf_class_model.set_params(verbose=100)

    if not X_train_sel.empty and not y_train_class.empty:
        eval_set_class = (X_test_sel, y_test_class) if not X_test_sel.empty and not y_test_class.empty else None
        try:
            # Ensure y_train_class is integer type for CatBoost
            clf_class_model.fit(X_train_sel, y_train_class.astype(int), eval_set=eval_set_class, plot=False)
            logging.info(f"clf_class best_iteration ({log_context_str}): {clf_class_model.get_best_iteration()}")
            if clf_class_model.get_best_score() and eval_set_class:
                validation_scores = clf_class_model.get_best_score().get('validation',
                                                                         clf_class_model.get_best_score().get(
                                                                             'validation_0'))
                if validation_scores and eval_metric_clf in validation_scores:
                    logging.info(
                        f"clf_class validation_{eval_metric_clf} ({log_context_str}): {validation_scores[eval_metric_clf]:.4f}")
            if clf_class_model is not None and hasattr(clf_class_model, 'get_feature_importance'):
                importances = clf_class_model.get_feature_importance(type='FeatureImportance')
                feature_importance_df = pd.DataFrame({'feature': X_train_sel.columns, 'importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
                logging.info(
                    f"\n🔥 Топ-10 признаков для clf_class ({log_context_str}):\n{feature_importance_df.head(10).to_string(index=False)}")
                try:
                    # Use file_prefix for feature importance log
                    fi_path = f"logs/feature_importance_clf_class_{file_prefix}.csv"
                    feature_importance_df.to_csv(fi_path, index=False)
                    logging.info(f"📁 Важность признаков для clf_class ({log_context_str}) сохранена в {fi_path}")
                except Exception as e_fi_save:
                    logging.error(f"Ошибка при сохранении важности признаков ({log_context_str}): {e_fi_save}")
        except Exception as e:
            logging.error(f"Ошибка при обучении clf_class_model ({log_context_str}): {e}", exc_info=True)
            clf_class_model = None
    else:
        logging.error(f"Обучение clf_class невозможно ({log_context_str}): X_train_sel или y_train_class пусты.")
        clf_class_model = None


    # --- Train Regression Models (Delta and Volatility) ---

    reg_delta_model = None
    logging.info(f"\n--- Статистика для reg_delta ({log_context_str}) ---")
    if not y_train_delta.empty: logging.info(
        f"y_train_delta ({len(y_train_delta)}) ({log_context_str}): min={y_train_delta.min():.6f}, max={y_train_delta.max():.6f}, mean={y_train_delta.mean():.6f}, std={y_train_delta.std():.6f}")
    if not y_test_delta.empty: logging.info(
        f"y_test_delta ({len(y_test_delta)}) ({log_context_str}): min={y_test_delta.min():.6f}, max={y_test_delta.max():.6f}, mean={y_test_delta.mean():.6f}, std={y_test_delta.std():.6f}")
    logging.info(f"Обучение reg_delta для {log_context_str}...")
    reg_delta_model = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=6,
        loss_function='RMSE', eval_metric='RMSE', custom_metric=['MAE'],
        min_data_in_leaf=10, task_type="GPU", devices='0',
        random_seed=42, verbose=100, early_stopping_rounds=50
    )
    if not X_train_sel.empty and not y_train_delta.empty:
        eval_set_delta = (X_test_sel, y_test_delta) if not X_test_sel.empty and not y_test_delta.empty else None
        try:
            reg_delta_model.fit(X_train_sel, y_train_delta, eval_set=eval_set_delta)
        except Exception as e:
            logging.error(f"Ошибка при обучении reg_delta_model ({log_context_str}): {e}", exc_info=True)
            reg_delta_model = None
    else:
        logging.error(f"Обучение reg_delta невозможно ({log_context_str}). X_train_sel или y_train_delta пусты.")
        reg_delta_model = None


    reg_vol_model = None
    logging.info(f"\n--- Статистика для reg_vol ({log_context_str}) ---")
    if not y_train_vol.empty: logging.info(
        f"y_train_vol ({len(y_train_vol)}) ({log_context_str}): min={y_train_vol.min():.6f}, max={y_train_vol.max():.6f}, mean={y_train_vol.mean():.6f}, std={y_train_vol.std():.6f}")
    if not y_test_vol.empty: logging.info(
        f"y_test_vol ({len(y_test_vol)}) ({log_context_str}): min={y_test_vol.min():.6f}, max={y_test_vol.max():.6f}, mean={y_test_vol.mean():.6f}, std={y_test_vol.std():.6f}")
    logging.info(f"Обучение reg_vol для {log_context_str}...")
    reg_vol_model = CatBoostRegressor(
        iterations=500, learning_rate=0.03, depth=6,
        loss_function='RMSE', eval_metric='RMSE', custom_metric=['MAE'],
        min_data_in_leaf=10, task_type="GPU", devices='0',
        random_seed=42, verbose=100, early_stopping_rounds=50
    )
    if not X_train_sel.empty and not y_train_vol.empty:
        eval_set_vol = (X_test_sel, y_test_vol) if not X_test_sel.empty and not y_test_vol.empty else None
        try:
            reg_vol_model.fit(X_train_sel, y_train_vol, eval_set=eval_set_vol)
        except Exception as e:
            logging.error(f"Ошибка при обучении reg_vol_model ({log_context_str}): {e}", exc_info=True)
            reg_vol_model = None
    else:
        logging.error(f"Обучение reg_vol невозможно ({log_context_str}). X_train_sel или y_train_vol пусты.")
        reg_vol_model = None


    # --- Train TP-Hit Classifier ---

    clf_tp_hit_model = None
    if train_tp_hit_model_flag and y_tp_hit_cv_pool is not None and not y_tp_hit_cv_pool.empty \
        and y_test_tp_hit is not None and not y_test_tp_hit.empty:

        logging.info(f"\nПодготовка к обучению clf_tp_hit для {log_context_str}...")

        # Ensure TP-hit targets are integer type
        y_train_tp_hit_int = y_train_tp_hit.astype(int)
        y_test_tp_hit_int = y_test_tp_hit.astype(int)
        y_tp_hit_cv_pool_int = y_tp_hit_cv_pool.astype(int)


        # Simple weight search using a small model
        best_f1 = -1
        best_weights = None
        weight_options = [1, 2, 5, 10, 15, 20, 30, 50] # Explore different weights for class 1

        logging.info(f"Поиск лучших весов для clf_tp_hit на тестовой выборке ({log_context_str})...")
        if X_test_sel.empty or y_test_tp_hit_int.empty:
             logging.warning(f"Тестовая выборка пуста для TP-hit ({log_context_str}). Пропуск поиска весов.")
             # Fallback to a default weight if search is skipped
             class_weights_tp_hit_dict_for_fit = {0: 1, 1: 10}
        else:
            for w in weight_options:
                class_weights_tp_hit_dict_for_fit_tmp = {0: 1, 1: w}
                clf_tmp = CatBoostClassifier(
                    iterations=200, # Use fewer iterations for fast weight search
                    learning_rate=0.05,
                    depth=4,
                    loss_function='Logloss',
                    eval_metric='F1', # Evaluate on F1 for TP-hit
                    random_seed=42,
                    task_type="GPU",
                    verbose=0,
                    class_weights=class_weights_tp_hit_dict_for_fit_tmp
                )
                try:
                    # Fit on train set, evaluate on test set
                    clf_tmp.fit(X_train_sel, y_train_tp_hit_int)
                    y_pred_tmp = clf_tmp.predict(X_test_sel)
                    # Ensure predictions are in the expected format (flat list/array)
                    y_pred_tmp_flat = [item[0] if isinstance(item, (np.ndarray, list)) and len(item) == 1 else item
                                          for item in y_pred_tmp]

                    score = f1_score(y_test_tp_hit_int, y_pred_tmp_flat, zero_division=0, pos_label=1)

                    logging.info(f"[TP-hit] F1 = {score:.4f} при весах {class_weights_tp_hit_dict_for_fit_tmp}")
                    if score > best_f1:
                        best_f1 = score
                        best_weights = class_weights_tp_hit_dict_for_fit_tmp
                except Exception as e:
                    logging.warning(f"[TP-hit] Ошибка при обучении с весами {class_weights_tp_hit_dict_for_fit_tmp} ({log_context_str}): {e}")
                    continue

        # If no best weights found (e.g., due to errors or empty test set), set a default
        class_weights_tp_hit_dict_for_fit = best_weights if best_weights else {0: 1, 1: 10}

        class_weights_tp_hit_list_for_cv = [
            class_weights_tp_hit_dict_for_fit.get(0, 1.0),
            class_weights_tp_hit_dict_for_fit.get(1, 1.0)
        ]
        logging.info(f"Веса для clf_tp_hit (для .fit()) ({log_context_str}): {class_weights_tp_hit_dict_for_fit}")
        logging.info(f"Веса для clf_tp_hit (для CV) ({log_context_str}): {class_weights_tp_hit_list_for_cv}")


        counts_tp_hit_train = y_train_tp_hit_int.value_counts().to_dict() if not y_train_tp_hit_int.empty else {}
        logging.info(f"Распределение классов в y_train_tp_hit (clf_tp_hit) ({log_context_str}): {counts_tp_hit_train}")


        best_iter_tp_hit = 500 # Default iterations
        # Check if CV is possible for TP-hit
        min_samples_for_tp_hit_cv = 5 # Minimum samples per class for Stratified KFold
        if use_stratified_cv_for_tp_hit and not X_cv_pool_sel.empty and not y_tp_hit_cv_pool_int.empty and \
                y_tp_hit_cv_pool_int.nunique() >= 2 and \
                (y_tp_hit_cv_pool_int.value_counts().min() if not y_tp_hit_cv_pool_int.empty else 0) >= min_samples_for_tp_hit_cv:

            logging.info(f"Запуск CV для clf_tp_hit ({log_context_str})...")
            cv_params_tp_hit = {
                'iterations': 1000, # Max iterations for CV
                'learning_rate': 0.03,
                'depth': 4,
                'loss_function': 'Logloss',
                'eval_metric': 'F1', # Use F1 for CV evaluation
                'early_stopping_rounds': 50,
                'random_seed': 42,
                'task_type': 'GPU', 'devices': '0',
                'verbose': 0,
                'class_weights': class_weights_tp_hit_list_for_cv # Use list for CV
            }
            strat_cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            try:
                pool_tp_hit_cv = Pool(
                    data=X_cv_pool_sel,
                    label=y_tp_hit_cv_pool_int.values,
                    feature_names=list(X_cv_pool_sel.columns)
                )
                cv_data_tp_hit = cv(pool_tp_hit_cv, params=cv_params_tp_hit, folds=strat_cv_folds, plot=False)
                metric_key_cv_tp_hit = f'test-{cv_params_tp_hit["eval_metric"]}-mean'

                if metric_key_cv_tp_hit in cv_data_tp_hit.columns:
                     # For F1, we maximize, so find idxmax
                    best_iter_tp_hit = cv_data_tp_hit[metric_key_cv_tp_hit].idxmax() + 1
                    # Add a buffer to the best iteration found
                    best_iter_tp_hit = max(best_iter_tp_hit, cv_params_tp_hit['early_stopping_rounds'] + 1) # Ensure at least early stopping rounds
                    best_iter_tp_hit = int(best_iter_tp_hit * 1.2) # Add 20% buffer

                    logging.info(
                        f"🔍 CatBoostCV (TP-hit) ({log_context_str}): best_iterations = {best_iter_tp_hit}, val_score ({cv_params_tp_hit['eval_metric']}) = {cv_data_tp_hit[metric_key_cv_tp_hit].iloc[-1]:.4f}")
                else:
                    logging.warning(f"Метрика '{cv_params_tp_hit['eval_metric']}' не найдена в результатах CV TP-hit ({log_context_str}). Доступные метрики: {list(cv_data_tp_hit.columns)}. Итерации по умолчанию.")
                    best_iter_tp_hit = 500 # Fallback to default
            except Exception as e:
                logging.error(f"Ошибка CV для clf_tp_hit ({log_context_str}): {e}. Итерации по умолчанию.", exc_info=True)
                best_iter_tp_hit = 500 # Fallback to default
        else:
            logging.info(f"CV для clf_tp_hit пропущен ({log_context_str}). use_stratified_cv_for_tp_hit={use_stratified_cv_for_tp_hit}, X_cv_pool_sel empty={X_cv_pool_sel.empty}, y_tp_hit_cv_pool_int empty={y_tp_hit_cv_pool_int.empty}, unique classes={y_tp_hit_cv_pool_int.nunique() if not y_tp_hit_cv_pool_int.empty else 'N/A'}, min class count={(y_tp_hit_cv_pool_int.value_counts().min() if not y_tp_hit_cv_pool_int.empty and y_tp_hit_cv_pool_int.nunique() > 0 else 'N/A')}. Итерации: {best_iter_tp_hit}")


        logging.info(f"Обучение clf_tp_hit ({log_context_str}) с {best_iter_tp_hit} итерациями...")
        # Use parameters found or default parameters
        clf_tp_hit_model = CatBoostClassifier(
            iterations=best_iter_tp_hit, learning_rate=0.03, depth=4,
            class_weights=class_weights_tp_hit_dict_for_fit, # Use dict for fit
            loss_function='Logloss', eval_metric='F1', random_state=42, verbose=100,
            task_type="GPU", devices='0', early_stopping_rounds=50
        )
        if not X_train_sel.empty and not y_train_tp_hit_int.empty:
            eval_set_tp_hit = (X_test_sel,
                               y_test_tp_hit_int) if not X_test_sel.empty and not y_test_tp_hit_int.empty else None
            try:
                clf_tp_hit_model.fit(X_train_sel, y_train_tp_hit_int, eval_set=eval_set_tp_hit)
                logging.info(f"clf_tp_hit best_iteration ({log_context_str}): {clf_tp_hit_model.get_best_iteration()}")
                if clf_tp_hit_model.get_best_score() and eval_set_tp_hit:
                    validation_scores_tp_hit = clf_tp_hit_model.get_best_score().get('validation',
                                                                             clf_tp_hit_model.get_best_score().get(
                                                                                 'validation_0'))
                    if validation_scores_tp_hit and 'F1' in validation_scores_tp_hit:
                        logging.info(
                            f"clf_tp_hit validation_F1 ({log_context_str}): {validation_scores_tp_hit['F1']:.4f}")

            except Exception as e:
                logging.error(f"Ошибка при обучении clf_tp_hit_model ({log_context_str}): {e}", exc_info=True)
                clf_tp_hit_model = None
        else:
            logging.error(f"Обучение clf_tp_hit невозможно ({log_context_str}). X_train_sel или y_train_tp_hit_int пусты.")
            clf_tp_hit_model = None

    elif train_tp_hit_model_flag:
        logging.warning(f"Не удалось подготовить данные для clf_tp_hit ({log_context_str}). Модель не будет обучена.")
        clf_tp_hit_model = None # Explicitly set to None if training was attempted but failed


    # --- Save Models ---
    logging.info(f"\n💾 Сохранение моделей для {log_context_str}...")
    if clf_class_model:
         try:
            joblib.dump(clf_class_model, f"models/{file_prefix}_clf_class.pkl")
            logging.info(f"Модель clf_class сохранена в models/{file_prefix}_clf_class.pkl")
         except Exception as e:
             logging.error(f"Ошибка при сохранении clf_class ({log_context_str}): {e}")
    else:
         logging.warning(f"Модель clf_class не обучена, пропуск сохранения для {log_context_str}.")

    if reg_delta_model:
         try:
            joblib.dump(reg_delta_model, f"models/{file_prefix}_reg_delta.pkl")
            logging.info(f"Модель reg_delta сохранена в models/{file_prefix}_reg_delta.pkl")
         except Exception as e:
             logging.error(f"Ошибка при сохранении reg_delta ({log_context_str}): {e}")
    else:
         logging.warning(f"Модель reg_delta не обучена, пропуск сохранения для {log_context_str}.")

    if reg_vol_model:
         try:
            joblib.dump(reg_vol_model, f"models/{file_prefix}_reg_vol.pkl")
            logging.info(f"Модель reg_vol сохранена в models/{file_prefix}_reg_vol.pkl")
         except Exception as e:
             logging.error(f"Ошибка при сохранении reg_vol ({log_context_str}): {e}")
    else:
         logging.warning(f"Модель reg_vol не обучена, пропуск сохранения для {log_context_str}.")


    if clf_tp_hit_model:
         try:
            joblib.dump(clf_tp_hit_model, f"models/{file_prefix}_clf_tp_hit.pkl")
            logging.info(f"Модель clf_tp_hit сохранена в models/{file_prefix}_clf_tp_hit.pkl")
         except Exception as e:
             logging.error(f"Ошибка при сохранении clf_tp_hit ({log_context_str}): {e}")
    else:
         logging.warning(f"Модель clf_tp_hit не обучена, пропуск сохранения для {log_context_str}.")


    # --- Evaluate Metrics on Test Set ---
    if X_test_sel.empty:
        logging.warning(f"X_test_sel пуст, невозможно рассчитать метрики для {log_context_str}.")
    else:
        metrics_output = f"\n📊  Метрики на тестовой выборке ({log_context_str}) с {len(feature_cols_final)} признаками:\n"

        # Metrics for clf_class
        report_class_str, acc_class_val = "N/A", "N/A"
        f1_class_val, precision_class_val = "N/A", "N/A"
        cm_df_str = "N/A"
        auc_class_val = "N/A"
        pr_auc_class_val = "N/A"

        if clf_class_model and not y_test_class.empty:
            try:
                # Ensure y_test_class is integer type for evaluation
                y_test_class_int = y_test_class.astype(int)

                proba_class_test = clf_class_model.predict_proba(X_test_sel)
                positive_class_label_int = 1 # 'UP' is mapped to 1
                model_classes_ = getattr(clf_class_model, 'classes_', [0, 1]) # Default to [0, 1] if not found
                try:
                    # Find the index corresponding to the positive class (1)
                    positive_class_idx = list(model_classes_).index(positive_class_label_int)
                except ValueError:
                    logging.warning(
                        f"Положительный класс {positive_class_label_int} не найден в model.classes_ ({model_classes_}) для {log_context_str}. Используется индекс 1 по умолчанию для вероятностей.")
                    positive_class_idx = 1 # Fallback index


                # Predict class based on threshold
                y_pred_class_flat = np.where(proba_class_test[:, positive_class_idx] > PREDICT_PROBA_THRESHOLD_CLASS,
                                             positive_class_label_int,
                                             1 - positive_class_label_int)

                # Ensure targets and predictions are flat for metrics
                y_test_class_flat = y_test_class_int.squeeze()
                y_pred_class_flat = pd.Series(y_pred_class_flat).squeeze()


                # Check unique values for reporting
                unique_labels_test_set = set(y_test_class_flat.unique()) if not y_test_class_flat.empty else set()
                unique_labels_pred_set = set(y_pred_class_flat.unique()) if not y_pred_class_flat.empty else set()
                all_present_labels_set = unique_labels_test_set | unique_labels_pred_set

                # Ensure target names align with labels [0, 1]
                report_labels_int = [0, 1]
                if not all_present_labels_set.issubset(set(report_labels_int)):
                     logging.warning(
                        f"Метки в y_test/y_pred ({all_present_labels_set}) отсутствуют в ожидаемых метках {set(report_labels_int)} для {log_context_str}. Classification report may fail or be incomplete.")
                     # Try to use labels present in data if standard ones fail
                     report_labels_int = sorted(list(all_present_labels_set))
                     target_names_for_report = [TARGET_CLASS_NAMES[i] for i in report_labels_int if i in [0,1]]
                     if len(target_names_for_report) != len(report_labels_int):
                          target_names_for_report = [f'Class_{i}' for i in report_labels_int]
                else:
                    target_names_for_report = TARGET_CLASS_NAMES # Use standard names if labels are 0 and 1

                try:
                    report_class_str = classification_report(y_test_class_flat, y_pred_class_flat,
                                                             labels=report_labels_int,
                                                             target_names=target_names_for_report,
                                                             digits=4, zero_division=0)
                except Exception as e_report:
                    logging.warning(f"Ошибка при создании classification_report для clf_class ({log_context_str}): {e_report}. Попытка рассчитать отдельные метрики.")
                    report_class_str = "Classification report failed."


                acc_class_val = accuracy_score(y_test_class_flat, y_pred_class_flat)
                # Calculate F1 and Precision for the positive class (1)
                f1_class_val = f1_score(y_test_class_flat, y_pred_class_flat, pos_label=positive_class_label_int,
                                        zero_division=0)
                precision_class_val = precision_score(y_test_class_flat, y_pred_class_flat,
                                                      pos_label=positive_class_label_int, zero_division=0)
                try:
                    # ROC AUC and PR AUC require probability scores
                    if len(np.unique(y_test_class_flat)) > 1:
                        y_score_probs = proba_class_test[:, positive_class_idx]
                        auc_class_val = roc_auc_score(y_test_class_flat, y_score_probs)
                        pr_auc_class_val = average_precision_score(y_test_class_flat, y_score_probs)
                    else:
                        logging.warning(
                            f"Не удалось рассчитать ROC AUC/PR AUC для {log_context_str}: в y_test_class_flat только один класс ({np.unique(y_test_class_flat)}).")
                        auc_class_val = "N/A (один класс в y_true)"
                        pr_auc_class_val = "N/A (один класс в y_true)"
                except Exception as e_auc:
                    logging.warning(f"Не удалось рассчитать ROC AUC/PR AUC для {log_context_str}: {e_auc}")
                    auc_class_val = "N/A (ошибка)"
                    pr_auc_class_val = "N/A (ошибка)"

                # Calculate Confusion Matrix
                try:
                    cm = confusion_matrix(y_test_class_flat, y_pred_class_flat, labels=[0, 1]) # Always use [0, 1] for CM labels if possible
                    cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in TARGET_CLASS_NAMES],
                                         columns=[f"Pred_{l}" for l in TARGET_CLASS_NAMES])
                    logging.info(f"🧠 Confusion Matrix (clf_class) для {log_context_str}:\n{cm_df}")
                    cm_df_str = cm_df.to_string()
                except Exception as e_cm:
                     logging.warning(f"Ошибка при создании Confusion Matrix для clf_class ({log_context_str}): {e_cm}")
                     cm_df_str = "N/A (ошибка)"


            except Exception as e:
                logging.error(f"Ошибка при расчете метрик clf_class для {log_context_str}: {e}", exc_info=True)
        else:
            logging.warning(
                f"Метрики clf_class не рассчитываются для {log_context_str} (модель не обучена или y_test_class пуст).")

        metrics_output += f"--- Классификатор направления (clf_class) [{eval_metric_clf}] (proba_threshold={PREDICT_PROBA_THRESHOLD_CLASS}) ---\n"
        metrics_output += report_class_str + "\n"
        metrics_output += f"   🎯 Accuracy (class): {acc_class_val:.4f}\n" if isinstance(acc_class_val,
                                                                                        float) else f"   🎯 Accuracy (class): {acc_class_val}\n"
        metrics_output += f"   🧠 F1-score (UP/1): {f1_class_val:.4f}\n" if isinstance(f1_class_val,
                                                                                      float) else f"   🧠 F1-score (UP/1): {f1_class_val}\n"
        metrics_output += f"   🎯 Precision (UP/1): {precision_class_val:.4f}\n" if isinstance(precision_class_val,
                                                                                              float) else f"   🎯 Precision (UP/1): {precision_class_val}\n"
        metrics_output += f"   🟦 ROC AUC (UP/1): {auc_class_val:.4f}\n" if isinstance(auc_class_val,
                                                                                      float) else f"   🟦 ROC AUC (UP/1): {auc_class_val}\n"
        metrics_output += f"   🔶 PR AUC (UP/1): {pr_auc_class_val:.4f}\n" if isinstance(pr_auc_class_val,
                                                                                       float) else f"   🔶 PR AUC (UP/1): {pr_auc_class_val}\n"
        metrics_output += f"   🧩 Confusion Matrix (clf_class):\n{cm_df_str}\n"


        # Metrics for reg_delta
        mae_delta_val = "N/A"
        if reg_delta_model and not y_test_delta.empty:
            try:
                y_pred_delta_test = reg_delta_model.predict(X_test_sel)
                mae_delta_val = mean_absolute_error(y_test_delta, y_pred_delta_test)
            except Exception as e:
                logging.error(f"Ошибка при расчете MAE Delta для {log_context_str}: {e}")
        else:
            logging.warning(f"MAE Delta не рассчитывается для {log_context_str} (модель не обучена или y_test_delta пуст).")
        metrics_output += f"   📈 MAE Delta: {mae_delta_val:.6f}\n" if isinstance(mae_delta_val,
                                                                                 float) else f"   📈 MAE Delta: {mae_delta_val}\n"


        # Metrics for reg_vol
        mae_vol_val = "N/A"
        if reg_vol_model and not y_test_vol.empty:
            try:
                y_pred_vol_test = reg_vol_model.predict(X_test_sel)
                mae_vol_val = mean_absolute_error(y_test_vol, y_pred_vol_test)
            except Exception as e:
                logging.error(f"Ошибка при расчете MAE Volatility для {log_context_str}: {e}")
        else:
            logging.warning(f"MAE Volatility не рассчитывается для {log_context_str} (модель не обучена или y_test_vol пуст).")
        metrics_output += f"   ⚡ MAE Volatility: {mae_vol_val:.6f}\n" if isinstance(mae_vol_val,
                                                                                    float) else f"   ⚡ MAE Volatility: {mae_vol_val}\n"


        # Metrics for clf_tp_hit
        if train_tp_hit_model_flag: # Check if TP-hit training was attempted
            metrics_output += "\n--- Классификатор TP-Hit (clf_tp_hit) ---\n"
            report_tp_hit_str, acc_tp_hit_val = "N/A", "N/A"
            f1_tp_hit_val = "N/A"
            cm_tp_hit_df_str = "N/A"

            # Check if the model exists and test data is available
            if clf_tp_hit_model and y_test_tp_hit is not None and not y_test_tp_hit.empty:
                try:
                    # Ensure y_test_tp_hit is integer type for evaluation
                    y_test_tp_hit_int = y_test_tp_hit.astype(int)

                    y_pred_tp_hit_test = clf_tp_hit_model.predict(X_test_sel)
                    # Ensure predictions are in the expected format (flat list/array)
                    y_pred_tp_hit_flat = [item[0] if isinstance(item, (np.ndarray, list)) and len(item) == 1 else item
                                          for item in y_pred_tp_hit_test]

                    # Ensure targets and predictions are flat and valid (not NaN)
                    y_test_tp_hit_flat = y_test_tp_hit_int.squeeze()
                    y_pred_tp_hit_flat = pd.Series(y_pred_tp_hit_flat).squeeze().astype(int) # Ensure int type

                    # Align indices and drop any potential NaNs introduced by alignment
                    # This might not be necessary if X_test_sel index matches y_test_tp_hit index
                    # but good practice if there were any previous data manipulations.
                    # Let's assume indices align because they came from the same split.
                    # Just ensure they are series for consistent handling.


                    if not y_test_tp_hit_flat.empty and not y_pred_tp_hit_flat.empty:
                         # Check unique values for reporting
                         unique_labels_test_set_tp = set(y_test_tp_hit_flat.unique())
                         unique_labels_pred_set_tp = set(y_pred_tp_hit_flat.unique())
                         all_present_labels_set_tp = unique_labels_test_set_tp | unique_labels_pred_set_tp

                         report_labels_tp_int = [0, 1]
                         target_names_tp = ['No TP Hit (0)', 'TP Hit (1)']

                         if not all_present_labels_set_tp.issubset(set(report_labels_tp_int)):
                              logging.warning(f"Метки в y_test/y_pred для TP-hit ({all_present_labels_set_tp}) отсутствуют в ожидаемых метках {set(report_labels_tp_int)} для {log_context_str}. Classification report may fail or be incomplete.")
                              report_labels_tp_int = sorted(list(all_present_labels_set_tp))
                              target_names_tp = [f'Class_{i}' for i in report_labels_tp_int]


                         try:
                            report_tp_hit_str = classification_report(y_test_tp_hit_flat, y_pred_tp_hit_flat,
                                                                      labels=report_labels_tp_int,
                                                                      target_names=target_names_tp,
                                                                      digits=4, zero_division=0)
                         except Exception as e_report_tp:
                             logging.warning(f"Ошибка при создании classification_report для clf_tp_hit ({log_context_str}): {e_report_tp}. Попытка рассчитать отдельные метрики.")
                             report_tp_hit_str = "Classification report failed."


                         acc_tp_hit_val = accuracy_score(y_test_tp_hit_flat, y_pred_tp_hit_flat)
                         # Calculate F1 for the positive class (1)
                         f1_tp_hit_val = f1_score(y_test_tp_hit_flat, y_pred_tp_hit_flat, zero_division=0, pos_label=1)

                         # Calculate Confusion Matrix
                         try:
                             cm_tp_hit = confusion_matrix(y_test_tp_hit_flat, y_pred_tp_hit_flat, labels=[0, 1]) # Always use [0, 1] for CM labels if possible
                             cm_tp_hit_df = pd.DataFrame(cm_tp_hit, index=["True_0", "True_1"], columns=["Pred_0", "Pred_1"])
                             logging.info(f"🧠 Confusion Matrix (clf_tp_hit) для {log_context_str}:\n{cm_tp_hit_df}")
                             cm_tp_hit_df_str = cm_tp_hit_df.to_string()
                         except Exception as e_cm_tp:
                              logging.warning(f"Ошибка при создании Confusion Matrix для clf_tp_hit ({log_context_str}): {e_cm_tp}")
                              cm_tp_hit_df_str = "N/A (ошибка)"

                    else:
                        report_tp_hit_str = "N/A (нет валидных данных для метрик)"
                        acc_tp_hit_val = "N/A (нет валидных данных)"
                        f1_tp_hit_val = "N/A (нет валидных данных)"
                        cm_tp_hit_df_str = "N/A (нет валидных данных)"
                        logging.warning(f"Нет валидных данных в y_test_tp_hit_flat или y_pred_tp_hit_flat для расчета метрик TP-hit ({log_context_str}).")

                except Exception as e:
                    logging.error(f"Ошибка при расчете метрик clf_tp_hit для {log_context_str}: {e}", exc_info=True)
            else:
                logging.warning(f"Метрики clf_tp_hit не рассчитываются для {log_context_str} (модель не обучена или y_test_tp_hit пуст).")


            metrics_output += report_tp_hit_str + "\n"
            metrics_output += f"   🎯 Accuracy (TP-hit): {acc_tp_hit_val:.4f}\n" if isinstance(acc_tp_hit_val,
                                                                                              float) else f"   🎯 Accuracy (TP-hit): {acc_tp_hit_val}\n"
            metrics_output += f"   🧠 F1-score (TP-hit): {f1_tp_hit_val:.4f}\n" if isinstance(f1_tp_hit_val,
                                                                                             float) else f"   🧠 F1-score (TP-hit): {f1_tp_hit_val}\n"
            metrics_output += f"   🧩 Confusion Matrix (clf_tp_hit):\n{cm_tp_hit_df_str}\n"


        print(metrics_output)
        # Use file_prefix for metrics log file
        log_file_path = f"logs/train_metrics_{file_prefix}.txt"
        # Append metrics if the file already exists from a previous run in the same session (e.g., group training)
        # If it's the start of the main script, it's cleared earlier.
        mode = "a" if os.path.exists(log_file_path) else "w"
        try:
            with open(log_file_path, mode, encoding="utf-8") as f_log:
                # Don't write header if appending
                # if mode == "w": f_log.write(f"=== Начало лога обучения для {log_context_str} ({pd.Timestamp.now()}) ===\n")
                f_log.write(metrics_output)
                f_log.write(f"\nИспользованные признаки ({len(feature_cols_final)} штук):\n" + ", ".join(
                    feature_cols_final) + "\n\n")
            logging.info(f"Метрики сохранены в лог: {log_file_path}")
        except Exception as e_log_save:
             logging.error(f"Ошибка при сохранении лога метрик ({log_context_str}): {e_log_save}")


    logging.info(f"✅  Обучение моделей для {log_context_str} завершено.")
    if train_tp_hit_model_flag and clf_tp_hit_model:
        logging.info(f"Напоминание: для predict_all.py рассмотрите использование predict_proba для clf_tp_hit ({log_context_str}) для гибкости.")


if __name__ == "__main__":
    # Define SYMBOL_GROUPS again if running directly, or rely on it being global
    # It's defined globally at the top, so this block is redundant here but harmless
    # SYMBOL_GROUPS = { ... } # Removed redundant definition

    parser = argparse.ArgumentParser(description="Обучение моделей CatBoost.")
    parser.add_argument('--tf', type=str, required=True, help="Таймфрейм (1m, 5m, ...)")
    parser.add_argument('--symbol-group', type=str, help="Псевдоним группы (например: top8, meme)")
    # Add --symbol argument
    parser.add_argument('--symbol', type=str, default=None, help="Символ (например: BTCUSDT)")
    args = parser.parse_args()

    # Determine which symbols/groups to process
    symbols_to_process = []
    if args.symbol_group:
        if args.symbol_group in SYMBOL_GROUPS:
            # If a symbol group is specified, train a single model for the group
            # The data loading logic in train_all_models already handles the group case
            # by looking for a file like features_top8_5m.pkl if symbol is set to 'top8'.
            symbols_to_process = [args.symbol_group] # Use the group name as the 'symbol' argument
            logging.info(f"🧩 Обучение групповой модели для: {args.symbol_group} → Объединенные данные символов: {SYMBOL_GROUPS[args.symbol_group]}")
        else:
            logging.error(f"Неизвестная группа: {args.symbol_group}. Доступные: {list(SYMBOL_GROUPS.keys())}")
            sys.exit(1)
    elif args.symbol:
        # If a single symbol is specified, train a model for that symbol
        symbols_to_process = [args.symbol]
        logging.info(f"🎯 Обучение модели для отдельного символа: {args.symbol}")
    else:
        logging.error("Не указан ни --symbol, ни --symbol-group. Укажите один из них.")
        sys.exit(1)


    # Process each symbol/group in the list
    for current_symbol in symbols_to_process:
        # Define prefix and context for main log file for the current task
        main_log_file_prefix = f"{current_symbol}_{args.tf}" if current_symbol else args.tf
        main_log_context_str = f"таймфрейма {args.tf}"
        if current_symbol:
            main_log_context_str += f", символа {current_symbol}"

        log_file_path_main = f"logs/train_metrics_{main_log_file_prefix}.txt"

        # Clear or start the log file for this specific training task
        try:
            with open(log_file_path_main, "w", encoding="utf-8") as f_clear:
                f_clear.write(f"=== Начало лога обучения для {main_log_context_str} ({pd.Timestamp.now()}) ===\n")
                f_clear.write(f"Запуск скрипта train_model.py для {main_log_context_str}\n\n")
        except Exception as e:
             logging.error(f"Ошибка при создании/очистке лог файла {log_file_path_main}: {e}")


        try:
            # Pass the current symbol (which might be a group name) and timeframe
            train_all_models(args.tf, current_symbol)
        except KeyboardInterrupt:
            print(f"\n[TrainModel] 🛑 Обучение для {main_log_context_str} прервано пользователем.")
            logging.info(f"Обучение для {main_log_context_str} прервано пользователем.")
            sys.exit(130)
        except Exception as e:
            logging.error(f"[TrainModel] 💥 Ошибка при обучении {main_log_context_str}: {e}", exc_info=True)
            # Continue to the next symbol/group if processing a list,
            # but if only one was specified, exit.
            if len(symbols_to_process) == 1:
                 sys.exit(1)
            else:
                 logging.warning(f"Продолжение обучения для следующих символов/групп после ошибки на {main_log_context_str}.")