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
import random

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [TrainModel] - %(message)s',
                    stream=sys.stdout)

# Убедимся, что директории существуют
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

TARGET_CLASS_NAMES = ['DOWN', 'UP']  # For display purposes in reports
PREDICT_PROBA_THRESHOLD_CLASS = 0.55


def train_all_models(tf_train):
    logging.info(f"🚀  Начало обучения моделей для таймфрейма: {tf_train}")
    features_path = f"data/features_{tf_train}.pkl"
    if not os.path.exists(features_path):
        logging.error(f"Файл признаков {features_path} не найден. Обучение для {tf_train} невозможно.")
        return

    try:
        df = pd.read_pickle(features_path)
    except Exception as e:
        logging.error(f"Ошибка чтения файла признаков {features_path}: {e}")
        return

    if df.empty:
        logging.error(f"Файл признаков {features_path} пуст. Обучение для {tf_train} невозможно.")
        return

    required_cols_for_dropna = ['target_class', 'delta', 'volatility']
    target_tp_hit_col = 'target_tp_hit'

    train_tp_hit_model_flag = False
    use_stratified_cv_for_tp_hit = False

    if target_tp_hit_col in df.columns:
        logging.info(f"Обнаружена колонка '{target_tp_hit_col}'. Попытка обучения модели TP-hit.")
        if df[target_tp_hit_col].isnull().all() or df[target_tp_hit_col].nunique() < 2:
            logging.warning(
                f"Колонка '{target_tp_hit_col}' содержит все NaN или только одно уникальное значение. Модель TP-hit НЕ будет обучена.")
        elif df[target_tp_hit_col].value_counts().min() < 10:
            logging.warning(
                f"Недостаточно примеров в одном из классов '{target_tp_hit_col}' ({df[target_tp_hit_col].value_counts().to_dict()}). "
                f"Stratified CV для TP-hit может быть пропущено или обучение будет стандартным."
            )
            train_tp_hit_model_flag = True
            use_stratified_cv_for_tp_hit = False
        else:
            train_tp_hit_model_flag = True
            use_stratified_cv_for_tp_hit = True
            logging.info(
                f"Распределение классов в '{target_tp_hit_col}' (до очистки NaN): {df[target_tp_hit_col].value_counts().to_dict()}")
        required_cols_for_dropna.append(target_tp_hit_col)
    else:
        logging.warning(
            f"Колонка '{target_tp_hit_col}' не найдена в {features_path}. "
            f"Модель TP-hit (clf_tp_hit) НЕ будет обучена."
        )

    df_cleaned = df.dropna(subset=required_cols_for_dropna).copy()
    if df_cleaned.empty:
        logging.error(
            f"DataFrame пуст после удаления NaN по колонкам {required_cols_for_dropna} для {tf_train}. Проверьте данные.")
        return

    logging.info(f"Размер DataFrame после очистки NaN: {df_cleaned.shape}")
    if train_tp_hit_model_flag and target_tp_hit_col in df_cleaned.columns:
        logging.info(
            f"Распределение классов в '{target_tp_hit_col}' (после очистки NaN): {df_cleaned[target_tp_hit_col].value_counts().to_dict()}")
        if df_cleaned[target_tp_hit_col].nunique() < 2 or df_cleaned[
            target_tp_hit_col].value_counts().min() < 5:
            logging.warning(
                f"После очистки NaN, недостаточно примеров или классов в '{target_tp_hit_col}' ({df_cleaned[target_tp_hit_col].value_counts().to_dict()}). "
                f"Stratified CV для TP-hit будет пропущено."
            )
            use_stratified_cv_for_tp_hit = False

    excluded_cols_for_features = ['timestamp', 'symbol', 'target_class', 'delta', 'volatility', 'future_close']
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
        logging.error(f"После удаления целевых и служебных колонок не осталось признаков для обучения для {tf_train}.")
        return

    logging.info(
        f"Начальное количество признаков для обучения: {len(feature_cols_initial)}. Первые 5: {feature_cols_initial[:5]}")

    X_all_data = df_cleaned[feature_cols_initial]
    y_class_all = df_cleaned['target_class']
    y_delta_all = df_cleaned['delta']
    y_vol_all = df_cleaned['volatility']
    y_tp_hit_all = df_cleaned[
        target_tp_hit_col] if train_tp_hit_model_flag and target_tp_hit_col in df_cleaned.columns else None

    if X_all_data.empty:
        logging.error(
            f"DataFrame X_all_data (признаки) пуст для {tf_train}. Проверьте feature_cols_initial и исходные данные.")
        return

    if len(X_all_data) < 20:
        logging.error(
            f"Слишком мало данных ({len(X_all_data)}) для разделения на train/test для {tf_train}. Обучение невозможно.")
        return

    try:
        X_train_cv_pool, X_test_full, y_class_train_cv_pool, y_test_class, \
            y_delta_train_cv_pool, y_test_delta, y_vol_train_cv_pool, y_test_vol = train_test_split(
            X_all_data, y_class_all, y_delta_all, y_vol_all,
            test_size=0.15, random_state=42, shuffle=False  # Ensure y_class_all is used here
        )

        X_cv_pool = X_train_cv_pool
        y_class_cv_pool = y_class_train_cv_pool  # At this point, these are string labels

        X_train_full = X_train_cv_pool
        y_train_class = y_class_train_cv_pool  # At this point, these are string labels
        y_train_delta = y_delta_train_cv_pool
        y_train_vol = y_vol_train_cv_pool

        if train_tp_hit_model_flag and y_tp_hit_all is not None:
            y_tp_hit_train_cv_pool = y_tp_hit_all.loc[X_train_cv_pool.index]
            y_test_tp_hit = y_tp_hit_all.loc[X_test_full.index]
            y_tp_hit_cv_pool = y_tp_hit_train_cv_pool
            y_train_tp_hit = y_tp_hit_train_cv_pool

            if not y_train_tp_hit.empty:
                logging.info(
                    f"Распределение классов в y_train_tp_hit: {y_train_tp_hit.value_counts().to_dict()}")
            if not y_test_tp_hit.empty:
                logging.info(f"Распределение классов в y_test_tp_hit: {y_test_tp_hit.value_counts().to_dict()}")
        else:
            y_train_tp_hit, y_test_tp_hit, y_tp_hit_cv_pool = None, None, None
            train_tp_hit_model_flag = False

    except ValueError as e:
        logging.error(
            f"Ошибка ValueError при разделении данных для {tf_train}: {e}. Возможно, слишком мало данных или проблемы с индексами.")
        return
    except Exception as e:
        logging.error(f"Непредвиденная ошибка при разделении данных для {tf_train}: {e}", exc_info=True)
        return

    # --- ВНЕСЕНИЕ ИЗМЕНЕНИЙ: Преобразование target_class в 0/1 ---
    logging.info("Преобразование target_class в 0/1 (DOWN=0, UP=1)...")
    label_mapping = {'DOWN': 0, 'UP': 1}

    # Применяем .map() к каждой серии. .map() возвращает новую серию.
    if not y_class_all.empty:
        y_class_all = y_class_all.map(label_mapping)
    if not y_class_cv_pool.empty:  # y_class_cv_pool is an alias for y_class_train_cv_pool initially
        y_class_cv_pool = y_class_cv_pool.map(label_mapping)
    if not y_train_class.empty:  # y_train_class is an alias for y_class_train_cv_pool initially
        y_train_class = y_train_class.map(label_mapping)
    if not y_test_class.empty:
        y_test_class = y_test_class.map(label_mapping)

    # Если y_class_train_cv_pool использовалась для создания y_class_cv_pool и y_train_class,
    # и они должны быть теми же объектами, то нужно было мапить y_class_train_cv_pool
    # и переприсваивать y_class_cv_pool = y_class_train_cv_pool, y_train_class = y_class_train_cv_pool
    # Однако, так как каждая из них мапится индивидуально, это покрывает все случаи.
    # Убедимся, что y_class_train_cv_pool также обновлена, если она используется где-то еще
    # или если y_class_cv_pool/y_train_class не являются ее прямыми потребителями после маппинга.
    # В данном коде y_class_cv_pool и y_train_class - это те, что используются далее.

    logging.info(
        f"Пример y_train_class после преобразования: {y_train_class.head().tolist() if not y_train_class.empty else 'пусто'}")
    logging.info(
        f"Распределение классов в y_train_class после преобразования: {y_train_class.value_counts().to_dict() if not y_train_class.empty else 'пусто'}")
    logging.info(
        f"Распределение классов в y_test_class после преобразования: {y_test_class.value_counts().to_dict() if not y_test_class.empty else 'пусто'}")
    logging.info(
        f"Распределение классов в y_class_cv_pool после преобразования: {y_class_cv_pool.value_counts().to_dict() if not y_class_cv_pool.empty else 'пусто'}")
    # --- КОНЕЦ ВНЕСЕНИЯ ИЗМЕНЕНИЙ ---

    logging.info(f"Размеры выборок: Train/CV Pool: {len(X_cv_pool)}, Test: {len(X_test_full)}")
    if X_cv_pool.empty or X_test_full.empty:
        logging.error(f"X_cv_pool (Train/CV) или X_test_full пусты после разделения для {tf_train}.")
        return

    logging.info(
        f"🚀 Начало отбора признаков (Top 20) для clf_class ({tf_train}) на основе {len(X_cv_pool)} образцов...")
    top20_features = []
    if X_cv_pool.empty or y_class_cv_pool.empty:
        logging.warning(
            f"X_cv_pool или y_class_cv_pool пусты перед отбором признаков. Отбор пропускается, используются все признаки.")
        top20_features = X_cv_pool.columns.tolist()
    elif len(X_cv_pool.columns) <= 20:
        logging.info(
            f"Количество признаков ({len(X_cv_pool.columns)}) меньше или равно 20. Отбор не требуется.")
        top20_features = X_cv_pool.columns.tolist()
    else:
        # y_class_cv_pool теперь содержит 0/1, что подходит для StratifiedKFold
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        min_samples_in_class = y_class_cv_pool.value_counts().min() if not y_class_cv_pool.empty else 0

        if min_samples_in_class < kf.get_n_splits():
            logging.warning(
                f"Недостаточно примеров в классе y_class_cv_pool ({y_class_cv_pool.value_counts().to_dict() if not y_class_cv_pool.empty else 'пусто'}) "
                f"для StratifiedKFold. Отбор признаков пропускается."
            )
            top20_features = X_cv_pool.columns.tolist()
        else:
            feature_importances_fi = np.zeros(X_cv_pool.shape[1])
            num_successful_folds = 0
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_cv_pool, y_class_cv_pool)):
                logging.info(f"Отбор признаков: фолд {fold + 1}/{kf.get_n_splits()}")
                X_fold_train, X_fold_val = X_cv_pool.iloc[train_idx], X_cv_pool.iloc[val_idx]
                y_fold_train, y_fold_val = y_class_cv_pool.iloc[train_idx], y_class_cv_pool.iloc[val_idx]

                if X_fold_val.empty or y_fold_val.empty:
                    logging.warning(
                        f"Валидационный набор пуст в фолде {fold + 1}. Пропускаем.")
                    continue
                clf_fi = CatBoostClassifier(
                    iterations=200, learning_rate=0.05,
                    early_stopping_rounds=25, verbose=False,
                    loss_function='Logloss',  # Logloss is suitable for 0/1 binary targets
                    task_type="GPU", devices='0', random_seed=42
                )
                clf_fi.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val))
                feature_importances_fi += clf_fi.get_feature_importance()
                num_successful_folds += 1

            if num_successful_folds > 0:
                fi_mean = pd.Series(feature_importances_fi / num_successful_folds,
                                    index=X_cv_pool.columns).sort_values(ascending=False)
                top20_features = fi_mean.head(20).index.tolist()
                logging.info(f"Топ-20 признаков для {tf_train}: {top20_features}")
            else:
                logging.warning(
                    f"Не удалось успешно завершить ни одного фолда для отбора признаков. Используются все.")
                top20_features = X_cv_pool.columns.tolist()

    X_train_sel = X_train_full[top20_features]
    X_test_sel = X_test_full[top20_features]
    X_cv_pool_sel = X_cv_pool[top20_features]

    logging.info(f"Количество признаков после отбора: {len(top20_features)}")
    feature_cols_final = top20_features
    try:
        features_list_path = f"models/{tf_train}_features_selected.txt"
        with open(features_list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(feature_cols_final))
        logging.info(f"Список отобранных признаков сохранен в {features_list_path}")
    except Exception as e:
        logging.error(f"Ошибка при сохранении списка признаков: {e}")

    # --- ВНЕСЕНИЕ ИЗМЕНЕНИЙ: Расчет class_weights для 0/1 классов ---
    logging.info(f"Расчет весов классов для clf_class ({tf_train}) на основе y_train_class (0/1)...")
    class_weights_final_for_catboost = {}
    if not y_train_class.empty:
        counts_class_in_train = y_train_class.value_counts().to_dict()  # e.g., {0: N_down, 1: N_up}
        total_class_in_train = sum(counts_class_in_train.values())

        # Calculate for class 0 (formerly 'DOWN')
        count_0 = counts_class_in_train.get(0, 0)
        if count_0 > 0:
            class_weights_final_for_catboost[0] = total_class_in_train / count_0
        else:
            class_weights_final_for_catboost[0] = 1.0
            logging.warning("Класс 0 ('DOWN') отсутствует в y_train_class. Вес установлен в 1.0.")

        # Calculate for class 1 (formerly 'UP')
        count_1 = counts_class_in_train.get(1, 0)
        if count_1 > 0:
            class_weights_final_for_catboost[1] = total_class_in_train / count_1
        else:
            class_weights_final_for_catboost[1] = 1.0
            logging.warning("Класс 1 ('UP') отсутствует в y_train_class. Вес установлен в 1.0.")

        logging.info(f"Распределение классов в y_train_class (0/1): {counts_class_in_train}")
    else:
        logging.warning(
            "y_train_class пуст, веса классов не могут быть рассчитаны. Используются веса по умолчанию (1.0).")
        class_weights_final_for_catboost = {0: 1.0, 1: 1.0}  # Default if no data

    logging.info(f"Веса для clf_class (CatBoost, ключи 0/1): {class_weights_final_for_catboost}")
    # --- КОНЕЦ ВНЕСЕНИЯ ИЗМЕНЕНИЙ ---

    clf_class_model = None
    num_unique_classes_clf = y_class_cv_pool.nunique() if not y_class_cv_pool.empty else 0
    if num_unique_classes_clf == 2:
        loss_func_clf = 'Logloss'  # Suitable for 0/1 binary classification
        eval_metric_clf = 'Accuracy'  # Or 'AUC', 'F1'
    elif num_unique_classes_clf == 1:
        logging.warning(
            f"В y_class_cv_pool только {num_unique_classes_clf} уникальный класс. "
            f"Обучение классификатора может быть неэффективным или невозможным. Используется Logloss/Accuracy."
        )
        loss_func_clf = 'Logloss'
        eval_metric_clf = 'Accuracy'
    else:  # 0 or more than 2 (should not happen for binary 'target_class' after mapping)
        logging.error(
            f"В y_class_cv_pool {num_unique_classes_clf} уникальных классов. "
            f"Проверьте данные. Используется Logloss/Accuracy по умолчанию."
        )
        loss_func_clf = 'Logloss'
        eval_metric_clf = 'Accuracy'

    logging.info(f"\n🔄 Запуск RandomSearch для clf_class ({tf_train})...")
    best_score_rs = -1
    best_params_rs = None
    num_rs_trials = 10

    default_cv_params_class = {
        'iterations': 700, 'learning_rate': 0.03, 'depth': 6,
        'loss_function': loss_func_clf, 'eval_metric': eval_metric_clf,
        'early_stopping_rounds': 50, 'random_seed': 42,
        'task_type': 'GPU', 'devices': '0', 'verbose': 0,
        'class_weights': class_weights_final_for_catboost,  # Now uses 0/1 keys
    }
    cv_params_class = default_cv_params_class.copy()
    best_iter_class_cv = cv_params_class['iterations']

    min_samples_for_cv = 5  # CatBoost default for n_splits=5 in cv
    if not X_cv_pool_sel.empty and not y_class_cv_pool.empty and \
            y_class_cv_pool.nunique() >= 2 and \
            (y_class_cv_pool.value_counts().min() if not y_class_cv_pool.empty else 0) >= min_samples_for_cv:
        for i in range(num_rs_trials):
            logging.info(f"RandomSearch для clf_class: попытка {i + 1}/{num_rs_trials}")
            trial_params = {
                'iterations': random.choice([300, 500, 700, 1000]),
                'learning_rate': random.choice([0.01, 0.03, 0.05, 0.07]),
                'depth': random.choice([4, 5, 6, 7, 8]),
                'l2_leaf_reg': random.choice([1, 3, 5, 7, 9]),
                'loss_function': loss_func_clf,
                'eval_metric': eval_metric_clf,
                'early_stopping_rounds': 50,
                'random_seed': 42,
                'task_type': 'GPU', 'devices': '0',
                'verbose': 0,
                'class_weights': class_weights_final_for_catboost,  # Uses 0/1 keys
            }
            try:
                # y_class_cv_pool is already 0/1
                current_pool = Pool(
                    data=X_cv_pool_sel,
                    label=y_class_cv_pool,
                    feature_names=list(X_cv_pool_sel.columns),
                    cat_features=[]  # явно указать, что категориальных нет
                )
                cv_data = cv(current_pool, params=trial_params, fold_count=5, shuffle=True, stratified=True, plot=False)
                metric_key_rs = f'test-{eval_metric_clf}-mean'
                if eval_metric_clf in ['Logloss', 'RMSE', 'MultiClass']:  # Metrics to minimize
                    current_score_rs = -cv_data[metric_key_rs].min()  # Maximize the negative of a minimization metric
                    current_best_iter_rs = cv_data[metric_key_rs].idxmin() + 1
                else:  # Metrics to maximize (Accuracy, AUC, F1)
                    current_score_rs = cv_data[metric_key_rs].max()
                    current_best_iter_rs = cv_data[metric_key_rs].idxmax() + 1

                trial_params['iterations'] = current_best_iter_rs
                if trial_params['iterations'] <= trial_params['early_stopping_rounds']:
                    trial_params['iterations'] = trial_params[
                                                     'early_stopping_rounds'] * 2 + 1  # Ensure more iterations than early stopping

                logging.info(
                    f"  RandomSearch Trial {i + 1}: params={ {k: v for k, v in trial_params.items() if k in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg']} }, score ({eval_metric_clf})={current_score_rs:.4f} (raw_cv_score: {cv_data[metric_key_rs].iloc[-1]:.4f})")
                if current_score_rs > best_score_rs:
                    best_score_rs = current_score_rs
                    best_params_rs = trial_params.copy()
                    logging.info(
                        f"  🎉 New best RandomSearch score: {best_score_rs:.4f} with params: { {k: v for k, v in best_params_rs.items() if k in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg']} }")
            except Exception as e:
                logging.warning(f"  Ошибка в RandomSearch trial {i + 1} ({trial_params}): {e}")
                continue
        if best_params_rs:
            cv_params_class = best_params_rs
            best_iter_class_cv = best_params_rs['iterations']  # This is already set in best_params_rs
            logging.info(
                f"🎯 Лучшие параметры для clf_class от RandomSearch: { {k: v for k, v in cv_params_class.items() if k in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg']} } с CV-score ({eval_metric_clf}): {best_score_rs:.4f}")
        else:
            logging.warning("RandomSearch не нашел лучших параметров, используются дефолтные.")
    else:
        logging.warning(
            f"RandomSearch для clf_class пропущен. Недостаточно данных или классов. X_cv_pool_sel empty: {X_cv_pool_sel.empty}, y_class_cv_pool empty: {y_class_cv_pool.empty}, unique classes: {y_class_cv_pool.nunique() if not y_class_cv_pool.empty else 'N/A'}, min class count: {(y_class_cv_pool.value_counts().min() if not y_class_cv_pool.empty and y_class_cv_pool.nunique() > 0 else 'N/A')}. Используются дефолтные параметры.")

    logging.info(
        f"\n🚀 Обучение финальной модели clf_class для {tf_train} с параметрами: { {k: v for k, v in cv_params_class.items() if k in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg']} }")
    clf_class_model = CatBoostClassifier(**cv_params_class)  # class_weights with 0/1 keys passed here
    clf_class_model.set_params(verbose=100)

    if not X_train_sel.empty and not y_train_class.empty:
        # y_train_class and y_test_class are already 0/1
        eval_set_class = (X_test_sel, y_test_class) if not X_test_sel.empty and not y_test_class.empty else None
        try:
            clf_class_model.fit(X_train_sel, y_train_class, eval_set=eval_set_class, plot=False)
            logging.info(f"clf_class best_iteration: {clf_class_model.get_best_iteration()}")
            if clf_class_model.get_best_score() and eval_set_class:
                validation_scores = clf_class_model.get_best_score().get('validation',
                                                                         clf_class_model.get_best_score().get(
                                                                             'validation_0'))
                if validation_scores:
                    logging.info(
                        f"clf_class validation_{eval_metric_clf}: {validation_scores[eval_metric_clf]:.4f}")
            if clf_class_model is not None and hasattr(clf_class_model, 'get_feature_importance'):
                importances = clf_class_model.get_feature_importance()
                feature_importance_df = pd.DataFrame({'feature': X_train_sel.columns, 'importance': importances})
                feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
                logging.info(
                    f"\n🔥 Топ-10 признаков для clf_class ({tf_train}):\n{feature_importance_df.head(10).to_string(index=False)}")
                try:
                    fi_path = f"logs/feature_importance_clf_class_{tf_train}.csv"
                    feature_importance_df.to_csv(fi_path, index=False)
                    logging.info(f"📁 Важность признаков для clf_class сохранена в {fi_path}")
                except Exception as e_fi_save:
                    logging.error(f"Ошибка при сохранении важности признаков: {e_fi_save}")
        except Exception as e:
            logging.error(f"Ошибка при обучении clf_class_model: {e}")
            clf_class_model = None
    else:
        logging.error("Обучение clf_class невозможно: X_train_sel или y_train_class пусты.")
        clf_class_model = None

    reg_delta_model = None
    logging.info(f"--- Статистика для reg_delta ({tf_train}) ---")
    if not y_train_delta.empty: logging.info(
        f"y_train_delta ({len(y_train_delta)}): min={y_train_delta.min():.6f}, max={y_train_delta.max():.6f}, mean={y_train_delta.mean():.6f}, std={y_train_delta.std():.6f}")
    if not y_test_delta.empty: logging.info(
        f"y_test_delta ({len(y_test_delta)}): min={y_test_delta.min():.6f}, max={y_test_delta.max():.6f}, mean={y_test_delta.mean():.6f}, std={y_test_delta.std():.6f}")
    logging.info(f"Обучение reg_delta для {tf_train}...")
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
            logging.error(f"Ошибка при обучении reg_delta_model: {e}")
            reg_delta_model = None
    else:
        logging.error("Обучение reg_delta невозможно.")
        reg_delta_model = None

    reg_vol_model = None
    logging.info(f"--- Статистика для reg_vol ({tf_train}) ---")
    if not y_train_vol.empty: logging.info(
        f"y_train_vol ({len(y_train_vol)}): min={y_train_vol.min():.6f}, max={y_train_vol.max():.6f}, mean={y_train_vol.mean():.6f}, std={y_train_vol.std():.6f}")
    if not y_test_vol.empty: logging.info(
        f"y_test_vol ({len(y_test_vol)}): min={y_test_vol.min():.6f}, max={y_test_vol.max():.6f}, mean={y_test_vol.mean():.6f}, std={y_test_vol.std():.6f}")
    logging.info(f"Обучение reg_vol для {tf_train}...")
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
            logging.error(f"Ошибка при обучении reg_vol_model: {e}")
            reg_vol_model = None
    else:
        logging.error("Обучение reg_vol невозможно.")
        reg_vol_model = None

    clf_tp_hit_model = None
    if train_tp_hit_model_flag and y_tp_hit_cv_pool is not None and y_test_tp_hit is not None:
        logging.info(f"\nПодготовка к обучению clf_tp_hit для {tf_train}...")
        class_weights_tp_hit_final_model = {0: 1, 1: 10}  # Assuming TP-hit is 0/1
        counts_tp_hit_train = y_train_tp_hit.value_counts().to_dict() if not y_train_tp_hit.empty else {}
        logging.info(f"Распределение классов в y_train_tp_hit (clf_tp_hit): {counts_tp_hit_train}")
        logging.info(f"Веса для clf_tp_hit: {class_weights_tp_hit_final_model}")
        best_iter_tp_hit = 300
        if use_stratified_cv_for_tp_hit and not X_cv_pool_sel.empty and not y_tp_hit_cv_pool.empty and \
                y_tp_hit_cv_pool.nunique() >= 2 and \
                (y_tp_hit_cv_pool.value_counts().min() if not y_tp_hit_cv_pool.empty else 0) >= min_samples_for_cv:
            logging.info("Запуск CV для clf_tp_hit...")
            cv_params_tp_hit = {
                'iterations': 500, 'learning_rate': 0.03, 'depth': 4,
                'loss_function': 'Logloss', 'eval_metric': 'Accuracy',
                'early_stopping_rounds': 50, 'random_seed': 42,
                'task_type': 'GPU', 'devices': '0', 'verbose': 0,
                'class_weights': class_weights_tp_hit_final_model
            }
            strat_cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            try:
                pool_tp_hit_cv = Pool(X_cv_pool_sel, y_tp_hit_cv_pool.astype(int),
                                      feature_names=list(X_cv_pool_sel.columns))
                cv_data_tp_hit = cv(pool_tp_hit_cv, params=cv_params_tp_hit, folds=strat_cv_folds, plot=False)
                metric_key_cv_tp_hit = f'test-{cv_params_tp_hit["eval_metric"]}-mean'
                if cv_params_tp_hit["eval_metric"] in ['Logloss', 'RMSE']:
                    best_iter_tp_hit = cv_data_tp_hit[metric_key_cv_tp_hit].idxmin() + 1
                else:
                    best_iter_tp_hit = cv_data_tp_hit[metric_key_cv_tp_hit].idxmax() + 1
                if best_iter_tp_hit <= cv_params_tp_hit['early_stopping_rounds']:
                    best_iter_tp_hit = cv_params_tp_hit['early_stopping_rounds'] * 2 + 1
                logging.info(
                    f"🔍 CatBoostCV (TP-hit): best_iterations = {best_iter_tp_hit}, val_score = {cv_data_tp_hit[metric_key_cv_tp_hit].iloc[-1]:.4f}")
            except Exception as e:
                logging.error(f"Ошибка CV для clf_tp_hit: {e}. Итерации по умолчанию.")
        else:
            logging.info(f"CV для clf_tp_hit пропущен. Итерации: {best_iter_tp_hit}")
        logging.info(f"Обучение clf_tp_hit с {best_iter_tp_hit} итерациями...")
        clf_tp_hit_model = CatBoostClassifier(
            iterations=best_iter_tp_hit, learning_rate=0.03, depth=4,
            class_weights=class_weights_tp_hit_final_model,
            loss_function='Logloss', eval_metric='Accuracy', random_state=42, verbose=100,
            task_type="GPU", devices='0', early_stopping_rounds=50
        )
        if not X_train_sel.empty and not y_train_tp_hit.empty:
            eval_set_tp_hit = (X_test_sel,
                               y_test_tp_hit.astype(int)) if not X_test_sel.empty and not y_test_tp_hit.empty else None
            try:
                clf_tp_hit_model.fit(X_train_sel, y_train_tp_hit.astype(int), eval_set=eval_set_tp_hit)
            except Exception as e:
                logging.error(f"Ошибка при обучении clf_tp_hit_model: {e}")
                clf_tp_hit_model = None
        else:
            logging.error("Обучение clf_tp_hit невозможно.")
            clf_tp_hit_model = None
    elif train_tp_hit_model_flag:
        logging.warning("Не удалось подготовить данные для clf_tp_hit.")

    if clf_class_model: joblib.dump(clf_class_model, f"models/{tf_train}_clf_class.pkl")
    if reg_delta_model: joblib.dump(reg_delta_model, f"models/{tf_train}_reg_delta.pkl")
    if reg_vol_model: joblib.dump(reg_vol_model, f"models/{tf_train}_reg_vol.pkl")
    if clf_tp_hit_model: joblib.dump(clf_tp_hit_model, f"models/{tf_train}_clf_tp_hit.pkl")

    if X_test_sel.empty:
        logging.warning("X_test_sel пуст, невозможно рассчитать метрики.")
    else:
        metrics_output = f"\n📊  Метрики на тестовой выборке ({tf_train}) с {len(top20_features)} признаками:\n"
        report_class_str, acc_class_val = "N/A", "N/A"
        f1_class_val, precision_class_val = "N/A", "N/A"
        cm_df_str = "N/A"
        auc_class_val = "N/A"

        if clf_class_model and not y_test_class.empty:  # y_test_class is now 0/1
            try:
                proba_class_test = clf_class_model.predict_proba(X_test_sel)
                # Assuming classes_ are [0, 1] and 1 corresponds to 'UP'
                # For binary classification, predict_proba typically returns [P(class_0), P(class_1)]
                # If model.classes_ is [0,1], then proba_class_test[:, 1] is P(class_1)
                # Positive class is 1 (formerly 'UP')
                positive_class_label_int = 1

                # Get the index of the positive class (1) from the model's learned classes
                model_classes_ = getattr(clf_class_model, 'classes_', [0, 1])  # Default to [0,1] if not found
                try:
                    positive_class_idx = list(model_classes_).index(positive_class_label_int)
                except ValueError:
                    logging.warning(
                        f"Положительный класс {positive_class_label_int} не найден в model.classes_ ({model_classes_}). Используется индекс 1 по умолчанию для вероятностей.")
                    positive_class_idx = 1  # Fallback, assumes second column is positive class

                y_pred_class_flat = np.where(proba_class_test[:, positive_class_idx] > PREDICT_PROBA_THRESHOLD_CLASS,
                                             positive_class_label_int,  # Predict 1 for UP
                                             1 - positive_class_label_int)  # Predict 0 for DOWN

                y_test_class_flat = y_test_class.squeeze()  # Already 0/1

                report_labels_int = [0, 1]  # Actual labels used in y_true, y_pred
                # TARGET_CLASS_NAMES = ['DOWN', 'UP'] for display in report

                unique_labels_test_set = set(y_test_class_flat.unique())
                unique_labels_pred_set = set(pd.Series(y_pred_class_flat).unique())
                all_present_labels_set = unique_labels_test_set | unique_labels_pred_set
                # Check against integer labels {0, 1}
                if not all_present_labels_set.issubset(set(report_labels_int)):
                    logging.warning(
                        f"Метки в y_test/y_pred ({all_present_labels_set}) отсутствуют в ожидаемых метках {set(report_labels_int)}")

                report_class_str = classification_report(y_test_class_flat, y_pred_class_flat,
                                                         labels=report_labels_int,
                                                         target_names=TARGET_CLASS_NAMES,  # Display 'DOWN', 'UP'
                                                         digits=4, zero_division=0)
                acc_class_val = accuracy_score(y_test_class_flat, y_pred_class_flat)

                # Calculate F1 and Precision for the positive class (1, formerly 'UP')
                f1_class_val = f1_score(y_test_class_flat, y_pred_class_flat, pos_label=positive_class_label_int,
                                        zero_division=0)
                precision_class_val = precision_score(y_test_class_flat, y_pred_class_flat,
                                                      pos_label=positive_class_label_int, zero_division=0)

                # Расчёт ROC AUC
                try:
                    # y_test_class_flat is already 0/1. This is y_true_binary.
                    if len(np.unique(y_test_class_flat)) > 1:
                        y_score_probs = proba_class_test[:, positive_class_idx]  # Probabilities for the positive class
                        auc_class_val = roc_auc_score(y_test_class_flat, y_score_probs)
                    else:
                        logging.warning(
                            f"Не удалось рассчитать ROC AUC: в y_test_class_flat только один класс ({np.unique(y_test_class_flat)}).")
                        auc_class_val = "N/A (один класс в y_true)"
                except Exception as e_auc:
                    logging.warning(f"Не удалось рассчитать ROC AUC: {e_auc}")
                    auc_class_val = "N/A (ошибка)"

                cm = confusion_matrix(y_test_class_flat, y_pred_class_flat, labels=report_labels_int)
                cm_df = pd.DataFrame(cm, index=[f"True_{l}" for l in TARGET_CLASS_NAMES],  # Display 'DOWN', 'UP'
                                     columns=[f"Pred_{l}" for l in TARGET_CLASS_NAMES])  # Display 'DOWN', 'UP'
                logging.info(f"🧠 Confusion Matrix (clf_class) для {tf_train}:\n{cm_df}")
                cm_df_str = cm_df.to_string()
            except Exception as e:
                logging.error(f"Ошибка при расчете метрик clf_class: {e}", exc_info=True)
        else:
            logging.warning("Метрики clf_class не рассчитываются (модель не обучена или y_test_class пуст).")

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
        metrics_output += f"   🧩 Confusion Matrix (clf_class):\n{cm_df_str}\n"

        mae_delta_val = "N/A"
        if reg_delta_model and not y_test_delta.empty:
            try:
                y_pred_delta_test = reg_delta_model.predict(X_test_sel)
                mae_delta_val = mean_absolute_error(y_test_delta, y_pred_delta_test)
            except Exception as e:
                logging.error(f"Ошибка при расчете MAE Delta: {e}")
        else:
            logging.warning("MAE Delta не рассчитывается.")
        metrics_output += f"   📈 MAE Delta: {mae_delta_val:.6f}\n" if isinstance(mae_delta_val,
                                                                                 float) else f"   📈 MAE Delta: {mae_delta_val}\n"

        mae_vol_val = "N/A"
        if reg_vol_model and not y_test_vol.empty:
            try:
                y_pred_vol_test = reg_vol_model.predict(X_test_sel)
                mae_vol_val = mean_absolute_error(y_test_vol, y_pred_vol_test)
            except Exception as e:
                logging.error(f"Ошибка при расчете MAE Volatility: {e}")
        else:
            logging.warning("MAE Volatility не рассчитывается.")
        metrics_output += f"   ⚡ MAE Volatility: {mae_vol_val:.6f}\n" if isinstance(mae_vol_val,
                                                                                    float) else f"   ⚡ MAE Volatility: {mae_vol_val}\n"

        if train_tp_hit_model_flag:
            metrics_output += "\n--- Классификатор TP-Hit (clf_tp_hit) ---\n"
            report_tp_hit_str, acc_tp_hit_val = "N/A", "N/A"
            if clf_tp_hit_model and y_test_tp_hit is not None and not y_test_tp_hit.empty:
                try:
                    y_pred_tp_hit_test = clf_tp_hit_model.predict(X_test_sel)
                    y_pred_tp_hit_flat = [item[0] if isinstance(item, (np.ndarray, list)) and len(item) == 1 else item
                                          for item in y_pred_tp_hit_test]
                    y_test_tp_hit_flat = y_test_tp_hit.squeeze().astype(int)
                    y_pred_tp_hit_flat = pd.Series(y_pred_tp_hit_flat).astype(int)
                    valid_indices = ~y_test_tp_hit_flat.isna() & ~y_pred_tp_hit_flat.isna()
                    y_test_tp_hit_flat_valid, y_pred_tp_hit_flat_valid = y_test_tp_hit_flat[valid_indices], \
                    y_pred_tp_hit_flat[
                        valid_indices]
                    if len(y_test_tp_hit_flat_valid) > 0:
                        report_tp_hit_str = classification_report(y_test_tp_hit_flat_valid, y_pred_tp_hit_flat_valid,
                                                                  digits=4,
                                                                  zero_division=0,
                                                                  target_names=['No TP Hit (0)', 'TP Hit (1)'],
                                                                  labels=[0, 1])
                        acc_tp_hit_val = accuracy_score(y_test_tp_hit_flat_valid, y_pred_tp_hit_flat_valid)
                    else:
                        report_tp_hit_str, acc_tp_hit_val = "N/A (нет валидных данных)", "N/A (нет валидных данных)"
                except Exception as e:
                    logging.error(f"Ошибка при расчете метрик clf_tp_hit: {e}", exc_info=True)
            else:
                logging.warning("Метрики clf_tp_hit не рассчитываются.")
            metrics_output += report_tp_hit_str + "\n"
            metrics_output += f"   🎯 Accuracy (TP-hit): {acc_tp_hit_val:.4f}\n" if isinstance(acc_tp_hit_val,
                                                                                              float) else f"   🎯 Accuracy (TP-hit): {acc_tp_hit_val}\n"

        print(metrics_output)
        log_file_path = f"logs/train_metrics_{tf_train}.txt"
        mode = "a" if os.path.exists(log_file_path) else "w"
        with open(log_file_path, mode, encoding="utf-8") as f_log:
            if mode == "w": f_log.write(f"=== Начало лога обучения для {tf_train} ({pd.Timestamp.now()}) ===\n")
            f_log.write(metrics_output)
            f_log.write(f"\nИспользованные признаки ({len(feature_cols_final)} штук):\n" + ", ".join(
                feature_cols_final) + "\n\n")
        logging.info(f"Метрики сохранены в лог: {log_file_path}")

    logging.info(f"✅  Модели для {tf_train} обучены.")
    if train_tp_hit_model_flag and clf_tp_hit_model:
        logging.info("Напоминание: для predict_all.py рассмотрите predict_proba для clf_tp_hit.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение моделей CatBoost.")
    parser.add_argument('--tf', type=str, required=True, help="Таймфрейм (1m, 5m, ...)")
    args = parser.parse_args()
    log_file_path_main = f"logs/train_metrics_{args.tf}.txt"
    # Ensure the main log file is cleared at the start of a specific tf run, if train_all_models appends.
    # The current logic in train_all_models handles its own log file appending/writing.
    # This initial write to log_file_path_main here will be overwritten if train_all_models also writes to it in 'w' mode.
    # It seems train_all_models opens its own log file path with "a" or "w", so this is fine.

    # Overwrite specific log file for this run at the beginning of __main__
    # This helps to have a clean log for each invocation of the script for a given --tf
    with open(log_file_path_main, "w", encoding="utf-8") as f_clear:
        f_clear.write(f"=== Начало основного лога для {args.tf} ({pd.Timestamp.now()}) ===\n")
        f_clear.write(f"Запуск скрипта train_model.py для таймфрейма {args.tf}\n")

    try:
        train_all_models(args.tf)
    except KeyboardInterrupt:
        print(f"\n[TrainModel] 🛑 Обучение для {args.tf} прервано.")
        logging.info(f"Обучение для {args.tf} прервано пользователем.")
        sys.exit(130)
    except Exception as e:
        logging.error(f"[TrainModel] 💥 Ошибка при обучении {args.tf}: {e}", exc_info=True)
        sys.exit(1)