# --- START OF FILE train_model.py ---

import pandas as pd
import numpy as np
import argparse
import os
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, accuracy_score, recall_score
from sklearn.metrics import f1_score, precision_score, confusion_matrix, roc_auc_score, average_precision_score
import joblib
import sys
import logging
import random
import json
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import optuna
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance

SYMBOL_GROUPS = {
    "top15": [
        "ETHUSDT", "SOLUSDT", "BNBUSDT", "BCHUSDT", "LTCUSDT",
        "BTCUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "XRPUSDT",
        "MATICUSDT", "LINKUSDT", "NEARUSDT", "ATOMUSDT", "TRXUSDT"
    ],
    "meme": [
        "DOGEUSDT", "PEPEUSDT", "FLOKIUSDT", "WIFUSDT", "SHIBUSDT"
    ],
    "top8": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
        "XRPUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT"
    ],
    "top3": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "top5": ["ETHUSDT", "SOLUSDT", "BNBUSDT", "BCHUSDT", "LTCUSDT"]
}

FEATURE_BLACKLIST = {
    "long": [],
    "short": [],
    "tp_hit": []
}

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [TrainModel] - %(message)s',
                    stream=sys.stdout)

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

N_CV_SPLITS = 5
DEFAULT_N_TOP_FEATURES = 20
DEFAULT_OPTUNA_TRIALS_CLASSIFIER = 30
DEFAULT_OPTUNA_TRIALS_REGRESSOR = 20
DEFAULT_MINORITY_WEIGHT_BOOST = 1.0


def find_optimal_threshold(model, X_val, y_val, pos_label=1):
    if X_val.empty or y_val.empty or y_val.nunique() < 2:
        logging.warning("Невозможно подобрать порог: валидационные данные пусты или содержат один класс.")
        return 0.5
    try:
        model_classes_list = list(model.classes_)
        if pos_label not in model_classes_list:
            logging.error(f"Класс {pos_label} не найден в model.classes_ ({model_classes_list}) для подбора порога.")
            return 0.5
        proba_col_idx = model_classes_list.index(pos_label)
        y_pred_proba = model.predict_proba(X_val)[:, proba_col_idx]
    except Exception as e:
        logging.error(f"Ошибка при predict_proba для подбора порога: {e}")
        return 0.5
    best_threshold = 0.5
    best_f1 = 0.0
    thresholds = np.arange(0.05, 0.96, 0.01)
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        current_f1 = f1_score(y_val, y_pred, pos_label=pos_label, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
    logging.info(f"Оптимальный порог найден: {best_threshold:.2f} с F1-score: {best_f1:.4f}")
    return best_threshold


class EarlyStoppingCallback:
    def __init__(self, patience=10, min_delta=0.0005, direction='maximize'):  # Добавлен direction
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.direction = direction
        if self.direction == 'maximize':
            self.best_value = -float('inf')
        else:  # minimize
            self.best_value = float('inf')

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        current_value = study.best_value
        if current_value is None: return

        improved = False
        if self.direction == 'maximize':
            if current_value > self.best_value + self.min_delta:
                improved = True
        else:  # minimize
            if current_value < self.best_value - self.min_delta:
                improved = True

        if improved:
            self.best_value = current_value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                logging.info(f"Optuna HPO остановлено досрочно: нет улучшения за {self.patience} попыток.")
                study.stop()


def _train_binary_classifier(
        X_cv_pool_full: pd.DataFrame, y_cv_pool_target: pd.Series,
        X_test_full: pd.DataFrame,
        y_test_target: pd.Series,
        target_name: str, log_context_str: str, file_prefix: str,
        n_top_features: int,
        n_optuna_trials: int,
        minority_weight_boost_factor: float,
        apply_smote_default: bool = True
):
    model_log_prefix = f"clf_{target_name}"
    logging.info(f"\n🚀 Начало обучения для {model_log_prefix} для {log_context_str}")

    if y_cv_pool_target.nunique() < 2:
        logging.error(f"Цель '{target_name}' < 2 уник. классов в CV пуле. Обучение {model_log_prefix} невозможно.")
        return None, [], 0.5

    # Инициализация X_val_for_threshold с колонками из X_cv_pool_full
    X_optuna_train_pool, y_optuna_train_pool = X_cv_pool_full.copy(), y_cv_pool_target.copy()  # Работаем с копиями
    X_val_for_threshold, y_val_for_threshold = pd.DataFrame(
        columns=X_cv_pool_full.columns if not X_cv_pool_full.empty else None), pd.Series(dtype='int')

    min_samples_for_val_split = N_CV_SPLITS * 20
    if len(X_cv_pool_full) > min_samples_for_val_split + (N_CV_SPLITS * 2):  # Достаточно данных для разделения
        val_threshold_size = int(len(X_cv_pool_full) * 0.20)
        val_threshold_size = max(N_CV_SPLITS * 2, val_threshold_size)
        val_threshold_size = min(val_threshold_size, len(X_cv_pool_full) - min_samples_for_val_split)

        if val_threshold_size > 0 and (len(X_cv_pool_full) - val_threshold_size) >= min_samples_for_val_split:
            split_idx_cv = len(X_cv_pool_full) - val_threshold_size
            X_optuna_train_pool = X_cv_pool_full.iloc[:split_idx_cv].copy()
            y_optuna_train_pool = y_cv_pool_target.iloc[:split_idx_cv].copy()
            X_val_for_threshold = X_cv_pool_full.iloc[split_idx_cv:].copy()
            y_val_for_threshold = y_cv_pool_target.iloc[split_idx_cv:].copy()
            logging.info(
                f"CV пул разделен: Optuna/Train: {len(X_optuna_train_pool)}, Валидация порога/калибровки: {len(X_val_for_threshold)}")

            if X_val_for_threshold.empty or y_val_for_threshold.empty or y_val_for_threshold.nunique() < 2:
                logging.warning(
                    "Валидационный набор для порога/калибровки невалиден (пуст или 1 класс). Используется весь CV пул для Optuna/Train.")
                X_optuna_train_pool, y_optuna_train_pool = X_cv_pool_full.copy(), y_cv_pool_target.copy()  # Возвращаем все для Optuna
                X_val_for_threshold, y_val_for_threshold = pd.DataFrame(columns=X_cv_pool_full.columns), pd.Series(
                    dtype='int')  # Сбрасываем val набор
        else:
            logging.warning(
                "Не удалось выделить валидационный набор (недостаточно данных после расчета val_threshold_size). Используется весь CV пул для Optuna/Train.")
    else:
        logging.warning(
            "CV пул слишком мал для выделения валидации для порога/калибровки. Используется весь CV пул для Optuna/Train.")

    selected_features = []
    # ... (код отбора признаков, как был, с исправленным условием в цикле)
    if X_optuna_train_pool.empty or y_optuna_train_pool.empty or len(X_optuna_train_pool.columns) == 0:
        logging.warning(f"X_optuna_train_pool пуст или нет признаков для {target_name}. Пропуск отбора.")
        if not X_cv_pool_full.empty and len(X_cv_pool_full.columns) > 0:
            selected_features = X_cv_pool_full.columns.tolist()
        else:
            return None, [], 0.5
    elif len(X_optuna_train_pool.columns) <= n_top_features:
        selected_features = X_optuna_train_pool.columns.tolist()
        logging.info(f"Признаков ({len(selected_features)}) <= {n_top_features}. Отбор не требуется.")
    else:
        tscv_fs = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        feature_importances_fi = np.zeros(X_optuna_train_pool.shape[1])
        num_successful_folds = 0
        logging.info(
            f"🚀 Отбор Топ-{n_top_features} признаков для {model_log_prefix} на {len(X_optuna_train_pool)} образцах...")
        for fold, (train_idx, val_idx) in enumerate(tscv_fs.split(X_optuna_train_pool, y_optuna_train_pool)):
            X_f, y_f = X_optuna_train_pool.iloc[train_idx], y_optuna_train_pool.iloc[train_idx]
            X_v, y_v = X_optuna_train_pool.iloc[val_idx], y_optuna_train_pool.iloc[val_idx]
            if X_f.empty or y_f.empty or X_v.empty or y_v.empty or \
                    y_f.nunique() < 2 or y_v.nunique() < 2:  # ИСПРАВЛЕННОЕ УСЛОВИЕ
                logging.warning(
                    f"Фолд {fold + 1} пропущен: недостаточно данных или классов в обучающей/валидационной выборке фолда.")
                continue
            clf_fi = CatBoostClassifier(iterations=200, learning_rate=0.05, early_stopping_rounds=25,
                                        loss_function='Logloss', eval_metric='F1', task_type="GPU", devices='0',
                                        random_seed=42, verbose=0)
            try:
                clf_fi.fit(X_f, y_f, eval_set=(X_v, y_v))
                feature_importances_fi += clf_fi.get_feature_importance(type='FeatureImportance')
                num_successful_folds += 1
            except Exception as e:
                logging.warning(f"Ошибка в фолде {fold + 1} отбора {target_name}: {e}")
        if num_successful_folds > 0:
            fi_mean = pd.Series(feature_importances_fi / num_successful_folds,
                                index=X_optuna_train_pool.columns).sort_values(ascending=False)
            selected_features = fi_mean.head(n_top_features).index.tolist()
            logging.info(f"Топ-{n_top_features} признаков для {model_log_prefix}: {selected_features}")
        else:
            selected_features = X_optuna_train_pool.columns.tolist()
            logging.warning(
                f"Отбор признаков для {target_name} не дал результатов. Все ({len(selected_features)}) используются.")

    if target_name in FEATURE_BLACKLIST:  # Применение черного списка
        # ... (код применения FEATURE_BLACKLIST, как был)
        current_blacklist = FEATURE_BLACKLIST[target_name]
        if current_blacklist:
            original_len = len(selected_features)
            selected_features_before_blacklist = selected_features.copy()
            selected_features = [f for f in selected_features if f not in current_blacklist]
            if len(selected_features) < original_len:
                logging.info(
                    f"Исключено по черному списку для {model_log_prefix}: {original_len - len(selected_features)} шт. Осталось: {len(selected_features)}")
            if not selected_features and original_len > 0:
                logging.warning(
                    f"После черного списка для {model_log_prefix} не осталось признаков! Используется исходный набор.")
                selected_features = selected_features_before_blacklist

    if not selected_features: return None, [], 0.5
    X_optuna_train_sel = X_optuna_train_pool[selected_features]
    X_val_for_threshold_sel = X_val_for_threshold[selected_features] if not X_val_for_threshold.empty else pd.DataFrame(
        columns=selected_features)

    # ... (Сохранение списка признаков) ...
    try:
        features_list_path = f"models/{file_prefix}_features_{target_name}_selected.txt"
        with open(features_list_path, "w", encoding="utf-8") as f:
            json.dump(selected_features, f, indent=2)
        logging.info(f"Отобранные признаки для {model_log_prefix} сохранены в {features_list_path}")
    except Exception as e:
        logging.error(f"Ошибка сохранения списка признаков: {e}")

    # ... (Optuna HPO, как был, с передачей n_optuna_trials и minority_weight_boost_factor в objective) ...
    logging.info(f"\n🔄 Начало Optuna HPO для {model_log_prefix} ({log_context_str}) на {n_optuna_trials} попыток...")
    trial_counter = 0

    def objective(trial):
        nonlocal trial_counter;
        trial_counter += 1
        logging.info(f"Optuna HPO для {model_log_prefix}: Попытка {trial_counter}/{n_optuna_trials}")
        current_y_train_for_weights = y_optuna_train_pool
        counts_trial = current_y_train_for_weights.value_counts().to_dict()
        total_trial, num_classes_trial = sum(counts_trial.values()), len(counts_trial) if len(counts_trial) > 0 else 1
        class_weights_objective = {}
        minority_label_obj = 1 if counts_trial.get(1, 0) < counts_trial.get(0, 0) else (
            0 if counts_trial.get(0, 0) < counts_trial.get(1, 0) else -1)
        for cls_label, count_val in counts_trial.items():
            base_weight = total_trial / (num_classes_trial * count_val) if count_val > 0 else 1.0
            class_weights_objective[
                cls_label] = base_weight * minority_weight_boost_factor if cls_label == minority_label_obj and minority_label_obj != -1 else base_weight
        if 0 not in class_weights_objective: class_weights_objective[0] = 1.0 * (
            minority_weight_boost_factor if 0 == minority_label_obj and minority_label_obj != -1 else 1.0)
        if 1 not in class_weights_objective: class_weights_objective[1] = 1.0 * (
            minority_weight_boost_factor if 1 == minority_label_obj and minority_label_obj != -1 else 1.0)
        class_weights_list_objective = [class_weights_objective.get(0, 1.0), class_weights_objective.get(1, 1.0)]
        params = {'iterations': trial.suggest_int('iterations', 200, 1500, step=100),
                  'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                  'depth': trial.suggest_int('depth', 4, 8),
                  'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),  # Ограничил depth
                  'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                  'random_strength': trial.suggest_float('random_strength', 1e-9, 1.0, log=True),
                  'border_count': trial.suggest_categorical('border_count', [32, 64, 128]),
                  # 'rsm': trial.suggest_float('rsm', 0.6, 0.95), # <--- ИСПРАВЛЕНИЕ: Закомментировано или удалено
                  'loss_function': 'Logloss', 'eval_metric': 'F1', 'early_stopping_rounds': 50, 'random_seed': 42,
                  'task_type': 'GPU', 'devices': '0', 'verbose': 0,
                  'class_weights': class_weights_list_objective}
        tscv_optuna = TimeSeriesSplit(n_splits=N_CV_SPLITS)
        try:
            cv_results = cv(Pool(X_optuna_train_sel, y_optuna_train_pool.astype(int),
                                 feature_names=list(X_optuna_train_sel.columns)), params, folds=tscv_optuna, plot=False)
            mean_f1 = cv_results[f'test-F1-mean'].iloc[-1]
            logging.info(f"  Optuna Попытка {trial_counter}: F1={mean_f1:.4f}, Параметры: {trial.params}")
            return mean_f1
        except Exception as e_cv:
            logging.error(
                f"  Ошибка в Optuna Попытке {trial_counter} во время CV: {e_cv}\n  Параметры: {trial.params}"); return -1.0

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    early_stopping_cb = EarlyStoppingCallback(patience=10, min_delta=0.0005)
    try:
        study.optimize(objective, n_trials=n_optuna_trials, n_jobs=1, callbacks=[early_stopping_cb])
        best_params_optuna = study.best_params
        logging.info(
            f"🎯 Лучшие параметры Optuna для {model_log_prefix}: {best_params_optuna}, F1_cv_score: {study.best_value:.4f}")
    except Exception as e:
        logging.warning(f"Optuna для {model_log_prefix} с ошибкой: {e}. Дефолтные параметры."); best_params_optuna = {}
    final_model_params = {'loss_function': 'Logloss', 'eval_metric': 'F1', 'early_stopping_rounds': 50,
                          'random_seed': 42, 'task_type': 'GPU', 'devices': '0', 'verbose': 100}
    final_model_params.update(best_params_optuna)
    counts_final_train = y_cv_pool_target.value_counts().to_dict()
    total_final_train, num_classes_final_train = sum(counts_final_train.values()), len(counts_final_train) if len(
        counts_final_train) > 0 else 1
    class_weights_final_fit = {}
    minority_label_final = 1 if counts_final_train.get(1, 0) < counts_final_train.get(0, 0) else (
        0 if counts_final_train.get(0, 0) < counts_final_train.get(1, 0) else -1)
    for cls_label, count_val in counts_final_train.items():
        base_weight = total_final_train / (num_classes_final_train * count_val) if count_val > 0 else 1.0
        class_weights_final_fit[
            cls_label] = base_weight * minority_weight_boost_factor if cls_label == minority_label_final and minority_label_final != -1 else base_weight
    if 0 not in class_weights_final_fit: class_weights_final_fit[0] = 1.0 * (
        minority_weight_boost_factor if 0 == minority_label_final and minority_label_final != -1 else 1.0)
    if 1 not in class_weights_final_fit: class_weights_final_fit[1] = 1.0 * (
        minority_weight_boost_factor if 1 == minority_label_final and minority_label_final != -1 else 1.0)
    final_model_params['class_weights'] = class_weights_final_fit
    logging.info(f"Финальные веса классов для {model_log_prefix} (fit): {class_weights_final_fit}")

    X_train_for_final_model_sel = X_cv_pool_full[selected_features]
    y_train_for_final_model_target = y_cv_pool_target
    apply_smote_final = apply_smote_default
    if target_name == "tp_hit":  # ... (логика SMOTE для tp_hit)
        counts_tp_hit_train = y_train_for_final_model_target.value_counts()
        if len(counts_tp_hit_train) == 2 and counts_tp_hit_train.get(1, 0) >= counts_tp_hit_train.get(0, 0):
            logging.info(
                f"Для {model_log_prefix} класс 1 не миноритарный ({counts_tp_hit_train.to_dict()}). SMOTE пропущен.")
            apply_smote_final = False
    if apply_smote_final:  # ... (исправленная логика SMOTE)
        min_class_count = y_train_for_final_model_target.value_counts().min()
        value_counts_before_smote = y_train_for_final_model_target.value_counts().to_dict()
        if y_train_for_final_model_target.nunique() < 2:
            logging.warning(f"Пропуск SMOTE для {target_name}: только один класс.")
        elif min_class_count <= 1:
            logging.warning(
                f"Пропуск SMOTE для {target_name}: в минорном классе <=1 пример ({min_class_count}). Распределение: {value_counts_before_smote}")
        else:
            k_val = max(1, min(4, min_class_count - 1))
            smote = SMOTE(random_state=42, k_neighbors=k_val)
            logging.info(
                f"Применение SMOTE для {model_log_prefix} с k_neighbors={k_val}. До: {value_counts_before_smote}")
            try:
                X_train_for_final_model_sel, y_train_for_final_model_target = smote.fit_resample(
                    X_train_for_final_model_sel, y_train_for_final_model_target)
                logging.info(
                    f"После SMOTE для {target_name}: {pd.Series(y_train_for_final_model_target).value_counts().to_dict()}")
            except Exception as e:
                logging.warning(f"SMOTE для {target_name} не удался: {e}. Обучение на оригинальных.")

    logging.info(f"\n🚀 Обучение финальной модели {model_log_prefix}...")
    final_model_uncalibrated = CatBoostClassifier(**final_model_params)
    # Проверка eval_set перед использованием
    can_use_eval_set_for_final_fit = not X_val_for_threshold_sel.empty and \
                                     not y_val_for_threshold.empty and \
                                     y_val_for_threshold.nunique() >= 2
    eval_set_for_final_fit = (X_val_for_threshold_sel,
                              y_val_for_threshold.astype(int)) if can_use_eval_set_for_final_fit else None
    if not can_use_eval_set_for_final_fit:
        logging.warning(
            f"Нет валидного eval_set для early stopping финальной модели {model_log_prefix}. Early stopping отключен (если iterations не задан явно и велик).")
        # Если iterations не задан в best_params_optuna, CatBoost может использовать дефолтные 1000.
        # Если iterations очень большой, а eval_set нет, обучение будет до конца.
        # final_model_params.pop('early_stopping_rounds', None) # Можно явно убрать, но CatBoost сам обработает

    if not X_train_for_final_model_sel.empty and not y_train_for_final_model_target.empty:
        try:
            final_model_uncalibrated.fit(X_train_for_final_model_sel, y_train_for_final_model_target.astype(int),
                                         eval_set=eval_set_for_final_fit, plot=False)
        except Exception as e:
            logging.error(f"Ошибка обучения {model_log_prefix}: {e}",
                          exc_info=True); return None, selected_features, 0.5
    else:
        logging.error(f"Обучение {model_log_prefix} невозможно: X или y пусты."); return None, selected_features, 0.5

    final_model_calibrated = final_model_uncalibrated
    # Проверка данных для калибровки и сам процесс
    can_calibrate = not X_val_for_threshold_sel.empty and \
                    not y_val_for_threshold.empty and \
                    y_val_for_threshold.nunique() >= 2
    if target_name in ["tp_hit", "long", "short"] and can_calibrate:
        logging.info(
            f"Калибровка вероятностей для {model_log_prefix} на X_val_for_threshold (классы: {y_val_for_threshold.value_counts().to_dict()})...")
        try:
            # Попробуем sigmoid, он более устойчив
            calibrated_clf = CalibratedClassifierCV(final_model_uncalibrated, method='sigmoid', cv=None)
            calibrated_clf.fit(X_val_for_threshold_sel, y_val_for_threshold.astype(int))
            final_model_calibrated = calibrated_clf
            logging.info(f"Калибровка (sigmoid) для {model_log_prefix} завершена.")
        except ValueError as ve:  # Ловим ValueError, который может возникнуть, если predict_proba возвращает один класс
            logging.warning(f"Ошибка калибровки (sigmoid) для {model_log_prefix}: {ve}. Попытка с isotonic.")
            try:
                calibrated_clf_iso = CalibratedClassifierCV(final_model_uncalibrated, method='isotonic', cv=None)
                calibrated_clf_iso.fit(X_val_for_threshold_sel, y_val_for_threshold.astype(int))
                final_model_calibrated = calibrated_clf_iso
                logging.info(f"Калибровка (isotonic) для {model_log_prefix} завершена.")
            except Exception as e_iso:
                logging.warning(
                    f"Ошибка калибровки (isotonic) для {model_log_prefix}: {e_iso}. Используется некалиброванная модель.")
        except Exception as e_cal:  # Другие ошибки калибровки
            logging.warning(
                f"Общая ошибка калибровки для {model_log_prefix}: {e_cal}. Используется некалиброванная модель.")
    elif target_name in ["tp_hit", "long", "short"]:
        logging.warning(
            f"Пропуск калибровки для {model_log_prefix}: нет валидных данных (X_val_for_threshold_sel пуст или y_val_for_threshold имеет <2 классов).")

    optimal_threshold = 0.5
    if can_calibrate:  # Используем то же условие, что и для калибровки
        optimal_threshold = find_optimal_threshold(final_model_calibrated, X_val_for_threshold_sel,
                                                   y_val_for_threshold.astype(int))
    else:
        logging.warning(
            f"Невозможно подобрать оптимальный порог для {model_log_prefix} (нет валидных данных), используется 0.5.")

    # Permutation Importance (используем X_test_full, если X_val_for_threshold_sel невалиден)
    # ... (код Permutation Importance, как был) ...
    perm_eval_data_valid = False
    if not X_val_for_threshold_sel.empty and not y_val_for_threshold.empty and y_val_for_threshold.nunique() >= 2:
        X_perm_eval, y_perm_eval = X_val_for_threshold_sel, y_val_for_threshold  # Предпочитаем валидационный
        perm_eval_data_valid = True
        logging.info(f"Permutation Importance будет рассчитана на X_val_for_threshold_sel для {model_log_prefix}.")
    elif not X_test_full[selected_features].empty and not y_test_target.empty and y_test_target.nunique() >= 2:
        X_perm_eval, y_perm_eval = X_test_full[selected_features], y_test_target
        perm_eval_data_valid = True
        logging.warning(
            f"X_val_for_threshold_sel невалиден, Permutation Importance будет рассчитана на X_test_full для {model_log_prefix}.")
    else:
        logging.warning(
            f"Нет валидных данных ни в X_val_for_threshold_sel, ни в X_test_full для Permutation Importance {model_log_prefix}. Пропуск.")

    if perm_eval_data_valid:
        logging.info(f"Расчет Permutation Importance для {model_log_prefix}...")
        try:
            perm_importance_result = permutation_importance(final_model_calibrated, X_perm_eval,
                                                            y_perm_eval.astype(int), n_repeats=10, random_state=42,
                                                            scoring='f1_weighted')
            sorted_idx = perm_importance_result.importances_mean.argsort()[::-1]
            perm_importance_df = pd.DataFrame({'feature': X_perm_eval.columns[sorted_idx],
                                               'importance_mean': perm_importance_result.importances_mean[sorted_idx],
                                               'importance_std': perm_importance_result.importances_std[sorted_idx]})
            logging.info(
                f"\nPermutation Importance для {model_log_prefix} (топ-10):\n{perm_importance_df.head(10).to_string(index=False)}")
            perm_importance_path = f"logs/perm_importance_{file_prefix}_{target_name}.csv"
            perm_importance_df.to_csv(perm_importance_path, index=False)
            logging.info(f"Permutation Importance сохранена в {perm_importance_path}")
        except Exception as e:
            logging.warning(f"Ошибка расчета Permutation Importance для {model_log_prefix}: {e}")

    return final_model_calibrated, selected_features, optimal_threshold


# --- train_all_models ---
def train_all_models(tf_train, symbol=None, cli_args=None):
    # ... (код функции train_all_models, как был, с передачей cli_args в _train_binary_classifier
    # и использованием cli_args.optuna_trials_reg для регрессоров)
    file_prefix = f"{symbol}_{tf_train}" if symbol else tf_train
    log_context_str = f"таймфрейма: {tf_train}" + (f", символа: {symbol}" if symbol else " (общая модель)")
    logging.info(f"🚀 Начало обучения моделей для {log_context_str}")

    features_path = f"data/features_{symbol}_{tf_train}.pkl" if symbol else f"data/features_{tf_train}.pkl"
    if not os.path.exists(features_path): logging.error(f"Файл признаков {features_path} не найден."); return
    try:
        df = pd.read_pickle(features_path)
        if 'timestamp' in df.columns: df = df.sort_values(by='timestamp').reset_index(drop=True); logging.info(
            f"Данные из {features_path} отсортированы.")
    except Exception as e:
        logging.error(f"Ошибка чтения {features_path}: {e}"); return
    if df.empty: logging.error(f"Файл {features_path} пуст."); return

    required_targets_for_models = ['target_long', 'target_short', 'delta', 'volatility']
    if any(t not in df.columns for t in required_targets_for_models): logging.error(
        f"Отсутствуют таргеты в {features_path}."); return

    df_cleaned = df.dropna(subset=required_targets_for_models).copy()
    target_tp_hit_col = 'target_tp_hit'
    train_tp_hit_model_flag = False
    if target_tp_hit_col in df.columns:
        if not df_cleaned[target_tp_hit_col].isnull().all() and df_cleaned[target_tp_hit_col].nunique(dropna=True) >= 2:
            df_cleaned.dropna(subset=[target_tp_hit_col], inplace=True)
            tp_hit_counts = df_cleaned[target_tp_hit_col].value_counts()
            if tp_hit_counts.min() >= N_CV_SPLITS:
                train_tp_hit_model_flag = True; logging.info(
                    f"'{target_tp_hit_col}'. Распределение: {tp_hit_counts.to_dict()}")
            else:
                logging.warning(
                    f"'{target_tp_hit_col}' мало примеров ({tp_hit_counts.to_dict()}). TP-hit не будет обучен.")
        else:
            logging.warning(f"'{target_tp_hit_col}' все NaN или 1 класс. TP-hit не будет обучен.")
    else:
        logging.warning(f"'{target_tp_hit_col}' не найден. TP-hit не будет обучен.")

    if df_cleaned.empty: logging.error("DataFrame пуст после NaN clear."); return
    logging.info(f"Размер DataFrame после NaN clear: {df_cleaned.shape}")

    excluded_cols_for_features = ['timestamp', 'symbol', 'target_long', 'target_short', 'delta', 'volatility',
                                  'future_close', 'future_max_high', 'future_min_low']
    if train_tp_hit_model_flag and target_tp_hit_col in df_cleaned.columns: excluded_cols_for_features.append(
        target_tp_hit_col)
    potential_leak_cols = ['target_class', 'target_up']
    for col in potential_leak_cols:
        if col in df_cleaned.columns and col not in excluded_cols_for_features: excluded_cols_for_features.append(
            col); logging.warning(f"'{col}' исключена.")
    feature_cols_initial = [c for c in df_cleaned.columns if c not in excluded_cols_for_features and \
                            df_cleaned[c].dtype in [np.int64, np.float64, np.int32, np.float32, bool] and \
                            not pd.api.types.is_datetime64_any_dtype(df_cleaned[c])]
    if not feature_cols_initial: logging.error(f"Нет признаков для обучения для {log_context_str}."); return
    logging.info(
        f"Начальное кол-во признаков ({log_context_str}): {len(feature_cols_initial)}. Первые 5: {feature_cols_initial[:5]}")

    X_all_data = df_cleaned[feature_cols_initial].copy()
    y_long_all, y_short_all = df_cleaned['target_long'].astype(int), df_cleaned['target_short'].astype(int)
    y_delta_all, y_vol_all = df_cleaned['delta'], df_cleaned['volatility']
    y_tp_hit_all = df_cleaned[target_tp_hit_col].astype(
        int) if train_tp_hit_model_flag and target_tp_hit_col in df_cleaned.columns else None

    min_data_needed_for_robust_split = N_CV_SPLITS * 10
    if len(X_all_data) < min_data_needed_for_robust_split: logging.warning(
        f"Данных ({len(X_all_data)}) может быть мало.")
    test_set_size_ratio, min_test_samples = 0.15, N_CV_SPLITS * 2
    actual_test_size = int(len(X_all_data) * test_set_size_ratio)
    if actual_test_size < min_test_samples:
        if len(X_all_data) > min_test_samples * 2:
            actual_test_size = min_test_samples; logging.warning(
                f"Тестовая выборка мала. Используется {actual_test_size}.")
        else:
            logging.error(f"Мало данных ({len(X_all_data)}) для теста. Прервано."); return
    split_idx_main = len(X_all_data) - actual_test_size
    X_cv_pool, X_test_full = X_all_data.iloc[:split_idx_main].copy(), X_all_data.iloc[split_idx_main:].copy()
    y_long_cv_pool, y_test_long = y_long_all.iloc[:split_idx_main].copy(), y_long_all.iloc[split_idx_main:].copy()
    y_short_cv_pool, y_test_short = y_short_all.iloc[:split_idx_main].copy(), y_short_all.iloc[split_idx_main:].copy()
    y_delta_cv_pool, y_test_delta = y_delta_all.iloc[:split_idx_main].copy(), y_delta_all.iloc[split_idx_main:].copy()
    y_vol_cv_pool, y_test_vol = y_vol_all.iloc[:split_idx_main].copy(), y_vol_all.iloc[split_idx_main:].copy()
    y_tp_hit_cv_pool = y_tp_hit_all.iloc[:split_idx_main].copy() if y_tp_hit_all is not None else None
    y_test_tp_hit = y_tp_hit_all.iloc[split_idx_main:].copy() if y_tp_hit_all is not None else None
    logging.info(f"Размеры: CV Pool: {len(X_cv_pool)}, Финальный Тест: {len(X_test_full)}")
    if X_cv_pool.empty or X_test_full.empty: logging.error("CV Pool или Тест пусты. Прервано."); return

    optimal_thresholds = {}
    clf_long_model, features_long, opt_thresh_long = None, [], 0.5
    clf_short_model, features_short, opt_thresh_short = None, [], 0.5
    reg_delta_model, reg_vol_model = None, None
    clf_tp_hit_model, opt_thresh_tp_hit = None, 0.5

    n_top_features_to_use = cli_args.n_top_features if cli_args else DEFAULT_N_TOP_FEATURES
    optuna_trials_clf_to_use = cli_args.optuna_trials_clf if cli_args else DEFAULT_OPTUNA_TRIALS_CLASSIFIER
    optuna_trials_reg_to_use = cli_args.optuna_trials_reg if cli_args else DEFAULT_OPTUNA_TRIALS_REGRESSOR
    minority_boost_to_use = cli_args.minority_weight_boost if cli_args else DEFAULT_MINORITY_WEIGHT_BOOST

    if not (cli_args and cli_args.skip_long):
        clf_long_model, features_long, opt_thresh_long = _train_binary_classifier(X_cv_pool, y_long_cv_pool,
                                                                                  X_test_full, y_test_long, "long",
                                                                                  log_context_str, file_prefix,
                                                                                  n_top_features_to_use,
                                                                                  optuna_trials_clf_to_use,
                                                                                  minority_boost_to_use)
        if clf_long_model: optimal_thresholds["long"] = opt_thresh_long
    else:
        logging.info(f"Пропуск обучения clf_long для {log_context_str}.")
        model_path, features_path_txt, threshold_path = f"models/{file_prefix}_clf_long.pkl", f"models/{file_prefix}_features_long_selected.txt", f"models/{file_prefix}_optimal_thresholds.json"
        if os.path.exists(model_path) and os.path.exists(features_path_txt):
            try:
                clf_long_model, features_long = joblib.load(model_path), json.load(open(features_path_txt, "r"))
                if os.path.exists(threshold_path): opt_thresh_long = json.load(open(threshold_path, "r")).get("long",
                                                                                                              0.5)
                logging.info(f"Загружена clf_long. Порог: {opt_thresh_long}")
                if clf_long_model: optimal_thresholds["long"] = opt_thresh_long
            except Exception as e:
                logging.error(
                    f"Ошибка загрузки clf_long: {e}."); clf_long_model, features_long, opt_thresh_long = None, [], 0.5
        else:
            logging.warning(f"Пропуск clf_long, модель/признаки не найдены.")

    if not (cli_args and cli_args.skip_short):
        clf_short_model, features_short, opt_thresh_short = _train_binary_classifier(X_cv_pool, y_short_cv_pool,
                                                                                     X_test_full, y_test_short, "short",
                                                                                     log_context_str, file_prefix,
                                                                                     n_top_features_to_use,
                                                                                     optuna_trials_clf_to_use,
                                                                                     minority_boost_to_use)
        if clf_short_model: optimal_thresholds["short"] = opt_thresh_short
    else:
        logging.info(f"Пропуск обучения clf_short для {log_context_str}.")
        model_path, features_path_txt, threshold_path = f"models/{file_prefix}_clf_short.pkl", f"models/{file_prefix}_features_short_selected.txt", f"models/{file_prefix}_optimal_thresholds.json"
        if os.path.exists(model_path) and os.path.exists(features_path_txt):
            try:
                clf_short_model, features_short = joblib.load(model_path), json.load(open(features_path_txt, "r"))
                if os.path.exists(threshold_path): opt_thresh_short = json.load(open(threshold_path, "r")).get("short",
                                                                                                               0.5)
                logging.info(f"Загружена clf_short. Порог: {opt_thresh_short}")
                if clf_short_model: optimal_thresholds["short"] = opt_thresh_short
            except Exception as e:
                logging.error(
                    f"Ошибка загрузки clf_short: {e}."); clf_short_model, features_short, opt_thresh_short = None, [], 0.5
        else:
            logging.warning(f"Пропуск clf_short, модель/признаки не найдены.")

    if not features_long:
        features_long_path_check = f"models/{file_prefix}_features_long_selected.txt"
        if os.path.exists(features_long_path_check):
            try:
                with open(features_long_path_check, "r") as f:
                    features_long = json.load(f)
                logging.info(f"Загружен список признаков clf_long (обучение было пропущено).")
            except Exception as e:
                logging.error(f"Не удалось загрузить признаки clf_long: {e}")
        else:
            logging.warning("Обучение clf_long пропущено, файл признаков не найден.")

    active_feature_set_for_others = features_long if features_long else feature_cols_initial
    if not active_feature_set_for_others:
        logging.error(f"Нет признаков для регрессоров/TP-hit для {log_context_str}.")
    else:
        X_cv_pool_others_sel, X_test_others_sel = X_cv_pool[active_feature_set_for_others], X_test_full[
            active_feature_set_for_others]
        if not (cli_args and cli_args.skip_regressors):
            def objective_regressor(trial, X_tr, y_tr, X_v, y_v, name, trial_num, total_trials):
                logging.info(f"Optuna HPO для {name}: Попытка {trial_num}/{total_trials}")
                params = {  # ... (параметры регрессора) ...
                    'iterations': trial.suggest_int('iterations', 200, 1000, step=100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    'depth': trial.suggest_int('depth', 3, 8),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
                    'loss_function': 'RMSE', 'eval_metric': 'MAE', 'random_seed': 42,
                    'task_type': 'GPU', 'devices': '0', 'verbose': 0, 'early_stopping_rounds': 30}
                model_r = CatBoostRegressor(**params)
                try:
                    model_r.fit(X_tr, y_tr, eval_set=(X_v, y_v), plot=False)
                    preds_r = model_r.predict(X_v)
                    mae_r = mean_absolute_error(y_v, preds_r)
                    logging.info(f"  Optuna Попытка {trial_num} для {name}: MAE={mae_r:.6f}, Параметры: {trial.params}")
                    return mae_r
                except Exception as e_cv_r:
                    logging.error(
                        f"  Ошибка в Optuna Попытке {trial_num} для {name}: {e_cv_r}\n  Параметры: {trial.params}")
                    return float('inf')

            val_size_reg = int(len(X_cv_pool_others_sel) * 0.20)
            if val_size_reg < N_CV_SPLITS * 2: val_size_reg = min(len(X_cv_pool_others_sel) // 2, N_CV_SPLITS * 5)
            if len(X_cv_pool_others_sel) - val_size_reg >= N_CV_SPLITS * 5:
                split_idx_r = len(X_cv_pool_others_sel) - val_size_reg
                X_tr_r_opt, X_v_r_opt = X_cv_pool_others_sel.iloc[:split_idx_r], X_cv_pool_others_sel.iloc[split_idx_r:]
                y_d_tr_r_opt, y_d_v_r_opt = y_delta_cv_pool.iloc[:split_idx_r], y_delta_cv_pool.iloc[split_idx_r:]
                y_v_tr_r_opt, y_v_v_r_opt = y_vol_cv_pool.iloc[:split_idx_r], y_vol_cv_pool.iloc[split_idx_r:]

                if not X_v_r_opt.empty and not y_d_v_r_opt.empty and not y_v_v_r_opt.empty:
                    logging.info(f"\n--- Optuna HPO для reg_delta ({optuna_trials_reg_to_use} попыток) ---")
                    study_d = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
                    early_stopping_cb_delta = EarlyStoppingCallback(patience=7, min_delta=1e-5, direction='minimize')
                    study_d.optimize(
                        lambda t: objective_regressor(t, X_tr_r_opt, y_d_tr_r_opt, X_v_r_opt, y_d_v_r_opt, "reg_delta",
                                                      t.number + 1, optuna_trials_reg_to_use),
                        n_trials=optuna_trials_reg_to_use, n_jobs=1, callbacks=[early_stopping_cb_delta])
                    bp_d = study_d.best_params
                    reg_delta_model = CatBoostRegressor(**bp_d, loss_function='RMSE', eval_metric='MAE', random_seed=42,
                                                        task_type="GPU", devices='0', verbose=100,
                                                        early_stopping_rounds=30)
                    reg_delta_model.fit(X_cv_pool_others_sel, y_delta_cv_pool, eval_set=(X_test_others_sel,
                                                                                         y_test_delta) if not X_test_others_sel.empty and not y_test_delta.empty else None)
                    logging.info(f"Лучшие параметры reg_delta: {bp_d}, MAE_val_optuna: {study_d.best_value:.6f}")

                    logging.info(f"\n--- Optuna HPO для reg_vol ({optuna_trials_reg_to_use} попыток) ---")
                    study_v = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
                    early_stopping_cb_vol = EarlyStoppingCallback(patience=7, min_delta=1e-5, direction='minimize')
                    study_v.optimize(
                        lambda t: objective_regressor(t, X_tr_r_opt, y_v_tr_r_opt, X_v_r_opt, y_v_v_r_opt, "reg_vol",
                                                      t.number + 1, optuna_trials_reg_to_use),
                        n_trials=optuna_trials_reg_to_use, n_jobs=1, callbacks=[early_stopping_cb_vol])
                    bp_v = study_v.best_params
                    reg_vol_model = CatBoostRegressor(**bp_v, loss_function='RMSE', eval_metric='MAE', random_seed=42,
                                                      task_type="GPU", devices='0', verbose=100,
                                                      early_stopping_rounds=30)
                    reg_vol_model.fit(X_cv_pool_others_sel, y_vol_cv_pool, eval_set=(X_test_others_sel,
                                                                                     y_test_vol) if not X_test_others_sel.empty and not y_test_vol.empty else None)
                    logging.info(f"Лучшие параметры reg_vol: {bp_v}, MAE_val_optuna: {study_v.best_value:.6f}")
                else:
                    logging.warning(
                        "Валидационные данные для HPO регрессоров пусты или содержат один класс. Обучение с параметрами по умолчанию.")
                    # ... (блок дефолтного обучения регрессоров)
                    reg_delta_model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6,
                                                        loss_function='RMSE', eval_metric='MAE', task_type="GPU",
                                                        devices='0', random_seed=42, verbose=100,
                                                        early_stopping_rounds=50)
                    if not X_cv_pool_others_sel.empty and not y_delta_cv_pool.empty: reg_delta_model.fit(
                        X_cv_pool_others_sel, y_delta_cv_pool, eval_set=(X_test_others_sel,
                                                                         y_test_delta) if not X_test_others_sel.empty and not y_test_delta.empty else None)
                    reg_vol_model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, loss_function='RMSE',
                                                      eval_metric='MAE', task_type="GPU", devices='0', random_seed=42,
                                                      verbose=100, early_stopping_rounds=50)
                    if not X_cv_pool_others_sel.empty and not y_vol_cv_pool.empty: reg_vol_model.fit(
                        X_cv_pool_others_sel, y_vol_cv_pool, eval_set=(X_test_others_sel,
                                                                       y_test_vol) if not X_test_others_sel.empty and not y_test_vol.empty else None)
            else:  # Дефолтные для регрессоров, если данных для HPO мало
                logging.warning("Мало данных для Optuna HPO регрессоров. Обучение с параметрами по умолчанию.")
                reg_delta_model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, loss_function='RMSE',
                                                    eval_metric='MAE', task_type="GPU", devices='0', random_seed=42,
                                                    verbose=100, early_stopping_rounds=50)
                if not X_cv_pool_others_sel.empty and not y_delta_cv_pool.empty: reg_delta_model.fit(
                    X_cv_pool_others_sel, y_delta_cv_pool, eval_set=(X_test_others_sel,
                                                                     y_test_delta) if not X_test_others_sel.empty and not y_test_delta.empty else None)
                reg_vol_model = CatBoostRegressor(iterations=500, learning_rate=0.03, depth=6, loss_function='RMSE',
                                                  eval_metric='MAE', task_type="GPU", devices='0', random_seed=42,
                                                  verbose=100, early_stopping_rounds=50)
                if not X_cv_pool_others_sel.empty and not y_vol_cv_pool.empty: reg_vol_model.fit(X_cv_pool_others_sel,
                                                                                                 y_vol_cv_pool,
                                                                                                 eval_set=(
                                                                                                     X_test_others_sel,
                                                                                                     y_test_vol) if not X_test_others_sel.empty and not y_test_vol.empty else None)
        else:  # Загрузка регрессоров
            logging.info(f"Пропуск обучения регрессоров для {log_context_str}.")
            if os.path.exists(f"models/{file_prefix}_reg_delta.pkl"): reg_delta_model = joblib.load(
                f"models/{file_prefix}_reg_delta.pkl")
            if os.path.exists(f"models/{file_prefix}_reg_vol.pkl"): reg_vol_model = joblib.load(
                f"models/{file_prefix}_reg_vol.pkl")
            if reg_delta_model and reg_vol_model:
                logging.info("Загружены существующие модели регрессоров.")
            else:
                logging.warning("Пропуск регрессоров, но существующие модели не найдены.")

        if not (cli_args and cli_args.skip_tp_hit):
            if train_tp_hit_model_flag and y_tp_hit_cv_pool is not None:
                if y_tp_hit_cv_pool.nunique() >= 2 and y_tp_hit_cv_pool.value_counts().min() >= N_CV_SPLITS:
                    clf_tp_hit_model, _, opt_thresh_tp_hit = _train_binary_classifier(X_cv_pool_others_sel,
                                                                                      y_tp_hit_cv_pool,
                                                                                      X_test_others_sel, y_test_tp_hit,
                                                                                      "tp_hit", log_context_str,
                                                                                      file_prefix,
                                                                                      n_top_features_to_use,
                                                                                      optuna_trials_clf_to_use,
                                                                                      minority_boost_to_use)
                    if clf_tp_hit_model: optimal_thresholds["tp_hit"] = opt_thresh_tp_hit
                else:
                    logging.warning(f"Мало данных/классов для clf_tp_hit. Пропуск.")
        else:  # Загрузка clf_tp_hit
            logging.info(f"Пропуск обучения clf_tp_hit для {log_context_str}.")
            model_path, threshold_path = f"models/{file_prefix}_clf_tp_hit.pkl", f"models/{file_prefix}_optimal_thresholds.json"
            if os.path.exists(model_path):
                clf_tp_hit_model = joblib.load(model_path)
                if os.path.exists(threshold_path): opt_thresh_tp_hit = json.load(open(threshold_path, "r")).get(
                    "tp_hit", 0.5)
                logging.info(f"Загружена clf_tp_hit. Порог: {opt_thresh_tp_hit}")
                if clf_tp_hit_model: optimal_thresholds["tp_hit"] = opt_thresh_tp_hit
            else:
                logging.warning("Пропуск clf_tp_hit, модель не найдена.")

    thresholds_path = f"models/{file_prefix}_optimal_thresholds.json"
    try:
        existing_thresholds = {}
        if os.path.exists(thresholds_path):
            try:
                with open(thresholds_path, "r", encoding="utf-8") as f:
                    existing_thresholds = json.load(f)
                if not isinstance(existing_thresholds, dict): existing_thresholds = {}
            except json.JSONDecodeError:
                existing_thresholds = {}; logging.warning(f"Файл {thresholds_path} поврежден, будет перезаписан.")
        existing_thresholds.update(optimal_thresholds)
        with open(thresholds_path, "w", encoding="utf-8") as f:
            json.dump(existing_thresholds, f, indent=2)
        logging.info(f"Оптимальные пороги сохранены/обновлены в {thresholds_path}: {existing_thresholds}")
    except Exception as e:
        logging.error(f"Ошибка сохранения файла с порогами {thresholds_path}: {e}")

    logging.info(f"\n💾 Сохранение моделей для {log_context_str}...")
    if clf_long_model: joblib.dump(clf_long_model, f"models/{file_prefix}_clf_long.pkl"); logging.info(
        "Модель clf_long сохранена.")
    if clf_short_model: joblib.dump(clf_short_model, f"models/{file_prefix}_clf_short.pkl"); logging.info(
        "Модель clf_short сохранена.")
    if reg_delta_model: joblib.dump(reg_delta_model, f"models/{file_prefix}_reg_delta.pkl"); logging.info(
        "Модель reg_delta сохранена.")
    if reg_vol_model: joblib.dump(reg_vol_model, f"models/{file_prefix}_reg_vol.pkl"); logging.info(
        "Модель reg_vol сохранена.")
    if clf_tp_hit_model: joblib.dump(clf_tp_hit_model, f"models/{file_prefix}_clf_tp_hit.pkl"); logging.info(
        "Модель clf_tp_hit сохранена.")

    metrics_output = f"\n📊 Метрики на тестовой выборке ({log_context_str}):\n"

    def get_classification_metrics_with_threshold(model, X_eval, y_eval, model_name, threshold,
                                                  class_names_map={0: 'NO', 1: 'YES'}):
        if model is None or X_eval.empty or y_eval.empty or y_eval.nunique() < 2:
            return f"--- {model_name} ---\nНет данных/модели или 1 класс в y_eval.\n"
        out_str = f"--- {model_name} (оптимальный порог={threshold:.2f}) ---\n"
        try:
            model_classes_list = list(model.classes_)
            if 1 not in model_classes_list:
                logging.warning(
                    f"Класс 1 отсутствует в модели {model_name}. Метрики для класса 1 могут быть некорректны.")
                y_proba_positive = np.zeros(len(X_eval))
            else:
                proba_col_idx = model_classes_list.index(1)
                y_proba_positive = model.predict_proba(X_eval)[:, proba_col_idx]
            y_pred = (y_proba_positive >= threshold).astype(int)
            present_labels = sorted(list(set(y_eval.unique()) | set(pd.Series(y_pred).unique())))
            if len(present_labels) < 2:
                out_str += f"Только один класс ({present_labels[0]}) присутствует в y_true/y_pred. Полный отчет невозможен.\n"
                out_str += f"   Accuracy: {accuracy_score(y_eval, y_pred):.4f}\n"
                if present_labels[0] == 1 and 1 in model_classes_list:
                    out_str += f"   F1-score ({class_names_map.get(1, '1')}): {f1_score(y_eval, y_pred, pos_label=1, zero_division=0):.4f}\n"
                    out_str += f"   Precision ({class_names_map.get(1, '1')}): {precision_score(y_eval, y_pred, pos_label=1, zero_division=0):.4f}\n"
                    out_str += f"   Recall ({class_names_map.get(1, '1')}): {recall_score(y_eval, y_pred, pos_label=1, zero_division=0):.4f}\n"
            else:
                target_names_for_report = [class_names_map.get(lbl, f"Класс_{lbl}") for lbl in present_labels]
                out_str += classification_report(y_eval, y_pred, labels=present_labels,
                                                 target_names=target_names_for_report, digits=4, zero_division=0) + "\n"
                out_str += f"   Accuracy: {accuracy_score(y_eval, y_pred):.4f}\n"
                if 1 in present_labels and 1 in model_classes_list:
                    out_str += f"   F1-score ({class_names_map.get(1, '1')}): {f1_score(y_eval, y_pred, pos_label=1, zero_division=0):.4f}\n"
                    out_str += f"   Precision ({class_names_map.get(1, '1')}): {precision_score(y_eval, y_pred, pos_label=1, zero_division=0):.4f}\n"
                    out_str += f"   Recall ({class_names_map.get(1, '1')}): {recall_score(y_eval, y_pred, pos_label=1, zero_division=0):.4f}\n"
                if y_eval.nunique() > 1 and 1 in model_classes_list:
                    out_str += f"   ROC AUC: {roc_auc_score(y_eval, y_proba_positive):.4f}\n"
                    out_str += f"   PR AUC: {average_precision_score(y_eval, y_proba_positive):.4f}\n"
                else:
                    out_str += "   ROC AUC: N/A (один класс в y_true или класс 1 отсутствует в модели)\n"
                    out_str += "   PR AUC: N/A (один класс в y_true или класс 1 отсутствует в модели)\n"
            cm_labels = [0, 1] if all(l in present_labels for l in [0, 1]) else present_labels
            cm_target_names = [class_names_map.get(lbl, f"Класс_{lbl}") for lbl in cm_labels]
            if len(cm_labels) >= 2:
                cm = confusion_matrix(y_eval, y_pred, labels=cm_labels)
                cm_df = pd.DataFrame(cm, index=[f"True_{n}" for n in cm_target_names],
                                     columns=[f"Pred_{n}" for n in cm_target_names])
                out_str += f"   Confusion Matrix:\n{cm_df.to_string()}\n"
            else:
                out_str += "   Confusion Matrix: N/A (менее 2 классов для матрицы)\n"
        except Exception as e:
            out_str += f"Ошибка метрик для {model_name}: {e}\n"
        return out_str

    if clf_long_model and features_long and not y_test_long.empty:
        X_test_long_sel = X_test_full[features_long] if features_long and all(
            f in X_test_full.columns for f in features_long) else pd.DataFrame()
        if not X_test_long_sel.empty:
            metrics_output += get_classification_metrics_with_threshold(clf_long_model, X_test_long_sel, y_test_long,
                                                                        "Классификатор Long",
                                                                        optimal_thresholds.get("long", 0.5),
                                                                        {0: 'NO_LONG', 1: 'LONG'})
        else:
            metrics_output += "--- Классификатор Long ---\nПропущено из-за отсутствия признаков в тесте.\n"
    if clf_short_model and features_short and not y_test_short.empty:
        X_test_short_sel = X_test_full[features_short] if features_short and all(
            f in X_test_full.columns for f in features_short) else pd.DataFrame()
        if not X_test_short_sel.empty:
            metrics_output += get_classification_metrics_with_threshold(clf_short_model, X_test_short_sel, y_test_short,
                                                                        "Классификатор Short",
                                                                        optimal_thresholds.get("short", 0.5),
                                                                        {0: 'NO_SHORT', 1: 'SHORT'})
        else:
            metrics_output += "--- Классификатор Short ---\nПропущено из-за отсутствия признаков в тесте.\n"

    if active_feature_set_for_others and not X_test_others_sel.empty:
        if reg_delta_model and not y_test_delta.empty: metrics_output += f"   📈 MAE Delta: {mean_absolute_error(y_test_delta, reg_delta_model.predict(X_test_others_sel)):.6f}\n"
        if reg_vol_model and not y_test_vol.empty: metrics_output += f"   ⚡ MAE Volatility: {mean_absolute_error(y_test_vol, reg_vol_model.predict(X_test_others_sel)):.6f}\n"
        if clf_tp_hit_model and y_test_tp_hit is not None and not y_test_tp_hit.empty: metrics_output += get_classification_metrics_with_threshold(
            clf_tp_hit_model, X_test_others_sel, y_test_tp_hit, "Классификатор TP-Hit",
            optimal_thresholds.get("tp_hit", 0.5), {0: 'NO_TP_HIT', 1: 'TP_HIT'})

    print(metrics_output)
    log_file_path = f"logs/train_metrics_{file_prefix}.txt"
    try:
        with open(log_file_path, "a", encoding="utf-8") as f_log:
            f_log.write(metrics_output)
            if features_long: f_log.write(
                f"\nИспользованные признаки для LONG ({len(features_long)}):\n" + ", ".join(features_long) + "\n")
            if features_short: f_log.write(
                f"\nИспользованные признаки для SHORT ({len(features_short)}):\n" + ", ".join(features_short) + "\n")
            if active_feature_set_for_others and (reg_delta_model or reg_vol_model or clf_tp_hit_model):
                f_log.write(
                    f"\nИспользованные признаки для регрессоров/TP-hit ({len(active_feature_set_for_others)}):\n" + ", ".join(
                        active_feature_set_for_others) + "\n")
            f_log.write("\n" + "=" * 50 + "\n\n")
        logging.info(f"Метрики сохранены в лог: {log_file_path}")
    except Exception as e:
        logging.error(f"Ошибка сохранения лога метрик: {e}")
    logging.info(f"✅ Обучение моделей для {log_context_str} завершено.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение моделей CatBoost.")
    parser.add_argument('--tf', type=str, required=True, help="Таймфрейм (1m, 5m, ...)")
    parser.add_argument('--symbol-group', type=str, help="Псевдоним группы (например: top5, meme)")
    parser.add_argument('--symbol', type=str, default=None, help="Символ (например: BTCUSDT)")
    # --- Аргументы для динамического управления ---
    parser.add_argument('--n-top-features', type=int, default=DEFAULT_N_TOP_FEATURES,
                        help=f"Количество топ-признаков для отбора (default: {DEFAULT_N_TOP_FEATURES})")
    parser.add_argument('--optuna-trials-clf', type=int, default=DEFAULT_OPTUNA_TRIALS_CLASSIFIER,
                        help=f"Количество попыток Optuna для классификаторов (default: {DEFAULT_OPTUNA_TRIALS_CLASSIFIER})")
    parser.add_argument('--optuna-trials-reg', type=int, default=DEFAULT_OPTUNA_TRIALS_REGRESSOR,
                        help=f"Количество попыток Optuna для регрессоров (default: {DEFAULT_OPTUNA_TRIALS_REGRESSOR})")
    parser.add_argument('--minority-weight-boost', type=float, default=DEFAULT_MINORITY_WEIGHT_BOOST,
                        help=f"Коэффициент усиления веса миноритарного класса (default: {DEFAULT_MINORITY_WEIGHT_BOOST})")
    # --- Аргументы для пропуска этапов ---
    parser.add_argument('--skip-long', action='store_true', help="Пропустить обучение clf_long")
    parser.add_argument('--skip-short', action='store_true', help="Пропустить обучение clf_short")
    parser.add_argument('--skip-regressors', action='store_true', help="Пропустить обучение регрессоров")
    parser.add_argument('--skip-tp-hit', action='store_true', help="Пропустить обучение clf_tp_hit")
    args = parser.parse_args()

    entities_to_process = []
    is_group_run = False

    if args.symbol_group:
        if args.symbol_group in SYMBOL_GROUPS:
            entities_to_process = [args.symbol_group]
            is_group_run = True
            logging.info(f"🧩 Обучение групповой модели для: {args.symbol_group}")
        else:
            logging.error(f"Неизвестная группа: {args.symbol_group}. Доступные: {list(SYMBOL_GROUPS.keys())}")
            sys.exit(1)
    elif args.symbol:
        if args.symbol.lower() in SYMBOL_GROUPS:
            args.symbol_group = args.symbol.lower()
            entities_to_process = [args.symbol_group]
            is_group_run = True
            logging.info(
                f"🧩 Аргумент --symbol ('{args.symbol}') распознан как группа. Обучение для группы: {args.symbol_group}")
            args.symbol = None
        else:
            entities_to_process = [args.symbol.upper()]
            is_group_run = False
            logging.info(f"🎯 Обучение модели для отдельного символа: {args.symbol.upper()}")
    else:
        logging.error("Не указан ни --symbol, ни --symbol-group. Укажите один из них.")
        sys.exit(1)

    for entity_name in entities_to_process:
        current_log_file_prefix = f"{entity_name}_{args.tf}"
        current_log_context_str = f"таймфрейма {args.tf}, " + \
                                  (
                                      f"группы {entity_name}" if is_group_run or args.symbol_group else f"символа {entity_name}")

        current_main_log_path = f"logs/train_metrics_{current_log_file_prefix}.txt"
        if not (args.skip_long and args.skip_short and args.skip_regressors and args.skip_tp_hit):
            try:
                with open(current_main_log_path, "w", encoding="utf-8") as f_clear:
                    f_clear.write(
                        f"=== Начало лога обучения для {current_log_context_str} ({pd.Timestamp.now()}) ===\n\n")
            except Exception as e:
                logging.error(f"Ошибка создания/очистки лог-файла {current_main_log_path}: {e}")

        try:
            train_all_models(args.tf, entity_name, args)
        except KeyboardInterrupt:
            logging.info(f"🛑 Обучение для {current_log_context_str} прервано.")
            sys.exit(130)
        except Exception as e:
            logging.error(f"💥 Критическая ошибка при обучении {current_log_context_str}: {e}", exc_info=True)
            if len(entities_to_process) == 1:
                sys.exit(1)
            else:
                logging.warning(f"Продолжение со следующим после ошибки на {current_log_context_str}.")