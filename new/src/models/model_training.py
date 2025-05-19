# src/models/model_training.py

import pandas as pd
import numpy as np
import logging
import random

# Import CatBoost components
from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from sklearn.model_selection import StratifiedKFold # Needed for stratified CV in tuning

# Import config
from src.utils.config import get_config

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Load Configuration ---
config = get_config()
TRAINING_CONFIG = config['training']

# --- Constants from Config ---
CATBOOST_DEFAULT_ITERATIONS = TRAINING_CONFIG['catboost']['default_iterations']
CATBOOST_DEFAULT_LEARNING_RATE = TRAINING_CONFIG['catboost']['default_learning_rate']
CATBOOST_DEFAULT_DEPTH = TRAINING_CONFIG['catboost']['default_depth']
CATBOOST_EARLY_STOPPING_ROUNDS = TRAINING_CONFIG['catboost']['early_stopping_rounds']
CATBOOST_GPU_DEVICES = TRAINING_CONFIG['catboost']['gpu_devices']
RANDOM_STATE = TRAINING_CONFIG['random_state']
MIN_SAMPLES_FOR_STRATIFY = TRAINING_CONFIG.get('min_samples_per_class_stratify', 5)
FEATURE_SELECTION_N_SPLITS = TRAINING_CONFIG['catboost']['feature_selection_n_splits'] # Use this for CV splits
RANDOM_SEARCH_TRIALS = TRAINING_CONFIG['catboost']['random_search_trials']
TP_HIT_WEIGHT_OPTIONS = TRAINING_CONFIG['catboost']['tp_hit_weight_options']
TP_HIT_WEIGHT_SEARCH_ITERATIONS = TRAINING_CONFIG['catboost']['tp_hit_weight_search_iterations']
TP_HIT_WEIGHT_SEARCH_DEPTH = TRAINING_CONFIG['catboost']['tp_hit_weight_search_depth']


# --- Training Functions ---

def train_classification_model(X_train, y_train, X_test, y_test, log_context_str):
    """
    Trains the classification model (clf_class) including hyperparameter tuning via Random Search.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (0/1 integers).
        X_test (pd.DataFrame): Test features (for evaluation set).
        y_test (pd.Series): Test target (0/1 integers) (for evaluation set).
        log_context_str (str): Logging context string.

    Returns:
        CatBoostClassifier or None: The trained model instance or None if training failed.
    """
    logger.info(f"\n--- Обучение классификатора направления (clf_class) для {log_context_str} ---")

    if X_train.empty or y_train.empty:
        logger.error(f"Обучение clf_class невозможно ({log_context_str}): Тренировочные данные пусты.")
        return None
    if not pd.api.types.is_integer_dtype(y_train):
         logger.error(f"Обучение clf_class невозможно ({log_context_str}): Целевая переменная y_train не является целочисленной.")
         return None

    # Calculate class weights based on the training set
    class_weights_dict_for_fit = {}
    counts_class_in_train = y_train.value_counts().to_dict()
    total_class_in_train = sum(counts_class_in_train.values())

    if total_class_in_train > 0:
        count_0 = counts_class_in_train.get(0, 0)
        class_weights_dict_for_fit[0] = total_class_in_train / (2 * max(1, count_0)) # Avoid division by zero
        count_1 = counts_class_in_train.get(1, 0)
        class_weights_dict_for_fit[1] = total_class_in_train / (2 * max(1, count_1)) # Avoid division by zero
        logger.info(f"Распределение классов в y_train (0/1) ({log_context_str}): {counts_class_in_train}")
    else:
        logger.warning(f"y_train пуст, веса классов не могут быть рассчитаны ({log_context_str}). Используются веса по умолчанию (1.0).")
        class_weights_dict_for_fit = {0: 1.0, 1: 1.0}

    # CatBoost CV expects class weights as a list [weight_for_class_0, weight_for_class_1]
    class_weights_list_for_cv = [
        class_weights_dict_for_fit.get(0, 1.0),
        class_weights_dict_for_fit.get(1, 1.0)
    ]
    logger.debug(f"Веса классов для fit: {class_weights_dict_for_fit}, для CV: {class_weights_list_for_cv}")


    # Determine feasibility of stratified CV
    num_unique_classes = y_train.nunique()
    min_samples_in_class = y_train.value_counts().min() if num_unique_classes > 0 else 0
    stratified_cv_possible = (num_unique_classes >= 2) and (min_samples_in_class >= FEATURE_SELECTION_N_SPLITS) # Use FI n_splits for CV

    if num_unique_classes < 2:
         logger.warning(f"В y_train только {num_unique_classes} уникальный класс ({log_context_str}). Невозможно обучить классификатор.")
         return None # Cannot train a classifier with < 2 classes

    # Determine loss and eval metric based on unique classes (should be 2 here)
    loss_func_clf = 'Logloss' # Standard for binary classification
    eval_metric_clf = 'F1' # Using F1 as primary metric for imbalanced data
    if num_unique_classes > 2: # Just in case, though target_class mapping should prevent this
         loss_func_clf = 'MultiClass'
         eval_metric_clf = 'MultiClass'
         logger.warning(f"Обнаружено > 2 уникальных классов ({num_unique_classes}) в y_train для {log_context_str}. Используется MultiClass loss/metric.")


    # --- Random Search for Hyperparameters ---
    best_score_rs = -float('inf') # Maximize score
    best_params_rs = None
    num_rs_trials = TRAINING_CONFIG.get('random_search_trials', 10)

    # Base parameters for Random Search and final training
    base_clf_params = {
        'iterations': CATBOOST_DEFAULT_ITERATIONS,
        'learning_rate': CATBOOST_DEFAULT_LEARNING_RATE,
        'depth': CATBOOST_DEFAULT_DEPTH,
        'early_stopping_rounds': CATBOOST_EARLY_STOPPING_ROUNDS,
        'random_seed': RANDOM_STATE,
        'task_type': 'GPU',
        'devices': CATBOOST_GPU_DEVICES,
        'verbose': 0, # Keep verbose off during search
        'loss_function': loss_func_clf,
        'eval_metric': eval_metric_clf,
        'class_weights': class_weights_list_for_cv, # Use list for CV in this section
    }
    # Add any specific overrides from config if they exist
    base_clf_params.update(TRAINING_CONFIG['catboost'].get('class_model_params', {}))


    if stratified_cv_possible and len(X_train) > CATBOOST_EARLY_STOPPING_ROUNDS * 2: # Ensure enough data for RS CV
        logger.info(f"Запуск RandomSearch ({num_rs_trials} попыток) для clf_class ({log_context_str})...")
        # Define parameter distributions for Random Search
        param_distributions = {
            'iterations': [int(base_clf_params['iterations'] * 0.5), base_clf_params['iterations'], int(base_clf_params['iterations'] * 1.5), int(base_clf_params['iterations'] * 2)],
            'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
            'bagging_temperature': [round(random.uniform(0.1, 1.0), 2) for _ in range(num_rs_trials * 2)], # More options
            'depth': [4, 5, base_clf_params['depth'], 7, 8, 9, 10],
            'l2_leaf_reg': [1, 3, 5, 7, 9, 11, 15, 20],
        }

        # Create a Pool for CV
        cv_pool_clf = Pool(
            data=X_train,
            label=y_train.values, # Use the integer target
            feature_names=list(X_train.columns),
            cat_features=[] # Assuming no categorical features
        )
        stratified_kf_cv = StratifiedKFold(n_splits=FEATURE_SELECTION_N_SPLITS, shuffle=True, random_state=RANDOM_STATE) # Use same splits as FI

        for i in range(num_rs_trials):
            logger.info(f"RandomSearch для clf_class ({log_context_str}): попытка {i + 1}/{num_rs_trials}")
            trial_params = base_clf_params.copy()
            trial_params.update({ # Override base params with random choices
                'iterations': random.choice(param_distributions['iterations']),
                'learning_rate': random.choice(param_distributions['learning_rate']),
                'bagging_temperature': random.choice(param_distributions['bagging_temperature']),
                'depth': random.choice(param_distributions['depth']),
                'l2_leaf_reg': random.choice(param_distributions['l2_leaf_reg']),
            })

            try:
                cv_data = cv(cv_pool_clf, params=trial_params, folds=stratified_kf_cv, plot=False)

                metric_key_rs = f'test-{eval_metric_clf}-mean'
                if metric_key_rs not in cv_data.columns:
                     logger.warning(f"Метрика '{eval_metric_clf}' не найдена в результатах CV RandomSearch ({log_context_str}). Доступные метрики: {list(cv_data.columns)}. Пропускаем попытку.")
                     continue

                # Find best iteration based on the eval_metric_clf
                if eval_metric_clf in ['Logloss', 'RMSE', 'MultiClass']: # Metrics to minimize
                    current_score_rs = -cv_data[metric_key_rs].min() # Negate min for maximization comparison
                    current_best_iter_rs = cv_data[metric_key_rs].idxmin() + 1
                else: # Assume metric to maximize (Accuracy, F1, AUC, etc.)
                    current_score_rs = cv_data[metric_key_rs].max()
                    current_best_iter_rs = cv_data[metric_key_rs].idxmax() + 1

                # Adjust final iterations to be based on the best iteration found in CV, with a buffer
                final_iterations = max(int(current_best_iter_rs * 1.2), CATBOOST_EARLY_STOPPING_ROUNDS + 1) # Add buffer and minimum
                trial_params['iterations'] = final_iterations # Use this for the final model if these params are chosen

                # Log trial result
                params_for_logging = {k: v for k, v in trial_params.items() if k in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'bagging_temperature']}
                last_cv_score = cv_data[metric_key_rs].iloc[-1] if not cv_data.empty else np.nan # Score from last iteration
                best_cv_score = cv_data[metric_key_rs].loc[cv_data[metric_key_rs].idxmax() if eval_metric_clf not in ['Logloss', 'RMSE', 'MultiClass'] else cv_data[metric_key_rs].idxmin()] if not cv_data.empty else np.nan # Best score found in CV
                logger.info(f"  RandomSearch Trial {i + 1} ({log_context_str}): params={params_for_logging}, CV score ({eval_metric_clf}) = {best_cv_score:.4f} (last: {last_cv_score:.4f}), Best Iter = {current_best_iter_rs}")

                # Check if this trial's score is better than the current best
                if current_score_rs > best_score_rs:
                    best_score_rs = current_score_rs
                    best_params_rs = trial_params.copy()
                    logger.info(f"  🎉 New best RandomSearch score ({log_context_str}): {best_score_rs:.4f} with params: {params_for_logging}")

            except Exception as e:
                params_causing_error = {k: v for k, v in trial_params.items()}
                logger.warning(
                    f"  Ошибка в RandomSearch trial {i + 1} ({log_context_str}) ({params_causing_error}): {e}")
                continue # Continue to the next trial

        if best_params_rs:
            # Use the best parameters found, but ensure class_weights is a dict for .fit
            final_clf_params = best_params_rs.copy()
            final_clf_params['class_weights'] = class_weights_dict_for_fit # Use dict for fit
            logger.info(
                f"🎯 Лучшие параметры для clf_class от RandomSearch ({log_context_str}): { {k: v for k, v in final_clf_params.items() if k in ['iterations', 'learning_rate', 'depth', 'l2_leaf_reg', 'bagging_temperature']} } с CV-score ({eval_metric_clf}): {best_score_rs:.4f}")
        else:
            logger.warning(f"RandomSearch не нашел лучших параметров для {log_context_str}, используются дефолтные.")
            # Ensure default params use the dict for fit
            final_clf_params = base_clf_params.copy()
            final_clf_params['iterations'] = CATBOOST_DEFAULT_ITERATIONS # Revert to default max iterations
            final_clf_params['class_weights'] = class_weights_dict_for_fit # Use dict for fit
            # Add any specific overrides from config if they exist - apply again to ensure dict weights are used
            final_clf_params.update(TRAINING_CONFIG['catboost'].get('class_model_params', {}))

    else:
        logger.warning(
            f"RandomSearch для clf_class пропущен для {log_context_str}. Недостаточно данных или классов для Stratified CV. Using default parameters."
        )
        # Ensure default params use the dict for fit
        final_clf_params = base_clf_params.copy()
        final_clf_params['iterations'] = CATBOOST_DEFAULT_ITERATIONS # Revert to default max iterations
        final_clf_params['class_weights'] = class_weights_dict_for_fit # Use dict for fit
        # Add any specific overrides from config if they exist - apply again to ensure dict weights are used
        final_clf_params.update(TRAINING_CONFIG['catboost'].get('class_model_params', {}))


    # Train the final clf_class model
    clf_class_model = None
    if not X_train.empty and not y_train.empty:
        logger.info(f"🚀 Обучение финальной модели clf_class для {log_context_str} с {final_clf_params.get('iterations')} итерациями...")
        try:
            clf_class_model = CatBoostClassifier(**final_clf_params)
            # Set verbose for the final training run
            clf_class_model.set_params(verbose=100)

            eval_set_class = (X_test, y_test) if not X_test.empty and not y_test.empty else None
            clf_class_model.fit(X_train, y_train, eval_set=eval_set_class, plot=False)

            logger.info(f"clf_class best_iteration ({log_context_str}): {clf_class_model.get_best_iteration()}")
            if clf_class_model.get_best_score() and eval_set_class:
                validation_scores = clf_class_model.get_best_score().get('validation',
                                                                         clf_class_model.get_best_score().get(
                                                                             'validation_0'))
                if validation_scores and eval_metric_clf in validation_scores:
                    logger.info(
                        f"clf_class validation_{eval_metric_clf} ({log_context_str}): {validation_scores[eval_metric_clf]:.4f}")
        except Exception as e:
            logger.error(f"Ошибка при обучении clf_class_model ({log_context_str}): {e}", exc_info=True)
            clf_class_model = None
    else:
        logger.error(f"Обучение clf_class невозможно ({log_context_str}): X_train или y_train пусты.")
        clf_class_model = None

    return clf_class_model


def train_regression_model(X_train, y_train, X_test, y_test, model_type, log_context_str):
    """
    Trains a regression model (reg_delta or reg_vol).

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (float).
        X_test (pd.DataFrame): Test features (for evaluation set).
        y_test (pd.Series): Test target (float) (for evaluation set).
        model_type (str): 'reg_delta' or 'reg_vol'.
        log_context_str (str): Logging context string.

    Returns:
        CatBoostRegressor or None: The trained model instance or None if training failed.
    """
    logger.info(f"\n--- Обучение регрессора ({model_type}) для {log_context_str} ---")

    if X_train.empty or y_train.empty:
        logger.error(f"Обучение {model_type} невозможно ({log_context_str}): Тренировочные данные пусты.")
        return None

    if not y_train.empty: logging.info(
        f"y_train ({model_type}, {len(y_train)}) ({log_context_str}): min={y_train.min():.6f}, max={y_train.max():.6f}, mean={y_train.mean():.6f}, std={y_train.std():.6f}")

    # Base parameters for regression models
    base_reg_params = {
        'iterations': CATBOOST_DEFAULT_ITERATIONS,
        'learning_rate': CATBOOST_DEFAULT_LEARNING_RATE,
        'depth': CATBOOST_DEFAULT_DEPTH,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'custom_metric': ['MAE'],
        'min_data_in_leaf': 10, # Default from old script, could move to config
        'task_type': "GPU",
        'devices': CATBOOST_GPU_DEVICES,
        'random_seed': RANDOM_STATE,
        'verbose': 100,
        'early_stopping_rounds': CATBOOST_EARLY_STOPPING_ROUNDS
    }
    # Add any specific overrides from config if they exist
    base_reg_params.update(TRAINING_CONFIG['catboost'].get('regression_model_params', {}))

    reg_model = None
    try:
        reg_model = CatBoostRegressor(**base_reg_params)
        eval_set_reg = (X_test, y_test) if not X_test.empty and not y_test.empty else None
        reg_model.fit(X_train, y_train, eval_set=eval_set_reg)
        logger.info(f"{model_type} best_iteration ({log_context_str}): {reg_model.get_best_iteration()}")
    except Exception as e:
        logger.error(f"Ошибка при обучении {model_type}_model ({log_context_str}): {e}", exc_info=True)
        reg_model = None

    return reg_model


def train_tp_hit_model(X_train, y_train, X_test, y_test, log_context_str):
    """
    Trains the TP-hit classification model (clf_tp_hit) including weight search and iteration tuning.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (0/1 integers).
        X_test (pd.DataFrame): Test features (for evaluation set).
        y_test (pd.Series): Test target (0/1 integers) (for evaluation set).
        log_context_str (str): Logging context string.

    Returns:
        CatBoostClassifier or None: The trained model instance or None if training failed.
    """
    logger.info(f"\n--- Обучение классификатора TP-Hit (clf_tp_hit) для {log_context_str} ---")

    if X_train.empty or y_train.empty:
        logger.error(f"Обучение clf_tp_hit невозможно ({log_context_str}): Тренировочные данные пусты.")
        return None
    if not pd.api.types.is_integer_dtype(y_train):
         logger.error(f"Обучение clf_tp_hit невозможно ({log_context_str}): Целевая переменная y_train не является целочисленной.")
         return None


    counts_tp_hit_train = y_train.value_counts().to_dict() if not y_train.empty else {}
    logger.info(f"Распределение классов в y_train (TP-hit, 0/1) ({log_context_str}): {counts_tp_hit_train}")

    # Determine feasibility of stratified CV for TP-hit
    num_unique_classes = y_train.nunique()
    min_samples_in_class = y_train.value_counts().min() if num_unique_classes > 0 else 0
    stratified_cv_possible = (num_unique_classes >= 2) and (min_samples_in_class >= FEATURE_SELECTION_N_SPLITS) # Use FI n_splits for CV

    if num_unique_classes < 2:
         logger.warning(f"В y_train (TP-hit) только {num_unique_classes} уникальный класс ({log_context_str}). Невозможно обучить классификатор TP-hit.")
         return None # Cannot train a classifier with < 2 classes


    # --- Simple Weight Search using CV ---
    best_f1_cv = -1
    best_weights_dict = None
    weight_options = TRAINING_CONFIG.get('tp_hit_weight_options', [1, 2, 5, 10])

    if stratified_cv_possible:
         logger.info(f"Поиск лучших весов для clf_tp_hit на CV ({log_context_str})...")

         pool_tp_hit_cv_weights = Pool(
             data=X_train,
             label=y_train.values,
             feature_names=list(X_train.columns)
         )
         strat_cv_folds_weights = StratifiedKFold(n_splits=FEATURE_SELECTION_N_SPLITS, shuffle=True, random_state=RANDOM_STATE) # Use same splits as FI

         for w in weight_options:
             class_weights_list_tmp = [1, w] # {0: 1, 1: w}
             clf_tmp = CatBoostClassifier(
                 iterations=TP_HIT_WEIGHT_SEARCH_ITERATIONS,
                 learning_rate=CATBOOST_DEFAULT_LEARNING_RATE,
                 depth=TP_HIT_WEIGHT_SEARCH_DEPTH,
                 loss_function='Logloss',
                 eval_metric='F1', # Evaluate on F1 for TP-hit
                 random_seed=RANDOM_STATE,
                 task_type="GPU",
                 devices=CATBOOST_GPU_DEVICES,
                 verbose=0, # Keep verbose low
                 class_weights=class_weights_list_tmp
             )
             try:
                 cv_data_weights = cv(pool_tp_hit_cv_weights, params=clf_tmp.get_params(), folds=strat_cv_folds_weights, plot=False)
                 metric_key_weights = 'test-F1-mean'

                 if metric_key_weights in cv_data_weights.columns:
                      current_f1_cv = cv_data_weights[metric_key_weights].max() # Find max F1 across iterations
                      logger.debug(f"[TP-hit Weight Search] F1 = {current_f1_cv:.4f} при весах {class_weights_list_tmp}")
                      if current_f1_cv > best_f1_cv:
                          best_f1_cv = current_f1_cv
                          best_weights_dict = {0: 1, 1: w} # Store as dict for .fit
             except Exception as e:
                 logger.warning(f"[TP-hit Weight Search] Ошибка при CV с весами {class_weights_list_tmp} ({log_context_str}): {e}")
                 continue # Continue to the next weight option

    # If no best weights found, set a default (e.g., if CV not possible or failed)
    class_weights_tp_hit_dict_for_fit = best_weights_dict if best_weights_dict else {0: 1, 1: 10} # Default 1:10 weight for class 1
    class_weights_tp_hit_list_for_cv = [ # Prepare list format for CV if needed later
        class_weights_tp_hit_dict_for_fit.get(0, 1.0),
        class_weights_tp_hit_dict_for_fit.get(1, 1.0)
    ]
    logger.info(f"Выбранные веса для clf_tp_hit ({log_context_str}): {class_weights_tp_hit_dict_for_fit}")


    # --- Determine Iterations for Final Model using CV ---
    best_iter_tp_hit = CATBOOST_DEFAULT_ITERATIONS # Default max iterations

    if stratified_cv_possible and len(X_train) > CATBOOST_EARLY_STOPPING_ROUNDS * 2: # Ensure enough data for CV
        logger.info(f"Запуск CV для определения итераций для clf_tp_hit ({log_context_str})...")
        cv_params_tp_hit = {
            'iterations': CATBOOST_DEFAULT_ITERATIONS, # Max iterations for CV
            'learning_rate': CATBOOST_DEFAULT_LEARNING_RATE,
            'depth': CATBOOST_DEFAULT_DEPTH,
            'loss_function': 'Logloss',
            'eval_metric': 'F1', # Use F1 for CV evaluation
            'early_stopping_rounds': CATBOOST_EARLY_STOPPING_ROUNDS,
            'random_seed': RANDOM_STATE,
            'task_type': 'GPU', 'devices': CATBOOST_GPU_DEVICES,
            'verbose': 0, # Keep verbose low
            'class_weights': class_weights_tp_hit_list_for_cv # Use list for CV
        }
        # Add any specific overrides from config if they exist
        cv_params_tp_hit.update(TRAINING_CONFIG['catboost'].get('tp_hit_model_params', {}))


        pool_tp_hit_cv = Pool(
            data=X_train,
            label=y_train.values,
            feature_names=list(X_train.columns)
        )
        strat_cv_folds = StratifiedKFold(n_splits=FEATURE_SELECTION_N_SPLITS, shuffle=True, random_state=RANDOM_STATE) # Use same splits as FI
        try:
            cv_data_tp_hit = cv(pool_tp_hit_cv, params=cv_params_tp_hit, folds=strat_cv_folds, plot=False)
            metric_key_cv_tp_hit = f'test-{cv_params_tp_hit["eval_metric"]}-mean'

            if metric_key_cv_tp_hit in cv_data_tp_hit.columns:
                 # For F1, we maximize, so find idxmax
                best_iter_tp_hit = cv_data_tp_hit[metric_key_cv_tp_hit].idxmax() + 1
                # Add a buffer to the best iteration found, ensure at least early stopping rounds
                best_iter_tp_hit = max(best_iter_tp_hit, CATBOOST_EARLY_STOPPING_ROUNDS + 1)
                best_iter_tp_hit = int(best_iter_tp_hit * 1.2) # Add 20% buffer

                logger.info(
                    f"🔍 CatBoostCV (TP-hit) ({log_context_str}): best_iterations = {best_iter_tp_hit}, val_score ({cv_params_tp_hit['eval_metric']}) = {cv_data_tp_hit[metric_key_cv_tp_hit].iloc[-1]:.4f}")
            else:
                logger.warning(f"Метрика '{cv_params_tp_hit['eval_metric']}' not found in CV results for TP-hit ({log_context_str}). Available: {list(cv_data_tp_hit.columns)}. Using default iterations.")
                best_iter_tp_hit = CATBOOST_DEFAULT_ITERATIONS # Fallback to default max
        except Exception as e:
            logger.error(f"Error during CV for clf_tp_hit ({log_context_str}): {e}. Using default iterations.", exc_info=True)
            best_iter_tp_hit = CATBOOST_DEFAULT_ITERATIONS # Fallback to default max
    else:
        logger.info(f"CV for clf_tp_hit skipped ({log_context_str}). Stratified CV possible: {stratified_cv_possible}. Using default iterations: {best_iter_tp_hit}")


    # --- Train the final clf_tp_hit model ---
    clf_tp_hit_model = None
    if not X_train.empty and not y_train.empty:
        logger.info(f"🚀 Обучение финальной модели clf_tp_hit ({log_context_str}) с {best_iter_tp_hit} итерациями...")
        # Use parameters found or default parameters, ensuring dict weights for fit
        final_tp_hit_params = {
            'iterations': best_iter_tp_hit,
            'learning_rate': CATBOOST_DEFAULT_LEARNING_RATE,
            'depth': CATBOOST_DEFAULT_DEPTH,
            'loss_function': 'Logloss',
            'eval_metric': 'F1',
            'random_seed': RANDOM_STATE,
            'early_stopping_rounds': CATBOOST_EARLY_STOPPING_ROUNDS,
            'task_type': "GPU",
            'devices': CATBOOST_GPU_DEVICES,
            'class_weights': class_weights_tp_hit_dict_for_fit, # Use dict for fit
        }
        # Add any specific overrides from config if they exist
        final_tp_hit_params.update(TRAINING_CONFIG['catboost'].get('tp_hit_model_params', {}))

        try:
            clf_tp_hit_model = CatBoostClassifier(**final_tp_hit_params)
            clf_tp_hit_model.set_params(verbose=100) # Set verbose for final fit

            eval_set_tp_hit = (X_test, y_test) if not X_test.empty and not y_test.empty else None
            clf_tp_hit_model.fit(X_train, y_train, eval_set=eval_set_tp_hit)

            logger.info(f"clf_tp_hit best_iteration ({log_context_str}): {clf_tp_hit_model.get_best_iteration()}")
            if clf_tp_hit_model.get_best_score() and eval_set_tp_hit:
                validation_scores_tp_hit = clf_tp_hit_model.get_best_score().get('validation',
                                                                         clf_tp_hit_model.get_best_score().get(
                                                                             'validation_0'))
                if validation_scores_tp_hit and 'F1' in validation_scores_tp_hit:
                    logger.info(
                        f"clf_tp_hit validation_F1 ({log_context_str}): {validation_scores_tp_hit['F1']:.4f}")

        except Exception as e:
            logger.error(f"Ошибка при обучении clf_tp_hit_model ({log_context_str}): {e}", exc_info=True)
            clf_tp_hit_model = None
    else:
        logger.error(f"Обучение clf_tp_hit невозможно ({log_context_str}): X_train или y_train пусты.")
        clf_tp_hit_model = None

    return clf_tp_hit_model


# Example usage (optional, for testing the module directly)
if __name__ == "__main__":
    # In a real run, logging and config would be set up by the entry script
    # For direct testing, set it up here
    from src.utils.logging_setup import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__) # Re-get logger after setup
    config = get_config() # Re-load config after setup

    print("Testing src/models/model_training.py")

    # Create dummy data for testing training functions
    n_samples = 2000
    n_features = 50
    # Create imbalanced data for classification and tp_hit
    n_up = int(n_samples * 0.35)
    n_down = n_samples - n_up
    n_tp_hit = int(n_samples * 0.25)
    n_no_tp_hit = n_samples - n_tp_hit

    dummy_X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feature_{i}' for i in range(n_features)])
    dummy_y_class = pd.Series([1] * n_up + [0] * n_down).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) # Shuffle
    dummy_y_delta = pd.Series(np.random.randn(n_samples) * 0.01)
    dummy_y_vol = pd.Series(np.random.rand(n_samples) * 0.02)
    dummy_y_tp_hit = pd.Series([1] * n_tp_hit + [0] * n_no_tp_hit).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) # Shuffle

    # Add some NaNs to simulate real data issues
    for col in dummy_X.columns:
         dummy_X.loc[dummy_X.sample(frac=0.05).index, col] = np.nan
    dummy_y_class.loc[dummy_y_class.sample(frac=0.02).index] = np.nan
    dummy_y_delta.loc[dummy_y_delta.sample(frac=0.02).index] = np.nan
    dummy_y_vol.loc[dummy_y_vol.sample(frac=0.02).index] = np.nan
    dummy_y_tp_hit.loc[dummy_y_tp_hit.sample(frac=0.02).index] = np.nan


    # Align data based on valid classification target
    valid_class_indices = dummy_y_class.dropna().index
    X_aligned = dummy_X.loc[valid_class_indices]
    y_class_aligned = dummy_y_class.loc[valid_class_indices]
    y_delta_aligned = dummy_y_delta.loc[valid_class_indices]
    y_vol_aligned = dummy_y_vol.loc[valid_class_indices]
    y_tp_hit_aligned = dummy_y_tp_hit.loc[valid_class_indices] # Will contain NaNs if TP-hit target had NaNs

    # Perform train/test split (stratified on classification target)
    try:
        X_train, X_test, y_train_class, y_test_class, \
        y_train_delta, y_test_delta, \
        y_train_vol, y_test_vol, \
        y_train_tp_hit, y_test_tp_hit = train_test_split(
            X_aligned, y_class_aligned, y_delta_aligned, y_vol_aligned, y_tp_hit_aligned,
            test_size=TRAINING_CONFIG['test_size'], random_state=RANDOM_STATE, shuffle=True, stratify=y_class_aligned
        )
    except ValueError as e:
        print(f"Error splitting dummy data: {e}. Check class distribution.")
        logger.error(f"Error splitting dummy data: {e}", exc_info=True)
        sys.exit(1)


    log_context = "dummy_symbol, dummy_tf"

    # Train Classification Model
    print("\n--- Training Classification Model ---")
    trained_clf_class = train_classification_model(X_train, y_train_class, X_test, y_test_class, log_context)
    if trained_clf_class:
        print("Classification model trained successfully.")
    else:
        print("Classification model training failed.")

    # Train Regression Model (Delta)
    print("\n--- Training Regression Model (Delta) ---")
    trained_reg_delta = train_regression_model(X_train, y_train_delta, X_test, y_test_delta, 'reg_delta', log_context)
    if trained_reg_delta:
        print("Delta regression model trained successfully.")
    else:
        print("Delta regression model training failed.")

    # Train Regression Model (Volatility)
    print("\n--- Training Regression Model (Volatility) ---")
    trained_reg_vol = train_regression_model(X_train, y_train_vol, X_test, y_test_vol, 'reg_vol', log_context)
    if trained_reg_vol:
        print("Volatility regression model trained successfully.")
    else:
        print("Volatility regression model training failed.")

    # Train TP-Hit Model (requires non-empty, >=2 class data)
    print("\n--- Training TP-Hit Model ---")
    # TP-hit data might still contain NaNs after the split based on target_class
    # Drop NaNs specifically for the TP-hit training subset
    tp_hit_train_clean_indices = y_train_tp_hit.dropna().index
    X_train_tp_hit_clean = X_train.loc[tp_hit_train_clean_indices]
    y_train_tp_hit_clean = y_train_tp_hit.loc[tp_hit_train_clean_indices].astype(int)

    tp_hit_test_clean_indices = y_test_tp_hit.dropna().index
    X_test_tp_hit_clean = X_test.loc[tp_hit_test_clean_indices]
    y_test_tp_hit_clean = y_test_tp_hit.loc[tp_hit_test_clean_indices].astype(int)

    if y_train_tp_hit_clean.nunique() < 2 or y_train_tp_hit_clean.value_counts().min() < MIN_SAMPLES_FOR_STRATIFY:
        print(f"Skipping TP-hit training: Not enough clean data or classes for training ({y_train_tp_hit_clean.shape}, {y_train_tp_hit_clean.value_counts().to_dict()})")
        trained_clf_tp_hit = None
    else:
         trained_clf_tp_hit = train_tp_hit_model(X_train_tp_hit_clean, y_train_tp_hit_clean, X_test_tp_hit_clean, y_test_tp_hit_clean, log_context)
         if trained_clf_tp_hit:
             print("TP-hit classification model trained successfully.")
         else:
             print("TP-hit classification model training failed.")


    print("\nModel training module test finished.")