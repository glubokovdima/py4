# core/training/trainer.py
"""
Trains CatBoost models (classifier for direction, regressors for delta and volatility,
and optionally a classifier for TP-hit probability) using preprocessed features.
Includes feature selection, optional hyperparameter search (RandomSearch),
model evaluation, and saving of trained models and feature lists.
This module is based on the logic from the original train_model.py.
"""
import pandas as pd
import numpy as np
import argparse
import os
import sys
import logging
import json  # For saving feature list in JSON format as well
import random  # For RandomSearch

from catboost import CatBoostClassifier, CatBoostRegressor, Pool, cv
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, mean_absolute_error, accuracy_score,
    f1_score, precision_score, confusion_matrix,
    roc_auc_score, average_precision_score  # For AUC and PR-AUC
)
import joblib  # For saving models (via model_ops helper)

from ..helpers.utils import load_config
from ..helpers.data_io import load_features_pkl, save_feature_list_to_txt
from ..helpers.model_ops import save_model_joblib  # Using direct save via helper

# Configure logging for this module
logger = logging.getLogger(__name__)

# Global config dictionary, to be loaded
CONFIG = {}

# --- Default Configuration (can be overridden by config.yaml) ---
DEFAULT_DATA_DIR_FEATURES = "data"  # Input directory for features_*.pkl files
DEFAULT_MODELS_DIR = "models"  # Output directory for trained models
DEFAULT_LOGS_DIR = "logs"  # Output directory for training logs/metrics

# For model training (can be tuned in config.yaml)
DEFAULT_TARGET_CLASS_NAMES_BINARY = ['DOWN', 'UP']  # For binary classification reports
DEFAULT_PREDICT_PROBA_THRESHOLD_CLASS = 0.55  # For converting class probas to final prediction
DEFAULT_CV_N_SPLITS = 5
DEFAULT_RS_N_TRIALS = 10  # Number of RandomSearch trials


# --- Helper to get config values ---
def _get_training_config_value(key, sub_section=None, default_value=None):
    """Safely retrieves a value from CONFIG, optionally from a sub-section."""
    global CONFIG
    if not CONFIG:
        CONFIG = load_config()
        if not CONFIG:
            logger.error("Trainer: Configuration not loaded. Using hardcoded defaults.")
            if key == "target_class_names_binary": return DEFAULT_TARGET_CLASS_NAMES_BINARY
            if key == "predict_proba_threshold_class": return DEFAULT_PREDICT_PROBA_THRESHOLD_CLASS
            # Add more direct defaults as needed
            return default_value

    source_dict = CONFIG
    if sub_section:
        source_dict = CONFIG.get(sub_section, {})

    return source_dict.get(key, default_value)


def _select_top_n_features_cv(
        X_pool_data, y_pool_labels, n_features_to_select=20,
        catboost_params_for_fi=None, cv_n_splits=DEFAULT_CV_N_SPLITS,
        log_context_str=""
):
    """
    Selects top N features using CatBoost feature importance averaged over CV folds.
    """
    logger.info(f"🚀 Starting feature selection (Top {n_features_to_select}) for {log_context_str} "
                f"on {len(X_pool_data)} samples using {cv_n_splits}-Fold CV...")

    if X_pool_data.empty or y_pool_labels.empty:
        logger.warning(f"X_pool_data or y_pool_labels is empty for feature selection ({log_context_str}). "
                       "Skipping selection, will use all available features.")
        return X_pool_data.columns.tolist()

    if len(X_pool_data.columns) <= n_features_to_select:
        logger.info(f"Number of available features ({len(X_pool_data.columns)}) is less than or equal to "
                    f"{n_features_to_select}. No feature selection needed. Using all available features for {log_context_str}.")
        return X_pool_data.columns.tolist()

    # Default CatBoost params for feature importance calculation if not provided
    if catboost_params_for_fi is None:
        catboost_params_for_fi = {
            'iterations': 200, 'learning_rate': 0.05, 'depth': 6,
            'loss_function': 'Logloss', 'eval_metric': 'Accuracy',  # Or F1
            'early_stopping_rounds': 25, 'random_seed': 42,
            'task_type': "GPU", 'devices': '0', 'verbose': 0  # Keep verbose low for FI
        }

    # Ensure class_weights is a list if present in params for CV
    if 'class_weights' in catboost_params_for_fi and isinstance(catboost_params_for_fi['class_weights'], dict):
        # Assuming binary classification [0, 1] for weights mapping
        weights_list = [catboost_params_for_fi['class_weights'].get(0, 1.0), catboost_params_for_fi['class_weights'].get(1, 1.0)]
        catboost_params_for_fi_cv = {**catboost_params_for_fi, 'class_weights': weights_list}
    else:
        catboost_params_for_fi_cv = catboost_params_for_fi

    # Prepare for StratifiedKFold
    # Ensure y_pool_labels are integers for CatBoost and StratifiedKFold
    y_pool_labels_int = y_pool_labels.astype(int)
    min_samples_per_class = y_pool_labels_int.value_counts().min() if not y_pool_labels_int.empty else 0

    if y_pool_labels_int.nunique() < 2 or min_samples_per_class < cv_n_splits:
        logger.warning(f"Not enough samples in each class of y_pool_labels ({y_pool_labels_int.value_counts().to_dict()}) "
                       f"or only one class ({y_pool_labels_int.nunique()}) for Stratified {cv_n_splits}-Fold CV for feature selection ({log_context_str}). "
                       "Skipping selection, using all features.")
        return X_pool_data.columns.tolist()

    kf = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)

    # CatBoost Pool for feature importance calculation
    # Assuming no categorical features are explicitly passed here for simplicity.
    # If cat_features are used, they need to be identified and passed to Pool.
    fi_pool = Pool(data=X_pool_data, label=y_pool_labels_int, feature_names=list(X_pool_data.columns))

    try:
        # Perform cross-validation to get feature importances
        # Note: CatBoost's cv function itself doesn't directly return aggregated feature importances
        # We need to train on folds and average them, or use select_features from CatBoost (more complex setup)
        # Simpler approach: train a model on the full pool_data (or a large part of it) and get FI.
        # For CV-averaged FI, we'd loop through folds:

        feature_importances_sum = np.zeros(X_pool_data.shape[1])
        num_successful_folds = 0

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_pool_data, y_pool_labels_int)):
            logger.debug(f"Feature selection ({log_context_str}): Fold {fold + 1}/{cv_n_splits}")
            X_fold_train, X_fold_val = X_pool_data.iloc[train_idx], X_pool_data.iloc[val_idx]
            y_fold_train, y_fold_val = y_pool_labels_int.iloc[train_idx], y_pool_labels_int.iloc[val_idx]

            if X_fold_val.empty or y_fold_val.empty or y_fold_val.nunique() < 2:
                logger.warning(f"Validation set in fold {fold + 1} is empty or has only one class ({log_context_str}). Skipping fold for FI.")
                continue

            # Use a copy of params, ensuring class_weights is a DICT for .fit() if that's what model expects
            fit_params_fi = catboost_params_for_fi.copy()

            model_fi = CatBoostClassifier(**fit_params_fi)
            try:
                model_fi.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val), plot=False)
                feature_importances_sum += model_fi.get_feature_importance(type='FeatureImportance')
                num_successful_folds += 1
            except Exception as e_fold:
                logger.warning(f"Error during CatBoost training in FI fold {fold + 1} ({log_context_str}): {e_fold}")

        if num_successful_folds > 0:
            fi_averaged = pd.Series(feature_importances_sum / num_successful_folds, index=X_pool_data.columns)
            fi_averaged_sorted = fi_averaged.sort_values(ascending=False)
            top_features = fi_averaged_sorted.head(n_features_to_select).index.tolist()
            logger.info(f"Top {len(top_features)} features selected for {log_context_str} based on CV average: {top_features[:5]}...")
            return top_features
        else:
            logger.warning(f"No successful folds completed for feature importance calculation ({log_context_str}). Using all features.")
            return X_pool_data.columns.tolist()

    except Exception as e:
        logger.error(f"Error during feature selection process ({log_context_str}): {e}", exc_info=True)
        return X_pool_data.columns.tolist()  # Fallback to all features


def _perform_catboost_cv_random_search(
        X_cv_data, y_cv_labels,  # Data for CV and RandomSearch
        base_catboost_params,  # Base parameters including loss, eval_metric, task_type
        param_grid_rs,  # Parameter grid for RandomSearch
        n_trials_rs=DEFAULT_RS_N_TRIALS,
        cv_n_splits=DEFAULT_CV_N_SPLITS,
        log_context_str=""
):
    """Performs RandomSearch using CatBoost's cv function."""
    logger.info(f"🔄 Starting RandomSearch ({n_trials_rs} trials, {cv_n_splits}-Fold CV) for {log_context_str}...")

    best_cv_score = -float('inf') if base_catboost_params['eval_metric'] not in ['Logloss', 'RMSE'] else float('inf')
    best_params_found = base_catboost_params.copy()  # Start with base
    best_iterations_from_cv = base_catboost_params.get('iterations', 500)  # Default iterations

    if X_cv_data.empty or y_cv_labels.empty:
        logger.warning(f"CV data or labels empty for RandomSearch ({log_context_str}). Returning base params.")
        return best_params_found, best_iterations_from_cv

    y_cv_labels_int = y_cv_labels.astype(int)
    min_samples_per_class_cv = y_cv_labels_int.value_counts().min() if not y_cv_labels_int.empty else 0

    if y_cv_labels_int.nunique() < 2 or min_samples_per_class_cv < cv_n_splits:
        logger.warning(f"Not enough samples/classes in y_cv_labels for Stratified {cv_n_splits}-Fold CV ({log_context_str}). "
                       "RandomSearch skipped. Returning base params.")
        return best_params_found, best_iterations_from_cv

    # CatBoost Pool for CV
    cv_pool = Pool(data=X_cv_data, label=y_cv_labels_int, feature_names=list(X_cv_data.columns))
    stratified_kf_cv = StratifiedKFold(n_splits=cv_n_splits, shuffle=True, random_state=42)

    for i in range(n_trials_rs):
        trial_params = base_catboost_params.copy()
        # Randomly pick parameters from the grid
        for param_name, values_list in param_grid_rs.items():
            trial_params[param_name] = random.choice(values_list)

        # Ensure class_weights is a list for CatBoost CV if provided in base_params
        if 'class_weights' in trial_params and isinstance(trial_params['class_weights'], dict):
            weights_list_trial = [trial_params['class_weights'].get(0, 1.0), trial_params['class_weights'].get(1, 1.0)]
            trial_params_for_cv = {**trial_params, 'class_weights': weights_list_trial}
        else:
            trial_params_for_cv = trial_params

        logger.debug(f"RS Trial {i + 1}/{n_trials_rs} ({log_context_str}) with params: "
                     f"iter={trial_params_for_cv.get('iterations')}, "
                     f"lr={trial_params_for_cv.get('learning_rate')}, "
                     f"depth={trial_params_for_cv.get('depth')}, "
                     f"l2={trial_params_for_cv.get('l2_leaf_reg')}, "
                     f"bag_temp={trial_params_for_cv.get('bagging_temperature')}")
        try:
            cv_results_df = cv(cv_pool, params=trial_params_for_cv, folds=stratified_kf_cv, plot=False)

            eval_metric_name = base_catboost_params['eval_metric']
            metric_col_name_test = f'test-{eval_metric_name}-mean'

            if metric_col_name_test not in cv_results_df.columns:
                logger.warning(f"Metric '{metric_col_name_test}' not found in CV results ({log_context_str}). "
                               f"Available: {list(cv_results_df.columns)}. Skipping trial.")
                continue

            # Get score and best iteration from this CV run
            if eval_metric_name in ['Logloss', 'RMSE']:  # Metrics to minimize
                current_trial_best_score = cv_results_df[metric_col_name_test].min()
                current_trial_best_iter = cv_results_df[metric_col_name_test].idxmin() + 1  # 0-indexed to 1-indexed
                is_better = current_trial_best_score < best_cv_score
            else:  # Metrics to maximize (F1, Accuracy, AUC, etc.)
                current_trial_best_score = cv_results_df[metric_col_name_test].max()
                current_trial_best_iter = cv_results_df[metric_col_name_test].idxmax() + 1
                is_better = current_trial_best_score > best_cv_score

            logger.debug(f"  RS Trial {i + 1} score ({eval_metric_name}): {current_trial_best_score:.4f} "
                         f"at iter {current_trial_best_iter} (vs best: {best_cv_score:.4f})")

            if is_better:
                best_cv_score = current_trial_best_score
                best_params_found = trial_params  # Store params used for this trial (with dict weights for fit)
                best_iterations_from_cv = max(int(current_trial_best_iter * 1.1), base_catboost_params.get('early_stopping_rounds', 25) + 5)  # Add buffer, ensure > early_stopping
                logger.info(f"  🎉 New best RS score ({log_context_str}): {best_cv_score:.4f}. "
                            f"Best iterations set to: {best_iterations_from_cv}. "
                            f"Params: depth={best_params_found.get('depth')}, lr={best_params_found.get('learning_rate')}, "
                            f"l2={best_params_found.get('l2_leaf_reg')}, bag_temp={best_params_found.get('bagging_temperature')}")

        except Exception as e_rs_cv:
            logger.warning(f"Error in RandomSearch CV trial {i + 1} ({log_context_str}): {e_rs_cv}")
            continue  # To next trial

    if best_params_found == base_catboost_params and best_cv_score == (-float('inf') if base_catboost_params['eval_metric'] not in ['Logloss', 'RMSE'] else float('inf')):
        logger.warning(f"RandomSearch did not find improved parameters for {log_context_str}. Using base parameters.")
    else:
        logger.info(f"🎯 Best parameters from RandomSearch for {log_context_str} (score {best_cv_score:.4f}): "
                    f"depth={best_params_found.get('depth')}, lr={best_params_found.get('learning_rate')}, "
                    f"l2={best_params_found.get('l2_leaf_reg')}, bag_temp={best_params_found.get('bagging_temperature')}. "
                    f"Optimal iterations: {best_iterations_from_cv}")

    # Ensure the final best_params_found has 'iterations' set to best_iterations_from_cv
    best_params_found['iterations'] = best_iterations_from_cv
    return best_params_found, best_iterations_from_cv


# core/training/trainer.py (continued)

def main_train_logic(
        timeframe_to_train,
        symbol_or_group_filter=None,  # e.g., "BTCUSDT" or "top8"
        is_group_model=False  # True if symbol_or_group_filter is a group name
):
    """
    Main logic for training all models (classifier, regressors, optional TP-hit)
    for a given timeframe and symbol/group filter.

    Args:
        timeframe_to_train (str): The timeframe to train models for (e.g., '15m').
        symbol_or_group_filter (str, optional): The specific symbol or group name to train for.
                                                If None, trains a generic model for the timeframe.
        is_group_model (bool): True if symbol_or_group_filter represents a group.
                               Used for logging and potentially file naming conventions.
    """
    global CONFIG
    if not CONFIG:
        CONFIG = load_config()
        if not CONFIG:
            logger.critical("Trainer: Configuration not loaded. Aborting training.")
            return 1  # Error

    # --- Determine file prefix and logging context ---
    # file_naming_suffix is used for features path and model/log output names
    if symbol_or_group_filter:
        file_naming_suffix = f"{symbol_or_group_filter}_{timeframe_to_train}"
        log_context_str = f"TF: {timeframe_to_train}, Filter: {symbol_or_group_filter} ({'Group' if is_group_model else 'Symbol'})"
    else:  # Generic model for the timeframe
        file_naming_suffix = timeframe_to_train
        log_context_str = f"TF: {timeframe_to_train} (Generic Model)"

    logger.info(f"🚀 Starting model training for {log_context_str}")

    # --- Get Project Root and Directories from Config ---
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    features_data_dir_rel = _get_training_config_value("data_dir", default_value=DEFAULT_DATA_DIR_FEATURES)
    models_output_dir_rel = _get_training_config_value("models_dir", default_value=DEFAULT_MODELS_DIR)
    logs_output_dir_rel = _get_training_config_value("logs_dir", default_value=DEFAULT_LOGS_DIR)

    features_data_dir_abs = os.path.join(project_root, features_data_dir_rel)
    models_output_dir_abs = os.path.join(project_root, models_output_dir_rel)
    logs_output_dir_abs = os.path.join(project_root, logs_output_dir_rel)

    os.makedirs(models_output_dir_abs, exist_ok=True)
    os.makedirs(logs_output_dir_abs, exist_ok=True)

    # --- Load Features Data ---
    # Path construction: features_BTCUSDT_15m.pkl or features_top8_15m.pkl or features_15m.pkl
    features_file_path = os.path.join(features_data_dir_abs, f"features_{file_naming_suffix}.pkl")

    df_features = load_features_pkl(features_file_path)  # Uses helper
    if df_features is None or df_features.empty:
        logger.error(f"Feature file {features_file_path} not found or is empty. Training aborted for {log_context_str}.")
        return 1

    # --- Define Target Columns and Clean Data ---
    # These target column names must match those created in preprocessor.py
    target_col_class = 'target_class'  # Expected: 'UP', 'DOWN', (NaN for NEUTRAL)
    target_col_delta = 'delta'
    target_col_volatility = 'volatility'
    target_col_tp_hit = 'target_tp_hit'  # Optional, e.g., from target_tp_hit_50bps

    required_cols_for_training = [target_col_class, target_col_delta, target_col_volatility]
    # Check if TP-hit target exists and should be trained
    train_tp_hit_model_enabled = False
    if target_col_tp_hit in df_features.columns:
        if df_features[target_col_tp_hit].isnull().all() or df_features[target_col_tp_hit].nunique() < 2:
            logger.warning(f"'{target_col_tp_hit}' column has all NaNs or <2 unique values for {log_context_str}. TP-hit model will NOT be trained.")
        else:
            logger.info(f"'{target_col_tp_hit}' column found. TP-hit model will be trained for {log_context_str}.")
            required_cols_for_training.append(target_col_tp_hit)
            train_tp_hit_model_enabled = True
    else:
        logger.info(f"'{target_col_tp_hit}' column NOT found. TP-hit model will NOT be trained for {log_context_str}.")

    # Drop rows where any of the essential targets are NaN
    df_cleaned_features = df_features.dropna(subset=required_cols_for_training).copy()
    if df_cleaned_features.empty:
        logger.error(f"DataFrame is empty after dropping NaNs in target columns ({required_cols_for_training}) for {log_context_str}. Training aborted.")
        return 1
    logger.info(f"Data size after cleaning NaNs in targets for {log_context_str}: {df_cleaned_features.shape}")

    # --- Prepare Features (X) and Targets (y) ---
    # Exclude target columns, timestamp, symbol, and any other non-feature/leakage columns
    cols_to_exclude_from_X = required_cols_for_training + \
                             ['timestamp', 'symbol', 'future_close', 'target_up']  # target_up is direct leakage for target_class

    initial_feature_candidates = [col for col in df_cleaned_features.columns if col not in cols_to_exclude_from_X and \
                                  df_cleaned_features[col].dtype in [np.int64, np.float64, np.int32, np.float32, bool] and \
                                  not pd.api.types.is_datetime64_any_dtype(df_cleaned_features[col]) and \
                                  not pd.api.types.is_string_dtype(df_cleaned_features[col])]

    if not initial_feature_candidates:
        logger.error(f"No feature candidates found after exclusions for {log_context_str}. Training aborted.")
        return 1

    X_all = df_cleaned_features[initial_feature_candidates]
    # Target for classification: map 'DOWN' -> 0, 'UP' -> 1. NaN (NEUTRAL) rows already dropped.
    y_class_str_labels = df_cleaned_features[target_col_class]
    label_map_binary = {'DOWN': 0, 'UP': 1}
    y_class_mapped_binary = y_class_str_labels.map(label_map_binary)
    # Drop rows where mapping resulted in NaN (if target_class had values other than 'UP'/'DOWN')
    valid_binary_indices = y_class_mapped_binary.dropna().index
    X_all_for_binary_clf = X_all.loc[valid_binary_indices]
    y_class_final_binary = y_class_mapped_binary.loc[valid_binary_indices].astype(int)

    if X_all_for_binary_clf.empty or y_class_final_binary.empty:
        logger.error(f"No valid binary classification data remains after mapping/cleaning for {log_context_str}. Training aborted.")
        return 1

    # Other targets (use X_all_for_binary_clf's index to ensure alignment)
    y_delta_all = df_cleaned_features.loc[X_all_for_binary_clf.index, target_col_delta]
    y_vol_all = df_cleaned_features.loc[X_all_for_binary_clf.index, target_col_volatility]
    y_tp_hit_all = None
    if train_tp_hit_model_enabled:
        y_tp_hit_all = df_cleaned_features.loc[X_all_for_binary_clf.index, target_col_tp_hit].astype(int)

    # --- Train/Test Split (Stratified for Classification Targets) ---
    # We use X_all_for_binary_clf and y_class_final_binary for the main split
    # Other targets (delta, vol, tp_hit) will be sliced using the same indices.
    # test_size=0.15 was used in original script.
    try:
        X_train_cvpool, X_test_full, \
            y_class_train_cvpool, y_test_class, \
            y_delta_train_cvpool, y_test_delta, \
            y_vol_train_cvpool, y_test_vol = train_test_split(
            X_all_for_binary_clf, y_class_final_binary, y_delta_all, y_vol_all,
            test_size=0.15, random_state=42, shuffle=True, stratify=y_class_final_binary
        )

        y_tp_hit_train_cvpool, y_test_tp_hit = (None, None)
        if train_tp_hit_model_enabled and y_tp_hit_all is not None:
            y_tp_hit_train_cvpool = y_tp_hit_all.loc[X_train_cvpool.index]
            y_test_tp_hit = y_tp_hit_all.loc[X_test_full.index]

    except ValueError as e:  # e.g. if a class has too few samples for stratification
        logger.error(f"Error during train/test split for {log_context_str}: {e}. "
                     "Ensure sufficient samples for each class. Training aborted.", exc_info=True)
        return 1

    logger.info(f"Data split for {log_context_str}: Train/CV Pool size={len(X_train_cvpool)}, Test size={len(X_test_full)}")

    # --- Feature Selection for clf_class ---
    # Calculate class weights for FI model (based on y_class_train_cvpool)
    class_counts_fi = y_class_train_cvpool.value_counts().to_dict()
    total_samples_fi = len(y_class_train_cvpool)
    class_weights_fi_dict = {
        0: total_samples_fi / (2 * class_counts_fi.get(0, 1)) if class_counts_fi.get(0, 1) > 0 else 1.0,
        1: total_samples_fi / (2 * class_counts_fi.get(1, 1)) if class_counts_fi.get(1, 1) > 0 else 1.0
    }
    fi_catboost_params = {  # Params for the temporary model used for feature importance
        'iterations': 250, 'learning_rate': 0.05, 'depth': 6,
        'loss_function': 'Logloss', 'eval_metric': 'F1',  # Using F1 for FI on potentially imbalanced data
        'early_stopping_rounds': 30, 'random_seed': 42,
        'task_type': "GPU", 'devices': '0', 'verbose': 0,
        'class_weights': class_weights_fi_dict  # Pass dict for .fit()
    }

    # Select top features using the Train/CV pool data
    final_selected_feature_cols = _select_top_n_features_cv(
        X_train_cvpool, y_class_train_cvpool,
        n_features_to_select=20,  # Can be made configurable
        catboost_params_for_fi=fi_catboost_params,  # Pass params for fit
        log_context_str=f"{log_context_str} (clf_class FI)"
    )

    if not final_selected_feature_cols:
        logger.error(f"Feature selection resulted in an empty list for {log_context_str}. Training aborted.")
        return 1

    # Save the selected feature list (TXT and JSON)
    features_list_txt_path = os.path.join(models_output_dir_abs, f"{file_naming_suffix}_features_selected.txt")
    save_feature_list_to_txt(final_selected_feature_cols, features_list_txt_path)  # Uses helper

    features_list_json_path = os.path.join(models_output_dir_abs, f"{file_naming_suffix}_features_selected.json")
    try:
        with open(features_list_json_path, "w", encoding="utf-8") as f_json:
            json.dump(final_selected_feature_cols, f_json, ensure_ascii=False, indent=2)
        logger.info(f"Selected features also saved to JSON: {features_list_json_path}")
    except Exception as e_json:
        logger.error(f"Error saving selected features to JSON {features_list_json_path}: {e_json}")

    # Filter datasets to use only selected features
    X_train_final = X_train_cvpool[final_selected_feature_cols]
    X_test_final = X_test_full[final_selected_feature_cols]

    # --- Train clf_class (Binary Classifier: UP/DOWN) ---
    logger.info(f"\n--- Training clf_class for {log_context_str} ---")
    # Class weights for the main training (based on y_class_train_cvpool)
    # These are the same as class_weights_fi_dict calculated above
    clf_class_weights_dict = class_weights_fi_dict
    logger.info(f"Class weights for clf_class training ({log_context_str}): {clf_class_weights_dict}")

    # Base CatBoost parameters for clf_class RandomSearch
    # Note: class_weights here is a DICT, _perform_catboost_cv_random_search will handle list conversion for cv
    base_params_clf_rs = {
        'iterations': 700, 'learning_rate': 0.03, 'depth': 6,  # Initial guesses
        'loss_function': 'Logloss', 'eval_metric': 'F1',  # F1 is good for (im)balanced binary
        'early_stopping_rounds': 50, 'random_seed': 42,
        'task_type': 'GPU', 'devices': '0', 'verbose': 0,  # verbose 0 for RS trials
        'class_weights': clf_class_weights_dict  # Pass dict for fit, helper converts for CV
    }
    # Parameter grid for RandomSearch for clf_class
    param_grid_clf_rs = {
        'iterations': [300, 500, 700, 1000, 1500],  # Max iterations for a trial
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'depth': [4, 5, 6, 7, 8, 9, 10],
        'l2_leaf_reg': [1, 3, 5, 7, 10, 15, 20],
        'bagging_temperature': [0.0, 0.2, 0.5, 0.8, 1.0]  # Temperature for BayesianBootstrap
    }

    # Perform RandomSearch (using the same Train/CV pool data, X_train_final, y_class_train_cvpool)
    best_clf_params_from_rs, best_clf_iterations = _perform_catboost_cv_random_search(
        X_train_final, y_class_train_cvpool,
        base_params_clf_rs, param_grid_clf_rs,
        n_trials_rs=_get_training_config_value("rs_n_trials", default_value=DEFAULT_RS_N_TRIALS),
        cv_n_splits=_get_training_config_value("cv_n_splits", default_value=DEFAULT_CV_N_SPLITS),
        log_context_str=f"{log_context_str} (clf_class RS)"
    )

    # Train final clf_class model with best params (or base if RS failed)
    final_clf_class_model = CatBoostClassifier(**best_clf_params_from_rs)  # iterations already set by RS helper
    final_clf_class_model.set_params(verbose=100)  # Set verbose for final training run

    eval_set_clf_class = (X_test_final, y_test_class) if not X_test_final.empty and not y_test_class.empty else None
    try:
        final_clf_class_model.fit(X_train_final, y_class_train_cvpool, eval_set=eval_set_clf_class, plot=False)
        logger.info(f"Final clf_class model trained for {log_context_str}. Best iter: {final_clf_class_model.get_best_iteration()}.")
        save_model_joblib(final_clf_class_model, os.path.join(models_output_dir_abs, f"{file_naming_suffix}_clf_class.pkl"))
    except Exception as e_clf_final:
        logger.error(f"Error training final clf_class model for {log_context_str}: {e_clf_final}", exc_info=True)
        final_clf_class_model = None  # Ensure it's None if training failed

    # --- Train reg_delta ---
    logger.info(f"\n--- Training reg_delta for {log_context_str} ---")
    # Note: RandomSearch is not implemented for regressors here for brevity, but could be added.
    params_reg_delta = {
        'iterations': 500, 'learning_rate': 0.03, 'depth': 6,
        'loss_function': 'RMSE', 'eval_metric': 'MAE',  # Using MAE as eval metric
        'early_stopping_rounds': 50, 'random_seed': 42,
        'task_type': "GPU", 'devices': '0', 'verbose': 100,
        'min_data_in_leaf': 10  # From original script
    }
    model_reg_delta = CatBoostRegressor(**params_reg_delta)
    eval_set_delta = (X_test_final, y_test_delta) if not X_test_final.empty and not y_test_delta.empty else None
    try:
        model_reg_delta.fit(X_train_final, y_delta_train_cvpool, eval_set=eval_set_delta, plot=False)
        save_model_joblib(model_reg_delta, os.path.join(models_output_dir_abs, f"{file_naming_suffix}_reg_delta.pkl"))
    except Exception as e_reg_d:
        logger.error(f"Error training reg_delta model for {log_context_str}: {e_reg_d}", exc_info=True)
        model_reg_delta = None

    # --- Train reg_vol ---
    logger.info(f"\n--- Training reg_vol for {log_context_str} ---")
    params_reg_vol = params_reg_delta.copy()  # Use similar params as delta, can be tuned separately
    model_reg_vol = CatBoostRegressor(**params_reg_vol)
    eval_set_vol = (X_test_final, y_vol_train_cvpool) if not X_test_final.empty and not y_vol_train_cvpool.empty else None
    try:
        model_reg_vol.fit(X_train_final, y_vol_train_cvpool, eval_set=eval_set_vol, plot=False)
        save_model_joblib(model_reg_vol, os.path.join(models_output_dir_abs, f"{file_naming_suffix}_reg_vol.pkl"))
    except Exception as e_reg_v:
        logger.error(f"Error training reg_vol model for {log_context_str}: {e_reg_v}", exc_info=True)
        model_reg_vol = None

    # --- Train clf_tp_hit (Optional) ---
    model_clf_tp_hit = None
    if train_tp_hit_model_enabled and y_tp_hit_train_cvpool is not None and not y_tp_hit_train_cvpool.empty:
        logger.info(f"\n--- Training clf_tp_hit for {log_context_str} ---")
        # Class weights for TP-hit model
        tp_hit_counts = y_tp_hit_train_cvpool.value_counts().to_dict()
        tp_total_samples = len(y_tp_hit_train_cvpool)
        tp_hit_weights_dict = {
            0: tp_total_samples / (2 * tp_hit_counts.get(0, 1)) if tp_hit_counts.get(0, 1) > 0 else 1.0,
            1: tp_total_samples / (2 * tp_hit_counts.get(1, 1)) if tp_hit_counts.get(1, 1) > 0 else 1.0  # Weight for class 1 (TP Hit)
        }
        # Simple weight search from original script (can be expanded or use RandomSearch like clf_class)
        # For simplicity, using a fixed aggressive weight or one found by a quick test.
        # Example: if TP hits (class 1) are rare, give them higher weight.
        # Original script had a loop for weight search, this is simplified here.
        # A common strategy is inverse frequency, or empirically found best weight.
        # Let's use a configurable or a simple heuristic:
        best_tp_hit_weight_for_class1 = tp_hit_weights_dict.get(1, 10.0)  # Default to 10 if calc fails or use a fixed value

        # Final weights for TP-hit model training
        final_tp_hit_weights_dict = {0: 1.0, 1: best_tp_hit_weight_for_class1}  # Weight class 1 more
        logger.info(f"Class weights for clf_tp_hit training ({log_context_str}): {final_tp_hit_weights_dict}")

        params_clf_tp_hit = {
            'iterations': 500, 'learning_rate': 0.03, 'depth': 4,  # Typically simpler model
            'loss_function': 'Logloss', 'eval_metric': 'F1',
            'early_stopping_rounds': 50, 'random_seed': 42,
            'task_type': "GPU", 'devices': '0', 'verbose': 100,
            'class_weights': final_tp_hit_weights_dict
        }
        model_clf_tp_hit = CatBoostClassifier(**params_clf_tp_hit)
        eval_set_tp_hit = (X_test_final, y_test_tp_hit) if y_test_tp_hit is not None and not y_test_tp_hit.empty else None
        try:
            model_clf_tp_hit.fit(X_train_final, y_tp_hit_train_cvpool, eval_set=eval_set_tp_hit, plot=False)
            save_model_joblib(model_clf_tp_hit, os.path.join(models_output_dir_abs, f"{file_naming_suffix}_clf_tp_hit.pkl"))
        except Exception as e_clf_tp:
            logger.error(f"Error training clf_tp_hit model for {log_context_str}: {e_clf_tp}", exc_info=True)
            model_clf_tp_hit = None
    elif train_tp_hit_model_enabled:
        logger.warning(f"TP-hit model training skipped for {log_context_str} due to lack of valid training data for it.")

    # --- Evaluate Models on Test Set and Log Metrics ---
    logger.info(f"\n📊 Evaluating models on test set for {log_context_str}...")
    metrics_summary = f"\n=== Metrics Report for {log_context_str} ===\n"
    metrics_summary += f"Timestamp: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
    metrics_summary += f"Features file: {features_file_path}\n"
    metrics_summary += f"Number of selected features: {len(final_selected_feature_cols)}\n"
    metrics_summary += f"Train/CV Pool size: {len(X_train_final)}, Test size: {len(X_test_final)}\n"

    # Evaluate clf_class
    if final_clf_class_model and not X_test_final.empty and not y_test_class.empty:
        try:
            y_pred_proba_class_test = final_clf_class_model.predict_proba(X_test_final)
            # Use threshold from config or default for converting probas to class labels
            proba_threshold = _get_training_config_value("predict_proba_threshold_class", sub_section="model_training", default_value=DEFAULT_PREDICT_PROBA_THRESHOLD_CLASS)

            # Assuming positive class (UP) is at index 1 if model.classes_ is [0, 1] or ['DOWN', 'UP']
            positive_class_idx_clf = 1  # Default
            clf_model_classes = getattr(final_clf_class_model, 'classes_', [0, 1])
            if 1 in clf_model_classes:
                positive_class_idx_clf = list(clf_model_classes).index(1)
            elif 'UP' in clf_model_classes:
                positive_class_idx_clf = list(clf_model_classes).index('UP')

            y_pred_class_test = (y_pred_proba_class_test[:, positive_class_idx_clf] > proba_threshold).astype(int)

            target_names_binary = _get_training_config_value("target_class_names_binary", sub_section="model_training", default_value=DEFAULT_TARGET_CLASS_NAMES_BINARY)

            metrics_summary += f"\n--- clf_class (Binary UP/DOWN, Threshold: {proba_threshold:.2f}) ---\n"
            metrics_summary += classification_report(y_test_class, y_pred_class_test, target_names=target_names_binary, digits=4, zero_division=0)
            metrics_summary += f"Accuracy: {accuracy_score(y_test_class, y_pred_class_test):.4f}\n"
            if len(np.unique(y_test_class)) > 1:  # AUC requires at least two classes in y_true
                metrics_summary += f"ROC AUC (UP): {roc_auc_score(y_test_class, y_pred_proba_class_test[:, positive_class_idx_clf]):.4f}\n"
                metrics_summary += f"PR AUC (UP): {average_precision_score(y_test_class, y_pred_proba_class_test[:, positive_class_idx_clf]):.4f}\n"
            cm_clf = confusion_matrix(y_test_class, y_pred_class_test, labels=[0, 1])  # Ensure labels for CM
            metrics_summary += f"Confusion Matrix (clf_class):\n{pd.DataFrame(cm_clf, index=[f'True_{n}' for n in target_names_binary], columns=[f'Pred_{n}' for n in target_names_binary])}\n"
        except Exception as e_eval_clf:
            metrics_summary += f"Error evaluating clf_class: {e_eval_clf}\n"
    else:
        metrics_summary += "\n--- clf_class: Not evaluated (model or test data missing) ---\n"

    # Evaluate reg_delta
    if model_reg_delta and not X_test_final.empty and not y_test_delta.empty:
        try:
            y_pred_delta_test = model_reg_delta.predict(X_test_final)
            mae_delta = mean_absolute_error(y_test_delta, y_pred_delta_test)
            metrics_summary += f"\n--- reg_delta ---\nMAE: {mae_delta:.6f}\n"
        except Exception as e_eval_delta:
            metrics_summary += f"Error evaluating reg_delta: {e_eval_delta}\n"
    else:
        metrics_summary += "\n--- reg_delta: Not evaluated ---\n"

    # Evaluate reg_vol
    if model_reg_vol and not X_test_final.empty and not y_vol_train_cvpool.empty:  # y_vol_train_cvpool was a typo, should be y_test_vol
        try:
            y_pred_vol_test = model_reg_vol.predict(X_test_final)
            mae_vol = mean_absolute_error(y_test_vol, y_pred_vol_test)  # Corrected to y_test_vol
            metrics_summary += f"\n--- reg_vol ---\nMAE: {mae_vol:.6f}\n"
        except Exception as e_eval_vol:
            metrics_summary += f"Error evaluating reg_vol: {e_eval_vol}\n"
    else:
        metrics_summary += "\n--- reg_vol: Not evaluated ---\n"

    # Evaluate clf_tp_hit
    if model_clf_tp_hit and y_test_tp_hit is not None and not X_test_final.empty and not y_test_tp_hit.empty:
        try:
            y_pred_tp_hit_test = model_clf_tp_hit.predict(X_test_final)
            metrics_summary += f"\n--- clf_tp_hit ---\n"
            metrics_summary += classification_report(y_test_tp_hit, y_pred_tp_hit_test, target_names=['No TP Hit (0)', 'TP Hit (1)'], digits=4, zero_division=0)
            metrics_summary += f"Accuracy (TP-hit): {accuracy_score(y_test_tp_hit, y_pred_tp_hit_test):.4f}\n"
            cm_tp_hit = confusion_matrix(y_test_tp_hit, y_pred_tp_hit_test, labels=[0, 1])
            metrics_summary += f"Confusion Matrix (clf_tp_hit):\n{pd.DataFrame(cm_tp_hit, index=['True_0', 'True_1'], columns=['Pred_0', 'Pred_1'])}\n"
        except Exception as e_eval_tp:
            metrics_summary += f"Error evaluating clf_tp_hit: {e_eval_tp}\n"
    elif train_tp_hit_model_enabled:  # If training was enabled but model is None or test data missing
        metrics_summary += "\n--- clf_tp_hit: Not evaluated (model or test data missing, though training was attempted) ---\n"

    logger.info(metrics_summary)
    # Save metrics to a log file
    metrics_log_file_path = os.path.join(logs_output_dir_abs, f"train_metrics_{file_naming_suffix}.txt")
    try:
        with open(metrics_log_file_path, "w", encoding="utf-8") as f_log:  # Overwrite for each new training run
            f_log.write(metrics_summary)
            f_log.write("\nSelected Features:\n" + "\n".join(final_selected_feature_cols) + "\n")
        logger.info(f"Training metrics saved to: {metrics_log_file_path}")
    except Exception as e_log:
        logger.error(f"Error saving metrics log to {metrics_log_file_path}: {e_log}")

    logger.info(f"✅ Model training for {log_context_str} completed.")
    return 0  # Success


if __name__ == "__main__":
    # Load config first for defaults, then parse args
    CONFIG = load_config()
    if not CONFIG:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')
        logger.warning("Running trainer.py with default config values as config.yaml failed to load.")
    else:
        # Setup logging properly if config loaded (e.g. if it defines log levels/paths)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s [%(module)s] - %(message)s')

    parser = argparse.ArgumentParser(description="Train CatBoost models for crypto prediction.")
    parser.add_argument('--tf', type=str, required=True,
                        help="Timeframe to train models for (e.g., 15m).")
    parser.add_argument('--symbol', type=str, default=None,
                        help="Specific symbol to train a model for (e.g., BTCUSDT).")
    parser.add_argument('--symbol-group', type=str, default=None,
                        help="Symbol group key to train a group model for (e.g., top8).")

    args = parser.parse_args()

    # --- Determine training mode based on args ---
    filter_arg_value = None
    is_group_arg = False

    if args.symbol and args.symbol_group:
        logger.error("Cannot specify both --symbol and --symbol-group. Please choose one. Exiting.")
        sys.exit(1)
    elif args.symbol_group:
        # Check if group key is valid (from config or default)
        all_groups_cfg = _get_training_config_value("symbol_groups", default_value={})
        if args.symbol_group.lower() not in all_groups_cfg:
            logger.error(f"Unknown symbol group: '{args.symbol_group}'. Available in config: {list(all_groups_cfg.keys())}. Exiting.")
            sys.exit(1)
        filter_arg_value = args.symbol_group.lower()  # Use group key as filter
        is_group_arg = True
        logger.info(f"Training mode: Group model for '{filter_arg_value}'.")
    elif args.symbol:
        filter_arg_value = args.symbol.upper()  # Use symbol as filter
        is_group_arg = False
        logger.info(f"Training mode: Symbol-specific model for '{filter_arg_value}'.")
    else:
        # No symbol or group specified, train a generic model for the timeframe
        logger.info(f"Training mode: Generic model for timeframe '{args.tf}'.")
        # filter_arg_value remains None

    exit_code = 1  # Default to error
    try:
        exit_code = main_train_logic(
            timeframe_to_train=args.tf,
            symbol_or_group_filter=filter_arg_value,
            is_group_model=is_group_arg
        )
    except KeyboardInterrupt:
        context_exit = f"TF {args.tf}" + (f", Filter {filter_arg_value}" if filter_arg_value else "")
        logger.info(f"\n[Trainer] 🛑 Model training for {context_exit} interrupted by user.")
        sys.exit(130)
    except Exception as e:
        context_exit = f"TF {args.tf}" + (f", Filter {filter_arg_value}" if filter_arg_value else "")
        logger.critical(f"[Trainer] 💥 Unexpected critical error during training for {context_exit}: {e}", exc_info=True)
        sys.exit(1)

    sys.exit(exit_code)
