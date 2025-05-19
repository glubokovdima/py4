# src/models/feature_selection.py

import pandas as pd
import numpy as np
import logging

# Import CatBoost and sklearn components needed for feature selection
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

# Import config
from src.utils.config import get_config

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Load Configuration ---
config = get_config()
TRAINING_CONFIG = config['training']

# --- Constants from Config ---
FEATURE_SELECTION_ITERATIONS = TRAINING_CONFIG['catboost']['feature_selection_iterations']
FEATURE_SELECTION_EARLY_STOPPING = TRAINING_CONFIG['catboost']['feature_selection_early_stopping']
FEATURE_SELECTION_N_SPLITS = TRAINING_CONFIG['catboost']['feature_selection_n_splits']
TOP_FEATURES_COUNT = TRAINING_CONFIG['catboost']['top_features_count']
CATBOOST_DEFAULT_LEARNING_RATE = TRAINING_CONFIG['catboost']['default_learning_rate']
CATBOOST_GPU_DEVICES = TRAINING_CONFIG['catboost']['gpu_devices']
RANDOM_STATE = TRAINING_CONFIG['random_state']
MIN_SAMPLES_FOR_STRATIFY = TRAINING_CONFIG.get('min_samples_per_class_stratify', 5)


# --- Feature Selection Function ---

def select_features(X_train_cv_pool, y_class_train_cv_pool, log_context_str):
    """
    Selects top features based on CatBoost Feature Importance using Stratified KFold.

    Args:
        X_train_cv_pool (pd.DataFrame): Feature DataFrame for training/CV pool.
        y_class_train_cv_pool (pd.Series): Target Series for classification (0/1)
                                           for training/CV pool.
        log_context_str (str): String describing the current training context
                               (e.g., "symbol BTCUSDT, tf 15m") for logging.

    Returns:
        list: A list of selected feature column names. Returns all columns if
              selection is skipped or fails.
    """
    top_features_count = TOP_FEATURES_COUNT
    initial_feature_cols = X_train_cv_pool.columns.tolist()
    feature_cols_final = initial_feature_cols # Default to all features

    min_samples_in_class_cv = y_class_train_cv_pool.value_counts().min() if not y_class_train_cv_pool.empty else 0
    n_splits_fi = FEATURE_SELECTION_N_SPLITS

    logger.info(f"🚀 Начало отбора признаков (Top {top_features_count}) для clf_class ({log_context_str}) на основе {len(X_train_cv_pool)} образцов...")

    # Check if feature selection is feasible
    if X_train_cv_pool.empty or y_class_train_cv_pool.empty:
        logger.warning(
            f"X_train_cv_pool или y_class_train_cv_pool пусты перед отбором признаков для {log_context_str}. Отбор пропускается, используются все признаки.")
    elif len(initial_feature_cols) <= top_features_count:
        logger.info(
            f"Количество признаков ({len(initial_feature_cols)}) меньше или равно {top_features_count} для {log_context_str}. Отбор не требуется.")
    elif min_samples_in_class_cv < n_splits_fi or y_class_train_cv_pool.nunique() < 2:
        logger.warning(
            f"Недостаточно примеров в классе ({min_samples_in_class_cv} < {n_splits_fi}) "
            f"или только один класс ({y_class_train_cv_pool.nunique()}) "
            f"в y_class_train_cv_pool для StratifiedKFold ({log_context_str}). Отбор признаков пропускается."
        )
    else:
        # Perform feature selection using Stratified KFold CV
        kf = StratifiedKFold(n_splits=n_splits_fi, shuffle=True, random_state=RANDOM_STATE)
        feature_importances_fi = np.zeros(X_train_cv_pool.shape[1])
        num_successful_folds = 0

        logger.info(f"Running Feature Importance CV ({n_splits_fi} folds) for clf_class ({log_context_str})...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_cv_pool, y_class_train_cv_pool)):
            X_fold_train, X_fold_val = X_train_cv_pool.iloc[train_idx], X_train_cv_pool.iloc[val_idx]
            y_fold_train, y_fold_val = y_class_train_cv_pool.iloc[train_idx], y_class_train_cv_pool.iloc[val_idx]

            if X_fold_val.empty or y_fold_val.empty:
                logger.warning(f"FI CV ({log_context_str}): Validation set empty in fold {fold + 1}. Skipping.")
                continue
            if y_fold_val.nunique() < 2:
                 logger.warning(f"FI CV ({log_context_str}): Validation set in fold {fold + 1} has only one class. Skipping.")
                 continue

            # Use minimal parameters for a quick FI run
            clf_fi = CatBoostClassifier(
                iterations=FEATURE_SELECTION_ITERATIONS, learning_rate=CATBOOST_DEFAULT_LEARNING_RATE,
                early_stopping_rounds=FEATURE_SELECTION_EARLY_STOPPING, verbose=0, # Keep verbose low
                loss_function='Logloss', eval_metric='Accuracy', # Use Logloss/Accuracy for FI
                task_type="GPU", devices=CATBOOST_GPU_DEVICES, random_seed=RANDOM_STATE
            )
            try:
                 clf_fi.fit(X_fold_train, y_fold_train, eval_set=(X_fold_val, y_fold_val))
                 feature_importances_fi += clf_fi.get_feature_importance(type='FeatureImportance')
                 num_successful_folds += 1
            except Exception as e:
                 logger.warning(f"Error during FI CV fold {fold + 1} ({log_context_str}): {e}")


        if num_successful_folds > 0:
            fi_mean = pd.Series(feature_importances_fi / num_successful_folds,
                                index=initial_feature_cols).sort_values(ascending=False)
            feature_cols_final = fi_mean.head(top_features_count).index.tolist()
            logger.info(f"Топ-{top_features_count} признаков для {log_context_str}: {feature_cols_final}")
        else:
            logger.warning(
                f"Не удалось успешно завершить ни одного фолда для отбора признаков ({log_context_str}). Используются все признаки.")
            feature_cols_final = initial_feature_cols # Fallback to all features

    # Ensure selected features are actually present in the input DataFrame columns
    feature_cols_final = [col for col in feature_cols_final if col in initial_feature_cols]
    if not feature_cols_final:
        logger.error(f"После отбора признаков не осталось ни одного признака, присутствующего в исходных данных ({log_context_str}).")
        # This is a critical error for training, returning an empty list might cause issues later.
        # Let's return the original columns if the selected list ends up empty.
        logger.warning("Returning all initial features as the selected list is empty.")
        return initial_feature_cols

    logger.info(f"Количество признаков после отбора ({log_context_str}): {len(feature_cols_final)}")
    return feature_cols_final

# Example usage (optional, for testing the module directly)
if __name__ == "__main__":
    # In a real run, logging and config would be set up by the entry script
    # For direct testing, set it up here
    from src.utils.logging_setup import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__) # Re-get logger after setup
    config = get_config() # Re-load config after setup

    print("Testing src/models/feature_selection.py")

    # Create dummy data for testing feature selection
    n_samples = 1000
    n_features = 30
    # Create imbalanced data
    n_up = int(n_samples * 0.3)
    n_down = n_samples - n_up

    dummy_X = pd.DataFrame(np.random.rand(n_samples, n_features), columns=[f'feature_{i}' for i in range(n_features)])
    # Create a target where some features are actually important
    dummy_X['feature_0'] = np.random.rand(n_samples) + np.where(np.arange(n_samples) < n_up, 1, 0) * 0.5 # Feature_0 important for UP
    dummy_X['feature_5'] = np.random.rand(n_samples) + np.where(np.arange(n_samples) >= n_up, 1, 0) * 0.5 # Feature_5 important for DOWN
    dummy_y = pd.Series(['UP'] * n_up + ['DOWN'] * n_down)
    dummy_y_mapped = dummy_y.map({'DOWN': 0, 'UP': 1}) # Map to 0/1 for CatBoost

    print(f"Dummy data shape: {dummy_X.shape}")
    print(f"Dummy target shape: {dummy_y_mapped.shape}")
    print(f"Dummy target distribution (0/1): {dummy_y_mapped.value_counts().to_dict()}")

    # Add some NaNs to simulate real data
    for col in dummy_X.columns:
         dummy_X.loc[dummy_X.sample(frac=0.02).index, col] = np.nan

    # Drop rows where target is NaN (not applicable in this dummy case, but good practice)
    # dummy_X = dummy_X.dropna(subset=['target_col']) # Not needed for this test structure

    # Simulate data split if needed, but select_features works on the 'train_cv_pool' directly
    # So we can just pass the dummy_X and dummy_y_mapped directly.

    log_context = "dummy_symbol, dummy_tf"

    try:
        selected_features = select_features(dummy_X, dummy_y_mapped, log_context)
        print(f"\nSelected features: {selected_features}")
        print(f"Number of selected features: {len(selected_features)}")

        # Verify if the important features were selected (feature_0 and feature_5)
        if 'feature_0' in selected_features and 'feature_5' in selected_features:
             print("✅ Important features (feature_0, feature_5) were selected.")
        else:
             print("⚠️ Important features (feature_0, feature_5) were NOT selected as expected.")

    except Exception as e:
        logger.error(f"Error during feature selection test: {e}", exc_info=True)

    print("\nFeature selection module test finished.")