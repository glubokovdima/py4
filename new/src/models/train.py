# src/models/train.py

import pandas as pd
import numpy as np
import os
import sys
import logging
import joblib
import json

# Import sklearn components needed for splitting
from sklearn.model_selection import train_test_split

# Import config and logging setup
from src.utils.config import get_config
# No need for setup_logging here, it's called by the entry script (scripts/train.py)

# Import functions from the new modules
from src.models.feature_selection import select_features
from src.models.model_training import (
    train_classification_model,
    train_regression_model,
    train_tp_hit_model
)
from src.models.evaluation import evaluate_models

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Load Configuration ---
config = get_config()
PATHS_CONFIG = config['paths']
TRAINING_CONFIG = config['training']
FEATURE_BUILDER_CONFIG = config['feature_builder'] # Need this for TP threshold / dropna cols
SYMBOL_GROUPS = config['symbol_groups'] # Need groups for fallback logic (though fallback is handled in predict)


# --- Constants from Config ---
TEST_SIZE = TRAINING_CONFIG['test_size']
RANDOM_STATE = TRAINING_CONFIG['random_state']
DROPNNA_COLS = FEATURE_BUILDER_CONFIG['dropna_columns'] # List of columns to dropna on


# Target name mapping (must be consistent with feature building/evaluation)
# In src/models/train.py, we mapped {'DOWN': 0, 'UP': 1}
# This is now handled in model_training, but mapping int->str for reports
# is in evaluation.py. Let's keep the map here for clarity on what 0/1 represents.
TARGET_CLASS_LABEL_MAP = {'DOWN': 0, 'UP': 1}


# --- Main Training Function ---

def train_models_for_group_or_symbol(tf_train, symbol_or_group_key):
    """
    Oбучает модели CatBoost для заданного таймфрейма и символа/группы.
    Orchestrates data loading, splitting, feature selection, training, and evaluation.

    Args:
        tf_train (str): Таймфрейм данных (например, '1m', '5m').
        symbol_or_group_key (str): Ключ символа ('BTCUSDT') или группы ('top8')
                                   или 'all' para processamento geral features file.
    """
    # Define a prefix for filenames based on symbol/group key and timeframe
    file_prefix = f"{symbol_or_group_key}_{tf_train}"

    # Define a string for logging context
    log_context_str = f"таймфрейма: {tf_train}, ключ: {symbol_or_group_key}"

    logger.info(f"🚀  Начало обучения моделей для {log_context_str}")

    # --- Load Features ---
    # Features file name depends on the symbol_or_group_key
    features_path = os.path.join(PATHS_CONFIG['data_dir'], f"features_{file_prefix}.pkl")

    # Strictly require the features file matching the key/tf
    if not os.path.exists(features_path):
         logger.error(f"❌ Файл признаков не найден: {os.path.basename(features_path)}. Обучение для {log_context_str} невозможно. Убедитесь, что preprocess.py был запущен с соответствующим ключом и таймфреймом.")
         return

    try:
        df = pd.read_pickle(features_path)
        logger.info(f"Загружены признаки из {os.path.basename(features_path)}. Размер: {df.shape}")
    except Exception as e:
        logger.error(f"Ошибка чтения файла признаков {features_path}: {e}")
        return

    if df.empty:
        logger.error(f"Файл признаков {os.path.basename(features_path)} пуст. Обучение для {log_context_str} невозможно.")
        return

    # --- Data Preparation ---

    # Define columns required to be non-null for training examples
    # These are typically the target variables and possibly key features
    required_cols_for_dropna = list(DROPNNA_COLS) # Use a copy of the list from config
    target_tp_hit_col = 'target_tp_hit' # Name from feature building

    train_tp_hit_model_flag = target_tp_hit_col in df.columns # Assume TP-hit model if column exists

    if train_tp_hit_model_flag:
        logger.info(f"Обнаружена колонка '{target_tp_hit_col}'. Будет предпринята попытка обучить модель TP-hit.")
        if target_tp_hit_col not in required_cols_for_dropna:
             required_cols_for_dropna.append(target_tp_hit_col)
    else:
        logger.warning(
            f"Колонка '{target_tp_hit_col}' не найдена в {os.path.basename(features_path)}. "
            f"Модель TP-hit (clf_tp_hit) НЕ будет обучена для {log_context_str}."
        )


    # Drop rows with NaN in required columns for training
    len_before_dropna = len(df)
    df_cleaned = df.dropna(subset=required_cols_for_dropna).copy()
    len_after_dropna = len(df_cleaned)

    if df_cleaned.empty:
        logger.error(
            f"DataFrame пуст после удаления NaN по колонкам {required_cols_for_dropna} для {log_context_str}. (Удалено {len_before_dropna - len_after_dropna} строк). Проверьте данные.")
        return

    logger.info(f"Размер DataFrame после очистки NaN ({len_before_dropna} -> {len_after_dropna}) для {log_context_str}: {df_cleaned.shape}")

    # Check distribution of target classes after cleaning
    if 'target_class' in df_cleaned.columns:
         logger.info(
             f"Распределение классов в 'target_class' (после очистки NaN) для {log_context_str}: {df_cleaned['target_class'].value_counts().to_dict()}")
    if train_tp_hit_model_flag and target_tp_hit_col in df_cleaned.columns:
        logger.info(
            f"Распределение классов в '{target_tp_hit_col}' (после очистки NaN) для {log_context_str}: {df_cleaned[target_tp_hit_col].value_counts().to_dict()}")


    # Define features (X) and targets (y)
    # Exclude original candle data, identifiers, and target variables
    excluded_cols_for_features = [
        'symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume', # Original data
        'future_close', 'delta', 'volatility', 'target_up', 'target_class', target_tp_hit_col # Target/Leakage variables
    ]

    # Ensure these are handled if they survived cleaning
    excluded_cols_for_features = [col for col in excluded_cols_for_features if col in df_cleaned.columns]


    feature_cols_initial = [col for col in df_cleaned.columns if col not in excluded_cols_for_features]
    # Keep only numerical features
    feature_cols_initial = [col for col in feature_cols_initial if df_cleaned[col].dtype in [np.int64, np.float64, np.int32, np.float32, bool]]
    # Ensure no datetime or string columns snuck in
    feature_cols_initial = [col for col in feature_cols_initial if not pd.api.types.is_datetime64_any_dtype(df_cleaned[col]) and not pd.api.types.is_string_dtype(df_cleaned[col])]


    if not feature_cols_initial:
        logger.error(
            f"После удаления целевых и служебных колонок не осталось признаков для обучения для {log_context_str}. Проверьте файл признаков и список исключений.")
        return

    logger.info(
        f"Начальное количество признаков для обучения ({log_context_str}): {len(feature_cols_initial)}. Первые 10: {feature_cols_initial[:10]}")

    X_all_data = df_cleaned[feature_cols_initial]
    y_class_all = df_cleaned['target_class'] # Still strings ('UP', 'DOWN', NaN)
    y_delta_all = df_cleaned['delta']
    y_vol_all = df_cleaned['volatility']
    y_tp_hit_all = df_cleaned[target_tp_hit_col].astype(int) if train_tp_hit_model_flag else pd.Series([], dtype=int) # Ensure int type


    if X_all_data.empty:
        logger.error(
            f"DataFrame X_all_data (признаки) пуст для {log_context_str}. Проверьте feature_cols_initial и исходные данные.")
        return

    if len(X_all_data) < 20: # Arbitrary minimum for split
        logger.error(
            f"Слишком мало данных ({len(X_all_data)}) для разделения на train/test для {log_context_str}. Обучение невозможно.")
        return

    # Map target_class to integers (0 for DOWN, 1 for UP) for CatBoost
    # This mapping must only be applied to the target series, not used for splitting/filtering
    y_class_all_mapped = y_class_all.map(TARGET_CLASS_LABEL_MAP).dropna()
    # Ensure X and other targets align with the indices where target_class is not NaN
    # This is crucial for a consistent train/test split
    aligned_indices = y_class_all_mapped.index

    X_aligned = X_all_data.loc[aligned_indices]
    y_delta_aligned = y_delta_all.loc[aligned_indices]
    y_vol_aligned = y_vol_all.loc[aligned_indices]
    y_tp_hit_aligned = y_tp_hit_all.loc[aligned_indices] # Will contain NaNs if TP-hit had NaNs before initial dropna

    if X_aligned.empty or y_class_all_mapped.empty:
         logger.error(f"Data is empty after aligning based on target_class for {log_context_str}. Check target distribution and NaNs.")
         return

    # Perform the train/test split
    try:
        # Stratify on the mapped target_class
        X_train_cv_pool, X_test_full, y_class_train_cv_pool_mapped, y_test_class_mapped = train_test_split(
            X_aligned, y_class_all_mapped,
            test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True, stratify=y_class_all_mapped
        )

        # Align other targets based on the indices from the classification split
        y_train_delta = y_delta_aligned.loc[X_train_cv_pool.index]
        y_test_delta = y_delta_aligned.loc[X_test_full.index]

        y_train_vol = y_vol_aligned.loc[X_train_cv_pool.index]
        y_test_vol = y_vol_aligned.loc[X_test_full.index]

        # Align TP-hit target. It might contain NaNs if the original TP-hit column had NaNs
        # even after the initial required_cols_for_dropna. NaNs are handled *within*
        # train_tp_hit_model by dropping before training.
        y_train_tp_hit = y_tp_hit_aligned.loc[X_train_cv_pool.index]
        y_test_tp_hit = y_tp_hit_aligned.loc[X_test_full.index]


    except ValueError as e:
        logger.error(
            f"Ошибка ValueError при разделении данных для {log_context_str}: {e}. Возможно, слишком мало данных или проблемы с распределением классов для стратификации.")
        if not y_class_all_mapped.empty:
             logger.error(f"Class distribution before split: {y_class_all_mapped.value_counts().to_dict()}")
        return
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при разделении данных для {log_context_str}: {e}", exc_info=True)
        return


    logger.info(f"Размеры выборок ({log_context_str}): Train/CV Pool: {len(X_train_cv_pool)}, Test: {len(X_test_full)}")
    if X_train_cv_pool.empty or X_test_full.empty:
        logger.error(f"X_train_cv_pool (Train/CV) or X_test_full are empty after split for {log_context_str}.")
        return

    # --- Feature Selection ---
    # Pass the training/CV pool data to the feature selection function
    feature_cols_final = select_features(
        X_train_cv_pool,
        y_class_train_cv_pool_mapped, # Use mapped target for FI
        log_context_str
    )

    if not feature_cols_final or not all(col in X_train_cv_pool.columns for col in feature_cols_final):
        logger.error(f"Отбор признаков не вернул действительный список признаков. Обучение невозможно.")
        return # Exit if feature selection failed

    # Select final features for training and testing sets based on the selected list
    X_train_sel = X_train_cv_pool[feature_cols_final]
    X_test_sel = X_test_full[feature_cols_final]

    logger.info(f"Количество признаков после отбора ({log_context_str}): {len(feature_cols_final)}")

    # Save the list of selected features
    try:
        model_output_dir = PATHS_CONFIG['models_dir']
        os.makedirs(model_output_dir, exist_ok=True) # Ensure models directory exists

        features_list_path = os.path.join(model_output_dir, f"{file_prefix}_features_selected.txt")
        with open(features_list_path, "w", encoding="utf-8") as f:
            f.write("\n".join(feature_cols_final))
        logger.info(f"Список отобранных признаков сохранен в {features_list_path}")

        features_json_path = os.path.join(model_output_dir, f"{file_prefix}_features_selected.json")
        try:
            with open(features_json_path, "w", encoding="utf-8") as f_json:
                json.dump(feature_cols_final, f_json, ensure_ascii=False, indent=2)
            logger.info(f"Список отобранных признаков также сохранен в {features_json_path}")
        except Exception as e_json:
            logger.error(f"Ошибка при сохранении признаков в JSON ({log_context_str}): {e_json}")

    except Exception as e:
        logger.error(f"Ошибка при сохранении списка признаков ({log_context_str}): {e}")


    # --- Train Models ---
    # Call the training functions for each model type
    trained_models = {}

    # Classification Model
    trained_models['clf_class'] = train_classification_model(
        X_train_sel, y_class_train_cv_pool_mapped, # Use selected features and mapped target
        X_test_sel, y_test_class_mapped, # Use selected features and mapped target for eval set
        log_context_str
    )

    # Regression Models (Delta and Volatility)
    trained_models['reg_delta'] = train_regression_model(
        X_train_sel, y_train_delta, # Use selected features
        X_test_sel, y_test_delta, # Use selected features for eval set
        'reg_delta',
        log_context_str
    )

    trained_models['reg_vol'] = train_regression_model(
        X_train_sel, y_train_vol, # Use selected features
        X_test_sel, y_test_vol, # Use selected features for eval set
        'reg_vol',
        log_context_str
    )

    # TP-Hit Classifier (Conditional)
    # Pass the data subsets that are potentially not NaN for TP-hit
    # train_tp_hit_model handles dropping NaNs within the function
    if train_tp_hit_model_flag:
        trained_models['clf_tp_hit'] = train_tp_hit_model(
            X_train_sel, y_train_tp_hit, # Use selected features
            X_test_sel, y_test_tp_hit, # Use selected features for eval set
            log_context_str
        )
    else:
        trained_models['clf_tp_hit'] = None # Model not trained


    # --- Save Models ---
    logger.info(f"\n💾 Сохранение обученных моделей для {log_context_str}...")
    model_output_dir = PATHS_CONFIG['models_dir']
    os.makedirs(model_output_dir, exist_ok=True) # Ensure models directory exists

    for model_type, model_instance in trained_models.items():
        if model_instance:
             try:
                model_path = os.path.join(model_output_dir, f"{file_prefix}_{model_type}.pkl")
                joblib.dump(model_instance, model_path)
                logger.info(f"Модель {model_type} сохранена в {model_path}")
             except Exception as e:
                 logger.error(f"Ошибка при сохранении модели {model_type} ({log_context_str}): {e}")
        else:
             logger.warning(f"Модель {model_type} не обучена, пропуск сохранения для {log_context_str}.")


    # --- Evaluate Metrics on Test Set ---
    # Pass the trained models and the test data subsets to the evaluation function
    evaluate_models(
        trained_models,
        X_test_sel,
        y_test_class_mapped, # Use mapped target for classification evaluation
        y_test_delta,
        y_test_vol,
        y_test_tp_hit, # Pass target for TP-hit (evaluation handles its NaNs)
        file_prefix,
        log_context_str
    )

    logger.info(f"✅  Обучение моделей для {log_context_str} завершено.")


# Example usage (optional, for testing the module directly)
if __name__ == "__main__":
    # In a real run, logging and config would be set up by the entry script (scripts/train.py)
    # For direct testing, set it up here
    # This block needs to simulate the input that train_models_for_group_or_symbol expects:
    # - A features file at data/features_{key}_{tf}.pkl
    # - Config loaded (handled by get_config)
    from src.utils.logging_setup import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__) # Re-get logger after setup
    config = get_config() # Re-load config after setup

    print("Testing src/models/train.py (Orchestrator)")

    # To test this orchestrator function, we need a features file.
    # Let's create a dummy features file if it doesn't exist.
    tf_test = '15m'
    symbol_or_group_test_key = 'test_orchestrator' # Use a dummy key

    dummy_features_path = os.path.join(PATHS_CONFIG['data_dir'], f"features_{symbol_or_group_test_key}_{tf_test}.pkl")

    if not os.path.exists(dummy_features_path):
        logger.warning(f"Dummy features file not found at {dummy_features_path}. Creating dummy data for testing.")
        # Create a dummy DataFrame with necessary columns and some NaNs
        n_samples = 2500 # Need enough samples for split, FI, CV, early stopping etc.
        n_features = 60
        dummy_data = {
            'symbol': [random.choice(['TESTA', 'TESTB', 'TESTC']) for _ in range(n_samples)],
            'timestamp': pd.to_datetime(pd.date_range('2023-01-01', periods=n_samples, freq='min')),
            'target_class': np.random.choice(['UP', 'DOWN', np.nan], n_samples, p=[0.4, 0.4, 0.2]), # Include NaNs
            'delta': np.random.randn(n_samples) * 0.01,
            'volatility': np.random.rand(n_samples) * 0.02,
            'target_tp_hit': np.random.choice([0, 1, np.nan], n_samples, p=[0.5, 0.4, 0.1]), # Include NaNs
             # Add some numerical features
            **{f'feature_{i}': np.random.randn(n_samples) * random.uniform(0.1, 1.0) for i in range(n_features)},
            # Add some columns that might be in excluded_cols but might be missing in dummy data
            'open': np.random.rand(n_samples), 'high': np.random.rand(n_samples),
            'low': np.random.rand(n_samples), 'close': np.random.rand(n_samples),
            'volume': np.random.rand(n_samples),
            'future_close': np.random.rand(n_samples), 'target_up': np.random.randint(0,2,n_samples),
        }
        df_dummy = pd.DataFrame(dummy_data)

        # Ensure sufficient non-NaNs for required_cols_for_dropna after creating some NaNs
        # For testing, ensure at least enough clean rows for split and minimums
        required_for_split = 100 # Arbitrary minimum needed for split and subsequent steps
        if len(df_dummy.dropna(subset=DROPNNA_COLS)) < required_for_split:
             logger.warning("Dummy data needs more clean rows for testing training.")
             # Simple way to add clean rows: fill NaNs in a subset
             clean_subset_indices = df_dummy.sample(n=required_for_split, replace=False, random_state=RANDOM_STATE).index
             for col in DROPNNA_COLS:
                 if col in df_dummy.columns:
                     df_dummy.loc[clean_subset_indices, col] = df_dummy.loc[clean_subset_indices, col].fillna(df_dummy[col].mean()) # Fill NaNs in subset with mean

             # Ensure target_class has both 0/1 in the clean subset
             clean_target_class = df_dummy.loc[clean_subset_indices, 'target_class']
             if clean_target_class.dropna().nunique() < 2:
                  df_dummy.loc[clean_subset_indices[0], 'target_class'] = 'UP'
                  if len(clean_subset_indices) > 1:
                       df_dummy.loc[clean_subset_indices[1], 'target_class'] = 'DOWN'

             # Ensure target_tp_hit has both 0/1 in the clean subset
             if 'target_tp_hit' in df_dummy.columns:
                 clean_target_tp_hit = df_dummy.loc[clean_subset_indices, 'target_tp_hit']
                 if clean_target_tp_hit.dropna().nunique() < 2:
                      df_dummy.loc[clean_subset_indices[2], 'target_tp_hit'] = 0
                      if len(clean_subset_indices) > 3:
                           df_dummy.loc[clean_subset_indices[3], 'target_tp_hit'] = 1


        os.makedirs(PATHS_CONFIG['data_dir'], exist_ok=True)
        try:
            df_dummy.to_pickle(dummy_features_path)
            logger.info(f"Dummy features file created at {dummy_features_path}")
        except Exception as e:
            logger.error(f"Failed to save dummy features file: {e}")
            # sys.exit(1) # Don't exit, maybe just warn


    try:
        # Run the main orchestrator function
        train_models_for_group_or_symbol(tf_test, symbol_or_group_test_key)
    except Exception as e:
        logger.error(f"Error during test training orchestration run: {e}", exc_info=True)
        # sys.exit(1) # Don't exit, just report error

    print("\nTrain orchestration module test finished.")