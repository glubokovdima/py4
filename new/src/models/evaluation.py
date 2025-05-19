# src/models/evaluation.py

import pandas as pd
import numpy as np
import os
import logging

# Import sklearn metrics
from sklearn.metrics import (
    classification_report, mean_absolute_error, accuracy_score,
    f1_score, precision_score, confusion_matrix, roc_auc_score,
    average_precision_score
)

# Import config
from src.utils.config import get_config

# Get logger for this module
logger = logging.getLogger(__name__)

# --- Load Configuration ---
config = get_config()
PATHS_CONFIG = config['paths']
# No specific evaluation metrics thresholds needed from config for this module,
# as the evaluation functions just report standard metrics.

# Target name mapping (must be consistent with feature building/training)
# In src/models/train.py, we mapped {'DOWN': 0, 'UP': 1}
TARGET_CLASS_NAMES_MAPPED = {0: 'DOWN', 1: 'UP'}


# --- Evaluation Function ---

def evaluate_models(models, X_test, y_test_class, y_test_delta, y_test_vol, y_test_tp_hit, file_prefix, log_context_str):
    """
    Evaluates trained models on the test set and logs the metrics.

    Args:
        models (dict): Dictionary containing trained model instances
                       (keys: 'clf_class', 'reg_delta', 'reg_vol', 'clf_tp_hit').
                       Model instance can be None if not trained.
        X_test (pd.DataFrame): Test features.
        y_test_class (pd.Series): Test target for classification (0/1 integers).
        y_test_delta (pd.Series): Test target for delta regression (float).
        y_test_vol (pd.Series): Test target for volatility regression (float).
        y_test_tp_hit (pd.Series): Test target for TP-hit classification (0/1 integers).
        file_prefix (str): Prefix for log file names (e.g., "symbol_tf").
        log_context_str (str): String describing the current training context for logging.
    """
    logger.info(f"\n📊  Начало оценки моделей на тестовой выборке ({log_context_str}):")

    if X_test.empty:
        logger.warning(f"X_test пуст, невозможно рассчитать метрики для {log_context_str}. Пропуск оценки.")
        return

    # Use a dedicated log file for metrics for this specific training run
    metrics_log_path = os.path.join(PATHS_CONFIG['logs_dir'], f"train_metrics_{file_prefix}.txt")
    os.makedirs(PATHS_CONFIG['logs_dir'], exist_ok=True) # Ensure log directory exists

    try:
        # Open the log file in append mode, as training logs might have already started it
        # Or, better, open in write mode to ensure a fresh metrics log per run
        # Let's open in write mode to overwrite if re-running training
        with open(metrics_log_path, "w", encoding="utf-8") as f_log:
            f_log.write(f"=== Метрики обучения для {log_context_str} ({pd.Timestamp.now()}) ===\n\n")

            # --- Metrics for clf_class ---
            f_log.write(f"--- Классификатор направления (clf_class) ---\n")
            clf_class_model = models.get('clf_class')
            if clf_class_model and not y_test_class.empty:
                try:
                    # Ensure y_test_class is integer type for evaluation
                    y_test_class_int = y_test_class.astype(int).squeeze() # Ensure flat series
                    y_pred_class_test = clf_class_model.predict(X_test)
                    proba_class_test = clf_class_model.predict_proba(X_test)

                    # Ensure predictions are flat
                    y_pred_class_flat = pd.Series(y_pred_class_test).squeeze()

                    # Handle potential single class in test set for reporting
                    # Use unique labels from both true and predicted for report labels
                    unique_labels = sorted(list(set(y_test_class_flat.unique()) | set(y_pred_class_flat.unique())))
                    # Map integer labels back to names for report
                    target_names_for_report = [TARGET_CLASS_NAMES_MAPPED.get(i, f'Class_{i}') for i in unique_labels]

                    try:
                        report_class_str = classification_report(y_test_class_flat, y_pred_class_flat,
                                                                 labels=unique_labels, # Use actual labels present
                                                                 target_names=target_names_for_report,
                                                                 digits=4, zero_division=0)
                        f_log.write(report_class_str + "\n")
                        logger.info(f"\nClassification Report (clf_class) for {log_context_str}:\n{report_class_str}")
                    except Exception as e_report:
                        f_log.write("Classification report failed.\n")
                        logger.warning(f"Ошибка при создании classification_report для clf_class ({log_context_str}): {e_report}. Попытка рассчитать отдельные метрики.")


                    # Calculate individual metrics
                    try:
                        acc_class_val = accuracy_score(y_test_class_flat, y_pred_class_flat)
                        f_log.write(f"Accuracy: {acc_class_val:.4f}\n")
                        logger.info(f"Accuracy (clf_class): {acc_class_val:.4f}")
                    except Exception: pass # Ignore if calculation fails

                    # Need to find the index of class 1 for F1, Precision, AUC
                    model_classes_clf = getattr(clf_class_model, 'classes_', [0, 1]) # Default to [0, 1] if .classes_ not found
                    try:
                        # Find the index corresponding to the integer label 1
                        positive_class_idx_clf = list(model_classes_clf).index(1)
                    except ValueError:
                        logger.warning(f"Positive class integer label '1' not found in clf_class.classes_ ({model_classes_clf}). Cannot compute F1/Precision/AUC for class 1.")
                        positive_class_idx_clf = None # Cannot proceed with class-specific metrics


                    if positive_class_idx_clf is not None:
                         # Check if class 1 is actually present in the true test labels before calculating pos_label metrics
                         if 1 in y_test_class_flat.unique():
                              try:
                                 f1_class_val = f1_score(y_test_class_flat, y_pred_class_flat, pos_label=1, zero_division=0)
                                 f_log.write(f"F1-score (UP/1): {f1_class_val:.4f}\n")
                                 logger.info(f"F1-score (UP/1, clf_class): {f1_class_val:.4f}")
                              except Exception: pass # Ignore if calculation fails

                              try:
                                 precision_class_val = precision_score(y_test_class_flat, y_pred_class_flat, pos_label=1, zero_division=0)
                                 f_log.write(f"Precision (UP/1): {precision_class_val:.4f}\n")
                                 logger.info(f"Precision (UP/1, clf_class): {precision_class_val:.4f}")
                              except Exception: pass # Ignore if calculation fails
                         else:
                              f_log.write("F1/Precision for UP/1 not calculated (class 1 not in y_test).\n")
                              logger.warning(f"Class 1 ('UP') not present in y_test_class_flat for {log_context_str}. Skipping F1/Precision for UP.")


                         # ROC AUC and PR AUC require probability scores for the positive class
                         if len(np.unique(y_test_class_flat)) > 1: # Need at least two classes in true labels
                              try:
                                  # Select probability column corresponding to integer label 1
                                  y_score_probs_clf = proba_class_test[:, positive_class_idx_clf]
                                  auc_class_val = roc_auc_score(y_test_class_flat, y_score_probs_clf)
                                  f_log.write(f"ROC AUC (UP/1): {auc_class_val:.4f}\n")
                                  logger.info(f"ROC AUC (UP/1, clf_class): {auc_class_val:.4f}")
                              except Exception as e_auc:
                                  f_log.write("ROC AUC calculation failed.\n")
                                  logger.warning(f"Не удалось рассчитать ROC AUC для clf_class ({log_context_str}): {e_auc}")

                              try:
                                  pr_auc_class_val = average_precision_score(y_test_class_flat, y_score_probs_clf)
                                  f_log.write(f"PR AUC (UP/1): {pr_auc_class_val:.4f}\n")
                                  logger.info(f"PR AUC (UP/1, clf_class): {pr_auc_class_val:.4f}")
                              except Exception as e_pr_auc:
                                   f_log.write("PR AUC calculation failed.\n")
                                   logger.warning(f"Не удалось рассчитать PR AUC для clf_class ({log_context_str}): {e_pr_auc}")

                         else:
                              f_log.write("ROC AUC/PR AUC not calculated (only one class in y_test).\n")
                              logger.warning(f"Не удалось рассчитать ROC AUC/PR AUC для clf_class ({log_context_str}): в y_test_class_flat только один класс.")


                    # Confusion Matrix
                    try:
                         # Use labels [0, 1] if both are present in test/pred for consistent CM structure
                         cm_labels_for_matrix = [0, 1]
                         # Ensure these labels are actually in the unique labels set before trying to create CM
                         if set(cm_labels_for_matrix).issubset(set(y_test_class_flat.unique()) | set(y_pred_class_flat.unique())):
                              cm = confusion_matrix(y_test_class_flat, y_pred_class_flat, labels=cm_labels_for_matrix)
                              cm_df = pd.DataFrame(cm,
                                                   index=[f"True_{TARGET_CLASS_NAMES_MAPPED.get(l, l)}" for l in cm_labels_for_matrix],
                                                   columns=[f"Pred_{TARGET_CLASS_NAMES_MAPPED.get(l, l)}" for l in cm_labels_for_matrix])
                              f_log.write("\nConfusion Matrix (clf_class):\n")
                              f_log.write(cm_df.to_string() + "\n")
                              logger.info(f"\nConfusion Matrix (clf_class) for {log_context_str}:\n{cm_df}")
                         else:
                              f_log.write("Confusion Matrix not calculated (not enough unique labels in test/pred).\n")
                              logger.warning(f"Недостаточно уникальных меток в y_test_class_flat/y_pred_class_flat для создания полной матрицы смещения ({log_context_str}).")

                    except Exception as e_cm:
                          f_log.write("Confusion Matrix calculation failed.\n")
                          logger.warning(f"Ошибка при создании Confusion Matrix для clf_class ({log_context_str}): {e_cm}")


                except Exception as e:
                    f_log.write(f"Error calculating clf_class metrics: {e}\n")
                    logger.error(f"Ошибка при расчете метрик clf_class для {log_context_str}: {e}", exc_info=True)
            else:
                f_log.write("Metrics not calculated (model not trained or test data empty).\n")
                logger.warning(
                    f"Метрики clf_class не рассчитываются для {log_context_str} (модель не обучена или y_test_class пуст).")


            # --- Metrics for reg_delta ---
            f_log.write("\n--- Регрессор дельты (reg_delta) ---\n")
            reg_delta_model = models.get('reg_delta')
            if reg_delta_model and not X_test.empty and not y_test_delta.empty:
                try:
                    y_pred_delta_test = reg_delta_model.predict(X_test)
                    mae_delta_val = mean_absolute_error(y_test_delta, y_pred_delta_test)
                    f_log.write(f"MAE Delta: {mae_delta_val:.6f}\n")
                    logger.info(f"MAE Delta: {mae_delta_val:.6f}")
                except Exception as e:
                    f_log.write(f"Error calculating MAE Delta: {e}\n")
                    logger.error(f"Ошибка при расчете MAE Delta для {log_context_str}: {e}")
            else:
                f_log.write("Metrics not calculated (model not trained or test data empty).\n")
                logger.warning(f"MAE Delta не рассчитывается для {log_context_str} (модель не обучена или y_test_delta пуст).")


            # --- Metrics for reg_vol ---
            f_log.write("\n--- Регрессор волатильности (reg_vol) ---\n")
            reg_vol_model = models.get('reg_vol')
            if reg_vol_model and not X_test.empty and not y_test_vol.empty:
                try:
                    y_pred_vol_test = reg_vol_model.predict(X_test)
                    mae_vol_val = mean_absolute_error(y_test_vol, y_pred_vol_test)
                    f_log.write(f"MAE Volatility: {mae_vol_val:.6f}\n")
                    logger.info(f"MAE Volatility: {mae_vol_val:.6f}")
                except Exception as e:
                    f_log.write(f"Error calculating MAE Volatility: {e}\n")
                    logger.error(f"Ошибка при расчете MAE Volatility для {log_context_str}: {e}")
            else:
                f_log.write("Metrics not calculated (model not trained or test data empty).\n")
                logger.warning(f"MAE Volatility не рассчитывается для {log_context_str} (модель не обучена или y_test_vol пуст).")


            # --- Metrics for clf_tp_hit ---
            clf_tp_hit_model = models.get('clf_tp_hit')
            if clf_tp_hit_model: # Only include section if TP-hit model was trained (and exists)
                 f_log.write("\n--- Классификатор TP-Hit (clf_tp_hit) ---\n")
                 if not X_test.empty and not y_test_tp_hit.empty:
                     try:
                         y_test_tp_hit_int = y_test_tp_hit.astype(int).squeeze() # Ensure flat series
                         y_pred_tp_hit_test = clf_tp_hit_model.predict(X_test)

                         # Ensure predictions are flat and valid (not NaN)
                         y_pred_tp_hit_flat = pd.Series(y_pred_tp_hit_test).squeeze().astype(int) # Ensure int type

                         if not y_test_tp_hit_flat.empty and not y_pred_tp_hit_flat.empty:
                              # Handle potential single class in test set for reporting
                              unique_labels_tp = sorted(list(set(y_test_tp_hit_flat.unique()) | set(y_pred_tp_hit_flat.unique())))
                              target_names_tp = {0: 'No TP Hit (0)', 1: 'TP Hit (1)'}
                              target_names_for_report_tp = [target_names_tp.get(i, f'Class_{i}') for i in unique_labels_tp]


                              try:
                                 report_tp_hit_str = classification_report(y_test_tp_hit_flat, y_pred_tp_hit_flat,
                                                                           labels=unique_labels_tp, # Use actual labels present
                                                                           target_names=target_names_for_report_tp,
                                                                           digits=4, zero_division=0)
                                 f_log.write(report_tp_hit_str + "\n")
                                 logger.info(f"\nClassification Report (clf_tp_hit) for {log_context_str}:\n{report_tp_hit_str}")
                              except Exception as e_report_tp:
                                 f_log.write("Classification report failed.\n")
                                 logger.warning(f"Ошибка при создании classification_report для clf_tp_hit ({log_context_str}): {e_report_tp}. Попытка рассчитать отдельные метрики.")

                              # Calculate individual metrics
                              try:
                                  acc_tp_hit_val = accuracy_score(y_test_tp_hit_flat, y_pred_tp_hit_flat)
                                  f_log.write(f"Accuracy: {acc_tp_hit_val:.4f}\n")
                                  logger.info(f"Accuracy (clf_tp_hit): {acc_tp_hit_val:.4f}")
                              except Exception: pass # Ignore if calculation fails

                              # Check if class 1 is actually present in the true test labels before calculating pos_label metrics
                              if 1 in y_test_tp_hit_flat.unique():
                                   try:
                                      # Calculate F1 for the positive class (1)
                                      f1_tp_hit_val = f1_score(y_test_tp_hit_flat, y_pred_tp_hit_flat, zero_division=0, pos_label=1)
                                      f_log.write(f"F1-score (TP-hit/1): {f1_tp_hit_val:.4f}\n")
                                      logger.info(f"F1-score (TP-hit/1, clf_tp_hit): {f1_tp_hit_val:.4f}")
                                   except Exception: pass # Ignore if calculation fails
                              else:
                                   f_log.write("F1 for TP-hit/1 not calculated (class 1 not in y_test).\n")
                                   logger.warning(f"Class 1 ('TP Hit') not present in y_test_tp_hit_flat for {log_context_str}. Skipping F1 for TP Hit.")


                              # Confusion Matrix
                              try:
                                   # Use labels [0, 1] if both are present in test/pred for consistent CM structure
                                   cm_labels_for_matrix_tp = [0, 1]
                                   if set(cm_labels_for_matrix_tp).issubset(set(y_test_tp_hit_flat.unique()) | set(y_pred_tp_hit_flat.unique())):
                                      cm_tp_hit = confusion_matrix(y_test_tp_hit_flat, y_pred_tp_hit_flat, labels=cm_labels_for_matrix_tp)
                                      cm_tp_hit_df = pd.DataFrame(cm_tp_hit,
                                                                   index=[f"True_{target_names_tp.get(l, l)}" for l in cm_labels_for_matrix_tp],
                                                                   columns=[f"Pred_{target_names_tp.get(l, l)}" for l in cm_labels_for_matrix_tp])
                                      f_log.write("\nConfusion Matrix (clf_tp_hit):\n")
                                      f_log.write(cm_tp_hit_df.to_string() + "\n")
                                      logger.info(f"\nConfusion Matrix (clf_tp_hit) for {log_context_str}:\n{cm_tp_hit_df}")
                                   else:
                                      f_log.write("Confusion Matrix not calculated (not enough unique labels in test/pred).\n")
                                      logger.warning(f"Недостаточно уникальных меток в y_test_tp_hit_flat/y_pred_tp_hit_flat для создания полной матрицы смещения ({log_context_str}).")

                              except Exception as e_cm_tp:
                                   f_log.write("Confusion Matrix calculation failed.\n")
                                   logger.warning(f"Ошибка при создании Confusion Matrix для clf_tp_hit ({log_context_str}): {e_cm_tp}")


                         else:
                             f_log.write("Metrics not calculated (no valid test/pred data after alignment).\n")
                             logger.warning(f"Нет валидных данных в y_test_tp_hit_flat или y_pred_tp_hit_flat для расчета метрик TP-hit ({log_context_str}).")

                     except Exception as e:
                         f_log.write(f"Error calculating clf_tp_hit metrics: {e}\n")
                         logger.error(f"Ошибка при расчете метрик clf_tp_hit для {log_context_str}: {e}", exc_info=True)
                 else:
                     f_log.write("Metrics not calculated (model not trained or test data empty).\n")
                     logger.warning(f"Метрики clf_tp_hit не рассчитываются для {log_context_str} (модель не обучена или y_test_tp_hit пуст).")
            else:
                 f_log.write("\n--- Классификатор TP-Hit (clf_tp_hit) ---\n")
                 f_log.write("Model was not trained (column missing or data insufficient).\n")
                 logger.info(f"TP-hit model was not trained for {log_context_str}. Skipping evaluation.")


            # Optionally write selected features list to metrics log for completeness
            # This requires passing the selected features list to this function
            # For now, it's saved separately by the train orchestrator function.
            # We could load it here, but passing it is cleaner. Let's add it as an argument.
            # Add feature_cols_final argument to evaluate_models function signature.
            # feature_cols_final = models.get('feature_cols_final') # Or pass it explicitly
            # if feature_cols_final:
            #     f_log.write(f"\nUsed Features ({len(feature_cols_final)}):\n" + ", ".join(feature_cols_final) + "\n")


        logger.info(f"✅ Оценка моделей завершена. Метрики сохранены в лог: {metrics_log_path}")

    except Exception as e_log_save:
        logger.error(f"Ошибка при создании/записи лога метрик ({metrics_log_path}): {e_log_save}")


# Example usage (optional, for testing the module directly)
if __name__ == "__main__":
    # In a real run, logging and config would be set up by the entry script
    # For direct testing, set it up here
    from src.utils.logging_setup import setup_logging
    setup_logging()
    logger = logging.getLogger(__name__) # Re-get logger after setup
    config = get_config() # Re-load config after setup

    print("Testing src/models/evaluation.py")

    # Create dummy test data and dummy models for testing evaluation functions
    n_samples_test = 500
    n_features_test = 20

    # Create dummy test data
    dummy_X_test = pd.DataFrame(np.random.rand(n_samples_test, n_features_test), columns=[f'feature_{i}' for i in range(n_features_test)])
    # Create dummy test targets
    dummy_y_test_class = pd.Series(np.random.randint(0, 2, n_samples_test)).astype(int) # Binary (0/1)
    dummy_y_test_delta = pd.Series(np.random.randn(n_samples_test) * 0.01) # Float
    dummy_y_test_vol = pd.Series(np.random.rand(n_samples_test) * 0.02) # Float
    dummy_y_test_tp_hit = pd.Series(np.random.randint(0, 2, n_samples_test)).astype(int) # Binary (0/1)

    # Add some NaNs to simulate real data issues
    dummy_y_test_class.loc[dummy_y_test_class.sample(frac=0.05).index] = np.nan
    dummy_y_test_delta.loc[dummy_y_test_delta.sample(frac=0.05).index] = np.nan
    dummy_y_test_vol.loc[dummy_y_test_vol.sample(frac=0.05).index] = np.nan
    dummy_y_test_tp_hit.loc[dummy_y_test_tp_hit.sample(frac=0.05).index] = np.nan

    # Drop rows with NaN in test targets to simulate the data passed for evaluation
    # Evaluation functions expect clean y_test data relevant to the model
    valid_indices_class = dummy_y_test_class.dropna().index
    X_test_class_aligned = dummy_X_test.loc[valid_indices_class]
    y_test_class_clean = dummy_y_test_class.loc[valid_indices_class].astype(int)

    valid_indices_delta = dummy_y_test_delta.dropna().index
    X_test_delta_aligned = dummy_X_test.loc[valid_indices_delta]
    y_test_delta_clean = dummy_y_test_delta.loc[valid_indices_delta]

    valid_indices_vol = dummy_y_test_vol.dropna().index
    X_test_vol_aligned = dummy_X_test.loc[valid_indices_vol]
    y_test_vol_clean = dummy_y_test_vol.loc[valid_indices_vol]

    valid_indices_tp_hit = dummy_y_test_tp_hit.dropna().index
    X_test_tp_hit_aligned = dummy_X_test.loc[valid_indices_tp_hit]
    y_test_tp_hit_clean = dummy_y_test_tp_hit.loc[valid_indices_tp_hit].astype(int)


    # Create dummy model objects (minimalistic, only need predict and predict_proba)
    class DummyClassifier:
        def predict(self, X):
            # Simulate prediction (e.g., predict 0 or 1)
            return np.random.randint(0, 2, len(X))
        def predict_proba(self, X):
            # Simulate probabilities (e.g., random probabilities)
            probs = np.random.rand(len(X), 2)
            probs = probs / probs.sum(axis=1, keepdims=True) # Normalize
            return probs
        # Add classes_ attribute needed for evaluation
        classes_ = np.array([0, 1])

    class DummyRegressor:
        def predict(self, X):
            # Simulate regression prediction
            return np.random.randn(len(X)) * 0.01

    # Instantiate dummy models
    dummy_models = {
        'clf_class': DummyClassifier(),
        'reg_delta': DummyRegressor(),
        'reg_vol': DummyRegressor(),
        'clf_tp_hit': DummyClassifier(), # Assume TP-hit is also binary classification
    }

    # Test evaluation with all models present
    print("\n--- Evaluating with all dummy models ---")
    evaluate_models(dummy_models, X_test_class_aligned, y_test_class_clean, y_test_delta_clean, y_test_vol_clean, y_test_tp_hit_clean, "dummy_all_models", "dummy_context")

    # Test evaluation with some models missing
    print("\n--- Evaluating with some dummy models missing ---")
    dummy_models_missing = {
        'clf_class': dummy_models['clf_class'],
        'reg_delta': None, # Missing
        'reg_vol': dummy_models['reg_vol'],
        'clf_tp_hit': None, # Missing
    }
    evaluate_models(dummy_models_missing, X_test_class_aligned, y_test_class_clean, y_test_delta_clean, y_test_vol_clean, y_test_tp_hit_clean, "dummy_missing_models", "dummy_context_missing")

    print("\nEvaluation module test finished.")