# core/backtesting/backtester.py
"""
Handles backtesting of classification models, calculates metrics,
and generates relevant plots.
"""
import pandas as pd
import joblib
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import logging
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Configuration Constants (Ideally from config.yaml) ---
# These are defaults if not overridden by function arguments or config
DEFAULT_OUTPUT_DIR_BACKTEST = 'logs'
DEFAULT_FEATURES_PATH_TEMPLATE = 'data/features_{tf}.pkl'  # Generic, may need suffix
DEFAULT_MODEL_PATH_TEMPLATE = 'models/{tf}_{model_type}.pkl'  # Generic for non-symbol specific model
DEFAULT_MODEL_FEATURES_LIST_TEMPLATE = "models/{tf}_features_selected.txt"  # For non-symbol specific model

# Target class names for reports and confusion matrix
TARGET_CLASS_NAMES_DEFAULT = ['DOWN', 'UP']  # Assuming binary classification (0: DOWN, 1: UP)


# If your model predicts more classes, update this and the mapping logic

def _load_model_for_backtest(model_path):
    """Loads a pre-trained model."""
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return None


def _load_features_for_backtest(features_path, tf_name):
    """Loads features data for backtesting."""
    if not os.path.exists(features_path):
        logger.error(f"Features file not found: {features_path} for timeframe {tf_name}.")
        return pd.DataFrame()
    try:
        df = pd.read_pickle(features_path)
        if df.empty:
            logger.warning(f"Features file {features_path} is empty for timeframe {tf_name}.")
        else:
            logger.info(f"Features loaded from {features_path} for {tf_name}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading features file {features_path} for {tf_name}: {e}")
        return pd.DataFrame()


def _load_feature_list(features_list_path, tf_name):
    """Loads the list of features used for training the model."""
    if not os.path.exists(features_list_path):
        logger.error(f"Feature list file '{features_list_path}' not found for {tf_name}.")
        return None
    try:
        with open(features_list_path, "r", encoding="utf-8") as f:
            feature_cols = [line.strip() for line in f if line.strip()]
        if not feature_cols:
            logger.error(f"Feature list file '{features_list_path}' is empty for {tf_name}.")
            return None
        logger.info(f"Feature list loaded from {features_list_path} for {tf_name} ({len(feature_cols)} features).")
        return feature_cols
    except Exception as e:
        logger.error(f"Error reading feature list file '{features_list_path}' for {tf_name}: {e}")
        return None


def _plot_confusion_matrix(cm, classes, tf_suffix, output_dir, normalize=False, title='Confusion Matrix'):
    """Prints and plots the confusion matrix."""
    plot_filename = f'backtest_conf_matrix_{"norm_" if normalize else ""}{tf_suffix}.png'
    plot_path = os.path.join(output_dir, plot_filename)

    if normalize:
        cm_sum = cm.sum(axis=1)[:, np.newaxis]
        # Handle division by zero if a row sums to 0 (no true samples for that class)
        cm_normalized = np.divide(cm.astype('float'), cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0)
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm_normalized

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title + f' ({tf_suffix})')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):  # Use np.ndindex for iterating over multi-dim arrays
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_path)
        logger.info(f"Confusion matrix saved: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save confusion matrix to {plot_path}: {e}")
    plt.close()


def _plot_daily_accuracy(df_results, tf_name, output_dir):
    """Plots daily accuracy over time."""
    plot_filename = f'backtest_accuracy_daily_{tf_name}.png'
    plot_path = os.path.join(output_dir, plot_filename)

    if df_results.empty or 'timestamp' not in df_results.columns:
        logger.warning(f"No data or 'timestamp' column for daily accuracy plot for {tf_name}.")
        return

    df_plot = df_results.copy()
    df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
    df_plot['date'] = df_plot['timestamp'].dt.date

    daily_accuracy_data = []
    for date_val, group in df_plot.groupby('date'):
        if not group.empty and 'true_label_idx' in group.columns and 'pred_label_idx' in group.columns:
            acc = (group['true_label_idx'] == group['pred_label_idx']).sum() / len(group)
            daily_accuracy_data.append({'date': date_val, 'accuracy': acc, 'samples': len(group)})

    if not daily_accuracy_data:
        logger.warning(f"No daily accuracy data to plot for {tf_name} after grouping.")
        return

    accuracy_by_day_df = pd.DataFrame(daily_accuracy_data).sort_values('date')

    plt.figure(figsize=(12, 6))
    plt.plot(accuracy_by_day_df['date'], accuracy_by_day_df['accuracy'], marker='o', linestyle='-', label='Daily Accuracy')

    if len(accuracy_by_day_df) >= 7:
        accuracy_by_day_df['accuracy_ma_7'] = accuracy_by_day_df['accuracy'].rolling(window=7, min_periods=1).mean()
        plt.plot(accuracy_by_day_df['date'], accuracy_by_day_df['accuracy_ma_7'], linestyle='--', label='7-day MA Accuracy')

    plt.title(f'Daily Accuracy — {tf_name}')
    plt.xlabel('Date')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(plot_path)
        logger.info(f"Daily accuracy plot saved: {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save daily accuracy plot to {plot_path}: {e}")
    plt.close()


def _analyze_confidence_bands(df_results, tf_name):
    """Analyzes accuracy within different confidence score bands."""
    if df_results.empty or 'confidence_score' not in df_results.columns:
        logger.warning(f"No data or 'confidence_score' for confidence analysis for {tf_name}.")
        return

    logger.info(f"\n--- Confidence Score Analysis ({tf_name}) ---")
    # Define confidence bins and labels
    # Original script uses TARGET_CLASS_NAMES for classification which implies 5 classes
    # predict_all.py calculates confidence as P(best) - P(second_best)
    # For binary, this is P(class_1) - P(class_0) if P(class_1) > P(class_0)
    bins = [0, 0.05, 0.1, 0.2, 0.4, 1.01]  # 1.01 to include 1.0
    labels = ["<0.05", "0.05-0.1", "0.1-0.2", "0.2-0.4", ">0.4"]
    df_analysis = df_results.copy()
    df_analysis['conf_group'] = pd.cut(df_analysis['confidence_score'], bins=bins, labels=labels, right=False)

    results_table = []
    for label_val in labels:
        group_df = df_analysis[df_analysis['conf_group'] == label_val]
        if group_df.empty:
            results_table.append((label_val, np.nan, 0, 0))
            continue

        correct_preds = (group_df['true_label_idx'] == group_df['pred_label_idx']).sum()
        total_in_group = len(group_df)
        acc = correct_preds / total_in_group if total_in_group > 0 else 0.0
        results_table.append((label_val, acc, total_in_group, correct_preds))

    print(f"{'Confidence Band':<18} | {'Accuracy':<9} | {'Samples':<9} | {'Correct':<9}")
    print("-" * 55)
    for label_print, acc_print, n_print, correct_print in results_table:
        acc_str = f"{acc_print:.4f}" if pd.notna(acc_print) else "N/A"
        print(f"{label_print:<18} | {acc_str:<9} | {n_print:<9} | {correct_print:<9}")
    print("-" * 55)


def run_backtest_on_data(
        df_features_full,
        model_clf,
        feature_cols,
        target_col='target_class',
        symbol_col='symbol',
        timestamp_col='timestamp',
        target_class_names=None,
        model_tf_identifier=""  # For logging/output naming
):
    """
    Runs backtest on provided data and model.

    Args:
        df_features_full (pd.DataFrame): DataFrame containing all features and target.
        model_clf: Trained classification model.
        feature_cols (list): List of feature column names to use.
        target_col (str): Name of the target column.
        symbol_col (str): Name of the symbol column.
        timestamp_col (str): Name of the timestamp column.
        target_class_names (list, optional): Names of the target classes for reports.
                                            Defaults to TARGET_CLASS_NAMES_DEFAULT.
        model_tf_identifier (str): Identifier string (e.g. timeframe or model name) for output files.

    Returns:
        pd.DataFrame: DataFrame with backtest results (timestamps, true labels, pred labels, confidence).
                      Returns empty DataFrame on failure.
    """
    if target_class_names is None:
        target_class_names = TARGET_CLASS_NAMES_DEFAULT

    logger.info(f"Starting backtest run for: {model_tf_identifier}")

    if df_features_full.empty:
        logger.error(f"Input DataFrame is empty for {model_tf_identifier}. Backtest aborted.")
        return pd.DataFrame()
    if not model_clf:
        logger.error(f"Model is not provided for {model_tf_identifier}. Backtest aborted.")
        return pd.DataFrame()
    if not feature_cols:
        logger.error(f"Feature columns list is empty for {model_tf_identifier}. Backtest aborted.")
        return pd.DataFrame()

    required_cols_in_df = feature_cols + [target_col, symbol_col, timestamp_col]
    missing_df_cols = [col for col in required_cols_in_df if col not in df_features_full.columns]
    if missing_df_cols:
        logger.error(f"Missing required columns in DataFrame for {model_tf_identifier}: {missing_df_cols}. Backtest aborted.")
        return pd.DataFrame()

    # Drop rows with NaNs in features or target
    df_cleaned = df_features_full.dropna(subset=feature_cols + [target_col]).copy()
    if df_cleaned.empty:
        logger.warning(f"No data remains after NaN cleaning for {model_tf_identifier}. Backtest aborted.")
        return pd.DataFrame()

    X_test_data = df_cleaned[feature_cols]
    y_true_labels = df_cleaned[target_col]  # These are string labels like 'UP', 'DOWN'

    # Map string labels to integer indices (0, 1, ...)
    # This mapping should align with how the model was trained (e.g., 'DOWN':0, 'UP':1)
    # For predict_backtest.py, the target_class_names was ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']
    # Here, we adapt to a simpler binary or a configurable list.

    # Create a mapping from the provided target_class_names to indices
    class_to_idx = {name: i for i, name in enumerate(target_class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    # Filter y_true_labels to only include those present in target_class_names
    # and map them to indices
    y_true_named_filtered = y_true_labels[y_true_labels.isin(target_class_names)]
    if y_true_named_filtered.empty:
        logger.error(f"No true labels remain after filtering by `target_class_names` for {model_tf_identifier}. Backtest aborted.")
        return pd.DataFrame()

    y_true_indices = y_true_named_filtered.map(class_to_idx)

    # Align X_test_data with the filtered y_true_indices
    X_test_data_aligned = X_test_data.loc[y_true_indices.index]
    if X_test_data_aligned.empty:
        logger.error(f"X_test_data is empty after aligning with filtered true labels for {model_tf_identifier}. Backtest aborted.")
        return pd.DataFrame()

    try:
        probas_all_classes = model_clf.predict_proba(X_test_data_aligned)
        y_pred_indices = probas_all_classes.argmax(axis=1)  # Predicted class indices
    except Exception as e:
        logger.error(f"Error during model prediction for {model_tf_identifier}: {e}")
        return pd.DataFrame()

    # Calculate confidence score: P(best_class) - P(second_best_class)
    # If only one class probability is returned (e.g. by some binary classifiers), confidence is that prob.
    confidence_scores = np.zeros(len(y_pred_indices))
    if probas_all_classes.shape[1] >= 2:
        sorted_probas = -np.sort(-probas_all_classes, axis=1)  # Sort probabilities in descending order
        confidence_scores = sorted_probas[:, 0] - sorted_probas[:, 1]
    elif probas_all_classes.shape[1] == 1:
        confidence_scores = probas_all_classes[:, 0]
    else:  # Should not happen with standard classifiers
        logger.warning(f"Unexpected shape for probabilities: {probas_all_classes.shape} for {model_tf_identifier}. Confidence set to 0.")

    df_results = pd.DataFrame({
        'timestamp': df_cleaned.loc[y_true_indices.index, timestamp_col],
        'symbol': df_cleaned.loc[y_true_indices.index, symbol_col],
        'true_label_idx': y_true_indices,
        'pred_label_idx': y_pred_indices,
        'confidence_score': confidence_scores
    })

    df_results['true_label_text'] = df_results['true_label_idx'].map(idx_to_class)
    df_results['pred_label_text'] = df_results['pred_label_idx'].map(idx_to_class)

    # --- Metrics Calculation ---
    logger.info(f"\n--- Classification Report ({model_tf_identifier}) ---")
    # Ensure labels for report align with the number of unique classes predicted/true
    # and that target_names matches these labels.
    report_labels = sorted(list(set(y_true_indices.unique()) | set(y_pred_indices)))
    report_target_names = [idx_to_class.get(l, f"Class_{l}") for l in report_labels]

    try:
        class_report_str = classification_report(
            y_true_indices, y_pred_indices,
            labels=report_labels,
            target_names=report_target_names,
            zero_division=0,
            digits=4
        )
        print(class_report_str)
    except Exception as e:
        logger.error(f"Error generating classification report for {model_tf_identifier}: {e}")
        print("Could not generate classification report.")

    accuracy = accuracy_score(y_true_indices, y_pred_indices)
    logger.info(f"Overall Accuracy ({model_tf_identifier}): {accuracy:.4f}")

    # Confusion Matrix (using the original target_class_names for consistency if possible)
    cm_labels_for_plot = list(range(len(target_class_names)))
    cm = confusion_matrix(y_true_indices, y_pred_indices, labels=cm_labels_for_plot)

    return df_results, cm  # Return results and confusion matrix data


def main_backtest_logic(
        timeframe,
        features_path_template=DEFAULT_FEATURES_PATH_TEMPLATE,
        model_path_template=DEFAULT_MODEL_PATH_TEMPLATE,
        model_features_list_template=DEFAULT_MODEL_FEATURES_LIST_TEMPLATE,
        output_dir=DEFAULT_OUTPUT_DIR_BACKTEST,
        target_class_names_list=None,  # Allow overriding class names
        model_suffix=""  # e.g., "_top8" if loading a group model
):
    """
    Main orchestrator for running a backtest for a given timeframe.
    Loads data and model, runs backtest, saves results and plots.

    Args:
        timeframe (str): The timeframe to backtest (e.g., '15m').
        features_path_template (str): Template for features file path.
        model_path_template (str): Template for model file path.
        model_features_list_template (str): Template for feature list file path.
        output_dir (str): Directory to save outputs.
        target_class_names_list (list, optional): List of target class names for reports.
                                                  If None, uses TARGET_CLASS_NAMES_DEFAULT.
        model_suffix (str): Optional suffix for model and feature list files (e.g., "_top8", "_BTCUSDT").
    """
    if target_class_names_list is None:
        target_class_names_list = TARGET_CLASS_NAMES_DEFAULT

    logger.info(f"🚀 Starting backtest for timeframe: {timeframe}, model suffix: '{model_suffix}'")
    os.makedirs(output_dir, exist_ok=True)

    # Construct paths with suffix
    # Example: features_path = data/features_top8_15m.pkl if model_suffix is _top8
    # If model_suffix is empty, it becomes data/features_15m.pkl
    # The original predict_backtest.py used features_{tf}.pkl and models/{tf}_features.txt
    # This logic needs to be flexible for symbol-specific, group-specific, or generic models.

    # For predict_backtest.py, it seemed to use generic features and models per TF.
    # Let's assume model_suffix is primarily for the model name and its feature list,
    # while features_path might be more generic or also suffixed.
    # Sticking to predict_backtest.py's original logic for paths if model_suffix is empty:

    # If a suffix is provided (e.g. "_top8"), it implies a specific model and its feature list.
    # The features data itself might be from a suffixed file (e.g. features_top8_15m.pkl)
    # or a generic one (features_15m.pkl) that is then filtered.
    # For simplicity here, let's assume the features_path also gets the suffix if provided.

    # Path construction needs to be robust.
    # If suffix is like "top8", it becomes "top8_15m"
    # If suffix is like "BTCUSDT", it becomes "BTCUSDT_15m"
    # If suffix is empty, it becomes "15m"

    file_identifier_base = f"{model_suffix.replace('_', '')}_{timeframe}" if model_suffix else timeframe

    # Path for features data. Original predict_backtest.py used 'data/features_{tf}.pkl'
    # If a suffix is given, we might expect 'data/features_{suffix}_{tf}.pkl'
    actual_features_path = features_path_template.format(tf=file_identifier_base if model_suffix else timeframe)
    if model_suffix and not os.path.exists(actual_features_path):
        # Fallback for features if suffixed version not found, try generic
        generic_features_path = features_path_template.format(tf=timeframe)
        if os.path.exists(generic_features_path):
            logger.info(f"Suffixed features file {actual_features_path} not found. Using generic: {generic_features_path}")
            actual_features_path = generic_features_path
        else:
            logger.error(f"Neither suffixed ({actual_features_path}) nor generic features ({generic_features_path}) found.")
            return

    # Path for the model. Original: 'models/{tf}_clf_class.pkl'
    # With suffix: 'models/{suffix}_{tf}_clf_class.pkl'
    model_name_part = f"{model_suffix.replace('_', '')}_{timeframe}" if model_suffix else timeframe
    actual_model_path = model_path_template.format(tf=model_name_part, model_type='clf_class')

    # Path for the list of features the model was trained on. Original: 'models/{tf}_features.txt'
    # With suffix: 'models/{suffix}_{tf}_features_selected.txt' (assuming _selected from trainer)
    actual_feature_list_path = model_features_list_template.format(tf=model_name_part)
    if not os.path.exists(actual_feature_list_path) and model_suffix:
        # Fallback for feature list if suffixed one not found (e.g. model was trained on 'all' features)
        generic_feature_list_path = model_features_list_template.format(tf=timeframe)
        if os.path.exists(generic_feature_list_path):
            logger.info(f"Suffixed feature list {actual_feature_list_path} not found. Using generic: {generic_feature_list_path}")
            actual_feature_list_path = generic_feature_list_path

    df_all_features = _load_features_for_backtest(actual_features_path, timeframe)
    if df_all_features.empty:
        return

    model_clf_class = _load_model_for_backtest(actual_model_path)
    if not model_clf_class:
        return

    feature_columns = _load_feature_list(actual_feature_list_path, timeframe)
    if not feature_columns:
        return

    # The original predict_backtest.py used target_class names like 'STRONG DOWN', etc.
    # And 'target_class' as the target column.
    # For binary, we might use 'target_up' (0/1) or map 'target_class' ('DOWN'/'UP') to 0/1.
    # The run_backtest_on_data function expects string labels which it then maps.
    # Let's assume 'target_class' contains 'DOWN'/'UP' for binary, or more for multi-class.

    # Filter df_all_features if model_suffix corresponds to a specific symbol (e.g. "BTCUSDT")
    # and the features file was generic.
    # If model_suffix is a group like "top8", the features file might already be filtered,
    # or it might be generic and then we'd need to filter by symbols in that group.
    # For this script, predict_backtest.py did not have symbol/group filtering for data.
    # It processed all symbols in the features_{tf}.pkl file.

    df_backtest_results, cm_data = run_backtest_on_data(
        df_features_full=df_all_features,
        model_clf=model_clf_class,
        feature_cols=feature_columns,
        target_col='target_class',  # As in original predict_backtest.py
        target_class_names=target_class_names_list,
        model_tf_identifier=file_identifier_base  # For naming outputs
    )

    if not df_backtest_results.empty:
        output_csv_filename = f'backtest_results_{file_identifier_base}.csv'
        output_csv_path = os.path.join(output_dir, output_csv_filename)
        try:
            df_backtest_results.to_csv(output_csv_path, index=False)
            logger.info(f"Backtest results saved to: {output_csv_path}")
        except IOError as e:
            logger.error(f"Error saving backtest results CSV to {output_csv_path}: {e}")

        _plot_daily_accuracy(df_backtest_results, file_identifier_base, output_dir)
        _analyze_confidence_bands(df_backtest_results, file_identifier_base)

        # Plot confusion matrix (using cm_data returned from run_backtest_on_data)
        _plot_confusion_matrix(cm_data, classes=target_class_names_list, tf_suffix=file_identifier_base, output_dir=output_dir, title='Confusion Matrix')
        _plot_confusion_matrix(cm_data, classes=target_class_names_list, tf_suffix=file_identifier_base, output_dir=output_dir, normalize=True, title='Normalized Confusion Matrix')
    else:
        logger.warning(f"Backtest for {timeframe} (suffix: '{model_suffix}') yielded no results. Outputs not generated.")

    logger.info(f"✅ Backtest for timeframe: {timeframe} (suffix: '{model_suffix}') completed.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    parser = argparse.ArgumentParser(description="Backtest classification model performance.")
    parser.add_argument('--tf', type=str, required=True,
                        help="Timeframe for backtest (e.g., '15m').")
    # Add arguments for model suffix if needed, e.g., for symbol-specific or group models
    parser.add_argument('--model-suffix', type=str, default="",
                        help="Suffix for model files (e.g., 'top8', 'BTCUSDT'). If provided, paths will be adjusted.")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR_BACKTEST,
                        help=f"Directory to save logs and plots (default: {DEFAULT_OUTPUT_DIR_BACKTEST}).")

    # Allow specifying target class names if they differ from the default binary
    # Example: --target-classes "STRONG DOWN" DOWN NEUTRAL UP "STRONG UP"
    parser.add_argument('--target-classes', nargs='+', default=None,  # TARGET_CLASS_NAMES_DEFAULT,
                        help=f"Names of the target classes in order. Default for binary: {TARGET_CLASS_NAMES_DEFAULT}. For 5-class: 'STRONG DOWN' DOWN NEUTRAL UP 'STRONG UP'")

    args = parser.parse_args()

    # Determine target_class_names_list based on argument or default
    # The original predict_backtest.py used 5 classes for its TARGET_CLASS_NAMES
    if args.target_classes:
        selected_target_classes = args.target_classes
    else:
        # Default to 5 classes if not specified, to match original predict_backtest.py behavior
        selected_target_classes = ['STRONG DOWN', 'DOWN', 'NEUTRAL', 'UP', 'STRONG UP']
        logger.info(f"Using default 5-class target names: {selected_target_classes}")

    try:
        main_backtest_logic(
            timeframe=args.tf,
            output_dir=args.output_dir,
            model_suffix=args.model_suffix,  # Pass the suffix
            target_class_names_list=selected_target_classes
            # Potentially add args for features_path_template, model_path_template etc. if needed
        )
    except KeyboardInterrupt:
        logger.info(f"\n[Backtester] 🛑 Backtest for {args.tf} interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"[Backtester] 💥 Unexpected error during backtest for {args.tf}: {e}", exc_info=True)
        sys.exit(1)
