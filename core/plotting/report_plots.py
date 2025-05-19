# core/plotting/report_plots.py
"""
Contains functions for generating plots commonly used in model evaluation
reports and backtesting analysis, such as confusion matrices and
accuracy over time plots.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # For plot_daily_accuracy_custom
import os
import logging

logger = logging.getLogger(__name__)

# Default settings for plots (can be overridden or extended)
DEFAULT_CMAP = plt.cm.Blues
DEFAULT_FIGSIZE_CM = (8, 6)
DEFAULT_FIGSIZE_ACCURACY = (12, 6)


def plot_confusion_matrix_custom(
        cm_data,
        class_names,
        output_path,  # Full path including filename
        normalize=False,
        title_prefix='Confusion Matrix',
        cmap=DEFAULT_CMAP,
        figsize=DEFAULT_FIGSIZE_CM
):
    """
    Prints and plots the confusion matrix, saving it to a file.

    Args:
        cm_data (np.array): Confusion matrix data (e.g., from sklearn.metrics.confusion_matrix).
        class_names (list): List of class names for axis labels.
        output_path (str): Full path (including directory and filename) to save the plot.
        normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.
        title_prefix (str, optional): Prefix for the plot title. Defaults to 'Confusion Matrix'.
        cmap (matplotlib.colors.Colormap, optional): Colormap for the plot. Defaults to plt.cm.Blues.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (8, 6).

    Returns:
        bool: True if plotting and saving were successful, False otherwise.
    """
    if cm_data is None or not isinstance(cm_data, np.ndarray) or cm_data.ndim != 2:
        logger.error(f"Invalid cm_data provided for confusion matrix. Expected 2D numpy array. Got: {type(cm_data)}")
        return False
    if not class_names or len(class_names) != cm_data.shape[0]:
        logger.error(f"Number of class_names ({len(class_names) if class_names else 0}) "
                     f"does not match confusion matrix dimension ({cm_data.shape[0]}).")
        return False

    plot_title = title_prefix
    if normalize:
        plot_title = f"Normalized {title_prefix.lower()}"
        # Handle division by zero if a row sums to 0 (no true samples for that class)
        row_sums = cm_data.sum(axis=1)[:, np.newaxis]
        cm_plot_data = np.divide(cm_data.astype('float'), row_sums,
                                 out=np.zeros_like(cm_data, dtype=float),
                                 where=row_sums != 0)
    else:
        cm_plot_data = cm_data.astype(int)  # Ensure integer display if not normalizing

    try:
        plt.figure(figsize=figsize)
        plt.imshow(cm_plot_data, interpolation='nearest', cmap=cmap)
        plt.title(plot_title)
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45, ha="right")  # Improved label alignment
        plt.yticks(tick_marks, class_names)

        fmt = '.2f' if normalize else 'd'
        thresh = cm_plot_data.max() / 2.

        for i, j in np.ndindex(cm_plot_data.shape):  # Use np.ndindex for multi-dim arrays
            plt.text(j, i, format(cm_plot_data[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm_plot_data[i, j] > thresh else "black",
                     fontsize=10)  # Adjust fontsize as needed

        plt.tight_layout()  # Adjust layout to prevent labels from overlapping
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory for plot: {output_dir}")

        plt.savefig(output_path, dpi=150)  # Save with good resolution
        logger.info(f"Confusion matrix plot saved to: {output_path}")
        plt.close()  # Close the figure to free memory
        return True
    except Exception as e:
        logger.error(f"Error generating or saving confusion matrix plot to {output_path}: {e}", exc_info=True)
        plt.close()  # Ensure figure is closed even on error
        return False


def plot_daily_accuracy_custom(
        df_results_with_ts,  # DataFrame with 'timestamp', 'true_label_idx', 'pred_label_idx'
        output_path,  # Full path including filename
        tf_identifier_str="",  # String like timeframe or model name for title
        plot_ma_window=7,  # Moving average window, or None to disable
        figsize=DEFAULT_FIGSIZE_ACCURACY
):
    """
    Plots daily accuracy over time and saves it to a file.
    Optionally includes a moving average of accuracy.

    Args:
        df_results_with_ts (pd.DataFrame): DataFrame containing backtest results.
            Required columns: 'timestamp' (datetime-like),
                              'true_label_idx' (integer true labels),
                              'pred_label_idx' (integer predicted labels).
        output_path (str): Full path (including directory and filename) to save the plot.
        tf_identifier_str (str, optional): Identifier string (e.g., timeframe, model name)
                                           to include in the plot title. Defaults to "".
        plot_ma_window (int, optional): Window size for calculating moving average of accuracy.
                                        If None or < 2, MA plot is disabled. Defaults to 7.
        figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (12, 6).

    Returns:
        bool: True if plotting and saving were successful, False otherwise.
    """
    required_cols = ['timestamp', 'true_label_idx', 'pred_label_idx']
    if df_results_with_ts.empty or not all(col in df_results_with_ts.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_results_with_ts.columns]
        logger.warning(f"DataFrame for daily accuracy plot is empty or missing columns: {missing}. "
                       f"Plot for '{tf_identifier_str}' will not be generated.")
        return False

    df_plot = df_results_with_ts.copy()
    try:
        df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'])
        df_plot['date'] = df_plot['timestamp'].dt.date
    except Exception as e:
        logger.error(f"Error converting 'timestamp' to datetime for daily accuracy plot: {e}")
        return False

    daily_accuracy_data = []
    # Group by date and calculate accuracy for each day
    for date_val, group in df_plot.groupby('date'):
        if not group.empty:
            correct_predictions = (group['true_label_idx'] == group['pred_label_idx']).sum()
            total_samples = len(group)
            if total_samples > 0:
                acc = correct_predictions / total_samples
                daily_accuracy_data.append({'date': date_val, 'accuracy': acc, 'samples': total_samples})
            else:
                # This case should ideally not happen if groupby yields non-empty groups
                daily_accuracy_data.append({'date': date_val, 'accuracy': np.nan, 'samples': 0})

    if not daily_accuracy_data:
        logger.warning(f"No daily accuracy data points to plot for '{tf_identifier_str}' after grouping.")
        return False

    accuracy_by_day_df = pd.DataFrame(daily_accuracy_data).sort_values('date')

    if accuracy_by_day_df['accuracy'].isnull().all():
        logger.warning(f"All daily accuracy values are NaN for '{tf_identifier_str}'. Plotting skipped.")
        return False

    try:
        plt.figure(figsize=figsize)
        plt.plot(accuracy_by_day_df['date'], accuracy_by_day_df['accuracy'], marker='o', linestyle='-', markersize=4, label='Daily Accuracy')

        # Plot Moving Average if window is valid
        if plot_ma_window and isinstance(plot_ma_window, int) and plot_ma_window >= 2 and len(accuracy_by_day_df) >= plot_ma_window:
            accuracy_by_day_df['accuracy_ma'] = accuracy_by_day_df['accuracy'].rolling(
                window=plot_ma_window, min_periods=1  # min_periods=1 to show MA from start
            ).mean()
            plt.plot(accuracy_by_day_df['date'], accuracy_by_day_df['accuracy_ma'], linestyle='--',
                     label=f'{plot_ma_window}-day MA Accuracy')

        title = f'Daily Accuracy'
        if tf_identifier_str:
            title += f' — {tf_identifier_str}'
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)  # Lighter grid
        plt.tight_layout()

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created directory for plot: {output_dir}")

        plt.savefig(output_path, dpi=150)
        logger.info(f"Daily accuracy plot saved to: {output_path}")
        plt.close()
        return True
    except Exception as e:
        logger.error(f"Error generating or saving daily accuracy plot to {output_path}: {e}", exc_info=True)
        plt.close()
        return False


# Example usage (if run directly for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    test_plot_dir = "temp_test_plots"
    os.makedirs(test_plot_dir, exist_ok=True)

    # --- Test plot_confusion_matrix_custom ---
    print("\n--- Testing Confusion Matrix Plot ---")
    cm_example_data = np.array([[100, 10, 5], [8, 120, 7], [3, 6, 90]])
    class_names_example = ['Class A', 'Class B', 'Class C']
    cm_output_path = os.path.join(test_plot_dir, "test_cm_plot.png")
    cm_norm_output_path = os.path.join(test_plot_dir, "test_cm_norm_plot.png")

    success_cm = plot_confusion_matrix_custom(cm_example_data, class_names_example, cm_output_path, title_prefix="Test CM")
    print(f"CM Plot Generation Success: {success_cm}")

    success_cm_norm = plot_confusion_matrix_custom(cm_example_data, class_names_example, cm_norm_output_path, normalize=True, title_prefix="Test CM (Normalized)")
    print(f"Normalized CM Plot Generation Success: {success_cm_norm}")

    # --- Test plot_daily_accuracy_custom ---
    print("\n--- Testing Daily Accuracy Plot ---")
    # Create dummy DataFrame for accuracy plot
    dates = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03',
                            '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08'])
    dummy_results_df = pd.DataFrame({
        'timestamp': dates,
        'true_label_idx': [0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
        'pred_label_idx': [0, 0, 1, 0, 1, 1, 0, 1, 0, 0]  # Some correct, some incorrect
    })
    acc_output_path = os.path.join(test_plot_dir, "test_daily_accuracy.png")
    success_acc = plot_daily_accuracy_custom(dummy_results_df, acc_output_path, tf_identifier_str="TestTF", plot_ma_window=3)
    print(f"Daily Accuracy Plot Generation Success: {success_acc}")

    # Test with empty data
    print("\n--- Testing Daily Accuracy Plot (Empty Data) ---")
    success_acc_empty = plot_daily_accuracy_custom(pd.DataFrame(), os.path.join(test_plot_dir, "empty.png"))
    print(f"Daily Accuracy Plot (Empty Data) Success (should be False): {success_acc_empty}")

    print(f"\nCheck plots in directory: {test_plot_dir}")
    # To auto-remove test dir:
    # import shutil
    # shutil.rmtree(test_plot_dir)
    # print(f"Cleaned up test directory: {test_plot_dir}")
