# core/helpers/data_io.py
"""
Helper functions for data input/output operations, such as
loading and saving features, samples, and feature lists.
"""
import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)


# --- Pickle file operations ---

def load_features_pkl(file_path):
    """
    Loads features from a pickle file.

    Args:
        file_path (str): Path to the .pkl file.

    Returns:
        pd.DataFrame: Loaded DataFrame, or an empty DataFrame if an error occurs or file not found.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Pickle file not found: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_pickle(file_path)
        logger.info(f"Successfully loaded features from: {file_path} (Shape: {df.shape})")
        return df
    except Exception as e:
        logger.error(f"Error loading pickle file {file_path}: {e}", exc_info=True)
        return pd.DataFrame()


def save_features_pkl(df_features, file_path):
    """
    Saves a DataFrame to a pickle file.

    Args:
        df_features (pd.DataFrame): DataFrame to save.
        file_path (str): Path to save the .pkl file.

    Returns:
        bool: True if successful, False otherwise.
    """
    if df_features.empty:
        logger.warning(f"DataFrame is empty. Skipping save to pickle: {file_path}")
        return False
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
        df_features.to_pickle(file_path)
        logger.info(f"Features successfully saved to: {file_path} (Shape: {df_features.shape})")
        return True
    except Exception as e:
        logger.error(f"Error saving pickle file to {file_path}: {e}", exc_info=True)
        return False


# --- CSV file operations ---

def save_sample_csv(df_sample, file_path, max_rows=1000):
    """
    Saves a sample of a DataFrame to a CSV file.
    Handles datetime conversion to string for CSV compatibility if 'timestamp' column exists.

    Args:
        df_sample (pd.DataFrame): DataFrame to sample and save.
        file_path (str): Path to save the .csv file.
        max_rows (int): Maximum number of rows for the sample.

    Returns:
        bool: True if successful, False otherwise.
    """
    if df_sample.empty:
        logger.warning(f"DataFrame for sample is empty. Skipping save to CSV: {file_path}")
        return False

    sample_to_save = df_sample.head(max_rows).copy()  # Take a sample and work on a copy

    # Convert timestamp to string if it exists and is datetime type
    if 'timestamp' in sample_to_save.columns and \
            pd.api.types.is_datetime64_any_dtype(sample_to_save['timestamp']):
        try:
            sample_to_save['timestamp'] = sample_to_save['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        except AttributeError:  # Handle cases where .dt might not be available (e.g., already string or mixed types)
            logger.debug(f"Could not format 'timestamp' column in sample for {file_path}, possibly not datetime.")
            pass  # Continue without formatting

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
        sample_to_save.to_csv(file_path, index=False, float_format='%.6f')  # Consistent float format
        logger.info(f"Sample data successfully saved to: {file_path} ({len(sample_to_save)} rows)")
        return True
    except Exception as e:
        logger.error(f"Error saving sample CSV file to {file_path}: {e}", exc_info=True)
        return False


# --- Feature list file operations (text files) ---

def load_feature_list_from_txt(file_path):
    """
    Loads a list of feature names from a text file (one feature per line).

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        list: List of feature names, or None if an error occurs or file not found.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Feature list text file not found: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            feature_list = [line.strip() for line in f if line.strip()]  # Read and strip empty lines/whitespace
        if not feature_list:
            logger.warning(f"Feature list file '{file_path}' is empty.")
            return []  # Return empty list if file is empty but exists
        logger.info(f"Successfully loaded {len(feature_list)} features from: {file_path}")
        return feature_list
    except Exception as e:
        logger.error(f"Error loading feature list from text file {file_path}: {e}", exc_info=True)
        return None


def save_feature_list_to_txt(feature_list, file_path):
    """
    Saves a list of feature names to a text file (one feature per line).

    Args:
        feature_list (list): List of feature names to save.
        file_path (str): Path to save the .txt file.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not feature_list:  # Check if the list itself is empty or None
        logger.warning(f"Feature list is empty. Skipping save to text file: {file_path}")
        return False
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
        with open(file_path, "w", encoding="utf-8") as f:
            for feature_name in feature_list:
                f.write(f"{feature_name}\n")
        logger.info(f"Feature list successfully saved to: {file_path} ({len(feature_list)} features)")
        return True
    except Exception as e:
        logger.error(f"Error saving feature list to text file {file_path}: {e}", exc_info=True)
        return False


# Example usage (if run directly for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    # Create dummy data for testing
    test_data_dir = "temp_test_data_io"
    os.makedirs(test_data_dir, exist_ok=True)

    dummy_df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0],
        'feature2': ['a', 'b', 'c'],
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-01 10:05:00', '2023-01-01 10:10:00'])
    })
    dummy_features = ['feature1', 'feature2', 'timestamp_related_feature']

    # Test save and load pkl
    pkl_path = os.path.join(test_data_dir, "test_features.pkl")
    print(f"\n--- Testing Pickle Save/Load (Path: {pkl_path}) ---")
    if save_features_pkl(dummy_df, pkl_path):
        loaded_df_pkl = load_features_pkl(pkl_path)
        if not loaded_df_pkl.empty:
            print("Pickle loaded successfully. Content sample:")
            print(loaded_df_pkl.head())
        else:
            print("Failed to load pickle or it was empty.")
    else:
        print("Failed to save pickle.")

    # Test save sample csv
    csv_path = os.path.join(test_data_dir, "test_sample.csv")
    print(f"\n--- Testing CSV Sample Save (Path: {csv_path}) ---")
    if save_sample_csv(dummy_df, csv_path, max_rows=2):
        print(f"Sample CSV saved. Check {csv_path}")
        # You can manually check the CSV content.
        # df_csv_check = pd.read_csv(csv_path)
        # print("CSV content check:")
        # print(df_csv_check)
    else:
        print("Failed to save sample CSV.")

    # Test save and load feature list txt
    txt_path = os.path.join(test_data_dir, "test_feature_list.txt")
    print(f"\n--- Testing Feature List TXT Save/Load (Path: {txt_path}) ---")
    if save_feature_list_to_txt(dummy_features, txt_path):
        loaded_list_txt = load_feature_list_from_txt(txt_path)
        if loaded_list_txt is not None:
            print(f"Feature list loaded successfully: {loaded_list_txt}")
        else:
            print("Failed to load feature list from TXT.")
    else:
        print("Failed to save feature list to TXT.")

    # Test with empty feature list
    print("\n--- Testing Save Empty Feature List ---")
    empty_txt_path = os.path.join(test_data_dir, "empty_feature_list.txt")
    save_feature_list_to_txt([], empty_txt_path)  # Should log a warning and not create file or create empty
    loaded_empty_list = load_feature_list_from_txt(empty_txt_path)
    if loaded_empty_list is not None:  # load_feature_list_from_txt returns [] for empty file
        print(f"Loading an empty (but existing) feature list file resulted in: {loaded_empty_list} (expected empty list or None if not created)")

    # Clean up dummy directory and files
    try:
        import shutil

        shutil.rmtree(test_data_dir)
        print(f"\nCleaned up test directory: {test_data_dir}")
    except Exception as e:
        print(f"Error cleaning up test directory {test_data_dir}: {e}")
