# core/helpers/model_ops.py
"""
Helper functions for model operations, such as loading and saving
machine learning models, typically using joblib.
"""
import joblib
import os
import logging

logger = logging.getLogger(__name__)


# --- Configuration (could be from global config if needed) ---
# Example: default model directory, though paths are usually passed directly.
# DEFAULT_MODELS_DIR = "models" # Not strictly needed here as paths are arguments

def load_model_simple(model_path):
    """
    Loads a single model from a file using joblib.

    Args:
        model_path (str): The full path to the model file.

    Returns:
        object: The loaded model object, or None if an error occurs or file not found.
    """
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}", exc_info=True)
        return None


def save_model_joblib(model_object, file_path, compress=3):
    """
    Saves a model object to a file using joblib.

    Args:
        model_object: The model object to save.
        file_path (str): The full path where the model will be saved.
        compress (int): Compression level for joblib (0-9). Default is 3.

    Returns:
        bool: True if successful, False otherwise.
    """
    if model_object is None:
        logger.warning(f"Model object is None. Skipping save to: {file_path}")
        return False
    try:
        # Ensure the directory for the model file exists
        model_dir = os.path.dirname(file_path)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created model directory: {model_dir}")

        joblib.dump(model_object, file_path, compress=compress)
        logger.info(f"Model successfully saved to: {file_path} (compression: {compress})")
        return True
    except Exception as e:
        logger.error(f"Error saving model to {file_path}: {e}", exc_info=True)
        return False


def load_model_with_fallback(
        base_model_path_template,  # e.g., "models/{filter_suffix}_{tf}_{model_type}.pkl"
        filter_suffix,  # e.g., "BTCUSDT", "top8", or "all" (for generic)
        timeframe,
        model_type,
        group_models_config=None,  # Dict like {"top8": ["BTCUSDT", ...], ...}
        generic_fallback_suffix="all"  # Suffix for the most generic model (e.g., features_all_15m.pkl -> model all_15m_...)
):
    """
    Attempts to load a model with a fallback strategy:
    1. Specific model (e.g., symbol-specific or group-specific if filter_suffix is a group).
    2. If filter_suffix was a symbol and it belongs to a group, try the group model.
    3. Generic model (using generic_fallback_suffix).

    Args:
        base_model_path_template (str): Template for model paths.
            Placeholders: {filter_suffix}, {tf}, {model_type}.
        filter_suffix (str): The primary suffix for the model (e.g., a symbol 'BTCUSDT', a group 'top8', or 'all').
        timeframe (str): The timeframe (e.g., '15m').
        model_type (str): The type of model (e.g., 'clf_class', 'reg_delta').
        group_models_config (dict, optional): Configuration of symbol groups.
            Example: {'top8': ['BTCUSDT', 'ETHUSDT'], 'meme': ['DOGEUSDT']}
        generic_fallback_suffix (str, optional): Suffix for the most generic fallback model. Defaults to "all".

    Returns:
        object: The loaded model, or None if no model could be loaded.
    """
    attempt_paths = []

    # 1. Attempt to load the model directly using filter_suffix
    #    This covers:
    #    - Symbol-specific model if filter_suffix is a symbol (e.g., BTCUSDT_15m_clf.pkl)
    #    - Group-specific model if filter_suffix is a group name (e.g., top8_15m_clf.pkl)
    #    - Generic model if filter_suffix is "all" (e.g., all_15m_clf.pkl)
    specific_model_path = base_model_path_template.format(
        filter_suffix=filter_suffix, tf=timeframe, model_type=model_type
    )
    attempt_paths.append(f"(Specific: {filter_suffix}) {specific_model_path}")
    model = load_model_simple(specific_model_path)
    if model:
        logger.info(f"Loaded model using specific/group filter '{filter_suffix}': {specific_model_path}")
        return model

    # 2. If filter_suffix was a symbol, check if it belongs to any group and try loading group model
    if group_models_config and filter_suffix not in group_models_config.keys():  # i.e. filter_suffix is likely a symbol
        for group_name, symbols_in_group in group_models_config.items():
            if filter_suffix.upper() in [s.upper() for s in symbols_in_group]:  # Symbol found in a group
                group_model_path = base_model_path_template.format(
                    filter_suffix=group_name, tf=timeframe, model_type=model_type
                )
                attempt_paths.append(f"(Group Fallback: {group_name} for symbol {filter_suffix}) {group_model_path}")
                model = load_model_simple(group_model_path)
                if model:
                    logger.info(f"Loaded group model '{group_name}' as fallback for symbol '{filter_suffix}': {group_model_path}")
                    return model
                break  # Symbol found in one group, no need to check other groups

    # 3. Attempt to load the generic fallback model (e.g., using "all" or a configured generic suffix)
    #    This is useful if neither specific nor group model was found, or if filter_suffix was already "all".
    #    Avoid re-trying "all" if filter_suffix was already "all" and failed in step 1.
    if filter_suffix.lower() != generic_fallback_suffix.lower():
        generic_model_path = base_model_path_template.format(
            filter_suffix=generic_fallback_suffix, tf=timeframe, model_type=model_type
        )
        attempt_paths.append(f"(Generic Fallback: {generic_fallback_suffix}) {generic_model_path}")
        model = load_model_simple(generic_model_path)
        if model:
            logger.info(f"Loaded generic fallback model '{generic_fallback_suffix}': {generic_model_path}")
            return model

    logger.warning(
        f"Model type '{model_type}' for TF '{timeframe}' with filter_suffix '{filter_suffix}' "
        f"not found after trying paths: {'; '.join(attempt_paths)}"
    )
    return None


# Example usage (if run directly for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    # Create dummy data for testing
    test_model_dir = "temp_test_model_ops"
    os.makedirs(test_model_dir, exist_ok=True)

    # Dummy model object (e.g., a simple dictionary or a scikit-learn model)
    from sklearn.linear_model import LogisticRegression

    dummy_model_obj = LogisticRegression()  # Just an example object

    # --- Test save_model_joblib and load_model_simple ---
    simple_model_path = os.path.join(test_model_dir, "simple_test_model.pkl")
    print(f"\n--- Testing Simple Save/Load (Path: {simple_model_path}) ---")
    if save_model_joblib(dummy_model_obj, simple_model_path):
        loaded_model = load_model_simple(simple_model_path)
        if loaded_model:
            print(f"Simple model loaded successfully. Type: {type(loaded_model)}")
            assert isinstance(loaded_model, LogisticRegression)
        else:
            print("Failed to load simple model.")
    else:
        print("Failed to save simple model.")

    # --- Test load_model_with_fallback ---
    print("\n--- Testing Fallback Logic ---")
    # Create some dummy model files to simulate fallback scenarios
    # Model paths will be like: temp_test_model_ops/{filter_suffix}_{tf}_{model_type}.pkl
    base_template = os.path.join(test_model_dir, "{filter_suffix}_{tf}_{model_type}.pkl")

    # Dummy group config
    test_groups = {
        "top_crypto": ["BTCUSDT", "ETHUSDT"],
        "altcoins": ["ADAUSDT", "SOLUSDT"]
    }

    # Scenario 1: Symbol-specific model exists
    save_model_joblib({"id": "BTC_model"}, base_template.format(filter_suffix="BTCUSDT", tf="1h", model_type="clf"))
    model_btc = load_model_with_fallback(base_template, "BTCUSDT", "1h", "clf", test_groups, "general")
    assert model_btc and model_btc["id"] == "BTC_model", "Fallback Scenario 1 Failed"
    print("Scenario 1 (Symbol Specific): Passed")

    # Scenario 2: Symbol-specific does NOT exist, but group model exists
    # (Ensure BTCUSDT_1h_reg does not exist for this test)
    save_model_joblib({"id": "TOP_CRYPTO_model_reg"}, base_template.format(filter_suffix="top_crypto", tf="1h", model_type="reg"))
    model_btc_group_fallback = load_model_with_fallback(base_template, "BTCUSDT", "1h", "reg", test_groups, "general")
    assert model_btc_group_fallback and model_btc_group_fallback["id"] == "TOP_CRYPTO_model_reg", "Fallback Scenario 2 Failed"
    print("Scenario 2 (Group Fallback for Symbol): Passed")

    # Scenario 3: Neither symbol nor group model exists, general model exists
    # (Ensure ADAUSDT_1h_clf and altcoins_1h_clf do not exist)
    save_model_joblib({"id": "GENERAL_model_clf"}, base_template.format(filter_suffix="general", tf="1h", model_type="clf"))
    model_ada_general_fallback = load_model_with_fallback(base_template, "ADAUSDT", "1h", "clf", test_groups, "general")
    assert model_ada_general_fallback and model_ada_general_fallback["id"] == "GENERAL_model_clf", "Fallback Scenario 3 Failed"
    print("Scenario 3 (General Fallback for Symbol): Passed")

    # Scenario 4: filter_suffix is a group, and group model exists
    save_model_joblib({"id": "ALTCOINS_model_vol"}, base_template.format(filter_suffix="altcoins", tf="4h", model_type="vol"))
    model_alt_group = load_model_with_fallback(base_template, "altcoins", "4h", "vol", test_groups, "general")
    assert model_alt_group and model_alt_group["id"] == "ALTCOINS_model_vol", "Fallback Scenario 4 Failed"
    print("Scenario 4 (Group Specific): Passed")

    # Scenario 5: filter_suffix is a group, group model does NOT exist, general model exists
    # (Ensure top_crypto_4h_vol does not exist)
    save_model_joblib({"id": "GENERAL_model_vol_4h"}, base_template.format(filter_suffix="general", tf="4h", model_type="vol"))
    model_top_general_fallback = load_model_with_fallback(base_template, "top_crypto", "4h", "vol", test_groups, "general")
    assert model_top_general_fallback and model_top_general_fallback["id"] == "GENERAL_model_vol_4h", "Fallback Scenario 5 Failed"
    print("Scenario 5 (General Fallback for Group): Passed")

    # Scenario 6: No models exist at all
    model_none = load_model_with_fallback(base_template, "XYZUSDT", "1d", "test", test_groups, "general")
    assert model_none is None, "Fallback Scenario 6 Failed"
    print("Scenario 6 (No Model Found): Passed")

    # Scenario 7: filter_suffix is "general" (or whatever generic_fallback_suffix is)
    # (Uses general_1h_clf from Scenario 3)
    model_general_direct = load_model_with_fallback(base_template, "general", "1h", "clf", test_groups, "general")
    assert model_general_direct and model_general_direct["id"] == "GENERAL_model_clf", "Fallback Scenario 7 Failed"
    print("Scenario 7 (Direct Generic Load): Passed")

    # Clean up dummy directory and files
    try:
        import shutil

        shutil.rmtree(test_model_dir)
        print(f"\nCleaned up test directory: {test_model_dir}")
    except Exception as e:
        print(f"Error cleaning up test directory {test_model_dir}: {e}")
