# core/pipelines/main_pipeline.py
"""
Defines and executes multi-step pipelines for the crypto prediction project.
Combines data updates, feature preprocessing, model training, and prediction.
This module is based on the logic from the original pipeline.py.
"""
import subprocess
import sys
import time
import argparse
import logging
import os

from ..helpers.utils import load_config, run_script  # Assuming run_script is still used for external scripts

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Configuration Constants (Defaults, can be overridden by config.yaml) ---
PYTHON_EXECUTABLE = sys.executable  # Use the same Python interpreter for sub-scripts
DEFAULT_PIPELINE_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
DEFAULT_PIPELINE_LOG_FILE = "logs/pipeline_execution.log"  # Changed from pipeline.log to avoid conflict with module name

# Global config dictionary, to be loaded
CONFIG = {}


def setup_pipeline_logging():
    """Sets up logging for pipeline operations."""
    global CONFIG
    if not CONFIG:  # Ensure config is loaded
        CONFIG = load_config()
        if not CONFIG:
            # Fallback basic logging if config fails
            logging.basicConfig(level=logging.INFO,
                                format='[%(asctime)s] %(levelname)s [Pipeline] — %(message)s',
                                handlers=[logging.StreamHandler(sys.stdout)])
            logger.error("Pipeline logging setup with defaults due to config load failure.")
            return

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    logs_dir_name = CONFIG.get('logs_dir', 'logs')  # Default to 'logs'
    log_file_name = CONFIG.get('pipeline_log_file', DEFAULT_PIPELINE_LOG_FILE).split('/')[-1]

    logs_dir_path = os.path.join(project_root, logs_dir_name)
    log_file_path = os.path.join(logs_dir_path, log_file_name)

    os.makedirs(logs_dir_path, exist_ok=True)

    # Clear existing handlers for this logger to avoid duplicate logs if re-run
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Also, prevent propagation to root if root has console handlers to avoid double console output
    # logger.propagate = False
    # Or, more robustly, check if root already has a streamhandler
    # root_has_console = any(isinstance(h, logging.StreamHandler) for h in logging.getLogger().handlers)
    # if root_has_console:
    #     logger.propagate = False

    log_level = logging.INFO  # Or from config
    logger.setLevel(log_level)
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s [Pipeline] — %(message)s')

    # File Handler
    try:
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')  # Append mode
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        print(f"Error setting up pipeline file logger for {log_file_path}: {e}")

    # Console Handler (always add for pipeline visibility)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info(f"Pipeline logging configured. Log file: {log_file_path}")


def _run_pipeline_step(description, command_list):
    """
    Wrapper for run_script specific to pipeline steps.
    Logs timing and handles exit conditions for the pipeline.
    """
    logger.info(f"Starting pipeline step: {description}")
    print(f"\n[Pipeline] 🔧  {description}...")  # Console output for user

    start_time_step = time.time()

    # run_script from helpers.utils already handles logging of command and success/failure
    # It also prints to console.
    # We assume external scripts are in the project root (one level above 'core')
    # If these scripts are moved into 'core', paths in command_list need adjustment
    # or direct function calls should be used.
    # Example: if historical_data_loader.py is now core.data_ingestion.historical_data_loader
    # command_list would be [PYTHON_EXECUTABLE, "core/data_ingestion/historical_data_loader.py", ...]

    # For now, assuming scripts are at project root as in original pipeline.py
    # e.g., "old_update_binance_data.py" implies it's in the same CWD or on PATH
    # To be robust, paths to scripts should be relative to project root.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Adjust script paths in command_list to be relative to project root
    # The second element is usually the script name.
    adjusted_command_list = list(command_list)
    if len(adjusted_command_list) > 1 and isinstance(adjusted_command_list[1], str) and \
            not os.path.isabs(adjusted_command_list[1]) and \
            not adjusted_command_list[1].startswith("core" + os.sep):  # Avoid double-prefixing if already specified

        # Check if the script exists at project root
        script_at_root = os.path.join(project_root, adjusted_command_list[1])
        if os.path.exists(script_at_root):
            adjusted_command_list[1] = script_at_root
        else:
            # If not at root, assume it's a Python module path that Python can find,
            # or it's a command on PATH.
            logger.debug(f"Script '{adjusted_command_list[1]}' not found at project root. Assuming it's a module or on PATH.")

    return_code = run_script(adjusted_command_list, description)  # run_script is from helpers
    duration_step = time.time() - start_time_step

    if return_code == 0:
        logger.info(f"Pipeline step '{description}' completed successfully (⏱  {duration_step:.1f}s).")
        print(f"[Pipeline] ✅  Step '{description}' — выполнено (⏱  {duration_step:.1f}s)")
    elif return_code == 130:  # Ctrl+C in child
        logger.warning(f"Pipeline step '{description}' interrupted by user (⏱  {duration_step:.1f}s). Exiting pipeline.")
        print(f"[Pipeline] 🔶  Step '{description}' прерван пользователем. Пайплайн остановлен.")
        sys.exit(130)
    else:  # Other errors
        logger.error(f"Pipeline step '{description}' failed with code {return_code} (⏱  {duration_step:.1f}s). Exiting pipeline.")
        print(f"[Pipeline] ❌  Ошибка в шаге '{description}' (код {return_code}). Пайплайн остановлен.")
        sys.exit(1)  # Exit pipeline on any error in a step


def main_pipeline_logic(
        timeframes_to_process,
        do_training=False,
        skip_final_predict=False,
        use_full_historical_update=False,
        # Add symbol/group filters if pipeline steps support them
        symbol_filter=None,
        symbol_group_filter=None
):
    """
    Main logic for executing a defined pipeline.

    Args:
        timeframes_to_process (list): List of timeframe strings to process.
        do_training (bool): If True, include model training step.
        skip_final_predict (bool): If True, skip the final prediction generation step.
        use_full_historical_update (bool): If True, run full historical data update,
                                           otherwise run incremental update.
        symbol_filter (str, optional): Symbol to filter processing by.
        symbol_group_filter (str, optional): Symbol group to filter processing by.
    """
    global CONFIG
    if not CONFIG:  # Ensure config is loaded if called directly
        CONFIG = load_config()
        if not CONFIG:
            logger.critical("Pipeline: Configuration not loaded. Aborting.")
            return 1  # Error

    # setup_pipeline_logging() # Logging is now set up once in main_cli or if __name__ == "__main__"

    logger.info(f"🚀 Starting pipeline for timeframes: {', '.join(timeframes_to_process)}")
    if use_full_historical_update:
        logger.info("Pipeline mode: Full historical data update selected.")
    if do_training:
        logger.info("Pipeline mode: Model training is ENABLED.")
    if skip_final_predict:
        logger.info("Pipeline mode: Final prediction step will be SKIPPED.")
    if symbol_filter:
        logger.info(f"Pipeline mode: Applying symbol filter: {symbol_filter}")
    if symbol_group_filter:
        logger.info(f"Pipeline mode: Applying symbol group filter: {symbol_group_filter}")

    # --- Step 1: Data Update ---
    if use_full_historical_update:
        # Assumes old_update_binance_data.py is at project root
        _run_pipeline_step(
            "📦  Полная загрузка исторических данных (old_update --all-tf-all-core)",
            [PYTHON_EXECUTABLE, "old_update_binance_data.py", "--all-tf-all-core"]  # Use the appropriate flag
        )
    else:
        # Assumes mini_update_binance_data.py is at project root
        _run_pipeline_step(
            f"📥  Инкрементальное обновление свечей для {', '.join(timeframes_to_process)}",
            [PYTHON_EXECUTABLE, "mini_update_binance_data.py", "--tf"] + timeframes_to_process
        )

    # --- Steps 2 & 3: Feature Preprocessing and Model Training (per timeframe) ---
    # Construct arguments for symbol/group filtering for preprocess and train steps
    filter_args_list = []
    if symbol_group_filter:
        filter_args_list = ["--symbol-group", symbol_group_filter]
    elif symbol_filter:
        filter_args_list = ["--symbol", symbol_filter]

    for tf_item in timeframes_to_process:
        # Step 2: Preprocess Features
        # Assumes preprocess_features.py is at project root
        _run_pipeline_step(
            f"⚙️  [{tf_item}] Построение признаков" + (f" для {symbol_group_filter or symbol_filter}" if filter_args_list else ""),
            [PYTHON_EXECUTABLE, "preprocess_features.py", "--tf", tf_item] + filter_args_list
        )

        # Step 3: Train Models (if enabled)
        if do_training:
            # Assumes train_model.py is at project root
            _run_pipeline_step(
                f"🧠  [{tf_item}] Обучение моделей" + (f" для {symbol_group_filter or symbol_filter}" if filter_args_list else ""),
                [PYTHON_EXECUTABLE, "train_model.py", "--tf", tf_item] + filter_args_list
            )

    # --- Step 4: Final Prediction (if not skipped) ---
    if not skip_final_predict:
        # Assumes predict_all.py is at project root
        # predict_all.py also takes --symbol or --symbol-group
        predict_all_args = ["--save"] + filter_args_list  # Pass along filters
        _run_pipeline_step(
            "🔮  Финальный прогноз (predict_all --save)" + (f" для {symbol_group_filter or symbol_filter}" if filter_args_list else ""),
            [PYTHON_EXECUTABLE, "predict_all.py"] + predict_all_args
        )

    logger.info("🎉 Pipeline execution finished successfully.")
    print("[Pipeline] 🎉 Пайплайн завершен.")
    return 0  # Success


if __name__ == "__main__":
    # This block allows running the pipeline module directly.
    # Load config and setup logging here for standalone execution.
    CONFIG = load_config()
    if not CONFIG:
        sys.exit("CRITICAL: Pipeline failed to load configuration. Exiting.")

    setup_pipeline_logging()  # Setup logging for this standalone run

    default_tfs_from_config = CONFIG.get('pipeline_default_timeframes', DEFAULT_PIPELINE_TIMEFRAMES)

    parser = argparse.ArgumentParser(description="Execute a multi-step crypto prediction pipeline.")
    parser.add_argument('--tf', nargs='*', default=default_tfs_from_config,
                        help=f"Timeframes for processing (e.g., --tf 5m 15m). Default: {' '.join(default_tfs_from_config)}")
    parser.add_argument('--train', action='store_true', help='Include model training step.')
    parser.add_argument('--skip-predict', action='store_true', help='Skip the final prediction generation step.')
    parser.add_argument('--full-update', action='store_true',
                        help='Run full historical data update instead of incremental.')
    parser.add_argument('--symbol', type=str, default=None,
                        help="Filter processing by a single symbol (e.g., BTCUSDT).")
    parser.add_argument('--symbol-group', type=str, default=None,
                        help="Filter processing by a symbol group (e.g., top8).")

    args = parser.parse_args()

    if args.symbol and args.symbol_group:
        logger.error("Cannot specify both --symbol and --symbol-group. Please choose one. Exiting.")
        sys.exit(1)

    try:
        exit_code = main_pipeline_logic(
            timeframes_to_process=args.tf,
            do_training=args.train,
            skip_final_predict=args.skip_predict,
            use_full_historical_update=args.full_update,
            symbol_filter=args.symbol,
            symbol_group_filter=args.symbol_group
        )
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n[Pipeline] 🛑 Pipeline execution interrupted by user (Ctrl+C).")
        sys.exit(130)
    except SystemExit as e:  # To catch sys.exit calls from _run_pipeline_step
        if e.code == 130:
            logger.warning("\n[Pipeline] 🛑 Pipeline interrupted due to Ctrl+C in a child process.")
        elif e.code == 1:
            logger.error("\n[Pipeline] 🛑 Pipeline interrupted due to an error in a child process.")
        sys.exit(e.code)  # Propagate the exit code
    except Exception as e:
        logger.critical(f"[Pipeline] 💥 Unexpected critical error in pipeline: {e}", exc_info=True)
        sys.exit(1)
