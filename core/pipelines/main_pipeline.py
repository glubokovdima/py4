# core/pipelines/main_pipeline.py
"""
Defines and executes multi-step pipelines for the crypto prediction project.
Combines data updates, feature preprocessing, model training, and prediction.
This module is based on the logic from the original pipeline.py.
"""
import subprocess  # Keep for _run_pipeline_step if any external scripts remain
import sys
import time
import argparse
import logging
import os

# --- Import functions from core modules ---
from ..helpers.utils import load_config, run_script  # run_script for external, print_header for pipeline steps
from ..data_ingestion.historical_data_loader import main_historical_load_logic
from ..data_ingestion.incremental_data_loader import main_incremental_load_logic
from ..features.preprocessor import main_preprocess_logic
from ..training.trainer import main_train_logic
from ..prediction.predictor import main_predict_all_logic

# from ..path.to.gpu_test_logic import main_gpu_test_logic # If gpu_test.py is refactored

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Configuration Constants (Defaults, can be overridden by config.yaml) ---
PYTHON_EXECUTABLE = sys.executable  # Use the same Python interpreter for sub-scripts
DEFAULT_PIPELINE_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
DEFAULT_PIPELINE_LOG_FILE = "logs/pipeline_execution.log"

# Global config dictionary, to be loaded
CONFIG = {}


def setup_pipeline_logging():
    """Sets up logging for pipeline operations."""
    global CONFIG
    if not CONFIG:
        CONFIG = load_config()
        if not CONFIG:
            logging.basicConfig(level=logging.INFO,
                                format='[%(asctime)s] %(levelname)s [Pipeline] — %(message)s',
                                handlers=[logging.StreamHandler(sys.stdout)])
            logger.error("Pipeline logging setup with defaults due to config load failure.")
            return

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    logs_dir_name = CONFIG.get('logs_dir', 'logs')
    log_file_name_rel = CONFIG.get('pipeline_log_file', DEFAULT_PIPELINE_LOG_FILE)
    log_file_name = log_file_name_rel.split('/')[-1] if '/' in log_file_name_rel else log_file_name_rel

    logs_dir_path = os.path.join(project_root, logs_dir_name)
    log_file_path = os.path.join(logs_dir_path, log_file_name)

    os.makedirs(logs_dir_path, exist_ok=True)

    # Configure the logger for this module ('core.pipelines.main_pipeline')
    # Avoid reconfiguring root logger if already done by CLI or another entry point.
    # logger.propagate = False # Stop messages from going to root logger to avoid double console output

    # Check if handlers are already configured for THIS logger to prevent duplication
    if not logger.handlers:
        log_level = logging.INFO
        logger.setLevel(log_level)
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s [Pipeline] — %(message)s')

        try:
            fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            print(f"Error setting up pipeline file logger for {log_file_path}: {e}")

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.info(f"Pipeline logging configured. Log file: {log_file_path}")
    else:
        logger.info(f"Pipeline logging already configured. Using existing handlers. Log file: {log_file_path}")


def _print_pipeline_header(title):
    """Prints a standardized header for pipeline steps to the console and logger."""
    from ..helpers.utils import print_header  # Local import to avoid circular if utils imports pipeline
    header_text = f"PIPELINE STEP: {title.upper()}"
    print_header(header_text, width=70)  # Print to console
    # logger.info(f"{'='*10} {header_text} {'='*10}") # Log distinct header


def _execute_step(description, func_to_call=None, func_args=None, func_kwargs=None, command_list=None):
    """
    Executes a pipeline step, either by calling a function directly or running a command.
    Handles timing, logging, and exits pipeline on failure.
    """
    _print_pipeline_header(description)  # Use the new header function for console
    logger.info(f"Starting pipeline step: {description}")  # Log the start

    start_time_step = time.time()
    return_code = -1  # Default error

    if func_to_call:
        if func_args is None: func_args = []
        if func_kwargs is None: func_kwargs = {}
        try:
            logger.info(f"    Calling function: {func_to_call.__name__}(args={func_args}, kwargs={func_kwargs})")
            # Assume direct functions return 0 on success, non-zero on failure or raise an exception
            ret_val = func_to_call(*func_args, **func_kwargs)
            return_code = 0 if ret_val is None or ret_val == 0 else int(ret_val)  # Standardize return
        except KeyboardInterrupt:  # Allow KeyboardInterrupt to propagate for pipeline exit
            logger.warning(f"Function call '{func_to_call.__name__}' interrupted by user.")
            raise  # Re-raise to be caught by main_pipeline_logic's try-except
        except Exception as e:
            logger.error(f"❌ Error during direct function call '{func_to_call.__name__}': {e}", exc_info=True)
            return_code = 1  # General error code
    elif command_list:
        # This part for running external scripts can be kept if some steps are still external
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        adjusted_command_list = list(command_list)
        if len(adjusted_command_list) > 1 and isinstance(adjusted_command_list[1], str) and \
                not os.path.isabs(adjusted_command_list[1]) and \
                not adjusted_command_list[1].startswith("core" + os.sep):
            script_at_root = os.path.join(project_root, adjusted_command_list[1])
            if os.path.exists(script_at_root):
                adjusted_command_list[1] = script_at_root
        return_code = run_script(adjusted_command_list, description)  # run_script from helpers
    else:
        logger.error(f"Pipeline step '{description}' misconfigured: no function or command.")
        sys.exit(1)  # Critical pipeline definition error

    duration_step = time.time() - start_time_step

    if return_code == 0:
        logger.info(f"Pipeline step '{description}' completed successfully (⏱  {duration_step:.1f}s).")
        print(f"[Pipeline] ✅  Step '{description}' — выполнено (⏱  {duration_step:.1f}s)")
    elif return_code == 130:
        logger.warning(f"Pipeline step '{description}' was interrupted by user (Ctrl+C signal) (⏱  {duration_step:.1f}s). Exiting pipeline.")
        print(f"[Pipeline] 🔶  Step '{description}' прерван пользователем. Пайплайн остановлен.")
        sys.exit(130)  # Propagate specific Ctrl+C code
    else:
        logger.error(f"Pipeline step '{description}' failed with code {return_code} (⏱  {duration_step:.1f}s). Exiting pipeline.")
        print(f"❌  Ошибка в шаге '{description}' (код {return_code}). Пайплайн остановлен.")
        sys.exit(1)  # Exit pipeline on any other error in a step
    return return_code  # Though sys.exit might have already occurred


def main_pipeline_logic(
        timeframes_to_process,
        do_training=False,
        skip_final_predict=False,
        use_full_historical_update=False,
        symbol_filter=None,
        symbol_group_filter=None
):
    """
    Main logic for executing a defined pipeline using direct function calls.
    """
    global CONFIG
    if not CONFIG:
        CONFIG = load_config()
        if not CONFIG:
            logger.critical("Pipeline: Configuration not loaded. Aborting.")
            return 1

            # Logging setup is expected to be done by the entry point (CLI or __main__)
    # If called directly, ensure logging is configured.
    # setup_pipeline_logging() # This might cause issues if called multiple times or from CLI

    logger.info(f"🚀 Starting pipeline for timeframes: {', '.join(timeframes_to_process)}")
    if use_full_historical_update:
        logger.info("Pipeline mode: Full historical data update.")
    else:
        logger.info("Pipeline mode: Incremental data update.")
    if do_training: logger.info("Pipeline mode: Model training ENABLED.")
    if skip_final_predict: logger.info("Pipeline mode: Final prediction SKIPPED.")

    filter_desc = "all symbols"
    if symbol_filter:
        filter_desc = f"symbol '{symbol_filter}'"
        logger.info(f"Pipeline filter: Symbol '{symbol_filter}'")
    elif symbol_group_filter:
        filter_desc = f"group '{symbol_group_filter}'"
        logger.info(f"Pipeline filter: Group '{symbol_group_filter}'")
    else:
        logger.info("Pipeline filter: Processing for all relevant symbols (no specific filter).")

    # --- Step 1: Data Update ---
    if use_full_historical_update:
        _execute_step(
            "📦  Полная загрузка исторических данных",
            func_to_call=main_historical_load_logic,
            func_kwargs={'run_all_core_symbols': True, 'timeframes_to_process': None}
        )
    else:
        _execute_step(
            f"📥  Инкрементальное обновление свечей для {', '.join(timeframes_to_process)}",
            func_to_call=main_incremental_load_logic,
            func_kwargs={'timeframes_to_process': timeframes_to_process}
        )

    # --- Steps 2 & 3: Feature Preprocessing and Model Training (per timeframe) ---
    for tf_item in timeframes_to_process:
        # Step 2: Preprocess Features
        step_desc_preprocess = f"⚙️  Построение признаков для [{tf_item}] ({filter_desc})"
        _execute_step(
            step_desc_preprocess,
            func_to_call=main_preprocess_logic,
            func_kwargs={
                'timeframe': tf_item,
                'symbol_filter': symbol_filter,
                'symbol_group_filter': symbol_group_filter
            }
        )

        # Step 3: Train Models (if enabled)
        if do_training:
            step_desc_train = f"🧠  Обучение моделей для [{tf_item}] ({filter_desc})"
            is_group = True if symbol_group_filter else False
            current_filter_for_train = symbol_group_filter if symbol_group_filter else symbol_filter
            _execute_step(
                step_desc_train,
                func_to_call=main_train_logic,
                func_kwargs={
                    'timeframe_to_train': tf_item,
                    'symbol_or_group_filter': current_filter_for_train,
                    'is_group_model': is_group
                }
            )

    # --- Step 4: Final Prediction (if not skipped) ---
    if not skip_final_predict:
        step_desc_predict = f"🔮  Финальный прогноз ({filter_desc}) с сохранением"
        _execute_step(
            step_desc_predict,
            func_to_call=main_predict_all_logic,
            func_kwargs={
                'save_output_flag': True,
                'symbol_filter': symbol_filter,
                'group_filter_key': symbol_group_filter  # main_predict_all_logic expects group_filter_key
            }
        )

    logger.info("🎉 Pipeline execution finished successfully.")
    print("[Pipeline] 🎉 Пайплайн завершен.")
    return 0  # Success


if __name__ == "__main__":
    # This block allows running the pipeline module directly.
    # Load config and setup logging here for standalone execution.
    CONFIG = load_config()
    if not CONFIG:
        # Fallback if config.yaml is missing, use basic logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s [PipelineStandalone] — %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])
        logger.error("CRITICAL: Pipeline (standalone) failed to load configuration. Using defaults where possible.")
    else:
        # If config loaded, setup logging as defined in the function
        # This ensures log file is created correctly based on config
        setup_pipeline_logging()

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
                        help="Filter processing by a symbol group key (e.g., top8).")

    args = parser.parse_args()

    if args.symbol and args.symbol_group:
        logger.error("Cannot specify both --symbol and --symbol-group. Please choose one. Exiting.")
        sys.exit(1)

    # Validate symbol_group if provided
    if args.symbol_group:
        all_groups_cfg = CONFIG.get('symbol_groups', {})
        if args.symbol_group.lower() not in all_groups_cfg:
            logger.error(f"Unknown symbol group for pipeline: '{args.symbol_group}'. Available: {list(all_groups_cfg.keys())}. Exiting.")
            sys.exit(1)

    exit_code = 1  # Default to error
    try:
        exit_code = main_pipeline_logic(
            timeframes_to_process=args.tf,
            do_training=args.train,
            skip_final_predict=args.skip_predict,
            use_full_historical_update=args.full_update,
            symbol_filter=args.symbol,
            symbol_group_filter=args.symbol_group.lower() if args.symbol_group else None  # Pass lowercased group key
        )
        # sys.exit(exit_code) # main_pipeline_logic will call sys.exit on errors or KeyboardInterrupt
    except KeyboardInterrupt:  # Catch interrupt if it happens in main_pipeline_logic or here
        logger.warning("\n[Pipeline] 🛑 Pipeline execution interrupted by user (Ctrl+C).")
        sys.exit(130)
    except SystemExit as e:  # To catch sys.exit calls from _execute_step or main_pipeline_logic itself
        # Log based on code if not already logged by _execute_step
        if e.code != 0 and e.code != 130: logger.error(f"[Pipeline] Pipeline exited with code {e.code}.")
        sys.exit(e.code)  # Propagate the exit code
    except Exception as e:
        logger.critical(f"[Pipeline] 💥 Unexpected critical error in pipeline __main__: {e}", exc_info=True)
        sys.exit(1)

    # If main_pipeline_logic completed without sys.exit (i.e. success), exit with its code
    sys.exit(exit_code)
