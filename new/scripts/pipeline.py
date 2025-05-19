# scripts/pipeline.py

import subprocess
import sys
import time
import argparse
import logging
import os

# Ensure the src directory is in the Python path if necessary (usually handled by running from project root)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import config and logging setup first
from src.utils.config import get_config
from src.utils.logging_setup import setup_logging

# --- Initial Setup ---
# Load configuration
config = get_config()
TIMEFRAMES_CONFIG = config['timeframes']
SYMBOL_GROUPS = config['symbol_groups'] # Needed for argument validation
PATHS_CONFIG = config['paths'] # Needed for logs path

# Configure logging for this script
# Note: In a pipeline via cli.py, logging might already be set up.
# Calling setup_logging again is generally safe if it uses dictConfig.
setup_logging()
logger = logging.getLogger(__name__) # Use logger specific to this module

# --- Constants from Config ---
DEFAULT_TIMEFRAMES = TIMEFRAMES_CONFIG['default']


# --- Utility Functions ---

def run_step(description, command_list):
    """
    Runs a command as a subprocess.

    Args:
        description (str): A human-readable description of the step.
        command_list (list): The command and its arguments as a list.

    Returns:
        int: The return code of the subprocess.
    """
    print(f"\n[Pipeline] 🔧  {description}...")
    logger.info(f"Старт шага: {description}")
    logger.info(f"Команда: {' '.join(command_list)}")
    start_time = time.time()

    try:
        # shell=False, command_list is passed directly
        # stdout=subprocess.PIPE, stderr=subprocess.PIPE will capture output
        # but we usually want to see the output of the subprocesses directly
        # in the console for tools like tqdm.
        # So, let's keep the default (None), which means stdout/stderr go to parent.
        result = subprocess.run(command_list, check=False) # check=False means we handle non-zero exit codes

        duration = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"Завершено: {description} (⏱  {duration:.1f}s)")
            print(f"[Pipeline] ✅  {description} — выполнено (⏱  {duration:.1f}s)")
        elif result.returncode == 130:  # Standard Unix exit code for Ctrl+C
            logger.warning(f"Шаг прерван пользователем: {description} (код {result.returncode}) (⏱  {duration:.1f}s)")
            print(f"[Pipeline] 🔶  Шаг прерван пользователем: {description} (код {result.returncode}) (⏱  {duration:.1f}s)")
        else:
            logger.error(f"Ошибка в шаге: {description} (код {result.returncode}) (⏱  {duration:.1f}s)")
            print(f"[Pipeline] ❌  Ошибка в шаге: {description} (код {result.returncode}) (⏱  {duration:.1f}s)")

        return result.returncode

    except FileNotFoundError:
        logger.error(f"Ошибка: Скрипт не найден. Убедитесь, что Python доступен и скрипт '{command_list[1]}' существует и находится в PATH или указан полный путь.")
        print(f"[Pipeline] ❌  Ошибка: Скрипт не найден ({command_list[1]})")
        return -1 # Custom error code for script not found
    except Exception as e:
        logger.error(f"Непредвиденная ошибка при выполнении '{description}': {e}", exc_info=True)
        print(f"[Pipeline] ❌  Непредвиденная ошибка при выполнении '{description}': {e}")
        return -2 # Custom error code for other exceptions


# --- Main Pipeline Logic ---

def main():
    parser = argparse.ArgumentParser(description="Пайплайн для обработки крипто-данных и прогнозирования.")
    parser.add_argument('--tf', nargs='*', default=DEFAULT_TIMEFRAMES,
                        help=f"Таймфреймы для обработки (например: --tf 5m 15m). По умолчанию: {' '.join(DEFAULT_TIMEFRAMES)}")
    parser.add_argument('--train', action='store_true', help='Включить обучение моделей (CatBoost) после построения признаков.')
    parser.add_argument('--skip-predict', action='store_true', help='Пропустить финальный прогноз после обучения.')

    # Control data update step
    update_group = parser.add_mutually_exclusive_group()
    update_group.add_argument('--full-update', action='store_true', help='Запустить полную историческую загрузку данных (scripts/update_data.py --full).')
    update_group.add_argument('--mini-update', action='store_true', help='Запустить инкрементальное обновление свежих данных (scripts/update_data.py --mini). (По умолчанию, если не указан --full-update)')

    # Filter by symbol or group (passed down to preprocess, train, predict)
    filter_group_pipeline = parser.add_mutually_exclusive_group()
    filter_group_pipeline.add_argument('--symbol', type=str, help="Обработать данные и модели только для отдельного символа.")
    filter_group_pipeline.add_argument('--symbol-group', type=str, help="Обработать данные и модели только для предопределенной группы символов.")


    args = parser.parse_args()

    # --- Determine parameters based on args ---
    timeframes_to_process = args.tf # Use argparse default
    # Validate specified timeframes
    allowed_timeframes = TIMEFRAMES_CONFIG['default']
    valid_tfs = [t for t in timeframes_to_process if t in allowed_timeframes]
    invalid_tfs = [t for t in timeframes_to_process if t not in allowed_timeframes]
    if invalid_tfs:
        logger.warning(f"Пайплайн игнорирует неверные таймфреймы: {', '.join(invalid_tfs)}. Допустимые: {', '.join(allowed_timeframes)}")
    if not valid_tfs:
        logger.error("Не указан ни один допустимый таймфрейм для пайплайна. Пайплайн прерван.")
        sys.exit(1)
    timeframes_to_process = valid_tfs # Use only valid timeframes

    # Determine symbol/group filter arguments to pass to subsequent scripts
    filter_args = []
    filter_description_suffix = ""
    if args.symbol:
        filter_args = ["--symbol", args.symbol.upper()] # Pass uppercase symbol
        filter_description_suffix = f" для символа {args.symbol.upper()}"
        # Note: update_data.py --symbol mode is for single symbol/tf.
        # The pipeline will still run update_data.py --full/--mini for CORE_SYMBOLS,
        # and the filter will be applied in preprocess/train/predict.
        logger.warning("В режиме пайплайна фильтр --symbol применяется только к шагам построения признаков, обучения и прогноза.")
    elif args.symbol_group:
        group_name = args.symbol_group.lower()
        if group_name not in SYMBOL_GROUPS and group_name != 'all': # Allow 'all' special key
            logger.error(f"Неизвестная группа символов '{args.symbol_group}'. Доступные группы: {list(SYMBOL_GROUPS.keys())} или 'all'. Пайплайн прерван.")
            sys.exit(1)
        filter_args = ["--symbol-group", group_name] # Pass lowercase group name
        filter_description_suffix = f" для группы {group_name}"
        logger.warning("В режиме пайплайна фильтр --symbol-group применяется только к шагам построения признаков, обучения и прогноза.")
    # If no filter args, filter_args remains empty, and subprocesses run without --symbol/--symbol-group


    print(f"[Pipeline] 🚀 Запуск пайплайна для таймфреймов: {', '.join(timeframes_to_process)}{filter_description_suffix}")
    if args.full_update:
        print("[Pipeline] Выбран режим полного обновления данных.")
    elif args.mini_update or (not args.full_update and not args.mini_update): # mini is default
         print("[Pipeline] Выбран режим инкрементального обновления данных (по умолчанию).")
    if args.train:
        print("[Pipeline] Обучение моделей включено.")
    if args.skip_predict:
        print("[Pipeline] Финальный прогноз будет пропущен.")


    # --- Step 1: Data Update ---
    update_script = os.path.join(os.path.dirname(__file__), "update_data.py")
    update_command = [sys.executable, update_script]

    if args.full_update:
        update_command.append("--full")
        # Note: --full updates CORE_SYMBOLS, --tf can filter which TFs to update
        update_command.extend(["--tf"] + timeframes_to_process) # Pass specified TFs to update script
        step_desc = f"📦  Полная загрузка исторических данных для {', '.join(timeframes_to_process)}"
    else: # Default is mini update
        update_command.append("--mini")
        # Note: --mini updates CORE_SYMBOLS, --tf can filter which TFs to update
        update_command.extend(["--tf"] + timeframes_to_process) # Pass specified TFs to update script
        step_desc = f"📥  Инкрементальное обновление свечей для {', '.join(timeframes_to_process)}"

    return_code = run_step(step_desc, update_command)
    if return_code != 0:
        # run_step already logs and prints error
        sys.exit(return_code)


    # --- Step 2: Preprocess Features ---
    preprocess_script = os.path.join(os.path.dirname(__file__), "preprocess.py")
    for tf_item in timeframes_to_process:
        preprocess_command = [sys.executable, preprocess_script, "--tf", tf_item]
        preprocess_command.extend(filter_args) # Pass symbol/group filter args

        step_desc = f"⚙️  [{tf_item}] Построение признаков{filter_description_suffix}"
        return_code = run_step(step_desc, preprocess_command)
        if return_code != 0:
            # run_step already logs and prints error. Exit pipeline.
            # If you want to continue processing other TFs despite an error,
            # you would move this sys.exit outside the TF loop and just log.
            # But typically, a failure to preprocess one TF means the pipeline should stop.
            sys.exit(return_code)


    # --- Step 3: Train Models (Conditional) ---
    if args.train:
        train_script = os.path.join(os.path.dirname(__file__), "train.py")
        # The train script expects a single key (--symbol or --symbol-group)
        # If filter_args is empty (no filter), we cannot call train.py directly.
        # Training models for 'all' requires a specific key, e.g., '--symbol-group all'.
        # If the pipeline is run *without* filter args, should it train 'all' models?
        # Or should it require a filter for training?
        # Let's make it require a filter for training for clarity.
        # If filter_args is empty, we skip training unless --symbol-group all is explicitly allowed.
        # The CLI parser already handles mutually exclusive symbols/groups.
        # We just need the *key* to pass to train.py.
        train_key = None
        if args.symbol:
            train_key = args.symbol.upper()
        elif args.symbol_group:
            train_key = args.symbol_group.lower()
            if train_key != 'all' and train_key not in SYMBOL_GROUPS:
                 # This validation should ideally happen earlier in argparse or when determining filter_args
                 # but double check here.
                 logger.error(f"Неизвестная группа символов для обучения: {args.symbol_group}. Пайплайн прерван.")
                 sys.exit(1)
        else:
             # This state should not be reached if --train implies a filter is needed
             # Let's add a check that if --train is used, either --symbol or --symbol-group must be provided.
             # Or, assume training 'all' is the default if --train but no filter.
             # Let's assume training 'all' is the default if --train is present but no filter is specified.
             train_key = 'all'
             filter_description_suffix = " (все символы)" # Update desc if training 'all' by default


        logger.info(f"Запуск обучения моделей для ключа '{train_key}' на ТФ: {', '.join(timeframes_to_process)}.")

        for tf_item in timeframes_to_process:
            train_command = [sys.executable, train_script, "--tf", tf_item]
            if args.symbol:
                 train_command.extend(["--symbol", train_key])
            elif args.symbol_group:
                 train_command.extend(["--symbol-group", train_key])
            else: # Default to training 'all' if --train is present but no filter
                 train_command.extend(["--symbol-group", 'all'])


            step_desc = f"🧠  [{tf_item}] Обучение моделей для ключа '{train_key}'"
            return_code = run_step(step_desc, train_command)
            if return_code != 0:
                sys.exit(return_code)
    else:
        logger.info("Шаг обучения моделей пропущен (--train не указан).")


    # --- Step 4: Generate Predictions (Conditional) ---
    if not args.skip_predict:
        predict_script = os.path.join(os.path.dirname(__file__), "predict.py")
        predict_command = [sys.executable, predict_script]
        predict_command.extend(["--tf"] + timeframes_to_process) # Pass specified TFs to predict script
        predict_command.extend(filter_args) # Pass symbol/group filter args
        predict_command.append("--save") # Always save predictions from pipeline? Or make it optional?
        # Let's make --save the default behavior in the pipeline for the predict step.

        step_desc = f"🔮  Финальный прогноз и сохранение{filter_description_suffix}"
        return_code = run_step(step_desc, predict_command)
        if return_code != 0:
            sys.exit(return_code)
    else:
        logger.info("Шаг генерации прогнозов пропущен (--skip-predict указан).")


    print("[Pipeline] 🎉 Пайплайн завершен.")


if __name__ == "__main__":
    # Ensure basic directories exist before running the pipeline
    os.makedirs(PATHS_CONFIG['data_dir'], exist_ok=True)
    os.makedirs(PATHS_CONFIG['logs_dir'], exist_ok=True)
    os.makedirs(PATHS_CONFIG['models_dir'], exist_ok=True)
    # database dir is handled by src.db.py init_db

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n[Pipeline] 🛑 Пайплайн прерван пользователем (Ctrl+C).")
        sys.exit(130)
    except SystemExit as e: # Catch sys.exit calls from run_step or validation errors
        if e.code == 130:
            logger.warning("\n[Pipeline] 🛑 Пайплайн прерван из-за Ctrl+C в дочернем процессе.")
        elif e.code != 0: # Any non-zero exit code indicates an error
            logger.error("\n[Pipeline] 🛑 Пайплайн прерван из-за ошибки в дочернем процессе или валидации.")
        sys.exit(e.code)
    except Exception as e:
        logger.error(f"[Pipeline] 💥 Непредвиденная ошибка в пайплайне: {e}", exc_info=True)
        print(f"\n[Pipeline] 💥 Непредвиденная ошибка в пайплайне: {e}")
        sys.exit(1) # Exit with a general error code