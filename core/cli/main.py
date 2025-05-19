# core/cli/main.py
"""
Main Command-Line Interface (CLI) for the Crypto Prediction project.
Provides a menu-driven interface to run various scripts and pipelines.
"""
import subprocess
import sys
import os
import shutil
import argparse
import logging

from ..helpers.utils import (
    load_config,
    print_header,
    run_script, # Этот run_script теперь должен быть исправлен
    select_timeframes_interactive,
    clear_training_artifacts_interactive,
    ensure_base_directories
)

# Import functions from other refactored modules
# We'll call these functions directly instead of subprocesses for internal operations
from ..data_ingestion.historical_data_loader import main_historical_load_logic # Example
from ..features.preprocessor import main_preprocess_logic # Example
from ..training.trainer import main_train_logic # Example
from ..prediction.predictor import main_predict_all_logic # Example
from ..backtesting.backtester import main_backtest_logic # Example
from ..pipelines.main_pipeline import main_pipeline_logic # Example
from ..analysis.symbol_selector import main_symbol_selection_logic # Example

# Configure logging for this module (can be basic, or more advanced if needed)
logger = logging.getLogger(__name__)
# Example of basic setup if not configured by a root logger:
# logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s [CLI] - %(message)s', stream=sys.stdout)


# --- Global Configuration (Loaded from YAML) ---
CONFIG = {}  # Will be loaded by load_config()
PYTHON_EXECUTABLE = sys.executable  # Use the same Python interpreter

# Default directory paths (can be overridden by config.yaml)
DEFAULT_MODELS_DIR = "models"
DEFAULT_LOGS_DIR = "logs"
DEFAULT_DATA_DIR = "data"
DEFAULT_DATABASE_DIR = "database"


# --- Helper functions specific to CLI or not yet moved to helpers ---
# run_script, print_header, select_timeframes_interactive, clear_training_artifacts_interactive,
# ensure_base_directories are now imported from core.helpers.utils

def setup_logging(log_level_str="INFO"):
    """Configures basic logging for the CLI application."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(levelname)s [%(module)s.%(funcName)s] - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
            # Optionally add a FileHandler if CLI actions need their own log file
            # logging.FileHandler("logs/cli_operations.log", mode='a', encoding='utf-8')
        ]
    )
    logger.info(f"CLI logging initialized at level {log_level_str}")


def _get_config_value(key, default_value=None):
    """Safely retrieves a value from the loaded CONFIG dictionary."""
    # Helper to access nested keys like 'prediction.target_class_names'
    keys = key.split('.')
    value = CONFIG
    try:
        for k in keys:
            if isinstance(value, dict):
                value = value[k]
            else:  # Handle cases where a sub-key is requested but parent is not a dict
                logger.warning(f"Config key '{k}' in '{key}' not found as dict. Path broken.")
                return default_value
        return value
    except (KeyError, TypeError) as e:
        # logger.debug(f"Config key '{key}' not found or not a dict, using default: {default_value}. Error: {e}")
        return default_value


def _get_full_path(config_key, default_dir_name):
    """
    Constructs a full path. Uses config_dir if available, otherwise default_dir_name.
    Paths are relative to the project root (where 'core' is).
    """
    # Assuming project root is one level above 'core' where this script resides
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    configured_dir = _get_config_value(f"directories.{config_key}", default_dir_name)

    # If configured_dir is already absolute, use it. Otherwise, join with project_root.
    if os.path.isabs(configured_dir):
        return configured_dir
    else:
        return os.path.join(project_root, configured_dir)


def main_menu():
    """
    Displays the main menu and handles user choices.
    """
    global CONFIG
    try:
        CONFIG = load_config()  # Load config at the start of the menu
        if not CONFIG:
            logger.error("Failed to load configuration. CLI cannot proceed reliably.")
            print("❌  Critical: Configuration file (config.yaml) not found or invalid. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        print(f"❌  Error loading configuration: {e}. Exiting.")
        sys.exit(1)

    # Get directory paths from config or use defaults
    # These paths are now constructed to be absolute or relative to project root
    # The ensure_base_directories function in helpers.utils will use these.
    models_dir_path = _get_full_path("models_dir", DEFAULT_MODELS_DIR)
    logs_dir_path = _get_full_path("logs_dir", DEFAULT_LOGS_DIR)
    data_dir_path = _get_full_path("data_dir", DEFAULT_DATA_DIR)
    database_dir_path = _get_full_path("database_dir", DEFAULT_DATABASE_DIR)
    update_log_file_path = os.path.join(data_dir_path, _get_config_value("update_log_file", "update_log.txt").split('/')[-1])

    # Pass these paths to ensure_base_directories
    # ensure_base_directories is now expected to take a list of paths to create
    ensure_base_directories([database_dir_path, data_dir_path, models_dir_path, logs_dir_path])

    # Get timeframe list and symbol groups from config
    core_timeframes = _get_config_value("core_timeframes_list", ['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
    symbol_groups_dict = _get_config_value("symbol_groups", {'top8': [], 'meme': [], 'defi': []})
    available_symbol_groups = list(symbol_groups_dict.keys())

    while True:
        print_header("Главное меню")
        print("--- 📦 Сбор и Обновление Данных ---")
        print("  1. Полная (глубокая) загрузка исторических данных (old_update)")
        print("  2. Инкрементальное обновление свежих данных (mini_update)")
        print("--- ⚙️  Обработка и Обучение Моделей ---")
        print("  3. Построить признаки для выбранных таймфреймов (preprocess_features)")
        print("  4. Обучить модели для выбранных таймфреймов (train_model)")
        print("--- 🚀 Пайплайны (Комбинированные операции) ---")
        print("  5. Пайплайн: Мини-обновление -> Признаки -> Обучение (для выбранных TF)")
        print("  6. Пайплайн: Полное обновление -> Признаки -> Обучение (для выбранных TF)")
        print("  7. ПОЛНЫЙ ПЕРЕСБОР: Очистка -> Полное обновление -> Признаки -> Обучение")
        print("--- 📊 Прогнозирование и Анализ ---")
        print("  8. Сгенерировать прогнозы и торговый план (predict_all --save)")
        print("  9. Запустить бэктест для одного таймфрейма (predict_backtest)")
        print("--- 🛠️  Утилиты ---")
        print(" 10. Проверить доступность GPU для CatBoost (gpu_test)")  # Assuming gpu_test.py exists at root
        print(" 11. ОЧИСТИТЬ все артефакты обучения (модели, логи, features)")
        print("  0. Выход")

        choice = input("Введите номер опции: ").strip()

        try:
            if choice == '1':
                # For external scripts, we still use subprocess
                # run_script now handles path resolution if historical_data_loader.py is in project root
                run_script([PYTHON_EXECUTABLE, "historical_data_loader.py", "--all-tf-all-core"], "Полная загрузка данных") # Changed from --all

            elif choice == '2':
                tfs = select_timeframes_interactive("Инкрементальное обновление", core_timeframes)
                if tfs:
                    # run_script now handles path resolution if incremental_data_loader.py is in project root
                    run_script([PYTHON_EXECUTABLE, "incremental_data_loader.py", "--tf"] + tfs,
                               f"Инкрементальное обновление для {', '.join(tfs)}")

            elif choice == '3':
                tfs = select_timeframes_interactive("Построение признаков", core_timeframes)
                if tfs:
                    group_or_symbol = input(
                        f"Введите группу ({'/'.join(available_symbol_groups)}) или символ (например: BTCUSDT),\n"
                        "или оставьте пустым для всех: "
                    ).strip()
                    group_args = []
                    description_suffix = "для всех"
                    if group_or_symbol:
                        if group_or_symbol.lower() in available_symbol_groups:
                            group_args = ["--symbol-group", group_or_symbol.lower()]
                            description_suffix = f"для группы {group_or_symbol.lower()}"
                        else:
                            group_args = ["--symbol", group_or_symbol.upper()]
                            description_suffix = f"для символа {group_or_symbol.upper()}"

                    for tf_item in tfs:
                        desc = f"Построение признаков {description_suffix} ({tf_item})"
                        # run_script now handles path resolution
                        if run_script([PYTHON_EXECUTABLE, "preprocess_features.py", "--tf", tf_item] + group_args, desc) != 0:
                            logger.warning(f"{desc} прервано или завершилось с ошибкой. Пропуск остальных таймфреймов.")
                            break

            elif choice == '4':
                tfs = select_timeframes_interactive("Обучение моделей", core_timeframes)
                if tfs:
                    group_or_symbol = input(
                        f"Введите группу ({'/'.join(available_symbol_groups)}) или символ (например: BTCUSDT),\n"
                        "или оставьте пустым для всех: "
                    ).strip()
                    symbol_arg_list = []
                    description_suffix = "для всех пар"
                    if group_or_symbol:
                        if group_or_symbol.lower() in available_symbol_groups:
                            symbol_arg_list = ["--symbol-group", group_or_symbol.lower()]
                            description_suffix = f"для группы {group_or_symbol.lower()}"
                        else:
                            symbol_arg_list = ["--symbol", group_or_symbol.upper()]
                            description_suffix = f"для символа {group_or_symbol.upper()}"

                    for tf_item in tfs:
                        desc = f"Обучение {description_suffix} ({tf_item})"
                        # run_script now handles path resolution
                        if run_script([PYTHON_EXECUTABLE, "train_model.py", "--tf", tf_item] + symbol_arg_list, desc) != 0:
                            logger.warning(f"{desc} прервано или завершилось с ошибкой. Пропуск остальных таймфреймов.")
                            break

            elif choice == '5':
                tfs = select_timeframes_interactive("Пайплайн: Мини-обновление -> Признаки -> Обучение", core_timeframes)
                if tfs:
                    # run_script now handles path resolution
                    run_script([PYTHON_EXECUTABLE, "pipeline.py", "--train", "--skip-predict", "--tf"] + tfs,
                               "Пайплайн (мини-обновление, признаки, обучение)")
            elif choice == '6':
                tfs = select_timeframes_interactive("Пайплайн: Полное обновление -> Признаки -> Обучение", core_timeframes)
                if tfs:
                    # run_script now handles path resolution
                    run_script(
                        [PYTHON_EXECUTABLE, "pipeline.py", "--full-update", "--train", "--skip-predict", "--tf"] + tfs,
                        "Пайплайн (полное обновление, признаки, обучение)")

            elif choice == '7':
                tfs = select_timeframes_interactive(
                    "ПОЛНЫЙ ПЕРЕСБОР: Очистка -> Полное обновление -> Признаки -> Обучение", core_timeframes)
                if tfs:
                    print_header("Начало ПОЛНОГО ПЕРЕСБОРА")
                    clear_training_artifacts_interactive(
                        models_dir_path, logs_dir_path, data_dir_path,
                        database_dir_path, update_log_file_path
                    )
                    # run_script now handles path resolution
                    if run_script([PYTHON_EXECUTABLE, "pipeline.py", "--full-update", "--train", "--skip-predict", "--tf"] + tfs,
                                  "Этап 1/1 (Полный пересбор): Обновление, Признаки, Обучение") == 0:
                        print_header("ПОЛНЫЙ ПЕРЕСБОР завершен успешно.")
                    else:
                        logger.error("ПОЛНЫЙ ПЕРЕСБОР прерван на одном из этапов.")

            elif choice == '8':
                print_header("Генерация прогнозов")
                group_or_symbol = input(
                    f"Введите группу ({'/'.join(available_symbol_groups)}) или символ (например: BTCUSDT),\n"
                    "или оставьте пустым для всех: "
                ).strip()
                predict_args = ["--save"]
                description_suffix = "для всех пар"
                if group_or_symbol:
                    group_or_symbol_lower = group_or_symbol.lower()
                    if group_or_symbol_lower in available_symbol_groups:
                        predict_args += ["--symbol-group", group_or_symbol_lower]
                        description_suffix = f"для группы {group_or_symbol_lower}"
                    else:
                        predict_args += ["--symbol", group_or_symbol.upper()]
                        description_suffix = f"для символа {group_or_symbol.upper()}"
                # run_script now handles path resolution
                run_script(
                    [PYTHON_EXECUTABLE, "predict_all.py"] + predict_args,
                    f"Генерация прогнозов {description_suffix}"
                )

            elif choice == '9':
                print_header("Запуск бэктеста")
                print("Доступные таймфреймы:", ", ".join(core_timeframes))
                tf_backtest = input(f"Введите таймфрейм для бэктеста (например, 15m) или 'q' для отмены: ").strip()
                if tf_backtest.lower() == 'q':
                    continue
                if tf_backtest in core_timeframes:
                    # run_script now handles path resolution
                    run_script([PYTHON_EXECUTABLE, "predict_backtest.py", "--tf", tf_backtest],
                               f"Бэктест для {tf_backtest}")
                else:
                    logger.warning(f"Некорректный таймфрейм: {tf_backtest}")

            elif choice == '10':
                # run_script now handles path resolution
                run_script([PYTHON_EXECUTABLE, "gpu_test.py"], "Проверка GPU")

            elif choice == '11':
                clear_training_artifacts_interactive(
                    models_dir_path, logs_dir_path, data_dir_path,
                    database_dir_path, update_log_file_path
                )

            elif choice == '0':
                logger.info("Выход из программы по выбору пользователя.")
                print("Выход из программы.")
                sys.exit(0)

            else:
                logger.warning(f"Неверный ввод в меню: {choice}")
                print("Неверный ввод. Пожалуйста, выберите номер из меню.")

        except KeyboardInterrupt:
            logger.info("Операция прервана пользователем (Ctrl+C в меню). Возврат в главное меню.")
            print("\nОперация прервана пользователем (Ctrl+C в меню). Возврат в главное меню.")
            continue
        except Exception as e:
            logger.critical(f"Критическая ошибка в main_menu: {e}", exc_info=True)
            continue

def main_cli_entry_point():
    """
    The main entry point for the CLI application.
    Initializes logging and starts the main menu.
    """
    setup_logging(log_level_str="INFO")

    try:
        main_menu()
    except KeyboardInterrupt:
        logger.info("Программа завершена пользователем (Ctrl+C).")
        print("\nПрограмма завершена пользователем (Ctrl+C).")
        sys.exit(0)
    except SystemExit as e:
        logger.info(f"Программа завершается с кодом: {e.code}")
        sys.exit(e.code)
    except Exception as e:
        logger.critical(f"Критическая ошибка вне главного меню: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main_cli_entry_point()
