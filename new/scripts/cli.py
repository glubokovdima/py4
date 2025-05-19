# scripts/cli.py

import subprocess
import sys
import os
import shutil
import argparse
import logging

# Ensure the src directory is in the Python path (usually handled by running from project root)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import config and logging setup first
from src.utils.config import get_config
from src.utils.logging_setup import setup_logging

# --- Initial Setup ---
# Load configuration
config = get_config()
PATHS_CONFIG = config['paths']
TIMEFRAMES_CONFIG = config['timeframes']
SYMBOL_GROUPS = config['symbol_groups']

# Configure logging for this script (this will set up handlers based on logging.yaml)
setup_logging()
logger = logging.getLogger(__name__) # Use logger specific to this module (scripts.cli)

# --- Constants from Config ---
CORE_TIMEFRAMES_LIST = TIMEFRAMES_CONFIG['default']
# Collect all known group names
KNOWN_SYMBOL_GROUPS = list(SYMBOL_GROUPS.keys())
# Add 'all' as a special key for --symbol-group in scripts if needed
# KNOWN_SYMBOL_GROUPS_WITH_ALL = KNOWN_SYMBOL_GROUPS + ['all'] # Decide if 'all' is a group or separate flag


# --- Helper Functions ---

def print_header(title):
    """Prints a formatted header to the console."""
    print("\n" + "=" * 60)
    print(f" {title.center(58)} ")
    print("=" * 60)

def run_script(script_name, command_args, description):
    """
    Runs a script as a subprocess.

    Args:
        script_name (str): The name of the script in the 'scripts' directory (e.g., 'update_data.py').
        command_args (list): List of arguments to pass to the script.
        description (str): A human-readable description of the step.

    Returns:
        int: The return code of the subprocess.
    """
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    command_list = [sys.executable, script_path] + command_args

    print(f"\n⏳  Запуск шага: {description}...")
    logger.info(f"Running step: {description}")
    logger.info(f"Command: {' '.join(command_list)}")

    try:
        # subprocess.run directs stdout/stderr to the parent process by default
        result = subprocess.run(command_list, check=False) # check=False allows us to handle return code

        if result.returncode == 0:
            print(f"✅  Успешно выполнено: {description}")
            logger.info(f"Step '{description}' completed successfully.")
        elif result.returncode == 130: # Standard Unix exit code for Ctrl+C
            print(f"🔶  Прервано пользователем: {description}")
            logger.warning(f"Step '{description}' interrupted by user (code 130).")
        else:
            print(f"❌  Ошибка (код {result.returncode}): {description}")
            logger.error(f"Step '{description}' failed with return code {result.returncode}.")

        return result.returncode

    except FileNotFoundError:
        print(f"❌  Ошибка: Скрипт '{script_name}' не найден. Убедитесь, что он существует в директории '{os.path.dirname(script_path)}' и Python доступен.")
        logger.error(f"Script '{script_name}' not found at '{script_path}'.")
        return -1 # Custom error code for script not found
    except Exception as e:
        print(f"❌  Непредвиденная ошибка при выполнении '{description}': {e}")
        logger.error(f"Unexpected error running step '{description}': {e}", exc_info=True)
        return -2 # Custom error code for other exceptions


def select_timeframes_interactive(prompt_message="Выберите таймфреймы"):
    """Interactively prompts user to select timeframes."""
    print_header(prompt_message)
    print("Доступные таймфреймы:", ", ".join(CORE_TIMEFRAMES_LIST))
    print("Введите таймфреймы через пробел (например, 15m 1h),")
    print("'all' для всех, или 'q' для отмены.")
    while True:
        selected_tfs_str = input(f"> ").strip()
        if selected_tfs_str.lower() == 'q':
            logger.info("Timeframe selection cancelled by user.")
            return None
        if not selected_tfs_str or selected_tfs_str.lower() == 'all':
            logger.info(f"Selected all timeframes: {CORE_TIMEFRAMES_LIST}")
            return CORE_TIMEFRAMES_LIST

        selected_tfs = []
        invalid_tfs = []
        for tf_input in selected_tfs_str.split():
            if tf_input in CORE_TIMEFRAMES_LIST:
                selected_tfs.append(tf_input)
            else:
                invalid_tfs.append(tf_input)

        if invalid_tfs:
            print(f"Предупреждение: Некорректные таймфреймы ({', '.join(invalid_tfs)}) будут проигнорированы.")
            logger.warning(f"Invalid timeframes entered: {invalid_tfs}. Ignoring.")

        if not selected_tfs:
            print("Не выбрано ни одного корректного таймфрейма. Попробуйте еще раз.")
            logger.warning("No valid timeframes selected.")
            continue
        # Sort selected TF for predictable order
        selected_tfs.sort(key=lambda x: CORE_TIMEFRAMES_LIST.index(x))
        print(f"Выбраны таймфреймы: {', '.join(selected_tfs)}")
        logger.info(f"Selected timeframes: {selected_tfs}")
        return selected_tfs


def select_symbol_filter_interactive(prompt_message="Выберите символ, группу или все"):
    """Interactively prompts user to select a symbol, group, or all symbols."""
    print_header(prompt_message)
    print(f"Доступные группы: {', '.join(KNOWN_SYMBOL_GROUPS)}")
    print("Введите символ (например: BTCUSDT),")
    print("имя группы (например: top8),")
    print("'all' для всех символов, или 'q' для отмены.")
    # Note: --symbol-list is not offered in the interactive menu for simplicity.

    while True:
        filter_str = input(f"> ").strip()
        if filter_str.lower() == 'q':
            logger.info("Symbol filter selection cancelled by user.")
            return None, None # Return None for both symbol and group args
        if not filter_str or filter_str.lower() == 'all':
            logger.info("Selected 'all' symbols.")
            return None, None # Return None for both args, indicating 'all'
        if filter_str.lower() in KNOWN_SYMBOL_GROUPS:
            group_name = filter_str.lower()
            print(f"Выбрана группа: {group_name}")
            logger.info(f"Selected symbol group: {group_name}")
            return None, ["--symbol-group", group_name] # Return group args
        # Assume it's a symbol if not a group or 'all'
        symbol_name = filter_str.upper() # Symbols are typically uppercase
        # Basic check if it looks like a symbol (e.g., contains USDT) - optional but can prevent typos
        # if 'USDT' not in symbol_name and len(symbol_name) < 4: # Very basic check
        #      print(f"'{filter_str}' не похоже на символ или известную группу. Попробуйте еще раз.")
        #      continue
        print(f"Выбран символ: {symbol_name}")
        logger.info(f"Selected symbol: {symbol_name}")
        return ["--symbol", symbol_name], None # Return symbol args


def clear_training_artifacts_interactive():
    """Interactively prompts user to confirm and clears training artifacts."""
    print_header("Очистка артефактов обучения")
    models_dir = PATHS_CONFIG['models_dir']
    logs_dir = PATHS_CONFIG['logs_dir']
    data_dir = PATHS_CONFIG['data_dir'] # Features and sample files are here

    print("\nВНИМАНИЕ! Это действие удалит:")
    print(f"  - Все файлы из директории '{models_dir}/'")
    print(f"  - Все файлы из директории '{logs_dir}/' (кроме update_log.txt)")
    print(f"  - Файлы 'features_*.pkl' и 'sample_*.csv' из директории '{data_dir}/'")
    print(f"\nПапка с базой данных '{PATHS_CONFIG['db']}' и файл '{os.path.join(data_dir, 'update_log.txt')}' НЕ будут затронуты.")

    confirm = input("Продолжить? (y/n): ").lower().strip()

    if confirm == 'y':
        print("\n🧹  Начинаем очистку...")
        logger.info("Starting cleanup of training artifacts.")

        # Clear Models Directory
        if os.path.exists(models_dir):
            try:
                # Use shutil.rmtree and then recreate the directory
                shutil.rmtree(models_dir)
                logger.info(f"Directory '{models_dir}' removed.")
            except Exception as e:
                print(f"    Не удалось удалить директорию моделей '{models_dir}': {e}")
                logger.error(f"Failed to remove models directory '{models_dir}': {e}")
            os.makedirs(models_dir, exist_ok=True) # Recreate the directory
            print(f"    Директория моделей '{models_dir}' очищена.")
        else:
            print(f"    Директория моделей '{models_dir}' не найдена (пропущено).")
            os.makedirs(models_dir, exist_ok=True) # Ensure it exists anyway

        # Clear Logs Directory (keeping update_log.txt)
        if os.path.exists(logs_dir):
            try:
                update_log_path = os.path.join(data_dir, "update_log.txt") # update_log is in data/
                # List all items first, then remove
                items_to_remove = [os.path.join(logs_dir, item) for item in os.listdir(logs_dir)]
                removed_count = 0
                for item_path in items_to_remove:
                    try:
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                            removed_count += 1
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                            removed_count += 1 # Count the directory removal
                    except Exception as e:
                         print(f"    Не удалось удалить '{item_path}': {e}")
                         logger.error(f"Failed to remove '{item_path}' during logs cleanup: {e}")

                print(f"    Содержимое директории логов '{logs_dir}' очищено (удалено {removed_count} элементов).")
                logger.info(f"Contents of '{logs_dir}' cleared ({removed_count} items removed).")

            except Exception as e:
                 print(f"    Не удалось просканировать директорию логов '{logs_dir}': {e}")
                 logger.error(f"Failed to list contents of '{logs_dir}' during cleanup: {e}")
        else:
            print(f"    Директория логов '{logs_dir}' не найдена (пропущено).")
        # Ensure logs directory exists after cleanup attempt
        os.makedirs(logs_dir, exist_ok=True)


        # Clear Features/Sample files from Data Directory
        if os.path.exists(data_dir):
            cleaned_feature_files_count = 0
            try:
                items_to_scan = [os.path.join(data_dir, item) for item in os.listdir(data_dir)]
                for item_path in items_to_scan:
                    if os.path.isfile(item_path):
                        item_name = os.path.basename(item_path)
                        # Check for features_*.pkl and sample_*.csv patterns
                        if (item_name.startswith("features_") and item_name.endswith(".pkl")) or \
                           (item_name.startswith("sample_") and item_name.endswith(".csv")):
                            try:
                                os.remove(item_path)
                                # logger.debug(f"Removed feature/sample file: {item_path}") # Too verbose
                                cleaned_feature_files_count += 1
                            except Exception as e:
                                print(f"    Не удалось удалить файл '{item_path}': {e}")
                                logger.error(f"Failed to remove feature/sample file '{item_path}': {e}")
            except Exception as e:
                 print(f"    Ошибка при сканировании директории данных '{data_dir}': {e}")
                 logger.error(f"Error scanning '{data_dir}' for feature/sample files: {e}")

            if cleaned_feature_files_count > 0:
                print(f"    Удалено {cleaned_feature_files_count} файлов признаков/сэмплов из '{data_dir}'.")
                logger.info(f"Removed {cleaned_feature_files_count} feature/sample files from '{data_dir}'.")
            else:
                print(f"    Файлы признаков/сэмплов в '{data_dir}' не найдены или уже очищены.")
                logger.info("No feature/sample files found for cleanup.")

        else:
            print(f"    Директория данных '{data_dir}' не найдена, пропуск удаления файлов признаков/сэмплов.")
            logger.warning(f"Data directory '{data_dir}' not found, skipping feature/sample cleanup.")

        print("✅  Очистка завершена.")
        logger.info("Cleanup process finished.")
    else:
        print("Очистка отменена.")
        logger.info("Cleanup cancelled by user.")


def ensure_base_directories():
    """Ensures necessary base directories exist."""
    print("Проверка и создание базовых директорий...")
    logger.info("Checking and creating base directories.")
    # DB directory path is from the config, extract directory name
    db_dir = os.path.dirname(PATHS_CONFIG['db'])
    dirs_to_check = [db_dir, PATHS_CONFIG['data_dir'], PATHS_CONFIG['models_dir'], PATHS_CONFIG['logs_dir']]
    for dir_path in dirs_to_check:
        # Ensure dir_path is not empty or current dir if db path is just a filename
        if dir_path and not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
                print(f"  Создана директория: {dir_path}")
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                 print(f"  ❌ Ошибка при создании директории {dir_path}: {e}")
                 logger.error(f"Failed to create directory {dir_path}: {e}")
        # else: logger.debug(f"Directory exists: {dir_path}") # Too verbose

    print("Проверка директорий завершена.")
    logger.info("Base directory check finished.")


# --- Main Menu ---
def main_menu():
    """Displays the main interactive menu and handles user input."""
    ensure_base_directories()

    while True:
        print_header("Главное меню")
        print("--- 📦 Сбор и Обновление Данных ---")
        print("  1. Полная (глубокая) загрузка исторических данных (update_data --full)")
        print("  2. Инкрементальное обновление свежих данных (update_data --mini)")
        print("--- ⚙️  Обработка и Обучение Моделей ---")
        print("  3. Построить признаки (preprocess)")
        print("  4. Обучить модели (train)")
        print("--- 🚀 Пайплайны (Комбинированные операции) ---")
        print("  5. Пайплайн: Мини-обновление -> Признаки -> Обучение (pipeline --mini --train)")
        print("  6. Пайплайн: Полное обновление -> Признаки -> Обучение (pipeline --full --train)")
        print("  7. ПОЛНЫЙ ПЕРЕСБОР: Очистка -> Полное обновление -> Признаки -> Обучение")
        print("--- 📊 Прогнозирование и Анализ ---")
        print("  8. Сгенерировать прогнозы и торговый план (predict)")
        print("  9. Запустить бэктест для одного таймфрейма (predict_backtest) [WIP]") # Mark as Work In Progress
        print("--- 🛠️  Утилиты ---")
        print(" 10. Проверить доступность GPU для CatBoost (gpu_test) [WIP]") # Mark as Work In Progress
        print(" 11. ОЧИСТИТЬ все артефакты обучения (модели, логи, features)")
        print("  0. Выход")
        print("-" * 60)

        choice = input("Введите номер опции: ").strip()
        logger.info(f"User selected menu option: {choice}")

        try:
            if choice == '1':
                tfs = select_timeframes_interactive("Таймфреймы для полной загрузки")
                if tfs is not None:
                     run_script("update_data.py", ["--full", "--tf"] + tfs, "Полная загрузка данных")

            elif choice == '2':
                tfs = select_timeframes_interactive("Таймфреймы для инкрементального обновления")
                if tfs is not None:
                     run_script("update_data.py", ["--mini", "--tf"] + tfs, "Инкрементальное обновление свежих данных")

            elif choice == '3':
                tfs = select_timeframes_interactive("Таймфреймы для построения признаков")
                if tfs is not None:
                    # Prompt for symbol/group filter
                    symbol_args, group_args = select_symbol_filter_interactive("Символ/Группа для построения признаков")
                    if symbol_args is not None or group_args is not None: # User didn't cancel filter selection
                         filter_args = symbol_args or group_args or [] # Use empty list if 'all' was selected
                         for tf_item in tfs:
                             desc = f"Построение признаков для {tf_item}" + ("" if not filter_args else f" ({filter_args[1]})")
                             if run_script("preprocess.py", ["--tf", tf_item] + filter_args, desc) != 0:
                                 print(f"Шаг '{desc}' завершился с ошибкой. Прерывание обработки остальных таймфреймов.")
                                 logger.error(f"Preprocessing failed for {tf_item} with filter {filter_args}. Aborting further TF processing.")
                                 break # Stop processing remaining TFs on error

            elif choice == '4':
                tfs = select_timeframes_interactive("Таймфреймы для обучения моделей")
                if tfs is not None:
                    # Prompt for symbol/group filter
                    # Train script requires --symbol or --symbol-group (or --symbol-group all)
                    print("Выберите символ или группу для обучения.")
                    print(f"  (Оставьте пустым или введите 'all' для обучения моделей для всех символов)")
                    symbol_args, group_args = select_symbol_filter_interactive(prompt_message="Символ/Группа для обучения")

                    # Determine the training key/args to pass to train.py
                    train_filter_args = []
                    if symbol_args:
                        train_filter_args = symbol_args # Pass --symbol <SYMBOL>
                        train_key_desc = symbol_args[1]
                    elif group_args:
                        train_filter_args = group_args # Pass --symbol-group <GROUP>
                        train_key_desc = group_args[1]
                        if train_key_desc.lower() not in KNOWN_SYMBOL_GROUPS and train_key_desc.lower() != 'all':
                             print(f"❌ Неизвестная группа для обучения: {train_key_desc}. Отмена.")
                             logger.error(f"Unknown group specified for training: {train_key_desc}. Cancelled.")
                             continue # Back to main menu
                    else: # User selected 'all' or left empty
                         train_filter_args = ["--symbol-group", "all"] # Use the standard key for 'all'
                         train_key_desc = "all"

                    for tf_item in tfs:
                         desc = f"Обучение моделей для {tf_item} ({train_key_desc})"
                         if run_script("train.py", ["--tf", tf_item] + train_filter_args, desc) != 0:
                            print(f"Шаг '{desc}' завершился с ошибкой. Прерывание обучения остальных таймфреймов.")
                            logger.error(f"Training failed for {tf_item} with filter {train_filter_args}. Aborting further TF training.")
                            break # Stop processing remaining TFs on error

            elif choice == '5':
                tfs = select_timeframes_interactive("Таймфреймы для пайплайна (Мини-обновление -> Признаки -> Обучение)")
                if tfs is not None:
                    # Prompt for symbol/group filter for preprocess/train steps in pipeline
                    symbol_args, group_args = select_symbol_filter_interactive("Символ/Группа для шагов Признаки/Обучение в пайплайне")
                    if symbol_args is not None or group_args is not None: # User didn't cancel filter selection
                        filter_args = symbol_args or group_args or [] # Use empty list if 'all' selected
                        filter_desc = "" if not filter_args else f" ({filter_args[1]})"
                        run_script("pipeline.py", ["--mini-update", "--train", "--tf"] + tfs + filter_args,
                                   f"Пайплайн (мини-обновление, признаки, обучение{filter_desc})")

            elif choice == '6':
                tfs = select_timeframes_interactive("Таймфреймы для пайплайна (Полное обновление -> Признаки -> Обучение)")
                if tfs is not None:
                    # Prompt for symbol/group filter for preprocess/train steps in pipeline
                    symbol_args, group_args = select_symbol_filter_interactive("Символ/Группа для шагов Признаки/Обучение в пайплайне")
                    if symbol_args is not None or group_args is not None: # User didn't cancel filter selection
                        filter_args = symbol_args or group_args or [] # Use empty list if 'all' selected
                        filter_desc = "" if not filter_args else f" ({filter_args[1]})"
                        run_script("pipeline.py", ["--full-update", "--train", "--tf"] + tfs + filter_args,
                                   f"Пайплайн (полное обновление, признаки, обучение{filter_desc})")

            elif choice == '7':
                tfs = select_timeframes_interactive(
                    "Таймфреймы для ПОЛНОГО ПЕРЕСБОРА (Очистка -> Полное обновление -> Признаки -> Обучение)")
                if tfs is not None:
                     print_header("Начало ПОЛНОГО ПЕРЕСБОРА")
                     logger.info("Starting FULL REBUILD pipeline.")
                     # Clear artifacts first
                     clear_training_artifacts_interactive()
                     # Run the pipeline with full update and training
                     # Full rebuild typically applies to ALL symbols, so no filter args here.
                     # If you wanted to rebuild *only* for a specific group/symbol, you'd add filter_args here.
                     # Let's assume full rebuild means for 'all' features/models.
                     if run_script("pipeline.py", ["--full-update", "--train", "--tf"] + tfs,
                                   "Этап 1/1 (Полный пересбор): Обновление, Признаки, Обучение для всех символов") == 0:
                         print_header("ПОЛНЫЙ ПЕРЕСБОР завершен успешно.")
                         logger.info("FULL REBUILD pipeline finished successfully.")
                     else:
                          print_header("ПОЛНЫЙ ПЕРЕСБОР прерван на одном из этапов.")
                          logger.error("FULL REBUILD pipeline aborted due to error.")


            elif choice == '8':
                # Generate Predictions
                tfs = select_timeframes_interactive("Таймфреймы для генерации прогнозов")
                if tfs is not None:
                    # Prompt for symbol/group filter
                    symbol_args, group_args = select_symbol_filter_interactive("Символ/Группа для генерации прогнозов")
                    if symbol_args is not None or group_args is not None: # User didn't cancel filter selection
                        filter_args = symbol_args or group_args or [] # Use empty list if 'all' selected
                        filter_desc = "" if not filter_args else f" ({filter_args[1]})"
                        # Add --save flag to save predictions by default from this menu option
                        run_script("predict.py", ["--tf"] + tfs + filter_args + ["--save"],
                                   f"Генерация прогнозов{filter_desc}")


            elif choice == '9':
                print("\n[WIP] Бэктест (predict_backtest) еще не реализован в новой структуре.")
                logger.info("Backtest option selected (WIP).")
                # TODO: Implement backtest script and call it here

            elif choice == '10':
                print("\n[WIP] Проверка GPU (gpu_test) еще не реализована в новой структуре.")
                logger.info("GPU Test option selected (WIP).")
                # TODO: Implement gpu_test script and call it here

            elif choice == '11':
                clear_training_artifacts_interactive()

            elif choice == '0':
                print("Выход из программы.")
                logger.info("Exiting program.")
                sys.exit(0)

            else:
                print("Неверный ввод. Пожалуйста, выберите номер опции из меню.")
                logger.warning(f"Invalid menu input: {choice}")

        except KeyboardInterrupt:
            print("\nОперация прервана пользователем (Ctrl+C). Возврат в главное меню.")
            logger.warning("Operation interrupted by user (Ctrl+C). Returning to main menu.")
            # subprocess.run should handle Ctrl+C and return 130, which run_step logs.
            # The outer loop catches it if Ctrl+C is pressed while in the menu prompt.
            continue # Go back to the menu loop
        except Exception as e:
            print(f"\nКритическая ошибка в main_cli: {e}")
            logger.critical(f"Critical error in main_cli menu loop: {e}", exc_info=True)
            # Decide whether to exit or return to menu on critical error
            # For now, return to menu is safer unless it's a SystemExit from subprocess
            continue # Go back to the menu loop

# --- Main Execution Block ---

if __name__ == "__main__":
    # This is the absolute entry point when running `python scripts/cli.py`
    # Set the current working directory to the project root
    # This ensures paths like ./config/, ./data/, etc. work correctly regardless
    # of where the script is executed from.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    # logger.info(f"Changed current working directory to: {os.getcwd()}") # Logger might not be fully set up yet

    # Now, set up logging using the config files
    # This should happen only once at the very beginning of the main entry point
    setup_logging()
    logger = logging.getLogger(__name__) # Re-get logger after setup

    logger.info("CLI started.")

    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем (Ctrl+C).")
        logger.info("Program terminated by user (Ctrl+C).")
        sys.exit(0)
    except SystemExit as e: # Catch sys.exit calls from run_step or validation errors
        if e.code != 0:
             logger.error(f"CLI exited with code {e.code}.")
        sys.exit(e.code)
    except Exception as e:
        print(f"\nКритическая ошибка вне меню: {e}")
        logger.critical(f"Critical error outside menu loop: {e}", exc_info=True)
        sys.exit(1)