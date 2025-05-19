# core/helpers/utils.py
"""
General utility functions for the Crypto Prediction project.
Includes configuration loading, CLI interaction helpers, time conversions,
and other miscellaneous helper functions.
"""
import yaml
import os
import sys
import shutil
import subprocess
import logging
import pandas as pd  # For type hinting and potentially some data ops
import numpy as np  # For classify_delta_value

logger = logging.getLogger(__name__)

# --- Configuration Loading ---
DEFAULT_CONFIG_PATH = 'core/config.yaml'  # Default path relative to project root


def load_config(config_path=None):
    """
    Loads the YAML configuration file.

    Args:
        config_path (str, optional): Path to the YAML configuration file.
            If None, tries to load from DEFAULT_CONFIG_PATH relative to project root,
            then from DEFAULT_CONFIG_PATH relative to current working directory.

    Returns:
        dict: A dictionary containing the configuration, or None if loading fails.
    """
    paths_to_try = []
    if config_path:
        paths_to_try.append(config_path)
    else:
        # Try path relative to project root (assuming utils.py is in core/helpers)
        project_root_config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),  # core/helpers
            "..",  # core
            "..",  # project root
            DEFAULT_CONFIG_PATH
        )
        paths_to_try.append(os.path.normpath(project_root_config_path))
        # Try path relative to current working directory
        paths_to_try.append(DEFAULT_CONFIG_PATH)

    for path in paths_to_try:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                logger.info(f"Configuration loaded successfully from: {path}")
                return config_data
            else:
                logger.debug(f"Config file not found at: {path}")
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration file {path}: {e}", exc_info=True)
            return None  # Critical error, stop trying
        except IOError as e:
            logger.error(f"Error reading configuration file {path}: {e}", exc_info=True)
            # Continue to next path if IOError on one, unless it was explicitly provided
            if config_path and path == config_path: return None
        except Exception as e:
            logger.error(f"Unexpected error loading configuration from {path}: {e}", exc_info=True)
            if config_path and path == config_path: return None

    logger.error(f"Configuration file could not be loaded from any of the tried paths: {paths_to_try}")
    return None


# --- CLI Interaction Helpers (from main_cli.py) ---

def print_header(title, width=50):
    """Prints a formatted header to the console."""
    print("\n" + "=" * width)
    print(f" {title.center(width - 2)} ")
    print("=" * width)


def run_script(command_list, description, python_executable=None):
    """
    Runs an external Python script using subprocess.

    Args:
        command_list (list): The command and its arguments as a list of strings.
                             The first element should be the Python executable,
                             followed by the script name, then script arguments.
        description (str): A description of the script/task being run.
        python_executable (str, optional): Path to the python interpreter.
                                           If None, sys.executable is used.

    Returns:
        int: The return code of the script. -1 if an exception occurred.
    """
    if not command_list:
        logger.error("Command list for run_script is empty.")
        return -1

    # Ensure the first element is the python executable if not already set
    # This is a bit redundant if the calling code already does this, but good for safety.
    actual_command = list(command_list)  # Make a copy
    if not actual_command[0].lower().endswith("python") and \
            not actual_command[0].lower().endswith("python.exe") and \
            not os.path.basename(actual_command[0]).lower().startswith("python"):
        py_exec = python_executable or sys.executable
        actual_command.insert(0, py_exec)

    logger.info(f"⏳ Запуск: {description}...")
    logger.info(f"    Команда: {' '.join(actual_command)}")
    print(f"\n⏳  Запуск: {description}...")
    print(f"    Команда: {' '.join(actual_command)}")  # Also print to console for CLI user

    try:
        # For simple pass-through of output, Popen or run without capture_output is better.
        # check=False allows manual handling of return codes.
        result = subprocess.run(actual_command, check=False)  # Runs and waits

        if result.returncode == 0:
            logger.info(f"✅  Успешно: {description}")
            print(f"✅  Успешно: {description}")
        elif result.returncode == 130:  # Ctrl+C in child process
            logger.warning(f"🔶  Прервано пользователем: {description}")
            print(f"🔶  Прервано пользователем: {description}")
        else:
            logger.error(f"❌  Ошибка (код {result.returncode}): {description}")
            print(f"❌  Ошибка (код {result.returncode}): {description}")
        return result.returncode
    except FileNotFoundError:
        logger.error(f"❌  Ошибка: Команда/скрипт не найден. Убедитесь, что Python доступен и скрипт '{actual_command[1] if len(actual_command) > 1 else 'N/A'}' существует.")
        print(f"❌  Ошибка: Команда не найдена. Убедитесь, что Python доступен и скрипт '{actual_command[1] if len(actual_command) > 1 else 'N/A'}' существует.")
        return -1
    except Exception as e:
        logger.error(f"❌  Непредвиденная ошибка при выполнении '{description}': {e}", exc_info=True)
        print(f"❌  Непредвиденная ошибка при выполнении '{description}': {e}")
        return -1


def select_timeframes_interactive(prompt_message="Выберите таймфреймы", available_timeframes=None):
    """
    Prompts the user to select timeframes interactively.

    Args:
        prompt_message (str): The message to display to the user.
        available_timeframes (list, optional): A list of valid timeframe strings.
            If None, a default list is used for validation.

    Returns:
        list or None: A list of selected timeframe strings, sorted,
                      or None if the user cancels or provides no valid input.
    """
    if available_timeframes is None:
        # Fallback if no list is provided (e.g., from config)
        available_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
        logger.debug("select_timeframes_interactive using default available_timeframes.")

    print_header(prompt_message)
    print(f"Доступные таймфреймы: {', '.join(available_timeframes)}")

    while True:
        selected_tfs_str = input(
            f"Введите таймфреймы через пробел (например, 15m 1h),\n"
            f"'all' для всех из списка ({len(available_timeframes)}), или 'q' для отмены: "
        ).strip().lower()  # Convert to lower for 'all' and 'q'

        if selected_tfs_str == 'q':
            logger.info("Выбор таймфреймов отменен пользователем.")
            return None
        if not selected_tfs_str or selected_tfs_str == 'all':
            logger.info(f"Выбраны все доступные таймфреймы: {available_timeframes}")
            return available_timeframes  # Return a copy

        selected_tfs = []
        invalid_tfs_input = []
        for tf_input in selected_tfs_str.split():
            if tf_input in available_timeframes:
                if tf_input not in selected_tfs:  # Avoid duplicates if user enters "15m 15m"
                    selected_tfs.append(tf_input)
            else:
                invalid_tfs_input.append(tf_input)

        if invalid_tfs_input:
            logger.warning(f"Некорректные таймфреймы в вводе: {', '.join(invalid_tfs_input)}. Будут проигнорированы.")
            print(f"Предупреждение: Некорректные таймфреймы ({', '.join(invalid_tfs_input)}) будут проигнорированы.")

        if not selected_tfs:
            logger.warning("Не выбрано ни одного корректного таймфрейма.")
            print("Не выбрано ни одного корректного таймфрейма. Попробуйте еще раз или введите 'q'.")
            continue

        # Sort selected timeframes based on their order in available_timeframes for consistency
        try:
            selected_tfs.sort(key=lambda x: available_timeframes.index(x))
        except ValueError:  # Should not happen if logic above is correct
            logger.error("Ошибка при сортировке выбранных таймфреймов. Используется алфавитная сортировка.")
            selected_tfs.sort()

        logger.info(f"Выбраны таймфреймы: {', '.join(selected_tfs)}")
        print(f"Выбраны таймфреймы: {', '.join(selected_tfs)}")
        return selected_tfs


def clear_training_artifacts_interactive(
        models_dir_path, logs_dir_path, data_features_dir_path,
        database_dir_path, update_log_file_path  # Pass full paths
):
    """
    Interactively prompts the user to clear training artifacts like models, logs,
    and feature files. Specific files/dirs like database and update_log are preserved.

    Args:
        models_dir_path (str): Path to the models directory.
        logs_dir_path (str): Path to the logs directory.
        data_features_dir_path (str): Path to the data directory containing features/samples.
        database_dir_path (str): Path to the database directory (will NOT be cleared).
        update_log_file_path (str): Path to the update_log.txt file (will NOT be cleared).
    """
    print_header("Очистка артефактов обучения")
    confirm = input(
        "ВНИМАНИЕ! Это действие удалит:\n"
        f"  - Все файлы и папки из директории '{models_dir_path}/'\n"
        f"  - Все файлы и папки из директории '{logs_dir_path}/'\n"
        f"  - Файлы 'features_*.pkl' и 'sample_*.csv' из директории '{data_features_dir_path}/'\n"
        f"Папка с базой данных '{database_dir_path}/' и файл '{update_log_file_path}' НЕ будут затронуты.\n"
        "Продолжить? (y/n): "
    ).lower()

    if confirm == 'y':
        logger.info("Начало очистки артефактов обучения по подтверждению пользователя.")
        print("\n🧹  Начинаем очистку...")

        # Clear contents of models and logs directories
        for dir_to_clear, description in [(models_dir_path, "моделей"), (logs_dir_path, "логов")]:
            if os.path.exists(dir_to_clear):
                try:
                    for item in os.listdir(dir_to_clear):
                        item_path = os.path.join(dir_to_clear, item)
                        if os.path.isfile(item_path) or os.path.islink(item_path):  # also remove symlinks
                            os.unlink(item_path)  # Use unlink for files and symlinks
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    logger.info(f"Содержимое директории {description} '{dir_to_clear}' очищено.")
                    print(f"    Содержимое директории {description} '{dir_to_clear}' очищено.")
                except Exception as e:
                    logger.error(f"Не удалось очистить содержимое директории {dir_to_clear}: {e}")
                    print(f"    Не удалось очистить содержимое директории {dir_to_clear}: {e}")
            else:
                logger.info(f"Директория {dir_to_clear} не найдена (пропущено).")
                print(f"    Директория {dir_to_clear} не найдена (пропущено).")
            # Ensure directory exists after clearing (it might have been removed if rmtree was used on the dir itself)
            os.makedirs(dir_to_clear, exist_ok=True)

        # Clear specific files from data_features_dir
        cleaned_feature_files_count = 0
        if os.path.exists(data_features_dir_path):
            try:
                for item in os.listdir(data_features_dir_path):
                    item_path = os.path.join(data_features_dir_path, item)
                    # Check if it's a file and matches patterns
                    if os.path.isfile(item_path) and \
                            ((item.startswith("features_") and item.endswith(".pkl")) or \
                             (item.startswith("sample_") and item.endswith(".csv"))):
                        # Crucially, do not delete the update_log.txt if it's in this directory
                        if item_path.lower() == update_log_file_path.lower():
                            logger.debug(f"Пропуск удаления файла лога обновлений: {item_path}")
                            continue
                        try:
                            os.remove(item_path)
                            logger.debug(f"Удален файл: {item_path}")
                            cleaned_feature_files_count += 1
                        except Exception as e:
                            logger.error(f"Не удалось удалить файл {item_path}: {e}")
                            print(f"    Не удалось удалить файл {item_path}: {e}")
            except Exception as e:
                logger.error(f"Ошибка при сканировании директории {data_features_dir_path}: {e}")
                print(f"    Ошибка при сканировании директории {data_features_dir_path}: {e}")

            if cleaned_feature_files_count > 0:
                logger.info(f"Удалено {cleaned_feature_files_count} файлов признаков/сэмплов из '{data_features_dir_path}'.")
                print(f"    Удалено {cleaned_feature_files_count} файлов признаков/сэмплов из '{data_features_dir_path}'.")
            else:
                logger.info(f"Файлы признаков/сэмплов в '{data_features_dir_path}' не найдены или уже очищены.")
                print(f"    Файлы признаков/сэмплов в '{data_features_dir_path}' не найдены или уже очищены.")
        else:
            logger.info(f"Директория {data_features_dir_path} не найдена, пропуск удаления файлов признаков/сэмплов.")
            print(f"    Директория {data_features_dir_path} не найдена, пропуск удаления файлов признаков/сэмплов.")

        logger.info("Очистка артефактов завершена.")
        print("✅  Очистка завершена.")
    else:
        logger.info("Очистка артефактов отменена пользователем.")
        print("Очистка отменена.")


def ensure_base_directories(dir_paths_list):
    """
    Creates a list of base directories if they don't exist.

    Args:
        dir_paths_list (list): A list of directory paths to create.
    """
    logger.info("Проверка и создание базовых директорий...")
    print("Проверка и создание базовых директорий...")
    for dir_path in dir_paths_list:
        if not dir_path:  # Skip if path is empty or None
            logger.warning("Получен пустой путь к директории, пропуск.")
            continue
        try:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)  # exist_ok=True is important
                logger.info(f"  Создана директория: {dir_path}")
                print(f"  Создана директория: {dir_path}")
            # else:
            #     logger.debug(f"  Директория существует: {dir_path}")
        except OSError as e:  # Catch potential OS errors during makedirs
            logger.error(f"  Ошибка при создании директории {dir_path}: {e}")
            print(f"  Ошибка при создании директории {dir_path}: {e}")
        except Exception as e:
            logger.error(f"  Непредвиденная ошибка при проверке/создании директории {dir_path}: {e}")
            print(f"  Непредвиденная ошибка при проверке/создании директории {dir_path}: {e}")

    logger.info("Проверка директорий завершена.")
    print("Проверка директорий завершена.")


# --- Timeframe and Data Conversion Helpers ---

def get_tf_ms(timeframe_str):
    """
    Converts a timeframe string (e.g., '1m', '1h', '1d') to milliseconds.

    Args:
        timeframe_str (str): The timeframe string.

    Returns:
        int: The timeframe in milliseconds, or 0 if conversion fails.
    """
    if not isinstance(timeframe_str, str) or not timeframe_str:
        logger.error(f"Invalid timeframe_str: '{timeframe_str}'. Must be a non-empty string.")
        return 0

    multipliers = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000}  # Added W, M for flexibility

    numeric_part_str = ""
    unit_part_str = ""

    for char in timeframe_str:
        if char.isdigit():
            numeric_part_str += char
        elif char.isalpha():
            unit_part_str += char
            # Allow for multi-char units if needed in future, but typically single like 'm', 'h'
            # For now, let's assume the first alphabetic char is the unit.
            break

    if not numeric_part_str or not unit_part_str:
        logger.error(f"Could not parse numeric value or unit from timeframe: '{timeframe_str}'")
        return 0

    try:
        numeric_value = int(numeric_part_str)
    except ValueError:
        logger.error(f"Could not convert numeric part '{numeric_part_str}' to int from timeframe: '{timeframe_str}'")
        return 0

    unit_key = unit_part_str[0].lower()  # Take first char of unit and lower case it
    if unit_key == 'm' and unit_part_str.lower() == 'mo':  # Distinguish 'Minutes' from 'Months'
        unit_key = 'M'  # Use 'M' for Month

    if unit_key not in multipliers:
        logger.error(f"Unknown time unit '{unit_part_str}' (parsed as '{unit_key}') in timeframe: '{timeframe_str}'")
        return 0

    return numeric_value * multipliers[unit_key] * 1000


# --- Data Classification Helpers ---

def classify_delta_value(delta_val, up_thresh=0.002, down_thresh=-0.002, neutral_is_nan=True):
    """
    Classifies a delta value into 'UP', 'DOWN', or NEUTRAL/NaN.
    NEUTRAL can be np.nan or a specific string like 'NEUTRAL'.

    Args:
        delta_val (float or pd.Series): The delta value(s) to classify.
        up_thresh (float): Threshold above which delta is considered 'UP'.
        down_thresh (float): Threshold below which delta is considered 'DOWN'.
        neutral_is_nan (bool): If True, NEUTRAL is represented as np.nan.
                               If False, NEUTRAL is represented as the string "NEUTRAL".

    Returns:
        str or np.nan or pd.Series: Classified label(s).
    """
    if isinstance(delta_val, pd.Series):
        # Apply classification element-wise for a Series
        def _classify_single(val):
            if pd.isna(val): return np.nan
            if val > up_thresh: return 'UP'
            if val < down_thresh: return 'DOWN'
            return np.nan if neutral_is_nan else 'NEUTRAL'

        return delta_val.apply(_classify_single)
    else:  # Single float value
        if pd.isna(delta_val): return np.nan
        if delta_val > up_thresh: return 'UP'
        if delta_val < down_thresh: return 'DOWN'
        return np.nan if neutral_is_nan else 'NEUTRAL'


# Example usage (if run directly for testing)
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - [%(module)s] - %(message)s')

    print("\n--- Testing load_config ---")
    # Create a dummy config for testing if it doesn't exist in expected default location
    dummy_cfg_path = "temp_test_config.yaml"
    if not os.path.exists(DEFAULT_CONFIG_PATH) and not os.path.exists(os.path.join("..", "..", DEFAULT_CONFIG_PATH)):
        with open(dummy_cfg_path, "w") as f_cfg:
            yaml.dump({"test_key": "test_value", "nested": {"num": 123}}, f_cfg)
        cfg = load_config(dummy_cfg_path)
        os.remove(dummy_cfg_path)
    else:
        cfg = load_config()  # Try default paths

    if cfg:
        print(f"Config loaded: {cfg}")
        assert cfg.get("test_key") == "test_value" if "test_key" in cfg else True
    else:
        print("Config not loaded (this might be normal if no default config exists for test).")

    print("\n--- Testing print_header ---")
    print_header("Test Header Utility")

    print("\n--- Testing select_timeframes_interactive ---")
    # This is interactive, so manual test or skip in automated tests
    # selected = select_timeframes_interactive(available_timeframes=['1h', '4h', '1d', '5m'])
    # print(f"Selected TFs: {selected}")

    print("\n--- Testing ensure_base_directories ---")
    test_dirs = ["temp_dir1/subdir", "temp_dir2"]
    ensure_base_directories(test_dirs)
    for d in test_dirs: assert os.path.exists(d)
    if os.path.exists("temp_dir1"): shutil.rmtree("temp_dir1")
    if os.path.exists("temp_dir2"): shutil.rmtree("temp_dir2")
    print("Base directories test passed and cleaned up.")

    print("\n--- Testing get_tf_ms ---")
    assert get_tf_ms("1m") == 60000
    assert get_tf_ms("15m") == 15 * 60000
    assert get_tf_ms("1h") == 3600000
    assert get_tf_ms("4h") == 4 * 3600000
    assert get_tf_ms("1d") == 86400000
    assert get_tf_ms("1w") == 7 * 86400000  # Week
    assert get_tf_ms("1mo") == 30 * 86400000  # Month (approx 30 days)
    assert get_tf_ms("5min") == 5 * 60000  # allow 'min'
    assert get_tf_ms("1hour") == 1 * 3600000  # allow 'hour'
    assert get_tf_ms("1day") == 1 * 86400000  # allow 'day'
    assert get_tf_ms("invalid") == 0
    assert get_tf_ms("m1") == 0  # number must come first
    print("get_tf_ms tests passed.")

    print("\n--- Testing classify_delta_value ---")
    assert classify_delta_value(0.003) == 'UP'
    assert classify_delta_value(-0.003) == 'DOWN'
    assert classify_delta_value(0.001) is np.nan
    assert classify_delta_value(0.001, neutral_is_nan=False) == 'NEUTRAL'
    assert pd.isna(classify_delta_value(np.nan))

    series_deltas = pd.Series([0.003, -0.005, 0.000, np.nan])
    classified_series = classify_delta_value(series_deltas)
    print(f"Classified series: {classified_series.tolist()}")
    expected_series = ['UP', 'DOWN', np.nan, np.nan]
    # Need to handle NaN comparison carefully for series assert
    pd.testing.assert_series_equal(classified_series, pd.Series(expected_series, dtype=object), check_dtype=False)
    print("classify_delta_value tests passed.")

    print("\n--- Testing run_script (example: python --version) ---")
    # This will print Python version to console
    ret_code = run_script([sys.executable, "--version"], "Проверка версии Python")
    assert ret_code == 0, f"run_script for python --version failed with code {ret_code}"
    print("run_script test (python --version) seemed to work.")
