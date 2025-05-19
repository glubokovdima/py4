# core/cli/main.py
"""
Main Command-Line Interface (CLI) for the Crypto Prediction project.
Provides a menu-driven interface to run various scripts and pipelines.
"""
import subprocess
import sys
import os
import shutil
import argparse  # Keep for potential future CLI args to this script itself
import logging

# --- Import functions from core modules ---
from ..helpers.utils import (
    load_config,
    print_header,
    run_script,  # Still used for scripts not yet fully integrated as direct calls
    select_timeframes_interactive,
    clear_training_artifacts_interactive,
    ensure_base_directories
)
from ..data_ingestion.historical_data_loader import main_historical_load_logic
from ..data_ingestion.incremental_data_loader import main_incremental_load_logic
from ..features.preprocessor import main_preprocess_logic
from ..training.trainer import main_train_logic
from ..prediction.predictor import main_predict_all_logic
from ..backtesting.backtester import main_backtest_logic
from ..pipelines.main_pipeline import main_pipeline_logic

# from ..analysis.symbol_selector import main_symbol_selection_logic # If you add a menu option for it
# from ..path.to.gpu_test_logic import main_gpu_test_logic # If gpu_test.py is refactored

# Configure logging for this module
logger = logging.getLogger(__name__)

# --- Global Configuration (Loaded from YAML) ---
CONFIG = {}
PYTHON_EXECUTABLE = sys.executable

# Default directory paths (can be overridden by config.yaml if keys exist)
# These are fallbacks if config doesn't specify them under 'directories'
DEFAULT_MODELS_DIR_NAME = "models"
DEFAULT_LOGS_DIR_NAME = "logs"
DEFAULT_DATA_DIR_NAME = "data"
DEFAULT_DATABASE_DIR_NAME = "database"
DEFAULT_UPDATE_LOG_FILENAME = "update_log.txt"  # Filename, dir comes from data_dir


def setup_logging(log_level_str="INFO"):
    """Configures basic logging for the CLI application."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    # Basic config that will apply if no other handlers are configured on root
    # If other modules configure their own loggers, this won't interfere badly.
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s] - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
            # Optionally add a FileHandler for CLI-specific operations
            # logging.FileHandler("logs/cli_main_operations.log", mode='a', encoding='utf-8')
        ],
        force=True  # If basicConfig was already called, force will reconfigure. Use with caution.
    )
    logger.info(f"CLI logging initialized at level {log_level_str}")


def _get_config_value(key_path, default_value=None):
    """
    Safely retrieves a value from the loaded CONFIG dictionary using a dot-separated key path.
    Example: _get_config_value("directories.models_dir", "models")
    """
    global CONFIG
    if not CONFIG:  # Should have been loaded by main_cli_entry_point
        logger.warning("Attempted to get config value, but CONFIG is not loaded. Using default.")
        return default_value

    keys = key_path.split('.')
    value = CONFIG
    try:
        for k in keys:
            if isinstance(value, dict):
                value = value[k]
            else:
                logger.debug(f"Config path broken at '{k}' in '{key_path}'. Parent not a dict.")
                return default_value
        return value
    except (KeyError, TypeError):
        # logger.debug(f"Config key '{key_path}' not found or type error, using default: {default_value}.")
        return default_value


def _get_project_relative_path(config_key_for_dir, default_dir_name, filename=None):
    """
    Constructs a full path relative to the project root.
    Uses config for directory name if available, otherwise default_dir_name.
    Optionally appends a filename.

    Args:
        config_key_for_dir (str): Dot-separated key for the directory in config (e.g., "directories.models_dir").
        default_dir_name (str): Default directory name if not in config.
        filename (str, optional): Filename to append to the directory path.

    Returns:
        str: Absolute path.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Get directory part from config or use default
    configured_dir_part = _get_config_value(config_key_for_dir, default_dir_name)

    # If configured_dir_part is already absolute, use it. Otherwise, join with project_root.
    if os.path.isabs(configured_dir_part):
        dir_abs_path = configured_dir_part
    else:
        dir_abs_path = os.path.join(project_root, configured_dir_part)

    if filename:
        return os.path.join(dir_abs_path, filename)
    return dir_abs_path


def _handle_function_call_result(return_code, success_msg, failure_msg_prefix):
    """Handles logging and printing for direct function call results."""
    if return_code == 0:
        logger.info(success_msg)
        print(f"✅  {success_msg}")
    elif return_code == 130:  # Specific code for user interruption if functions return it
        logger.warning(f"🔶  {failure_msg_prefix} прервано пользователем.")
        print(f"🔶  {failure_msg_prefix} прервано пользователем.")
    else:
        logger.error(f"❌  {failure_msg_prefix} завершилось с ошибкой (код: {return_code}).")
        print(f"❌  {failure_msg_prefix} завершилось с ошибкой (код: {return_code}).")
    return return_code  # Propagate for pipeline handling


def main_menu():
    """
    Displays the main menu and handles user choices.
    """
    # Config should be loaded by main_cli_entry_point before this menu runs.
    global CONFIG
    if not CONFIG:
        logger.critical("CLI Main Menu: CONFIG not loaded. This should not happen. Exiting.")
        print("❌  Критическая ошибка: Конфигурация не загружена. Выход.")
        sys.exit(1)

    # Get directory paths from config or use defaults
    models_dir_abs = _get_project_relative_path("directories.models_dir", DEFAULT_MODELS_DIR_NAME)
    logs_dir_abs = _get_project_relative_path("directories.logs_dir", DEFAULT_LOGS_DIR_NAME)
    data_dir_abs = _get_project_relative_path("directories.data_dir", DEFAULT_DATA_DIR_NAME)
    database_dir_abs = _get_project_relative_path("directories.database_dir", DEFAULT_DATABASE_DIR_NAME)

    # update_log_file is a filename within data_dir
    update_log_filename_cfg = _get_config_value("update_log_file", "").split('/')[-1] or DEFAULT_UPDATE_LOG_FILENAME
    update_log_file_abs = os.path.join(data_dir_abs, update_log_filename_cfg)

    # Ensure base directories exist
    ensure_base_directories([database_dir_abs, data_dir_abs, models_dir_abs, logs_dir_abs])

    # Get timeframe list and symbol groups from config
    core_timeframes = _get_config_value("core_timeframes_list", ['1m', '5m', '15m', '30m', '1h', '4h', '1d'])
    symbol_groups_dict = _get_config_value("symbol_groups", {})
    available_symbol_groups = list(symbol_groups_dict.keys())

    while True:
        print_header("Главное меню")
        print("--- 📦 Сбор и Обновление Данных ---")
        print("  1. Полная (глубокая) загрузка исторических данных")
        print("  2. Инкрементальное обновление свежих данных")
        print("--- ⚙️  Обработка и Обучение Моделей ---")
        print("  3. Построить признаки для выбранных таймфреймов")
        print("  4. Обучить модели для выбранных таймфреймов")
        print("--- 🚀 Пайплайны (Комбинированные операции) ---")
        print("  5. Пайплайн: Мини-обновление -> Признаки -> Обучение")
        print("  6. Пайплайн: Полное обновление -> Признаки -> Обучение")
        print("  7. ПОЛНЫЙ ПЕРЕСБОР: Очистка -> Полное обновление -> Признаки -> Обучение")
        print("--- 📊 Прогнозирование и Анализ ---")
        print("  8. Сгенерировать прогнозы и торговый план")
        print("  9. Запустить бэктест для одного таймфрейма")
        print("--- 🛠️  Утилиты ---")
        print(" 10. Проверить доступность GPU (gpu_test.py)")
        print(" 11. ОЧИСТИТЬ все артефакты обучения")
        print("  0. Выход")

        choice = input("Введите номер опции: ").strip()

        try:
            if choice == '1':
                print_header("Полная загрузка исторических данных")
                ret_code = main_historical_load_logic(run_all_core_symbols=True, timeframes_to_process=None)
                _handle_function_call_result(ret_code, "Полная загрузка данных успешно завершена.", "Полная загрузка данных")

            elif choice == '2':
                tfs = select_timeframes_interactive("Инкрементальное обновление", core_timeframes)
                if tfs:
                    print_header(f"Инкрементальное обновление для {', '.join(tfs)}")
                    ret_code = main_incremental_load_logic(timeframes_to_process=tfs)
                    _handle_function_call_result(ret_code, f"Инкрементальное обновление для {', '.join(tfs)} успешно завершено.", f"Инкрементальное обновление для {', '.join(tfs)}")

            elif choice == '3':  # Preprocess Features
                tfs = select_timeframes_interactive("Построение признаков", core_timeframes)
                if tfs:
                    group_or_sym_input = input(
                        f"Введите группу ({'/'.join(available_symbol_groups)}) или символ (например: BTCUSDT),\n"
                        "или оставьте пустым для всех: "
                    ).strip()

                    symbol_filter_val, group_filter_val = None, None
                    desc_suffix = "для всех символов"
                    if group_or_sym_input:
                        if group_or_sym_input.lower() in available_symbol_groups:
                            group_filter_val = group_or_sym_input.lower()
                            desc_suffix = f"для группы {group_filter_val}"
                        else:
                            symbol_filter_val = group_or_sym_input.upper()
                            desc_suffix = f"для символа {symbol_filter_val}"

                    print_header(f"Построение признаков {desc_suffix}")
                    all_tf_success = True
                    for tf_item in tfs:
                        logger.info(f"CLI: Запуск main_preprocess_logic для TF={tf_item}, Symbol={symbol_filter_val}, Group={group_filter_val}")
                        ret_code = main_preprocess_logic(
                            timeframe=tf_item,
                            symbol_filter=symbol_filter_val,
                            symbol_group_filter=group_filter_val
                        )
                        _handle_function_call_result(ret_code, f"Построение признаков для {tf_item} {desc_suffix} успешно.", f"Построение признаков для {tf_item} {desc_suffix}")
                        if ret_code != 0:
                            all_tf_success = False
                            logger.warning(f"Построение признаков для {tf_item} {desc_suffix} не удалось. Пропуск остальных ТФ для этой операции.")
                            break
                    if all_tf_success: print("✅ Все выбранные таймфреймы обработаны для построения признаков.")


            elif choice == '4':  # Train Models
                tfs = select_timeframes_interactive("Обучение моделей", core_timeframes)
                if tfs:
                    group_or_sym_input = input(
                        f"Введите группу ({'/'.join(available_symbol_groups)}) или символ (например: BTCUSDT),\n"
                        "или оставьте пустым для обучения общих моделей по ТФ (если применимо): "  # Adjusted prompt
                    ).strip()

                    filter_val, is_group = None, False
                    desc_suffix = "общих моделей по ТФ"  # Default if no symbol/group
                    if group_or_sym_input:
                        if group_or_sym_input.lower() in available_symbol_groups:
                            filter_val = group_or_sym_input.lower()
                            is_group = True
                            desc_suffix = f"моделей для группы {filter_val}"
                        else:
                            filter_val = group_or_sym_input.upper()
                            is_group = False
                            desc_suffix = f"моделей для символа {filter_val}"

                    print_header(f"Обучение {desc_suffix}")
                    all_tf_success_train = True
                    for tf_item in tfs:
                        logger.info(f"CLI: Запуск main_train_logic для TF={tf_item}, Filter={filter_val}, IsGroup={is_group}")
                        ret_code = main_train_logic(
                            timeframe_to_train=tf_item,
                            symbol_or_group_filter=filter_val,
                            is_group_model=is_group
                        )
                        _handle_function_call_result(ret_code, f"Обучение {desc_suffix} для {tf_item} успешно.", f"Обучение {desc_suffix} для {tf_item}")
                        if ret_code != 0:
                            all_tf_success_train = False
                            logger.warning(f"Обучение {desc_suffix} для {tf_item} не удалось. Пропуск остальных ТФ.")
                            break
                    if all_tf_success_train: print(f"✅ Все выбранные таймфреймы обработаны для обучения {desc_suffix}.")


            elif choice == '5':  # Pipeline: Mini-update -> Features -> Train
                tfs = select_timeframes_interactive("Пайплайн: Мини-обновление -> Признаки -> Обучение", core_timeframes)
                if tfs:
                    print_header("Запуск пайплайна: Мини-обновление -> Признаки -> Обучение")
                    # Pipeline logic now expects filters too
                    ret_code = main_pipeline_logic(
                        timeframes_to_process=tfs,
                        do_training=True,
                        skip_final_predict=True,  # As per original menu option
                        use_full_historical_update=False
                        # Add symbol/group filter selection here if you want pipelines to be filterable
                    )
                    _handle_function_call_result(ret_code, "Пайплайн (мини-обновление, признаки, обучение) успешно завершен.", "Пайплайн (мини-обновление, признаки, обучение)")

            elif choice == '6':  # Pipeline: Full-update -> Features -> Train
                tfs = select_timeframes_interactive("Пайплайн: Полное обновление -> Признаки -> Обучение", core_timeframes)
                if tfs:
                    print_header("Запуск пайплайна: Полное обновление -> Признаки -> Обучение")
                    ret_code = main_pipeline_logic(
                        timeframes_to_process=tfs,
                        do_training=True,
                        skip_final_predict=True,  # As per original menu option
                        use_full_historical_update=True
                    )
                    _handle_function_call_result(ret_code, "Пайплайн (полное обновление, признаки, обучение) успешно завершен.", "Пайплайн (полное обновление, признаки, обучение)")

            elif choice == '7':  # FULL RESET: Clear -> Full Update -> Features -> Train
                tfs = select_timeframes_interactive(
                    "ПОЛНЫЙ ПЕРЕСБОР: Очистка -> Полное обновление -> Признаки -> Обучение", core_timeframes)
                if tfs:
                    print_header("Начало ПОЛНОГО ПЕРЕСБОРА")
                    clear_training_artifacts_interactive(
                        models_dir_abs, logs_dir_abs, data_dir_abs,
                        database_dir_abs, update_log_file_abs  # Pass correct absolute paths
                    )
                    logger.info("CLI: Запуск полного пересбора (обновление, признаки, обучение)...")
                    ret_code = main_pipeline_logic(
                        timeframes_to_process=tfs,
                        do_training=True,
                        skip_final_predict=True,  # As per original
                        use_full_historical_update=True
                    )
                    if ret_code == 0:  # Check specific return code from pipeline
                        print_header("ПОЛНЫЙ ПЕРЕСБОР завершен успешно.")
                        logger.info("ПОЛНЫЙ ПЕРЕСБОР завершен успешно.")
                    else:
                        print_header("ПОЛНЫЙ ПЕРЕСБОР прерван или завершился с ошибкой.")
                        logger.error("ПОЛНЫЙ ПЕРЕСБОР прерван или завершился с ошибкой.")

            elif choice == '8':  # Generate predictions
                print_header("Генерация прогнозов")
                group_or_sym_predict_input = input(
                    f"Введите группу ({'/'.join(available_symbol_groups)}) или символ (например: BTCUSDT),\n"
                    "или оставьте пустым для всех: "
                ).strip()

                symbol_filter_pred, group_filter_pred = None, None
                desc_suffix_pred = "для всех пар"
                if group_or_sym_predict_input:
                    if group_or_sym_predict_input.lower() in available_symbol_groups:
                        group_filter_pred = group_or_sym_predict_input.lower()
                        desc_suffix_pred = f"для группы {group_filter_pred}"
                    else:
                        symbol_filter_pred = group_or_sym_predict_input.upper()
                        desc_suffix_pred = f"для символа {symbol_filter_pred}"

                logger.info(f"CLI: Запуск генерации прогнозов {desc_suffix_pred} (с сохранением)...")
                ret_code = main_predict_all_logic(
                    save_output_flag=True,
                    symbol_filter=symbol_filter_pred,
                    group_filter_key=group_filter_pred
                )
                _handle_function_call_result(ret_code, f"Генерация прогнозов {desc_suffix_pred} успешно завершена.", f"Генерация прогнозов {desc_suffix_pred}")

            elif choice == '9':  # Backtest
                print_header("Запуск бэктеста")
                # Get TFs choices for backtest from config (or use core_timeframes as fallback)
                backtest_tf_choices = _get_config_value("predict_backtest_timeframes_choices", core_timeframes)
                print("Доступные таймфреймы для бэктеста:", ", ".join(backtest_tf_choices))

                tf_backtest_input = input(f"Введите таймфрейм для бэктеста (например, 15m) или 'q' для отмены: ").strip()
                if tf_backtest_input.lower() == 'q':
                    continue
                if tf_backtest_input in backtest_tf_choices:
                    # Ask for model suffix for backtest (e.g. if testing a group model)
                    model_suffix_backtest = input(
                        "Введите суффикс модели для бэктеста (напр. top8, BTCUSDT, или оставьте пустым для общей модели по ТФ): "
                    ).strip()

                    logger.info(f"CLI: Запуск бэктеста для TF={tf_backtest_input}, ModelSuffix='{model_suffix_backtest}'...")
                    ret_code = main_backtest_logic(
                        timeframe=tf_backtest_input,
                        model_suffix=model_suffix_backtest if model_suffix_backtest else ""
                        # target_class_names_list can be passed if needed from config
                    )
                    _handle_function_call_result(ret_code, f"Бэктест для {tf_backtest_input} (модель: '{model_suffix_backtest or 'общая'}') успешно завершен.", f"Бэктест для {tf_backtest_input} (модель: '{model_suffix_backtest or 'общая'}')")
                else:
                    logger.warning(f"Некорректный таймфрейм для бэктеста: {tf_backtest_input}")

            elif choice == '10':  # GPU Test
                # This still calls an external script.
                # If gpu_test.py is refactored into core.utils.gpu_checker.main_gpu_test_logic()
                # then call that directly.
                print_header("Проверка GPU")
                run_script([PYTHON_EXECUTABLE, "gpu_test.py"], "Проверка GPU")  # Assumes gpu_test.py at project root

            elif choice == '11':  # Clear artifacts
                print_header("Очистка артефактов")
                clear_training_artifacts_interactive(
                    models_dir_abs, logs_dir_abs, data_dir_abs,
                    database_dir_abs, update_log_file_abs  # Pass correct absolute paths
                )

            elif choice == '0':  # Exit
                logger.info("Выход из программы по выбору пользователя.")
                print("Выход из программы.")
                sys.exit(0)

            else:  # Invalid input
                logger.warning(f"Неверный ввод в меню: {choice}")
                print("Неверный ввод. Пожалуйста, выберите номер из меню.")

        except KeyboardInterrupt:
            logger.info("Операция прервана пользователем (Ctrl+C в меню). Возврат в главное меню.")
            print("\nОперация прервана пользователем (Ctrl+C в меню). Возврат в главное меню.")
            continue  # To the main menu loop
        except SystemExit as se:  # Catch sys.exit from direct function calls if they use it for errors
            logger.info(f"Действие завершилось с кодом выхода: {se.code}. Возврат в главное меню.")
            if se.code != 0:
                print(f"⚠️  Действие завершилось с ошибкой (код: {se.code}).")
            # Continue to main menu unless it's an exit from choice '0'
        except Exception as e:
            logger.critical(f"Критическая ошибка в обработке выбора меню '{choice}': {e}", exc_info=True)
            print(f"\n❌ Произошла ошибка при выполнении опции '{choice}'. Пожалуйста, проверьте логи.")
            continue  # To the main menu loop


def main_cli_entry_point():
    """
    The main entry point for the CLI application.
    Initializes logging and configuration, then starts the main menu.
    """
    global CONFIG
    # Setup logging first, so config loading issues are logged.
    # Logging level can be made configurable later (e.g., from env var or CLI arg to this script)
    setup_logging(log_level_str="INFO")

    try:
        CONFIG = load_config()
        if not CONFIG:
            # This is critical. If config doesn't load, many defaults will be used,
            # which might not be what the user expects.
            logger.error("Критическая ошибка: Файл конфигурации (core/config.yaml) не найден или не может быть загружен. "
                         "Проверьте его наличие и корректность. Некоторые функции могут работать некорректно с настройками по умолчанию.")
            # Allow proceeding with defaults but with a clear warning.
            # Or sys.exit(1) if config is absolutely mandatory.
            # For now, let it proceed so user can see the menu, but functions might fail.
            CONFIG = {}  # Ensure CONFIG is a dict even if loading failed
    except Exception as e_cfg:
        logger.error(f"Не удалось загрузить конфигурацию при запуске CLI: {e_cfg}", exc_info=True)
        CONFIG = {}  # Ensure CONFIG is a dict

    try:
        main_menu()
    except KeyboardInterrupt:
        logger.info("Программа завершена пользователем (Ctrl+C из главного меню).")
        print("\nПрограмма завершена пользователем (Ctrl+C).")
        sys.exit(0)  # Exit code 0 for clean user interrupt
    except SystemExit as e:
        logger.info(f"Программа завершается с кодом: {e.code} (перехвачено из main_menu).")
        sys.exit(e.code)  # Propagate exit code
    except Exception as e:
        logger.critical(f"Непредвиденная критическая ошибка вне главного меню: {e}", exc_info=True)
        sys.exit(1)  # General error


if __name__ == "__main__":
    main_cli_entry_point()
