# scripts/predict.py

import argparse
import sys
import os
import logging

# Ensure the src directory is in the Python path if necessary (usually handled by running from project root)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import config and logging setup first
from src.utils.config import get_config
from src.utils.logging_setup import setup_logging

# Import the prediction functions from the models module
from src.models.predict import generate_predictions, save_predictions_and_plan, print_predictions_to_console

# --- Initial Setup ---
# Load configuration
config = get_config()
SYMBOL_GROUPS = config['symbol_groups'] # Need symbol groups for validation
TIMEFRAMES_CONFIG = config['timeframes'] # Need default timeframes

# Configure logging for this script
setup_logging()
logger = logging.getLogger(__name__) # Use logger specific to this module

# --- Command Line Interface ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Генерация прогнозов на основе обученных моделей.")
    parser.add_argument('--tf', nargs='+', help=f"Укажи таймфреймы для прогноза (например: 15m 1h). По умолчанию: {', '.join(TIMEFRAMES_CONFIG['default'])}")

    # Mutually exclusive group for specifying symbol or group filter
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('--symbol', type=str, help="Сгенерировать прогноз для отдельного символа (например: BTCUSDT).")
    filter_group.add_argument('--symbol-group', type=str, help=f"Сгенерировать прогноз для предопределенной группы символов из конфига (например: {', '.join(SYMBOL_GROUPS.keys())}).")
    # Note: If neither --symbol nor --symbol-group is specified, predictions will be generated for ALL symbols present in the features files.

    parser.add_argument('--save', action='store_true', help="Сохранять результаты прогнозов (latest_predictions_*.csv, trade_plan_*.csv, alerts_*.txt) в директорию logs.")

    args = parser.parse_args()

    # --- Determine Timeframes to Process ---
    timeframes_to_process = []
    if args.tf:
        # Validate specified timeframes against config defaults or a known list
        # Using config defaults as the source of truth for known timeframes
        allowed_timeframes = TIMEFRAMES_CONFIG['default']
        timeframes_to_process = [t for t in args.tf if t in allowed_timeframes]
        invalid_tfs = [t for t in args.tf if t not in allowed_timeframes]
        if invalid_tfs:
            logger.warning(f"Игнорируются неверные таймфреймы: {', '.join(invalid_tfs)}. Допустимые: {', '.join(allowed_timeframes)}")
        if not timeframes_to_process:
            logger.error("Не указан ни один допустимый таймфрейм. Прогноз не может быть сгенерирован.")
            # Print help or exit? Let's exit.
            sys.exit(1)
    else:
        # Use default timeframes from config if none specified
        timeframes_to_process = TIMEFRAMES_CONFIG['default']
        logger.info(f"Не указаны таймфреймы (--tf), используются по умолчанию: {', '.join(timeframes_to_process)}")


    # --- Determine Symbol/Group Filter and File Suffix ---
    symbol_filter = None
    group_filter = None
    files_suffix = "all" # Default suffix if no filter

    if args.symbol:
        symbol_filter = args.symbol.upper() # Ensure uppercase
        files_suffix = symbol_filter # Suffix is the symbol name
        logger.info(f"Прогноз для символа: {symbol_filter}.")
    elif args.symbol_group:
        group_name = args.symbol_group.lower() # Ensure lowercase
        if group_name in SYMBOL_GROUPS:
            group_filter = group_name
            files_suffix = group_filter # Suffix is the group name
            logger.info(f"Прогноз для группы символов: {group_filter}.")
        else:
            logger.error(f"Неизвестная группа символов: '{args.symbol_group}'. Доступные группы: {list(SYMBOL_GROUPS.keys())}. Прогноз не может быть сгенерирован.")
            sys.exit(1)
    else:
        # No filter specified - process all symbols
        files_suffix = "all" # Suffix is 'all'
        logger.info("Прогноз для всех символов (без фильтра по символу/группе).")


    # --- Run the prediction process ---
    try:
        # Call the main prediction function from the models module
        # Pass the determined timeframes and filter criteria
        predictions_data = generate_predictions(
            timeframes_to_process,
            symbol_filter=symbol_filter,
            group_filter=group_filter # Pass the group filter key
        )

        # --- Handle Output ---
        if args.save:
            # Call the save function from the models module
            save_predictions_and_plan(predictions_data, files_suffix)

        # Always print results to console
        print_predictions_to_console(predictions_data)


    except KeyboardInterrupt:
        logger.warning(f"\nГенерация прогнозов прервана пользователем (Ctrl+C).")
        sys.exit(130) # Standard Unix exit code for Ctrl+C
    except Exception as e:
        logger.error(f"💥 Непредвиденная ошибка при генерации прогнозов: {e}", exc_info=True)
        sys.exit(1) # Standard Unix exit code for general error