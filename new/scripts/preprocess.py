# scripts/preprocess.py

import pandas as pd
import os
import argparse
from tqdm import tqdm
import sys
import logging

# Ensure the src directory is in the Python path if necessary (usually handled by running from project root)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import config, logging setup, and db module
from src.utils.config import get_config
from src.utils.logging_setup import setup_logging
from src import db # Import the db module

# Import feature computation functions from the new module
from src.features.build import compute_all_features_for_symbol, compute_btc_features, apply_flat_filter

# --- Initial Setup ---
# Load configuration
config = get_config()
FEATURE_BUILDER_CONFIG = config['feature_builder']
PATHS_CONFIG = config['paths']
SYMBOL_GROUPS = config['symbol_groups'] # Get symbol groups from config

# Configure logging for this script
# Note: In a pipeline or via cli.py, logging might already be set up.
# Calling setup_logging again is generally safe if it uses dictConfig,
# but logging.basicConfig should be avoided if calling setup_logging.
# Let's assume setup_logging handles idempotency or is called once by the entry point.
setup_logging()
logger = logging.getLogger(__name__) # Use logger specific to this module

# --- Constants from Config ---
# We now get most feature-related constants from FEATURE_BUILDER_CONFIG
MIN_DATA_FOR_FEATURES = FEATURE_BUILDER_CONFIG.get('min_data_for_features', 100) # Default if not in config
DROPNNA_COLS = FEATURE_BUILDER_CONFIG['dropna_columns'] # List of columns to dropna on


# --- Main Script Logic ---

def main_preprocess(tf_arg, symbols_filter=None):
    """
    Main function to load data, compute features, filter, and save.

    Args:
        tf_arg (str): Timeframe key to process.
        symbols_filter (list, optional): List of symbols to filter by.
                                         If None, process all symbols in the loaded data.
    """
    logger.info(f"⚙️  Начало построения признаков для таймфрейма: {tf_arg}")

    # --- Determine target symbols and file suffix ---
    # symbols_filter is passed directly now based on CLI args in __main__
    # Let's derive the suffix from the symbols_filter list if provided, or default to 'all'
    model_name_suffix_for_files = "all"
    log_processing_details = " (все символы из БД для этого ТФ)"

    if symbols_filter is not None:
         # Try to find if the symbols_filter list matches a known group for naming
         matched_group_name = None
         # Convert symbols_filter to a set for efficient comparison
         symbols_filter_set = set(symbols_filter)
         for group_name, group_symbols in SYMBOL_GROUPS.items():
             if symbols_filter_set == set(group_symbols):
                 matched_group_name = group_name
                 break

         if matched_group_name:
             model_name_suffix_for_files = matched_group_name
             log_processing_details = f" группы '{matched_group_name}' (символы: {', '.join(symbols_filter)})"
         elif len(symbols_filter) == 1:
             model_name_suffix_for_files = symbols_filter[0] # Use symbol name as suffix
             log_processing_details = f" символа '{symbols_filter[0]}'"
         else:
             # For a custom list, maybe create a hash or just use a generic 'custom' + count/prefix?
             # Let's use 'custom_N_symbols' for clarity
             model_name_suffix_for_files = f"custom_{len(symbols_filter)}"
             log_processing_details = f" списка символов ({', '.join(symbols_filter)})"

         logger.info(f"Обработка для {tf_arg} для{log_processing_details}. Фильтрация по этому списку.")
    else:
        logger.info(f"Обработка для {tf_arg} для{log_processing_details}. Используются все загруженные символы.")

    logger.info(f"Используемый суффикс для имен файлов: {model_name_suffix_for_files}")


    # --- Load data ---
    # Load ALL symbols for the given TF initially. Filtering happens next.
    df_all_candles = db.load_candles(tf_arg)

    if df_all_candles.empty:
        logger.error(f"Нет данных из БД для {tf_arg}. Построение признаков прервано.")
        return

    # --- Filter data if symbols_filter is provided ---
    if symbols_filter is not None:
        original_symbols_count = df_all_candles['symbol'].nunique()
        original_rows_count = len(df_all_candles)

        if 'symbol' not in df_all_candles.columns:
            logger.error("Колонка 'symbol' отсутствует в загруженных данных. Фильтрация невозможна.")
            return

        df_all_candles = df_all_candles[df_all_candles['symbol'].isin(symbols_filter)].copy() # Use .copy()

        if df_all_candles.empty:
            logger.warning(f"После фильтрации по символам {symbols_filter} для ТФ {tf_arg} данных не осталось. "
                            f"(Исходно было {original_rows_count} строк для {original_symbols_count} символов). "
                            f"Обработка не будет продолжена.")
            return
        else:
            filtered_symbols_count = df_all_candles['symbol'].nunique()
            logger.info(f"Отфильтровано по символам. Осталось {len(df_all_candles)} строк для {filtered_symbols_count} символов.")
            symbols_present_after_filter = df_all_candles['symbol'].unique().tolist()
            missing_requested_symbols = [s for s in symbols_filter if s not in symbols_present_after_filter]
            if missing_requested_symbols:
                 logger.warning(f"Некоторые запрошенные символы отсутствуют в загруженных данных для {tf_arg}: {missing_requested_symbols}")


    # --- Prepare BTC features (if BTCUSDT is in the data after filtering) ---
    btc_features_prepared = None
    if 'BTCUSDT' in df_all_candles['symbol'].unique():
        logger.info("Подготовка признаков BTCUSDT...")
        df_btc_raw = df_all_candles[df_all_candles['symbol'] == 'BTCUSDT'].copy()
        # Call the function from src.features.build
        btc_features_prepared = compute_btc_features(df_btc_raw)

        if btc_features_prepared is None or btc_features_prepared.empty:
             logger.warning("BTCUSDT features could not be computed or resulted in an empty DataFrame.")
        else:
             logger.info(f"BTCUSDT features computed and ready for merge: {len(btc_features_prepared)} rows.")
    else:
        logger.info("BTCUSDT not present in the data for this TF/filter. Skipping BTC features.")


    # --- Compute features for each symbol ---
    df_to_process = df_all_candles # This is already filtered if symbols_filter was provided
    if df_to_process.empty:
        logger.error(
            f"Нет данных для обработки признаков на ТФ {tf_arg} для{log_processing_details} (df_to_process пуст после фильтрации).")
        return

    all_symbols_features = []
    unique_symbols_in_data = df_to_process['symbol'].unique()

    if len(unique_symbols_in_data) == 0:
         logger.warning(f"Нет уникальных символов для обработки в df_to_process для ТФ {tf_arg}.")
         return

    logger.info(f"Вычисляем признаки для {len(unique_symbols_in_data)} символов на ТФ {tf_arg}...")

    for symbol_val in tqdm(unique_symbols_in_data, desc=f"Признаки {tf_arg}", unit="symbol"):
        df_sym = df_to_process[df_to_process['symbol'] == symbol_val].copy()
        # No need to sort or set index here, compute_all_features_for_symbol handles it

        # Call the main feature computation function from src.features.build
        # This function handles checking MIN_DATA_FOR_FEATURES internally
        df_sym_features = compute_all_features_for_symbol(df_sym, tf_arg, btc_features_prepared)

        if not df_sym_features.empty:
            # compute_all_features_for_symbol returns with timestamp as index, reset it
            df_sym_features = df_sym_features.reset_index()
            df_sym_features['symbol'] = symbol_val # Ensure symbol column is present
            all_symbols_features.append(df_sym_features)
        else:
             logger.debug(f"Feature computation for {symbol_val} on {tf_arg} resulted in an empty DataFrame or insufficient data after checks.")


    if not all_symbols_features:
        logger.error(f"Не удалось рассчитать признаки ни для одного символа на ТФ {tf_arg} для{log_processing_details}. Итоговый файл признаков не будет создан.")
        return

    full_features_df = pd.concat(all_symbols_features).reset_index(drop=True)
    logger.info(f"Объединены признаки для всех символов на {tf_arg}. Всего строк до фильтрации и dropna: {len(full_features_df)}.")


    # --- Apply Flat Filter ---
    # Apply the flat filter to the combined DataFrame.
    # apply_flat_filter expects timestamp as a column, so ensure it's reset if needed.
    # compute_all_features_for_symbol returns with index reset, so this is fine.
    len_before_flat_filter = len(full_features_df)
    # Call the function from src.features.build
    full_features_df = apply_flat_filter(full_features_df)
    len_after_flat_filter = len(full_features_df)
    if len_before_flat_filter > len_after_flat_filter:
        logger.info(f"Удалено {len_before_flat_filter - len_after_flat_filter} строк после применения плоского фильтра.")
    else:
        logger.debug("Плоский фильтр не удалил ни одной строки.")


    # --- Drop NaNs ---
    # Drop rows with NaN in essential columns required for training
    len_before_dropna = len(full_features_df)
    if DROPNNA_COLS and not full_features_df.empty:
         # Check if DROPNNA_COLS actually exist in the DataFrame
         cols_to_drop_subset = [col for col in DROPNNA_COLS if col in full_features_df.columns]
         if cols_to_drop_subset:
              full_features_df.dropna(subset=cols_to_drop_subset, inplace=True)
              len_after_dropna = len(full_features_df)
              if len_before_dropna > len_after_dropna:
                   logger.info(f"Удалено {len_before_dropna - len_after_dropna} строк из-за NaN в ключевых колонках ({', '.join(cols_to_drop_subset)}).")
              else:
                   logger.debug("dropna на ключевых колонках не удалил ни одной строки.")
         else:
              logger.warning(f"Ни одна из колонок для dropna ({', '.join(DROPNNA_COLS)}) не найдена в DataFrame. dropna пропущен.")
              len_after_dropna = len_before_dropna # No change in length
    else:
        logger.warning("Список колонок для dropna пуст или DataFrame пуст. dropna пропущен.")
        len_after_dropna = len_before_dropna # No change in length


    if full_features_df.empty:
        logger.error(f"Итоговый DataFrame признаков для {tf_arg} для{log_processing_details} пуст после фильтрации и dropna. Файлы признаков не будут созданы.")
        return

    logger.info(f"Расчет признаков для {tf_arg} завершен. Итого строк после фильтрации и dropna: {len(full_features_df)}.")


    # --- Save results ---
    os.makedirs(PATHS_CONFIG['data_dir'], exist_ok=True)

    output_pickle_path = os.path.join(PATHS_CONFIG['data_dir'], f"features_{model_name_suffix_for_files}_{tf_arg}.pkl")
    output_sample_csv_path = os.path.join(PATHS_CONFIG['data_dir'], f"sample_{model_name_suffix_for_files}_{tf_arg}.csv")

    try:
        full_features_df.to_pickle(output_pickle_path)
        logger.info(f"💾  Признаки сохранены: {output_pickle_path}, форма: {full_features_df.shape}")
    except Exception as e:
        logger.error(f"Ошибка сохранения Pickle файла {output_pickle_path}: {e}")

    try:
        sample_size = min(1000, len(full_features_df))
        if sample_size > 0:
            # Ensure timestamp is in a readable format for CSV sample
            df_sample = full_features_df.head(sample_size).copy()
            if 'timestamp' in df_sample.columns and pd.api.types.is_datetime64_any_dtype(df_sample['timestamp']):
                df_sample['timestamp'] = df_sample['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            # Save with timestamp column for sample visibility
            df_sample.to_csv(output_sample_csv_path, index=False)
            logger.info(f"📄  Сэмпл данных сохранен: {output_sample_csv_path} ({sample_size} строк)")
    except Exception as e:
        logger.error(f"Ошибка сохранения CSV сэмпла {output_sample_csv_path}: {e}")

    logger.info(f"✅  Завершено построение признаков для таймфрейма: {tf_arg} для{log_processing_details}")


# --- Command Line Interface ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Построение признаков на основе данных свечей из SQLite.")
    parser.add_argument('--tf', type=str, required=True, help='Таймфрейм для обработки (например: 5m, 15m)')
    # Allow specifying symbol, symbol-group, or a list of symbols
    symbol_group = parser.add_mutually_exclusive_group()
    symbol_group.add_argument('--symbol', type=str, default=None, help="Символ (например: BTCUSDT)")
    symbol_group.add_argument('--symbol-group', type=str, help="Псевдоним группы монет из конфига (например: top8, meme)")
    symbol_group.add_argument('--symbol-list', nargs='+', help="Список символов через пробел (например: BTCUSDT ETHUSDT ...)")


    args = parser.parse_args()

    # Determine the list of symbols to process based on arguments
    symbols_to_process = None # Default to None to process all
    log_filter_source = None

    if args.symbol_group:
        group_name = args.symbol_group.lower()
        if group_name in SYMBOL_GROUPS:
            symbols_to_process = SYMBOL_GROUPS[group_name]
            log_filter_source = f"группа '{group_name}'"
        else:
            logger.error(f"Неизвестная группа символов: {args.symbol_group}. Доступные группы: {list(SYMBOL_GROUPS.keys())}")
            sys.exit(1)
    elif args.symbol_list:
        symbols_to_process = [s.upper() for s in args.symbol_list] # Ensure uppercase
        log_filter_source = f"список символов ({len(symbols_to_process)})"
    elif args.symbol:
        symbols_to_process = [args.symbol.upper()] # Ensure uppercase
        log_filter_source = f"символ '{args.symbol.upper()}'"

    if log_filter_source:
         logger.info(f"Фильтр символов задан через {log_filter_source}.")
    else:
         logger.info("Фильтр символов не задан. Будут обработаны все символы, найденные в БД для данного TF.")


    try:
        main_preprocess(args.tf, symbols_to_process)
    except KeyboardInterrupt:
        logger.warning(f"\nПостроение признаков для {args.tf} прервано пользователем (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 Непредвиденная ошибка при построении признаков для {args.tf}: {e}", exc_info=True)
        sys.exit(1)