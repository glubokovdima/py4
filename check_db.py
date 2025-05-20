import sqlite3
import os
import pandas as pd # Используем pandas для удобного вывода таблицы

# --- Конфигурация (скопирована из вашего old_update_binance_data.py) ---
DB_PATH = 'database/market_data.db'
TIMEFRAMES = {
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',  
    '4h': '4h',
    '1d': '1d'
}
# Если у вас в old_update_binance_data.py есть еще 1m и 5m, добавьте их сюда:
# TIMEFRAMES['1m'] = '1m' 
# TIMEFRAMES['5m'] = '5m'
# Или просто определите TIMEFRAMES как список ключей, если значения не важны для этого скрипта
# TIMEFRAMES_KEYS = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

def get_db_stats():
    """
    Подключается к базе данных SQLite и выводит статистику по таблицам таймфреймов.
    """
    if not os.path.exists(DB_PATH):
        print(f"Ошибка: Файл базы данных '{DB_PATH}' не найден.")
        return

    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        print(f"Подключено к базе данных: {DB_PATH}\n")
        
        stats_data = []

        # Получаем список всех таблиц в БД, чтобы проверить существование
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        all_db_tables = [row[0] for row in cursor.fetchall()]

        for tf_key in TIMEFRAMES.keys(): # Или TIMEFRAMES_KEYS, если вы определили список
            table_name = f"candles_{tf_key}"
            
            if table_name not in all_db_tables:
                print(f"Таблица '{table_name}' не найдена в базе данных.")
                stats_data.append({'Таймфрейм (Таблица)': table_name, 'Всего свечей': 'Не найдена', 'Уникальных символов': 'Не найдена'})
                continue

            # Подсчет общего количества свечей
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_candles = cursor.fetchone()[0]

            # Подсчет уникальных символов
            cursor.execute(f"SELECT COUNT(DISTINCT symbol) FROM {table_name}")
            unique_symbols = cursor.fetchone()[0]
            
            stats_data.append({
                'Таймфрейм (Таблица)': table_name,
                'Всего свечей': total_candles,
                'Уникальных символов': unique_symbols
            })
            
            # print(f"Статистика для таблицы '{table_name}':")
            # print(f"  Всего свечей: {total_candles}")
            # print(f"  Уникальных символов: {unique_symbols}")
            # print("-" * 30)

        if stats_data:
            df_stats = pd.DataFrame(stats_data)
            print("Общая статистика по таблицам таймфреймов:")
            print(df_stats.to_string(index=False))
        else:
            print("Не найдено таблиц для анализа или словарь TIMEFRAMES пуст.")

    except sqlite3.Error as e:
        print(f"Ошибка SQLite: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
    finally:
        if conn:
            conn.close()
            # print("\nСоединение с базой данных закрыто.")

if __name__ == '__main__':
    # Важно: если вы недавно добавили 1m и 5m в TIMEFRAMES 
    # в old_update_binance_data.py, но еще не запускали его,
    # то таблицы candles_1m и candles_5m могут еще не существовать.
    # Этот скрипт покажет "Не найдена" для них.
    
    # Добавим 1m и 5m в TIMEFRAMES для этого скрипта, если они есть в вашей БД
    # (если они определены в вашем old_update_binance_data.py)
    if '1m' not in TIMEFRAMES:
        TIMEFRAMES['1m'] = '1m'
    if '5m' not in TIMEFRAMES:
        TIMEFRAMES['5m'] = '5m'
        
    # Сортируем ключи для более предсказуемого вывода
    # (необязательно, но приятно)
    # CORE_TIMEFRAMES_ORDER = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    # sorted_tf_keys = [tf for tf in CORE_TIMEFRAMES_ORDER if tf in TIMEFRAMES]
    # for tf_key_missing in TIMEFRAMES.keys():
    #    if tf_key_missing not in sorted_tf_keys:
    #        sorted_tf_keys.append(tf_key_missing)
    # (Проще просто оставить порядок из словаря TIMEFRAMES)

    get_db_stats()