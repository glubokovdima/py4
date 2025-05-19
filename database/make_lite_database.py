import os
import sqlite3
import pandas as pd

# 🗂 Исходная и целевая базы
SOURCE_DB = 'database/market_data.db'
TARGET_DB = 'database/lite_market_data.db'

# 🎯 Целевые пары и таймфреймы
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
           'XRPUSDT', 'ADAUSDT', 'LINKUSDT', 'AVAXUSDT']
TIMEFRAMES = ['5m', '15m', '30m', '1h']

def make_lite_database():
    if not os.path.exists(SOURCE_DB):
        print(f"❌ Исходная база не найдена: {SOURCE_DB}")
        return

    # Создаем новую мини-базу
    if os.path.exists(TARGET_DB):
        os.remove(TARGET_DB)

    conn_src = sqlite3.connect(SOURCE_DB)
    conn_dst = sqlite3.connect(TARGET_DB)

    cursor = conn_src.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    available_tables = [row[0] for row in cursor.fetchall()]
    print("📋 Найдены таблицы:", available_tables)

    copied_tables = 0

    for tf in TIMEFRAMES:
        table_name = f"candles_{tf}"
        if table_name not in available_tables:
            print(f"⚠️ Пропуск: таблица {table_name} отсутствует в {SOURCE_DB}")
            continue

        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {table_name} WHERE symbol IN ({','.join(['?']*len(SYMBOLS))})",
                conn_src,
                params=SYMBOLS
            )

            if not df.empty:
                df.to_sql(table_name, conn_dst, index=False, if_exists='replace')
                print(f"✅ {table_name}: {len(df)} строк экспортировано.")
                copied_tables += 1
            else:
                print(f"⚠️ {table_name}: данных по символам не найдено.")
        except Exception as e:
            print(f"❌ Ошибка при обработке {table_name}: {e}")

    conn_src.close()
    conn_dst.close()

    if copied_tables > 0:
        print(f"\n📦 Готово: база '{TARGET_DB}' содержит {copied_tables} таблиц.")
    else:
        print(f"\n⚠️ Ничего не скопировано. Проверь содержимое исходной базы.")

if __name__ == '__main__':
    make_lite_database()
