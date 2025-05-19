import sqlite3
import pandas as pd
import numpy as np
import os
import argparse

DB_PATH = "database/market_data.db"
MIN_OBS = 5000     # минимальное количество свечей на каждом ТФ

def load_data(tf):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM candles_{tf}", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["tf"] = tf
    return df

def evaluate_symbols(df, tf):
    results = []
    grouped = df.groupby("symbol")
    for symbol, df_sym in grouped:
        if len(df_sym) < MIN_OBS:
            continue

        df_sym = df_sym.sort_values("timestamp").copy()
        df_sym["atr"] = df_sym["high"] - df_sym["low"]
        df_sym["log_return"] = np.log(df_sym["close"] / df_sym["close"].shift(1))

        avg_volume = df_sym["volume"].rolling(50).mean().iloc[-1]
        avg_atr = df_sym["atr"].rolling(50).mean().iloc[-1]
        volatility = df_sym["log_return"].std()

        results.append({
            "symbol": symbol,
            "tf": tf,
            "avg_volume": avg_volume,
            "avg_atr": avg_atr,
            "volatility": volatility,
            "score": avg_volume * avg_atr * volatility
        })
    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Анализ ликвидности криптопар по нескольким таймфреймам")
    parser.add_argument('--tf', nargs='+', required=True, help='Укажи таймфреймы, например: 1m 5m 15m')
    args = parser.parse_args()

    all_results = []

    for tf in args.tf:
        print(f"📊 Загрузка данных: {tf}")
        if not os.path.exists(DB_PATH):
            print("❌ База данных не найдена.")
            return
        try:
            df = load_data(tf)
            if df.empty:
                print(f"⚠️ Нет данных по {tf}")
                continue
            df_eval = evaluate_symbols(df, tf)
            all_results.append(df_eval)
        except Exception as e:
            print(f"❌ Ошибка при обработке {tf}: {e}")
            continue

    if not all_results:
        print("❌ Нет пригодных данных для анализа.")
        return

    df_combined = pd.concat(all_results)

    # Агрегация: усреднение метрик по ТФ
    df_summary = df_combined.groupby("symbol").agg({
        "avg_volume": "mean",
        "avg_atr": "mean",
        "volatility": "mean",
        "score": "mean"
    }).sort_values("score", ascending=False)

    top_symbols = df_summary.head(20).reset_index()

    print("\n🔥 Топ-20 самых активных монет (агрегировано по ТФ):")
    print(top_symbols.round(4).to_string(index=False))

    top_symbols["symbol"].to_csv("top_symbols.csv", index=False)
    print("\n💾 Сохранено в top_symbols.csv")

if __name__ == "__main__":
    main()
