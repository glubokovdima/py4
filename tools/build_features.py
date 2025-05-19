# build_features.py
import pandas as pd
import os

def preprocess_all(timeframe='15m'):
    # читаем CSV-файлы data_binance/*/{timeframe}.csv
    # собираем признаки, метки (target_up, delta, volatility)
    # сохраняем в data/features_{timeframe}.pkl
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', type=str, default='15m')
    args = parser.parse_args()
    preprocess_all(args.tf)
