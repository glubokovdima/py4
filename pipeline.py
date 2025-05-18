import subprocess
import sys
import time
import argparse
import logging
import os

# Логи
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename='logs/pipeline.log',
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s — %(message)s',
    handlers=[  # Добавляем вывод в консоль для pipeline.log тоже
        logging.FileHandler('logs/pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
PYTHON_EXECUTABLE = sys.executable  # Используем тот же интерпретатор

# Таймфреймы по умолчанию, если не переданы через аргументы
DEFAULT_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']


def run_step(description, command_list):  # Изменено на command_list
    print(f"\n[Pipeline] 🔧  {description}...")
    logging.info(f"Старт шага: {description}")
    logging.info(f"Команда: {' '.join(command_list)}")
    start_time = time.time()  # Переименовано во избежание конфликта с модулем time

    # shell=False, передаем команду как список
    result = subprocess.run(command_list, check=False)
    duration = time.time() - start_time

    if result.returncode == 0:
        logging.info(f"Завершено: {description} (⏱  {duration:.1f}s)")
        print(f"[Pipeline] ✅  {description} — выполнено (⏱  {duration:.1f}s)")
    elif result.returncode == 130:  # Ctrl+C в дочернем процессе
        logging.warning(f"Шаг прерван пользователем: {description} (⏱  {duration:.1f}s)")
        print(f"[Pipeline] 🔶  Шаг прерван пользователем: {description} (⏱  {duration:.1f}s)")
        sys.exit(130)  # Прерываем весь пайплайн
    else:
        logging.error(f"Ошибка в шаге: {description} (код {result.returncode})")
        print(f"[Pipeline] ❌  Ошибка в шаге: {description} (код {result.returncode}) (⏱  {duration:.1f}s)")
        sys.exit(1)  # Прерываем весь пайплайн при ошибке


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline для крипто-прогноза")
    parser.add_argument('--tf', nargs='*', default=DEFAULT_TIMEFRAMES,
                        help=f"Таймфреймы для обработки (например: --tf 5m 15m). По умолчанию: {' '.join(DEFAULT_TIMEFRAMES)}")
    parser.add_argument('--train', action='store_true', help='Включить обучение моделей (CatBoost)')
    parser.add_argument('--skip-predict', action='store_true', help='Пропустить финальный прогноз')
    parser.add_argument('--full-update', action='store_true',
                        help='Вместо mini обновления запускать old_update_binance_data.py --all')
    return parser.parse_args()


def main():
    args = parse_args()
    timeframes_to_process = args.tf

    print(f"[Pipeline] 🚀 Запуск пайплайна для таймфреймов: {', '.join(timeframes_to_process)}")
    if args.full_update:
        print("[Pipeline] Выбран режим полного обновления данных.")
    if args.train:
        print("[Pipeline] Обучение моделей включено.")
    if args.skip_predict:
        print("[Pipeline] Финальный прогноз будет пропущен.")

    if args.full_update:
        # Для old_update_binance_data.py передаем --all, если хотим обновить все ТФ из его списка
        # или можно передать конкретные --tf, если old_update_binance_data.py это поддерживает для --all
        # В текущей реализации old_update_binance_data.py --all обрабатывает все свои TIMEFRAMES
        run_step("📦  Полная загрузка исторических данных (old_update --all)",
                 [PYTHON_EXECUTABLE, "old_update_binance_data.py", "--all"])
    else:
        # mini_update_binance_data.py принимает --tf со списком
        run_step(f"📥  Инкрементальное обновление свечей для {', '.join(timeframes_to_process)}",
                 [PYTHON_EXECUTABLE, "mini_update_binance_data.py", "--tf"] + timeframes_to_process)

    # Построение признаков и обучение для каждого выбранного таймфрейма
    for tf_item in timeframes_to_process:
        run_step(f"⚙️  [{tf_item}] Построение признаков",
                 [PYTHON_EXECUTABLE, "preprocess_features.py", "--tf", tf_item])
        if args.train:
            run_step(f"🧠  [{tf_item}] Обучение моделей (CatBoost)",
                     [PYTHON_EXECUTABLE, "train_model.py", "--tf", tf_item])

    if not args.skip_predict:
        run_step("🔮  Финальный прогноз (predict_all --save)",
                 [PYTHON_EXECUTABLE, "predict_all.py", "--save"])

    print("[Pipeline] 🎉 Пайплайн завершен.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Pipeline] 🛑 Пайплайн прерван пользователем (Ctrl+C).")
        sys.exit(130)
    except SystemExit as e:  # Перехватываем sys.exit из run_step
        if e.code == 130:
            print("\n[Pipeline] 🛑 Пайплайн прерван из-за Ctrl+C в дочернем процессе.")
        elif e.code == 1:
            print("\n[Pipeline] 🛑 Пайплайн прерван из-за ошибки в дочернем процессе.")
        sys.exit(e.code)
    except Exception as e:
        logging.error(f"[Pipeline] 💥 Непредвиденная ошибка в пайплайне: {e}", exc_info=True)
        print(f"\n[Pipeline] 💥 Непредвиденная ошибка в пайплайне: {e}")
        sys.exit(1)