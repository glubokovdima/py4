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
    handlers=[
        logging.FileHandler('logs/pipeline.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
PYTHON_EXECUTABLE = sys.executable

DEFAULT_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']


def run_step(description, command_list):
    print(f"\n[Pipeline] 🔧  {description}...")
    logging.info(f"Старт шага: {description}")
    logging.info(f"Команда: {' '.join(command_list)}")
    start_time = time.time()

    result = subprocess.run(command_list, check=False)
    duration = time.time() - start_time

    if result.returncode == 0:
        logging.info(f"Завершено: {description} (⏱  {duration:.1f}s)")
        print(f"[Pipeline] ✅  {description} — выполнено (⏱  {duration:.1f}s)")
    elif result.returncode == 130:
        logging.warning(f"Шаг прерван пользователем: {description} (⏱  {duration:.1f}s)")
        print(f"[Pipeline] 🔶  Шаг прерван пользователем: {description} (⏱  {duration:.1f}s)")
        sys.exit(130)
    else:
        logging.error(f"Ошибка в шаге: {description} (код {result.returncode})")
        print(f"[Pipeline] ❌  Ошибка в шаге: {description} (код {result.returncode}) (⏱  {duration:.1f}s)")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline для крипто-прогноза")
    parser.add_argument('--tf', nargs='*', default=DEFAULT_TIMEFRAMES,
                        help=f"Таймфреймы для обработки (например: --tf 5m 15m). По умолчанию: {' '.join(DEFAULT_TIMEFRAMES)}")
    parser.add_argument('--train', action='store_true', help='Включить обучение моделей (CatBoost)')
    parser.add_argument('--skip-predict', action='store_true', help='Пропустить финальный прогноз')
    parser.add_argument('--full-update', action='store_true',
                        help='Вместо mini обновления запускать old_update_binance_data.py --all')
    
    # --- НОВЫЕ АРГУМЕНТЫ ---
    parser.add_argument('--symbol', type=str, default=None,
                        help="Один символ для обработки (например: BTCUSDT). Если указан, --symbol-group игнорируется.")
    parser.add_argument('--symbol-group', type=str, default=None,
                        help="Группа символов для обработки (например: top8, meme).")
    # -----------------------
    return parser.parse_args()


def main():
    args = parse_args()
    timeframes_to_process = args.tf

    # Формируем суффикс для описания и логов на основе symbol/symbol-group
    processing_target_desc = ""
    if args.symbol:
        processing_target_desc = f" для символа {args.symbol.upper()}"
    elif args.symbol_group:
        processing_target_desc = f" для группы {args.symbol_group.lower()}"
    else:
        processing_target_desc = " (все символы/группы по умолчанию)"


    print(f"[Pipeline] 🚀 Запуск пайплайна для таймфреймов: {', '.join(timeframes_to_process)}{processing_target_desc}")
    if args.full_update:
        print("[Pipeline] Выбран режим полного обновления данных.")
    if args.train:
        print("[Pipeline] Обучение моделей включено.")
    if args.skip_predict:
        print("[Pipeline] Финальный прогноз будет пропущен.")

    # --- Шаг 1: Обновление данных ---
    # Скрипты обновления данных (old_update, mini_update) пока не принимают --symbol/--symbol-group.
    # Если они будут модифицированы, здесь можно будет передавать эти аргументы.
    if args.full_update:
        run_step("📦  Полная загрузка исторических данных (old_update --all)",
                 [PYTHON_EXECUTABLE, "old_update_binance_data.py", "--all"])
    else:
        run_step(f"📥  Инкрементальное обновление свечей для {', '.join(timeframes_to_process)}",
                 [PYTHON_EXECUTABLE, "mini_update_binance_data.py", "--tf"] + timeframes_to_process)

    # --- Шаги 2 и 3: Построение признаков и Обучение (для каждого ТФ) ---
    for tf_item in timeframes_to_process:
        # --- Построение признаков ---
        preprocess_cmd = [PYTHON_EXECUTABLE, "preprocess_features.py", "--tf", tf_item]
        if args.symbol: # Если указан --symbol, он имеет приоритет
            preprocess_cmd.extend(["--symbol", args.symbol.upper()])
        elif args.symbol_group: # Иначе, если указана --symbol-group
            preprocess_cmd.extend(["--symbol-group", args.symbol_group.lower()])
        # Если ни --symbol, ни --symbol-group не указаны, preprocess_features.py обработает все по умолчанию
        
        desc_preprocess = f"⚙️  [{tf_item}{processing_target_desc}] Построение признаков"
        run_step(desc_preprocess, preprocess_cmd)

        # --- Обучение моделей ---
        if args.train:
            train_cmd = [PYTHON_EXECUTABLE, "train_model.py", "--tf", tf_item]
            # train_model.py ТРЕБУЕТ либо --symbol, либо --symbol-group
            if args.symbol:
                train_cmd.extend(["--symbol", args.symbol.upper()])
            elif args.symbol_group:
                train_cmd.extend(["--symbol-group", args.symbol_group.lower()])
            else:
                # Если ни символ, ни группа не указаны для пайплайна,
                # train_model.py может выдать ошибку или иметь свое поведение по умолчанию (например, обучить "общую" модель).
                # Для явности, если мы хотим обучать "общую" модель, train_model.py должен это поддерживать
                # без --symbol/--symbol-group или с каким-то специальным флагом.
                # В текущей реализации train_model.py (из предыдущего ответа) ожидает один из этих флагов.
                # Если мы хотим, чтобы пайплайн без этих флагов обучал "общую" модель,
                # то train_model.py должен быть изменен, чтобы это обрабатывать (например, суффикс "all").
                # Пока что, если эти флаги не переданы в пайплайн, train_model.py не будет вызван с ними.
                # Это может привести к ошибке в train_model.py, если он не настроен на работу без этих флагов.
                #
                # РЕШЕНИЕ: train_model.py должен уметь работать без --symbol/--symbol-group,
                # обрабатывая это как запрос на обучение "общей" модели (например, используя суффикс "all").
                # Если train_model.py строго требует один из флагов, то пайплайн должен либо
                # требовать их, либо иметь логику по умолчанию (например, не обучать, если они не заданы).
                #
                # Для данного обновления предположим, что если ни --symbol, ни --symbol-group не переданы в пайплайн,
                # то train_model.py будет вызван без этих флагов, и он должен сам решить, как это обработать
                # (например, обучить модель для "all").
                # Если train_model.py не может так работать, то нужно добавить ошибку здесь или
                # сделать --symbol или --symbol-group обязательными для пайплайна, если --train активен.
                #
                # Давайте сделаем так: если ни символ, ни группа не указаны, мы НЕ передаем эти флаги в train_model.
                # train_model.py из вашего предыдущего запроса уже умеет это обрабатывать,
                # устанавливая symbols_to_process = [None] (т.е. общая модель)
                # и используя files_suffix = tf_train (например, "1h_clf_long.pkl")
                #
                # ОБНОВЛЕНИЕ: train_model.py из предыдущего ответа требует либо --symbol, либо --symbol-group.
                # Значит, если они не переданы в пайплайн, и args.train активен, нужно выдать ошибку или пропустить обучение.
                # Для простоты, если они не переданы, мы не будем передавать эти флаги,
                # и train_model.py выдаст ошибку, что остановит пайплайн. Это явное поведение.
                # Альтернативно, можно не вызывать обучение, если флаги не заданы:
                if not args.symbol and not args.symbol_group:
                    logging.warning(f"Обучение для {tf_item} пропущено: не указан --symbol или --symbol-group для пайплайна.")
                    print(f"[Pipeline] ⚠️  Обучение для {tf_item} пропущено: не указан --symbol или --symbol-group.")
                    # continue # Пропустить этот ТФ для обучения, если это желательно
                else: # Только если символ или группа указаны, вызываем обучение
                    desc_train = f"🧠  [{tf_item}{processing_target_desc}] Обучение моделей (CatBoost)"
                    run_step(desc_train, train_cmd)

    # --- Шаг 4: Финальный прогноз ---
    if not args.skip_predict:
        predict_cmd = [PYTHON_EXECUTABLE, "predict_all.py", "--save"]
        if args.symbol:
            predict_cmd.extend(["--symbol", args.symbol.upper()])
        elif args.symbol_group:
            predict_cmd.extend(["--symbol-group", args.symbol_group.lower()])
        # Если ни --symbol, ни --symbol-group не указаны, predict_all.py обработает все по умолчанию.
        
        desc_predict = f"🔮  Финальный прогноз{processing_target_desc}"
        run_step(desc_predict, predict_cmd)

    print("[Pipeline] 🎉 Пайплайн завершен.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Pipeline] 🛑 Пайплайн прерван пользователем (Ctrl+C).")
        sys.exit(130)
    except SystemExit as e:
        if e.code == 130:
            print("\n[Pipeline] 🛑 Пайплайн прерван из-за Ctrl+C в дочернем процессе.")
        elif e.code == 1:
            print("\n[Pipeline] 🛑 Пайплайн прерван из-за ошибки в дочернем процессе.")
        sys.exit(e.code) # Передаем код выхода дальше
    except Exception as e:
        logging.error(f"[Pipeline] 💥 Непредвиденная ошибка в пайплайне: {e}", exc_info=True)
        print(f"\n[Pipeline] 💥 Непредвиденная ошибка в пайплайне: {e}")
        sys.exit(1)