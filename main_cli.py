# main_cli.py
import subprocess
import sys
import os
import shutil
import argparse

# --- Конфигурация ---
PYTHON_EXECUTABLE = sys.executable
# Таймфреймы можно вынести в общий конфигурационный файл или определить здесь
# Для примера, возьмем из mini_update_binance_data.py (но лучше иметь один источник истины)
CORE_TIMEFRAMES_LIST = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

# Допустимые группы символов для --symbol-group
# >>> ДОБАВЛЕНО согласно патчу
SYMBOL_GROUPS = ['top8', 'meme', 'defi']

# Директории
MODELS_DIR = "models"
LOGS_DIR = "logs"
DATA_FEATURES_DIR = "data"  # Папка, где хранятся features_*.pkl и sample_*.csv
DATABASE_DIR = "database"
# Лог-файл загрузки данных, который не должен удаляться при очистке артефактов обучения
UPDATE_LOG_FILE = os.path.join(DATA_FEATURES_DIR, "update_log.txt")


# --- Вспомогательные функции ---
def print_header(title):
    print("\n" + "=" * 50)
    print(f" {title.center(48)} ")
    print("=" * 50)


def run_script(command_list, description):
    print(f"\n⏳  Запуск: {description}...")
    print(f"    Команда: {' '.join(command_list)}")
    try:
        # Использование capture_output=True и text=True может помочь при отладке,
        # но для простого проброса вывода дочернего процесса в консоль лучше оставить как есть
        # result = subprocess.run(command_list, check=False, capture_output=True, text=True)
        result = subprocess.run(command_list, check=False) # check=False позволяет обработать код возврата вручную
        if result.returncode == 0:
            print(f"✅  Успешно: {description}")
        elif result.returncode == 130: # Код 130 обычно означает прерывание по Ctrl+C
            print(f"🔶  Прервано пользователем: {description}")
        else:
            print(f"❌  Ошибка (код {result.returncode}): {description}")
        return result.returncode
    except FileNotFoundError:
        print(f"❌  Ошибка: Команда не найдена. Убедитесь, что Python доступен и скрипт '{command_list[1]}' существует.")
        return -1 # Возвращаем код ошибки
    except Exception as e:
        print(f"❌  Непредвиденная ошибка при выполнении '{description}': {e}")
        import traceback
        traceback.print_exc() # Для детальной информации об ошибке
        return -1 # Возвращаем код ошибки


def select_timeframes_interactive(prompt_message="Выберите таймфреймы"):
    print_header(prompt_message)
    print("Доступные таймфреймы:", ", ".join(CORE_TIMEFRAMES_LIST))
    while True:
        selected_tfs_str = input(
            f"Введите таймфреймы через пробел (например, 15m 1h),\n'all' для всех, или 'q' для отмены: ").strip()
        if selected_tfs_str.lower() == 'q':
            return None
        if not selected_tfs_str or selected_tfs_str.lower() == 'all':
            return CORE_TIMEFRAMES_LIST

        selected_tfs = []
        invalid_tfs = []
        for tf_input in selected_tfs_str.split():
            if tf_input in CORE_TIMEFRAMES_LIST:
                selected_tfs.append(tf_input)
            else:
                invalid_tfs.append(tf_input)

        if invalid_tfs:
            print(f"Предупреждение: Некорректные таймфреймы ({', '.join(invalid_tfs)}) будут проигнорированы.")

        if not selected_tfs:
            print("Не выбрано ни одного корректного таймфрейма. Попробуйте еще раз.")
            continue
        # Сортируем выбранные TF для предсказуемого порядка
        selected_tfs.sort(key=lambda x: CORE_TIMEFRAMES_LIST.index(x))
        print(f"Выбраны таймфреймы: {', '.join(selected_tfs)}")
        return selected_tfs


def clear_training_artifacts_interactive():
    print_header("Очистка артефактов обучения")
    confirm = input(
        "ВНИМАНИЕ! Это действие удалит:\n"
        f"  - Все файлы из директории '{MODELS_DIR}/'\n"
        f"  - Все файлы из директории '{LOGS_DIR}/'\n"
        f"  - Файлы 'features_*.pkl' и 'sample_*.csv' из директории '{DATA_FEATURES_DIR}/'\n"
        f"Папка с базой данных '{DATABASE_DIR}/' и файл '{UPDATE_LOG_FILE}' НЕ будут затронуты.\n"
        "Продолжить? (y/n): "
    ).lower()

    if confirm == 'y':
        print("\n🧹  Начинаем очистку...")

        for dir_to_clear, description in [(MODELS_DIR, "моделей"), (LOGS_DIR, "логов")]:
            if os.path.exists(dir_to_clear):
                try:
                    # Удаляем только содержимое
                    for item in os.listdir(dir_to_clear):
                        item_path = os.path.join(dir_to_clear, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    print(f"    Содержимое директории {description} '{dir_to_clear}' очищено.")
                except Exception as e:
                    print(f"    Не удалось очистить содержимое директории {dir_to_clear}: {e}")
            else:
                print(f"    Директория {dir_to_clear} не найдена (пропущено).")
            # Убедимся, что директория существует после очистки содержимого (если она была пустой или не существовала)
            os.makedirs(dir_to_clear, exist_ok=True)

        if os.path.exists(DATA_FEATURES_DIR):
            cleaned_feature_files = 0
            try:
                for item in os.listdir(DATA_FEATURES_DIR):
                    item_path = os.path.join(DATA_FEATURES_DIR, item)
                    if os.path.isfile(item_path):
                        if (item.startswith("features_") and item.endswith(".pkl")) or \
                                (item.startswith("sample_") and item.endswith(".csv")):
                            try:
                                os.remove(item_path)
                                # print(f"    Удален файл: {item_path}") # Можно раскомментировать для детального лога
                                cleaned_feature_files += 1
                            except Exception as e:
                                print(f"    Не удалось удалить файл {item_path}: {e}")
            except Exception as e:
                 print(f"    Ошибка при сканировании директории {DATA_FEATURES_DIR}: {e}")

            if cleaned_feature_files > 0:
                print(f"    Удалено {cleaned_feature_files} файлов признаков/сэмплов из '{DATA_FEATURES_DIR}'.")
            else:
                print(f"    Файлы признаков/сэмплов в '{DATA_FEATURES_DIR}' не найдены или уже очищены.")
        else:
            print(f"    Директория {DATA_FEATURES_DIR} не найдена, пропуск удаления файлов признаков/сэмплов.")

        print("✅  Очистка завершена.")
    else:
        print("Очистка отменена.")


def ensure_base_directories():
    """Создает базовые директории, если их нет."""
    print("Проверка и создание базовых директорий...")
    for dir_path in [DATABASE_DIR, DATA_FEATURES_DIR, MODELS_DIR, LOGS_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"  Создана директория: {dir_path}")
        # else: # Можно закомментировать, чтобы не выводить для существующих папок
        #     print(f"  Директория существует: {dir_path}")
    print("Проверка директорий завершена.")

# --- Главное меню ---
def main_menu():
    ensure_base_directories()

    while True:
        print_header("Главное меню")
        print("--- 📦 Сбор и Обновление Данных ---")
        print("  1. Полная (глубокая) загрузка исторических данных (old_update)")
        print("  2. Инкрементальное обновление свежих данных (mini_update)")
        print("--- ⚙️  Обработка и Обучение Моделей ---")
        print("  3. Построить признаки для выбранных таймфреймов (preprocess_features)")
        print("  4. Обучить модели для выбранных таймфреймов (train_model)")
        print("--- 🚀 Пайплайны (Комбинированные операции) ---")
        print("  5. Пайплайн: Мини-обновление -> Признаки -> Обучение (для выбранных TF)")
        print("  6. Пайплайн: Полное обновление -> Признаки -> Обучение (для выбранных TF)")
        print("  7. ПОЛНЫЙ ПЕРЕСБОР: Очистка -> Полное обновление -> Признаки -> Обучение")
        print("--- 📊 Прогнозирование и Анализ ---")
        print("  8. Сгенерировать прогнозы и торговый план (predict_all --save)")
        print("  9. Запустить бэктест для одного таймфрейма (predict_backtest)")
        print("--- 🛠️  Утилиты ---")
        print(" 10. Проверить доступность GPU для CatBoost (gpu_test)")
        print(" 11. ОЧИСТИТЬ все артефакты обучения (модели, логи, features)")
        print("  0. Выход")

        choice = input("Введите номер опции: ").strip()

        try:
            if choice == '1':
                # Полная загрузка данных - обычно без фильтров по символам
                run_script([PYTHON_EXECUTABLE, "old_update_binance_data.py", "--all"], "Полная загрузка данных")

            elif choice == '2':
                # Инкрементальное обновление - может быть с фильтрами, но в этом меню пока без них
                tfs = select_timeframes_interactive("Инкрементальное обновление")
                if tfs:
                     # Здесь можно было бы добавить запрос символа/группы, но оставим пока как в оригинале
                     run_script([PYTHON_EXECUTABLE, "mini_update_binance_data.py", "--tf"] + tfs,
                               f"Инкрементальное обновление для {', '.join(tfs)}")

            elif choice == '3':
                tfs = select_timeframes_interactive("Построение признаков")
                if tfs:
                    # >>> ИЗМЕНЕНО согласно патчу
                    group_or_symbol = input(
                        f"Введите группу ({'/'.join(SYMBOL_GROUPS)}) или символ (например: BTCUSDT),\n"
                        "или оставьте пустым для всех: "
                    ).strip()
                    group_args = [] # Переименовано в group_args для соответствия патчу, хотя может быть и symbol
                    description_suffix = "для всех"
                    if group_or_symbol:
                        if group_or_symbol.lower() in SYMBOL_GROUPS:
                            group_args = ["--symbol-group", group_or_symbol.lower()]
                            description_suffix = f"для группы {group_or_symbol.lower()}"
                        else:
                            group_args = ["--symbol", group_or_symbol.upper()]
                            description_suffix = f"для символа {group_or_symbol.upper()}"

                    for tf_item in tfs:
                        desc = f"Построение признаков {description_suffix} ({tf_item})"
                        print(f"\n--- {desc} ---")
                        if run_script([PYTHON_EXECUTABLE, "preprocess_features.py", "--tf", tf_item] + group_args, desc) != 0:
                            print(f"{desc} прервано или завершилось с ошибкой. Пропуск остальных таймфреймов.")
                            break # Прерываем цикл по таймфреймам

            elif choice == '4':
                tfs = select_timeframes_interactive("Обучение моделей")
                if tfs:
                    # >>> ИЗМЕНЕНО согласно патчу
                    group_or_symbol = input(
                        f"Введите группу ({'/'.join(SYMBOL_GROUPS)}) или символ (например: BTCUSDT),\n"
                        "или оставьте пустым для всех: "
                    ).strip()
                    symbol_arg = [] # Переименовано в symbol_arg для соответствия патчу, хотя может быть и group
                    description_suffix = "для всех пар"
                    if group_or_symbol:
                         if group_or_symbol.lower() in SYMBOL_GROUPS:
                             symbol_arg = ["--symbol-group", group_or_symbol.lower()]
                             description_suffix = f"для группы {group_or_symbol.lower()}"
                         else:
                             symbol_arg = ["--symbol", group_or_symbol.upper()]
                             description_suffix = f"для символа {group_or_symbol.upper()}"


                    for tf_item in tfs:
                        desc = f"Обучение {description_suffix} ({tf_item})"
                        if run_script([PYTHON_EXECUTABLE, "train_model.py", "--tf", tf_item] + symbol_arg, desc) != 0:
                            print(f"{desc} прервано или завершилось с ошибкой. Пропуск остальных таймфреймов.")
                            break # Прерываем цикл по таймфреймам


            elif choice == '5':
                tfs = select_timeframes_interactive("Пайплайн: Мини-обновление -> Признаки -> Обучение")
                if tfs:
                     # Пайплайны пока без фильтров по символам в этом меню, но train_model и preprocess_features
                     # внутри пайплайна могли бы их использовать, если бы пайплайн их принимал.
                     # Текущий pipeline.py их не принимает, оставляем как есть.
                     run_script([PYTHON_EXECUTABLE, "pipeline.py", "--train", "--skip-predict", "--tf"] + tfs,
                               "Пайплайн (мини-обновление, признаки, обучение)")
            elif choice == '6':
                tfs = select_timeframes_interactive("Пайплайн: Полное обновление -> Признаки -> Обучение")
                if tfs:
                     # Пайплайны пока без фильтров по символам
                    run_script(
                        [PYTHON_EXECUTABLE, "pipeline.py", "--full-update", "--train", "--skip-predict", "--tf"] + tfs,
                        "Пайплайн (полное обновление, признаки, обучение)")

            elif choice == '7':
                tfs = select_timeframes_interactive(
                    "ПОЛНЫЙ ПЕРЕСБОР: Очистка -> Полное обновление -> Признаки -> Обучение")
                if tfs:
                    print_header("Начало ПОЛНОГО ПЕРЕСБОРА")
                    # Очистка
                    clear_training_artifacts_interactive()
                    # Полное обновление и Построение признаков (через pipeline)
                    # Пайплайн predict_all.py не запускает, только preprocess и train (если указано)
                    # Пайплайн train_model может принимать --symbol/--symbol-group, но текущий pipeline.py не передает эти флаги.
                    # Для полного пересбора, вероятно, мы всегда хотим обработать ВСЕ символы, так что фильтр не нужен.
                    if run_script([PYTHON_EXECUTABLE, "pipeline.py", "--full-update", "--train", "--skip-predict", "--tf"] + tfs,
                                  "Этап 1/1 (Полный пересбор): Обновление, Признаки, Обучение") == 0:
                        print_header("ПОЛНЫЙ ПЕРЕСБОР завершен успешно.")
                    else:
                         print("ПОЛНЫЙ ПЕРЕСБОР прерван на одном из этапов.")

            elif choice == '8':

            print_header("Генерация прогнозов")





                            +    group_or_symbol = input(

                                +        "Введите группу (top8/meme/defi) или символ (например: BTCUSDT),\n"


                                +).strip()

                            +    predict_args = ["--save"]

                            +
                            if group_or_symbol:

                            +
                            if group_or_symbol.lower() in SYMBOL_GROUPS:

                            +            predict_args += ["--symbol-group", group_or_symbol.lower()]

                            + else:

            +            predict_args += ["--symbol", group_or_symbol.upper()]

            run_script(

                + [PYTHON_EXECUTABLE, "predict_all.py"] + predict_args,

                +        f"Генерация прогнозов для {group_or_symbol or 'всех пар'}"

            )

            elif choice == '9':
                print_header("Запуск бэктеста")
                print("Доступные таймфреймы:", ", ".join(CORE_TIMEFRAMES_LIST))
                tf_backtest = input(f"Введите таймфрейм для бэктеста (например, 15m) или 'q' для отмены: ").strip()
                if tf_backtest.lower() == 'q':
                    continue
                if tf_backtest in CORE_TIMEFRAMES_LIST:
                     # Бэктест predict_backtest.py может принимать --symbol/--symbol-group
                     # Можно добавить запрос здесь, если требуется фильтрация бэктеста.
                     # Оставляем пока без фильтрации как в оригинале.
                    run_script([PYTHON_EXECUTABLE, "predict_backtest.py", "--tf", tf_backtest],
                               f"Бэктест для {tf_backtest}")
                else:
                    print(f"Некорректный таймфрейм: {tf_backtest}")

            elif choice == '10':
                run_script([PYTHON_EXECUTABLE, "gpu_test.py"], "Проверка GPU")

            elif choice == '11':
                clear_training_artifacts_interactive()

            elif choice == '0':
                print("Выход из программы.")
                sys.exit(0)

            else:
                print("Неверный ввод. Пожалуйста, выберите номер из меню.")

        except KeyboardInterrupt:
            print("\nОперация прервана пользователем (Ctrl+C в меню). Возврат в главное меню.")
            # Дочерние процессы, запущенные через subprocess.run, должны получить сигнал и корректно завершиться
            # Но если они зависнут, может потребоваться ручное завершение.
            continue
        except Exception as e:
            print(f"\nКритическая ошибка в main_cli: {e}")
            import traceback
            traceback.print_exc()
            # Можно не выходить, а вернуться в меню, если ошибка не фатальна
            # sys.exit(1)
            continue # Вернуться в меню после ошибки


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        print(f"\nКритическая ошибка вне меню: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)