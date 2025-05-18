# main_cli.py
import subprocess
import sys
import os
import shutil
import argparse  # Для CORE_TIMEFRAMES, если они будут в отдельном конфиге

# --- Конфигурация ---
PYTHON_EXECUTABLE = sys.executable
# Таймфреймы можно вынести в общий конфигурационный файл или определить здесь
# Для примера, возьмем из mini_update_binance_data.py (но лучше иметь один источник истины)
CORE_TIMEFRAMES_LIST = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

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
        result = subprocess.run(command_list, check=False)
        if result.returncode == 0:
            print(f"✅  Успешно: {description}")
        elif result.returncode == 130:
            print(f"🔶  Прервано пользователем: {description}")
        else:
            print(f"❌  Ошибка (код {result.returncode}): {description}")
        return result.returncode
    except FileNotFoundError:
        print(f"❌  Ошибка: Команда не найдена. Убедитесь, что Python доступен и скрипт '{command_list[1]}' существует.")
        return -1
    except Exception as e:
        print(f"❌  Непредвиденная ошибка при выполнении '{description}': {e}")
        return -1


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
                    for item in os.listdir(dir_to_clear):  # Удаляем содержимое, а не саму папку, если это важно для git
                        item_path = os.path.join(dir_to_clear, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    print(f"    Содержимое директории {description} '{dir_to_clear}' очищено.")
                except Exception as e:
                    print(f"    Не удалось очистить директорию {dir_to_clear}: {e}")
            else:
                print(f"    Директория {dir_to_clear} не найдена (пропущено).")
            os.makedirs(dir_to_clear, exist_ok=True)  # Создаем заново, если была удалена или не существовала

        if os.path.exists(DATA_FEATURES_DIR):
            cleaned_feature_files = 0
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
        else:
            print(f"  Директория существует: {dir_path}")
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
        print("  7. ПОЛНЫЙ ПЕРЕСБОР: Очистка -> Полное обновление -> Признаки -> Обучение")  # Новый пункт
        print("--- 📊 Прогнозирование и Анализ ---")
        print("  8. Сгенерировать прогнозы и торговый план (predict_all --save)")  # Было 7
        print("  9. Запустить бэктест для одного таймфрейма (predict_backtest)")  # Было 8
        print("--- 🛠️  Утилиты ---")
        print(" 10. Проверить доступность GPU для CatBoost (gpu_test)")  # Было 9
        print(" 11. ОЧИСТИТЬ все артефакты обучения (модели, логи, features)")  # Было 10
        print("  0. Выход")

        choice = input("Введите номер опции: ").strip()

        try:
            if choice == '1':
                run_script([PYTHON_EXECUTABLE, "old_update_binance_data.py", "--all"], "Полная загрузка данных")

            elif choice == '2':
                tfs = select_timeframes_interactive("Инкрементальное обновление")
                if tfs:
                    run_script([PYTHON_EXECUTABLE, "mini_update_binance_data.py", "--tf"] + tfs,
                               f"Инкрементальное обновление для {', '.join(tfs)}")

            elif choice == '3':
                tfs = select_timeframes_interactive("Построение признаков")
                if tfs:
                    for tf_item in tfs:
                        print(f"\n--- Построение признаков для {tf_item} ---")
                        if run_script([PYTHON_EXECUTABLE, "preprocess_features.py", "--tf", tf_item],
                                      f"Признаки для {tf_item}") != 0:
                            print(
                                f"Построение признаков для {tf_item} прервано или завершилось с ошибкой. Пропуск остальных.")
                            break
            elif choice == '4':
                tfs = select_timeframes_interactive("Обучение моделей")
                if tfs:
                    # Очистку перед обучением убрали отсюда, т.к. она должна быть раньше или как отдельная опция
                    for tf_item in tfs:
                        print(f"\n--- Обучение моделей для {tf_item} ---")
                        if run_script([PYTHON_EXECUTABLE, "train_model.py", "--tf", tf_item],
                                      f"Обучение для {tf_item}") != 0:
                            print(f"Обучение для {tf_item} прервано или завершилось с ошибкой. Пропуск остальных.")
                            break
            elif choice == '5':
                tfs = select_timeframes_interactive("Пайплайн: Мини-обновление -> Признаки -> Обучение")
                if tfs:
                    run_script([PYTHON_EXECUTABLE, "pipeline.py", "--train", "--skip-predict", "--tf"] + tfs,
                               "Пайплайн (мини-обновление, признаки, обучение)")
            elif choice == '6':
                tfs = select_timeframes_interactive("Пайплайн: Полное обновление -> Признаки -> Обучение")
                if tfs:
                    run_script(
                        [PYTHON_EXECUTABLE, "pipeline.py", "--full-update", "--train", "--skip-predict", "--tf"] + tfs,
                        "Пайплайн (полное обновление, признаки, обучение)")

            elif choice == '7':  # Новый пункт - ПОЛНЫЙ ПЕРЕСБОР
                tfs = select_timeframes_interactive(
                    "ПОЛНЫЙ ПЕРЕСБОР: Очистка -> Полное обновление -> Признаки -> Обучение")
                if tfs:
                    print_header("Начало ПОЛНОГО ПЕРЕСБОРА")
                    clear_training_artifacts_interactive()  # 1. Очистка
                    # 2. Полное обновление (через pipeline с флагом full-update, но без train на этом этапе)
                    if run_script([PYTHON_EXECUTABLE, "pipeline.py", "--full-update", "--skip-predict", "--tf"] + tfs,
                                  "Этап 1/2 (Полный пересбор): Полное обновление и Построение признаков") == 0:
                        # 3. Обучение (запускаем train_model.py отдельно для выбранных ТФ)
                        print_header("Этап 2/2 (Полный пересбор): Обучение моделей")
                        for tf_item in tfs:
                            if run_script([PYTHON_EXECUTABLE, "train_model.py", "--tf", tf_item],
                                          f"Обучение для {tf_item}") != 0:
                                print(
                                    f"Обучение для {tf_item} (в рамках полного пересбора) прервано. Пропуск остальных.")
                                break
                    else:
                        print("Полный пересбор прерван на этапе обновления/построения признаков.")


            elif choice == '8':  # Было 7
                run_script([PYTHON_EXECUTABLE, "predict_all.py", "--save"], "Генерация прогнозов")

            elif choice == '9':  # Было 8
                print_header("Запуск бэктеста")
                print("Доступные таймфреймы:", ", ".join(CORE_TIMEFRAMES_LIST))
                tf_backtest = input(f"Введите таймфрейм для бэктеста (например, 15m) или 'q' для отмены: ").strip()
                if tf_backtest.lower() == 'q':
                    continue
                if tf_backtest in CORE_TIMEFRAMES_LIST:
                    run_script([PYTHON_EXECUTABLE, "predict_backtest.py", "--tf", tf_backtest],
                               f"Бэктест для {tf_backtest}")
                else:
                    print(f"Некорректный таймфрейм: {tf_backtest}")

            elif choice == '10':  # Было 9
                run_script([PYTHON_EXECUTABLE, "gpu_test.py"], "Проверка GPU")

            elif choice == '11':  # Было 10
                clear_training_artifacts_interactive()

            elif choice == '0':
                print("Выход из программы.")
                sys.exit(0)

            else:
                print("Неверный ввод. Пожалуйста, выберите номер из меню.")

        except KeyboardInterrupt:
            print("\nОперация прервана пользователем (Ctrl+C в меню). Возврат в главное меню.")
            continue


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\nПрограмма завершена пользователем (Ctrl+C).")
        sys.exit(0)
    except Exception as e:
        print(f"\nКритическая ошибка в main_cli: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)