import os

# --- НАСТРОЙКИ (задаются здесь) ---
FOLDER_TO_SCAN = "./"  # Путь к папке, которую нужно сканировать
OUTPUT_FILE_NAME = "merged.txt"  # Имя выходного файла
INCLUDE_EXTENSIONS = ['.py', '.md', '.txt']  # Список расширений для включения (в нижнем регистре, с точкой).
# Если None или пустой список, включаются все файлы.
EXCLUDE_FILES_LIST = ['.gitignore', 'merge.py', 'merged.txt']  # Список файлов для исключения (регистронезависимо).


# ------------------------------------


def combine_files_to_txt(folder_path, output_file_path, extensions=None, exclude_files=None):
    """
    Объединяет содержимое файлов из указанной папки в один текстовый файл.
    Каждый блок в выходном файле будет содержать имя исходного файла и его содержимое.

    Args:
        folder_path (str): Путь к папке с файлами.
        output_file_path (str): Путь к выходному текстовому файлу.
        extensions (list, optional): Список расширений файлов для включения.
        exclude_files (list, optional): Список имен файлов для исключения.
    """
    if not os.path.isdir(folder_path):
        print(f"Ошибка: Папка '{folder_path}' не найдена.")
        return

    # Приводим имена исключаемых файлов к нижнему регистру для регистронезависимого сравнения
    # и делаем копию, чтобы не изменять оригинальный список (если он None)
    exclude_files_lower = [name.lower() for name in (exclude_files or [])]

    # Если extensions задан, также приводим к нижнему регистру
    processed_extensions = None
    if extensions:
        processed_extensions = [ext.lower() for ext in extensions]
        print(f"Будут обработаны файлы с расширениями: {processed_extensions}")

    if exclude_files_lower:
        print(f"Будут исключены файлы (регистронезависимо): {exclude_files_lower}")

    all_content = []
    file_separator = "\n" + "=" * 80 + "\n"  # Разделитель между файлами

    print(f"Сканирование папки: {folder_path}")

    for item_name in sorted(os.listdir(folder_path)):
        item_path = os.path.join(folder_path, item_name)

        if os.path.isfile(item_path):
            # 1. Проверка на исключение файла
            if item_name.lower() in exclude_files_lower:
                print(f"  Пропуск файла (в списке исключений): {item_name}")
                continue

            # 2. Проверка расширения, если указано
            if processed_extensions:  # Используем processed_extensions
                file_ext = os.path.splitext(item_name)[1].lower()
                if file_ext not in processed_extensions:
                    print(f"  Пропуск файла (неверное расширение): {item_name}")
                    continue

            print(f"  Чтение файла: {item_name}")
            try:
                with open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                file_block = f"--- Файл: {item_name} ---\n\n{content}\n"
                all_content.append(file_block)
            except Exception as e:
                print(f"    Ошибка при чтении файла '{item_name}': {e}")
        else:
            print(f"  Пропуск (это папка): {item_name}")

    if not all_content:
        print("Не найдено подходящих файлов для объединения (с учетом фильтров и исключений).")
        return

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write(file_separator.join(all_content))
        print(f"\nВсе подходящие файлы успешно объединены в: {output_file_path}")
    except Exception as e:
        print(f"Ошибка при записи в выходной файл '{output_file_path}': {e}")


if __name__ == "__main__":
    # Проверяем, что заданные расширения (если есть) корректны (начинаются с точки)
    # Это не строго обязательно, так как сравнение идет по os.path.splitext(...)[1].lower(),
    # но для консистентности в INCLUDE_EXTENSIONS лучше хранить их с точкой.
    final_extensions = None
    if INCLUDE_EXTENSIONS:
        final_extensions = []
        for ext_item in INCLUDE_EXTENSIONS:
            ext_item_lower = ext_item.lower()
            if not ext_item_lower.startswith('.'):
                final_extensions.append('.' + ext_item_lower)
            else:
                final_extensions.append(ext_item_lower)

    # Вызываем основную функцию с заданными параметрами
    combine_files_to_txt(FOLDER_TO_SCAN,
                         OUTPUT_FILE_NAME,
                         extensions=final_extensions,
                         exclude_files=EXCLUDE_FILES_LIST)