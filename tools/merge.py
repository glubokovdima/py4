import os

# --- НАСТРОЙКИ (задаются здесь) ---
FOLDER_TO_SCAN = "./core"  # Путь к папке, которую нужно сканировать
OUTPUT_FILE_NAME = "merged.txt"  # Имя выходного файла
INCLUDE_EXTENSIONS = ['.py']  # Список расширений для включения (в нижнем регистре, с точкой).
# Если None или пустой список, включаются все файлы.
EXCLUDE_FILES_LIST = ['.gitignore', 'merge.py', 'merged.txt', '__pycache__']  # Список файлов и папок для исключения (регистронезависимо).


# Добавил '__pycache__' как типичную папку для исключения


# ------------------------------------


def combine_files_to_txt_recursive(folder_path, output_file_path, extensions=None, exclude_items=None):
    """
    Объединяет содержимое файлов из указанной папки и всех ее подпапок в один текстовый файл.
    Каждый блок в выходном файле будет содержать относительный путь к файлу и его содержимое.

    Args:
        folder_path (str): Путь к папке с файлами.
        output_file_path (str): Путь к выходному текстовому файлу.
        extensions (list, optional): Список расширений файлов для включения.
        exclude_items (list, optional): Список имен файлов или папок для исключения.
    """
    if not os.path.isdir(folder_path):
        print(f"Ошибка: Папка '{folder_path}' не найдена.")
        return

    # Приводим имена исключаемых элементов к нижнему регистру для регистронезависимого сравнения
    exclude_items_lower = [name.lower() for name in (exclude_items or [])]

    processed_extensions = None
    if extensions:
        processed_extensions = [ext.lower() for ext in extensions]
        print(f"Будут обработаны файлы с расширениями: {processed_extensions}")

    if exclude_items_lower:
        print(f"Будут исключены файлы/папки (регистронезависимо): {exclude_items_lower}")

    all_content = []
    file_separator = "\n" + "=" * 80 + "\n"  # Разделитель между файлами

    print(f"\nНачало рекурсивного сканирования папки: {os.path.abspath(folder_path)}")

    # os.walk() проходит по дереву каталогов
    # dirpath - текущая директория
    # dirnames - список поддиректорий в dirpath
    # filenames - список файлов в dirpath
    for dirpath, dirnames, filenames in os.walk(folder_path):
        # Модифицируем dirnames на месте, чтобы исключить нежелательные директории из дальнейшего обхода
        # Важно: нужно изменять список dirnames, а не создавать новый
        # Используем обратный цикл для безопасного удаления элементов
        for i in range(len(dirnames) - 1, -1, -1):
            if dirnames[i].lower() in exclude_items_lower:
                print(f"  Исключение директории из обхода: {os.path.join(dirpath, dirnames[i])}")
                del dirnames[i]

        # Сортируем имена файлов и папок для предсказуемого порядка
        dirnames.sort()
        filenames.sort()

        print(f"  Сканирование директории: {dirpath}")

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            relative_file_path = os.path.relpath(file_path, folder_path)  # Путь относительно корневой папки сканирования

            # 1. Проверка на исключение файла по имени
            if filename.lower() in exclude_items_lower:
                print(f"    Пропуск файла (в списке исключений): {relative_file_path}")
                continue

            # 2. Проверка расширения, если указано
            if processed_extensions:
                file_ext = os.path.splitext(filename)[1].lower()
                if file_ext not in processed_extensions:
                    # print(f"    Пропуск файла (неверное расширение): {relative_file_path}") # Можно раскомментировать для подробного лога
                    continue

            print(f"    Чтение файла: {relative_file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                file_block = f"--- Файл: {relative_file_path} ---\n\n{content}\n"
                all_content.append(file_block)
            except Exception as e:
                print(f"      Ошибка при чтении файла '{relative_file_path}': {e}")

    if not all_content:
        print("\nНе найдено подходящих файлов для объединения (с учетом фильтров и исключений).")
        return

    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write(file_separator.join(all_content))
        print(f"\nВсе подходящие файлы успешно объединены в: {os.path.abspath(output_file_path)}")
    except Exception as e:
        print(f"Ошибка при записи в выходной файл '{output_file_path}': {e}")


if __name__ == "__main__":
    final_extensions = None
    if INCLUDE_EXTENSIONS:
        final_extensions = []
        for ext_item in INCLUDE_EXTENSIONS:
            ext_item_lower = ext_item.lower()
            if not ext_item_lower.startswith('.'):
                final_extensions.append('.' + ext_item_lower)
            else:
                final_extensions.append(ext_item_lower)

    # Обновляем имя функции и параметр exclude_files на exclude_items
    combine_files_to_txt_recursive(FOLDER_TO_SCAN,
                                   OUTPUT_FILE_NAME,
                                   extensions=final_extensions,
                                   exclude_items=EXCLUDE_FILES_LIST)
