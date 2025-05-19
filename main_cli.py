# run_cli.py (в корне проекта)
import sys
import os

# Добавляем корень проекта в PYTHONPATH, чтобы Python мог найти пакет 'core'
# Это особенно важно, если вы запускаете скрипт не из корня проекта.
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    # Также можно добавить core, если импорты внутри core используют относительные пути от корня
    # sys.path.insert(0, os.path.join(project_root, "core"))


# Теперь можно импортировать из core
from core.cli.main import main_cli_entry_point

if __name__ == "__main__":
    main_cli_entry_point()
