@echo off
:: Переход в директорию, где находится main_cli.py (если нужно)
cd /d %~dp0

:: Запуск main_cli.py
python main_cli.py

pause