# Обновлённое Техническое Задание

## Содержание

1. [Общая архитектура и оркестрация](#architecture)
2. [Сбор и хранение данных](#data-ingestion)
3. [Feature Engineering](#feature-engineering)
4. [Обучение моделей](#model-training)
5. [Оптимизация гиперпараметров](#hyperparameter-tuning)
6. [Ансамблирование и расширение архитектур](#ensembling)
7. [Бэктестинг и оценка P\&L](#backtesting)
8. [Прогнозирование и логика принятия решений](#prediction)
9. [Мониторинг и автоматическое дообучение](#monitoring)
10. [Объяснимость и отчётность](#explainability)
11. [Качество кода и CI/CD](#code-quality)
12. [Продвинутая фильтрация и динамические метки](#advanced-labeling)
13. [Поэтапный план внедрения](#roadmap)

---

<a name="architecture"></a>

## 1. Общая архитектура и оркестрация

* **old\_update\_binance\_data.py / mini\_update\_binance\_data.py**: загрузка исторических и свежих данных (параллельно по символам, надежная работа с лимитами API).
* **SQLite DB** (`database/market_data.db`): табличная структура `candles_<tf>` с {`symbol`, `timestamp`, OHLCV}.
* **pipeline.py / main\_cli.py**: единый CLI для запуска загрузки, построения признаков, обучения, прогноза и бэктеста.
* **Выборка символов**: `select_top_symbols.py` для автоматического отбора по ликвидности и волатильности.

---

<a name="data-ingestion"></a>

## 2. Сбор и хранение данных

1. **Исторические свечи**: incremental vs full-update, управление ретрай-логикой, ThreadPoolExecutor.
2. **Хранилище**: SQLite, оптимизация индексов, защита от дублирующихся записей.
3. **Расширение**: возможность подключения Order Book для микроструктурных индикаторов.

---

<a name="feature-engineering"></a>

## 3. Feature Engineering (`preprocess_features.py`)

1. **Мульти-TF признаки**: агрегация 1m–15m + 1h, 4h.
2. **Технические индикаторы**: RSI, EMA(20/50), MACD, OBV, ATR, Bollinger Bands, GARCH-volatility.
3. **Лаг- и разворотные фичи**: pin-bar, doji, EMA-slope, RSI-cross, объемные спайки.
4. **Rolling-статистика**: mean/std по окнам 5/10/20, последовательные тренды.
5. **Adaptive labeling**: динамические пороги на основе квантилей и текущей волатильности.
6. **Фильтрация флэта**: отброс участков с низкой ATR и объемом.
7. **Внешние фичи**: признаки BTCUSDT (for cross-asset context).
8. **Конфигурируемость**: все константы (`TARGET_SHIFT`, пороги) вынести в YAML/JSON.

---

<a name="model-training"></a>

## 4. Обучение моделей (`train_model.py`)

1. **Два бинарных классификатора**: `target_long`, `target_short` вместо мультикласса.
2. **Регрессоры**: CatBoostRegressor для `delta` и `volatility`.
3. **ТП-Hit классификатор**: отдельная модель для успешного достижения TP.
4. **Временные CV**: `TimeSeriesSplit` / `PurgedKFold` вместо `StratifiedKFold`.
5. **Feature selection**: OOF-важность + SHAP + permutation importance → топ‑20.
6. **Стратегия разбиения**: train/test (15%) + CV pool.
7. **Class weights и балансировка**: inverse frequency, undersample/oversample (SMOTE).

---

<a name="hyperparameter-tuning"></a>

## 5. Оптимизация гиперпараметров

* **Bayesian Optimization**: Optuna / Hyperopt вместо RandomSearch.
* **Персонализация**: независимый поиск для каждого символа или группы.
* **Динамика пространства**: адаптация диапазонов под результаты проб.

---

<a name="ensembling"></a>

## 6. Ансамблирование и расширение архитектур

* **Stacking/Blending**: CatBoost + LightGBM + XGBoost + CNN/LSTM на изображениях свечек.
* **Multitask Neural Network**: единая модель для направления, дельты и волатильности.

---

<a name="backtesting"></a>

## 7. Бэктестинг и оценка P\&L

1. **Интеграция**: Backtrader/Zipline для учёта спреда, проскальзывания, комиссий.
2. **Метрики**: Sharpe, max drawdown, P\&L вместо MAE/F1.
3. **Pipeline**: автоматический сквозной запуск прогноза → бэктест → отчёт.

---

<a name="prediction"></a>

## 8. Прогнозирование и логика принятия решений (`predict_all.py`)

1. **Fallback загрузка моделей**: персональная → групповая → общая.
2. **Decision Logic**:

   * агрегирование сигналов multi‑TF (vote/scoring),
   * комбинирование `delta_model` и `delta_history`,
   * сигнал strength + conflict detection,
   * фильтрация по confidence, σHist и TP-hit probability.
3. **Trade plan & Alerts**: таблицы, CSV, `alerts.txt`, консольный вывод (tabulate) и топ‑сигналы.

---

<a name="monitoring"></a>

## 9. Мониторинг и автоматическое дообучение

* **Drift Detection**: контроль mean/std признаков и метрик.
* **Автоматические задачи**: `automations.create` для перетренировки на триггерах.
* **CI/CD**: Docker + MLflow/DVC, versioning моделей.

---

<a name="explainability"></a>

## 10. Объяснимость и отчётность

* **SHAP**/LIME в конце обучения → визуализация наиболее значимых фич.
* **Автоматические отчёты**: PDF/HTML с метриками, ROC, распределениями и важностью.

---

<a name="code-quality"></a>

## 11. Качество кода и CI/CD

* **Модульность**: разбиение монолитных скриптов на функции и модули.
* **Unit-тесты**: pytest для ключевых функций.
* **Конфиг**: YAML/JSON для порогов и параметров.
* **Документация**: docstrings, README, шаблоны Runbooks.

---

<a name="advanced-labeling"></a>

## 12. Продвинутая фильтрация и динамические метки

* **Adaptive labeling**: пороги на основе волатильности и квантилей.
* **Neutral class**: дополнительный класс `no-trade` в мультиклассе или бинарный подход.

---

<a name="roadmap"></a>

## 13. Поэтапный план внедрения

1. **Анализ признаков**: очистка «шумных» фич → OOF-важность + SHAP.
2. **Учет частоты сигналов**: балансировка классов, class\_weights, SMOTE/undersample.
3. **Настройка обучения**: learning\_rate 0.03, early\_stopping, подбор числа итераций.
4. **Проверка на отложенной выборке**: последние 15% истории.
5. **Расширенные фичи и модели**: multi‑TF, microstructure, ансамбли, CV с учётом времени.

---

*Документ подготовлен на основе текущей кодовой базы и предложенного ТЗ.*

---

## 14. Рефакторинг структуры проекта

**Цель:** разделить кодовую базу на логические модули, улучшить поддержку и переиспользование без изменения бизнес-логики.

### 14.1. Общие требования

1. Все исходные скрипты и модули перенести в папку `core/` в корне репозитория.
2. В `core/` создать файл `config.yaml` для всех настроек, констант и путей.
3. Переиспользуемые утилиты и функции вынести в `core/helpers/`.
4. Скрипты работы с загрузкой данных — в `core/data_ingestion/`.
5. Скрипты фичеринга — в `core/features/`.
6. Скрипты обучения моделей — в `core/training/`.
7. Скрипты генерации прогнозов — в `core/prediction/`.
8. CLI-энптрипойнты и батники — в `core/cli/`.
9. Оркестрационный пайплайн — в `core/pipeline/`.
10. Функции построения графиков (если имеются) — в `core/visualization/`.

### 14.2. Древовидная структура директорий

```
project_root/
└── core/
    ├── config.yaml
    ├── helpers/
    │   └── utils.py
    ├── data_ingestion/
    │   ├── old_update_binance_data.py
    │   └── select_top_symbols.py
    ├── features/
    │   ├── preprocess_features.py
    │   └── build_features.py
    ├── training/
    │   └── train_model.py
    ├── prediction/
    │   └── predict_all.py
    ├── cli/
    │   ├── main_cli.py
    │   └── run_cli.bat
    ├── pipeline/
    │   └── pipeline.py
    └── visualization/
        └── plotting.py  # (если есть код отрисовки)
```

### 14.3. Краткое описание содержимого файлов

* **config.yaml**: параметры Binance API, пути к БД, таймфреймы, пороги сигналов, гиперпараметры моделей.
* **helpers/utils.py**: вспомогательные функции (чтение/запись, логирование, преобразование форматов).
* **data\_ingestion/old\_update\_binance\_data.py**: логика incremental и full-update запроса свечей.
* **data\_ingestion/select\_top\_symbols.py**: алгоритм отбора символов по ликвидности и волатильности.
* **features/preprocess\_features.py**: расчёт технических индикаторов и формирование набора признаков.
* **features/build\_features.py**: объединение, фильтрация и сохранение итогового feature set.
* **training/train\_model.py**: загрузка данных, настройка CV, обучение и сохранение CatBoost/других моделей.
* **prediction/predict\_all.py**: загрузка моделей, построение прогнозов, применение логики сигналов.
* **cli/main\_cli.py**: парсинг аргументов командной строки и вызов соответствующих модулей.
* **cli/run\_cli.bat**: Windows-скрипт для запуска `main_cli.py`.
* **pipeline/pipeline.py**: оркестрация полного пайплайна: от загрузки до бэктеста.
* **visualization/plotting.py**: функции для отрисовки графиков и аннотаций сигналов (при наличии).

*Эти изменения позволят сохранить существующую бизнес-логику, но улучшат модульность, читаемость и поддержку кода.*
