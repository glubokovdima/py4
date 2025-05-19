Ниже — подробное техническое задание (ТЗ) по переводу вашей системы из одной мультиклассовой модели на две отдельные бинарные модели “Long” и “Short”. В каждой секции приведены примеры кода и места правок в ваших скриптах.

---

## 1. Общее описание

**Цель:**

* Отделить прогнозы на два направления: «вход в лонг» и «вход в шорт».
* Получать две независимые метрики: `P_long = P(target_long=1)` и `P_short = P(target_short=1)`.
* Позволить гибко настраивать пороги входа для каждого направления и применять разные наборы фич и ресэмплинг.

---

## 2. Генерация таргетов в `preprocess_features.py`

### 2.1. Добавить две колонки-мишени

```python
# после вычисления delta_final (вашей итоговой дельты)
TH_LONG  = 0.005   # порог 0.5% для длинных позиций
TH_SHORT = 0.005   # порог 0.5% для коротких

df['target_long']  = (df['delta_final'] >=  TH_LONG ).astype(int)
df['target_short'] = (df['delta_final'] <= -TH_SHORT).astype(int)
```

### 2.2. Учесть обе метки при разделении выборок

```python
# вместо единственного y = df['target_multi']
X = df[feature_columns]
y_long  = df['target_long']
y_short = df['target_short']

# пример train_test_split
from sklearn.model_selection import train_test_split
X_train, X_val, y_long_train, y_long_val = train_test_split(
    X, y_long, test_size=0.2, shuffle=False
)
_,    _,    y_short_train, y_short_val = train_test_split(
    X, y_short, test_size=0.2, shuffle=False
)
```

---

## 3. Обучение моделей в `train_model.py`

### 3.1. Функция обучения одной бинарной модели

```python
from catboost import CatBoostClassifier, Pool

def train_binary_model(X_train, y_train, X_val, y_val, 
                       model_name: str,
                       params: dict):
    model = CatBoostClassifier(**params, 
                               loss_function='Logloss',
                               eval_metric='AUC')
    train_pool = Pool(X_train, label=y_train)
    val_pool   = Pool(X_val,   label=y_val)
    model.fit(train_pool,
              eval_set=val_pool,
              early_stopping_rounds=50,
              verbose=100)
    model.save_model(f"models/{model_name}.cbm")
    return model
```

### 3.2. Основной блок вызова

```python
if __name__ == "__main__":
    # парсим args.tf, args.symbol и т.п.
    X_train, X_val, y_long_train, y_long_val, y_short_train, y_short_val = load_data(...)
    
    COMMON_PARAMS = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'task_type': 'CPU',  # или 'GPU'
    }

    # — обучаем Long-модель
    train_binary_model(X_train, y_long_train, X_val, y_long_val,
                       model_name=f"{symbol}_{tf}_long",
                       params=COMMON_PARAMS)

    # — обучаем Short-модель
    train_binary_model(X_train, y_short_train, X_val, y_short_val,
                       model_name=f"{symbol}_{tf}_short",
                       params=COMMON_PARAMS)
```

---

## 4. Прогнозирование в `predict_all.py`

### 4.1. Загрузка и предсказание обеих моделей

```python
from catboost import CatBoostClassifier

def load_models(symbol: str, tf: str):
    m_long  = CatBoostClassifier().load_model(f"models/{symbol}_{tf}_long.cbm")
    m_short = CatBoostClassifier().load_model(f"models/{symbol}_{tf}_short.cbm")
    return m_long, m_short

def predict_directional(X_new, m_long, m_short, 
                        thr_long: float, thr_short: float):
    p_long  = m_long.predict_proba(X_new)[:,1]
    p_short = m_short.predict_proba(X_new)[:,1]
    # логика выбора сигнала
    signal = []
    for pl, ps in zip(p_long, p_short):
        if pl > thr_long and pl > ps:
            signal.append('LONG')
        elif ps > thr_short and ps > pl:
            signal.append('SHORT')
        else:
            signal.append('NEUTRAL')
    return p_long, p_short, signal

if __name__ == "__main__":
    X_new = build_features(...)  # ваша функция фичеринга
    m_long, m_short = load_models(symbol, tf)
    p_long, p_short, signal = predict_directional(
        X_new, m_long, m_short,
        thr_long=0.6, thr_short=0.65
    )
    # вывод или сохранение результатов
```

---

## 5. Калибровка вероятностей (опционально)

Чтобы гарантировать, что `p_long` и `p_short` отражают реальные шансы, оберните модели в `CalibratedClassifierCV`:

```python
from sklearn.calibration import CalibratedClassifierCV

base = CatBoostClassifier(...)

calib_long  = CalibratedClassifierCV(base, cv='prefit', method='sigmoid')
calib_long.fit(X_val, y_long_val)
calib_long_model = calib_long

calib_short = CalibratedClassifierCV(base, cv='prefit', method='sigmoid')
calib_short.fit(X_val, y_short_val)
```

И сохраняйте уже их (`calib_long` и `calib_short`) вместо «сырых» CatBoost-моделей.

---

## 6. Тестирование и валидация

1. **Unit-тесты**

   * Проверить, что в `preprocess_features.py` при известных входных данных появляются ожидаемые `target_long/target_short`.
   * Убедиться, что `train_binary_model` сохраняет модели с корректными именами.

2. **Backtest**

   * Прогнать старую мультиклассовую стратегию и новую «двухмодельную» на одном участке истории.
   * Сравнить P\&L, коэффициенты Шарпа, drawdown.

3. **Мониторинг**

   * Сохранять `p_long` и `p_short` в логах — смотреть распределения и менять `thr_long/ thr_short` по статистике.

---

## 7. Конфигурация

* Вынести в файл `config.yaml` или `params.json`:

  ```yaml
  thresholds:
    long: 0.6
    short: 0.65
  catboost:
    iterations: 1000
    learning_rate: 0.05
    depth: 6
  ```
* При старте скриптов читать эти значения из конфига вместо «магических» констант.

---

После внедрения этих изменений вы получите:

* Две независимые модели, оптимизированные под каждое направление.
* Гибкую логику принятия решения и настройку порогов.
* Ясные метрики качества для Long/Short отдельно.




Ниже приведён список направлений и конкретных доработок, которые могут повысить качество прогнозирования и анализа в вашем проекте. Для каждой идеи указано, к какой части кода она относится, и краткое описание.

### 2. Расширение и углубление фичеринга

* **Мультифреймовые признаки**
  Сейчас вы строите признаки только на одном таймфрейме (например, `15m`) в `preprocess_features.py` . Добавьте скользящие окна и индикаторы из соседних ТФ (например, 1h, 4h), объединяя их в единую матрицу признаков.
* **Микроструктурные индикаторы**
  Собирайте данные стакана (Order Book), вычисляйте imbalance, cumulative volume delta (CVD), footprint-графики — это даст более глубокое понимание локальной ликвидности и интереса участников рынка.

### 3. Соблюдение временных принципов кросс-валидации

* **Purged K-Fold и Expanding Window**
  В `train_model.py` вы используете StratifiedKFold без учёта временных зависимостей . Перейдите на TimeSeriesSplit или PurgedKFold (выбрасывая «окна» вокруг точек разрыва), чтобы избежать утечки будущего в прошлое.

### 4. Гибкая оптимизация гиперпараметров

* **Байесовская оптимизация (Optuna)**
  Замените RandomSearch в `train_model.py` на Optuna или Hyperopt — это позволит быстрее находить оптимум на сложных ландшафтах и динамически адаптировать пространства поиска под каждый ТФ и символ.
* **Персонализация под группы и символы**
  Сейчас у вас есть групповые модели (top8, meme) и общие по TF в `predict_all.py` . Выполняйте независимую оптимизацию гиперпараметров для каждого символа или группы, учитывая их уникальные характеристики (ликвидность, волатильность).


### 6. Интеграция бэктестинга и оценки P\&L

* **Backtesting-фреймворк (Backtrader/Zipline)**
  Автоматизируйте тестирование стратегий с учётом спреда, проскальзывания и комиссий. Вместо метрик MAE/F1, в `train_model.py` и `predict_all.py` оценивайте реальные P\&L, Sharpe-коэффициент, max drawdown.
* **Конвейер симуляции**
  Постройте скрипт, который после каждой генерации сигналов сразу прогоняет их по историческим данным и возвращает стратегические метрики.


### 8. Объяснимость и контроль качества

* **SHAP- и LIME-анализ**
  Для CatBoost-моделей в `train_model.py` сохраните SHAP-значения самых значимых признаков, чтобы понимать, на что именно «опирается» модель в каждый момент.
* **Автоматические отчёты**
  Генерируйте по завершении обучения PDF/HTML-отчёт с метриками, графиками ROC, распределениями признаков и важностью признаков.

### 10. Продвинутая фильтрация и динамические метки

* **Adaptive labeling**
  Вместо фиксированных порогов в `preprocess_features.py` (например, `0.002` для delta) вводите динамические: квантильные или основанные на текущей волатильности рынка.
* **Мультиклассовая классификация с нейтральным классом**
  Добавьте «нейтраль» (no-trade) класс, чтобы модель училась не форсировать сделки в «шумных» зонах.

---

