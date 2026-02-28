# Автоматическое выявление трендов и закономерностей на рынках на основе анализа открытых новостных данных

Выпускная квалификационная работа (ВКР)

## Описание

Система автоматически собирает новостные данные, анализирует тональность публикаций, выявляет тематические тренды и оценивает их влияние на движение финансовых рынков. Результаты отображаются в интерактивном веб-дашборде.

**Ключевые возможности:**
- Анализ тональности новостей (FinBERT / RoBERTa)
- Тематическая кластеризация трендов (BERTopic)
- Композитный индекс влияния новости на рынок
- Корреляционный анализ с ценами акций (Event Study, лаговая корреляция)
- Интерактивный дашборд (Streamlit)
- Поддержка трёх источников данных: демо, исторические (DJIA 2008–2016), реальные (NewsAPI)

---

## Структура проекта

```
vkr-ermakova/
├── data/
│   └── stocknews/
│       └── Combined_News_DJIA.csv      # Исторический датасет DJIA
├── src/
│   ├── analysis/
│   │   ├── market_trend_analyzer.py    # Главный модуль анализа
│   │   ├── sentiment_analyzer.py       # Анализ тональности (FinBERT)
│   │   ├── trend_clusterer.py          # Кластеризация трендов (BERTopic)
│   │   ├── influence_index.py          # Композитный индекс влияния
│   │   ├── correlation_analyzer.py     # Корреляция с ценами акций
│   │   └── temporal_analyzer.py        # Временной анализ трендов
│   ├── data_collection/
│   │   ├── news_collector.py           # Сбор новостей (NewsAPI + RSS)
│   │   └── stock_collector.py          # Сбор цен акций (yfinance)
│   ├── nlp_processing/
│   │   ├── __init__.py
│   │   └── text_preprocessor.py        # Предобработка текста
│   └── visualization/
│       ├── dashboard.py                # Streamlit-приложение
│       ├── components.py               # Компоненты UI
│       ├── charts.py                   # Графики Plotly
│       ├── data_loader.py              # Загрузка данных для дашборда
│       └── config.py                   # Конфигурация визуализации
├── tests/                              # Тесты pytest
├── reports/
│   └── figures/                        # Графики EDA (генерируются автоматически)
├── eda_djia.py                         # Разведочный анализ датасета DJIA
├── app.py                              # Точка входа дашборда
├── requirements.txt
└── requirements-dev.txt
```

---

## Установка

### 1. Клонировать репозиторий

```bash
git clone <url>
cd vkr-ermakova
```

### 2. Создать виртуальное окружение

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Установить зависимости

```bash
pip install -r requirements.txt
```

> **Примечание:** PyTorch (`torch`) устанавливается автоматически. Если нужна версия с поддержкой GPU, установите вручную с [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Настроить переменные окружения (опционально)

Создайте файл `.env` в корне проекта:

```
NEWS_API_KEY=ваш_ключ_newsapi
```

Получить ключ NewsAPI можно бесплатно на [newsapi.org](https://newsapi.org).

---

## Запуск

### Дашборд

```bash
streamlit run app.py
```

Откроется в браузере по адресу `http://localhost:8501`.

### Разведочный анализ данных (EDA)

```bash
python eda_djia.py
```

Графики сохраняются в `reports/figures/`.

---

## Источники данных

| Режим | Описание | Требования |
|-------|----------|------------|
| Демо-данные | Встроенные тестовые новости | Не требуется |
| Исторические (DJIA) | 1989 торговых дней, 2008–2016 | Файл `data/stocknews/Combined_News_DJIA.csv` |
| Реальные (NewsAPI) | Актуальные новости | API-ключ NewsAPI |

Датасет DJIA: [Kaggle — Daily News for Stock Market Prediction](https://www.kaggle.com/datasets/aaron7sun/stocknews)

---

## Тестирование

```bash
# Все тесты
pytest tests/ -v

# Без медленных тестов (без загрузки NLP-модели)
pytest tests/ -v -m "not slow"

# С отчётом о покрытии
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Композитный индекс влияния

Индекс рассчитывается по формуле из литературного обзора:

```
Influence = 0.3 × Интенсивность + 0.4 × Тональность + 0.2 × Виральность + 0.1 × Авторитетность
```

| Компонент | Вес | Описание |
|-----------|-----|----------|
| Интенсивность | 0.3 | Нормированное число упоминаний |
| Тональность | 0.4 | Оценка FinBERT/RoBERTa в диапазоне [-1, 1] |
| Виральность | 0.2 | Репосты и лайки (если доступны) |
| Авторитетность | 0.1 | Вес источника (Reuters=1.0, блог=0.2) |

---

## Используемые модели

| Задача | Модель |
|--------|--------|
| Анализ тональности | `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis` |
| Альтернатива | `ProsusAI/finbert` |
| Кластеризация | BERTopic + sentence-transformers + UMAP + HDBSCAN |

---

## Требования к системе

- Python 3.9+
- RAM: минимум 8 ГБ (16 ГБ рекомендуется для NLP-моделей)
- Место на диске: ~5 ГБ (модели загружаются автоматически при первом запуске)
- GPU: опционально, ускоряет анализ тональности в 5–10 раз