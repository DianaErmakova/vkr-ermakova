"""
conftest.py — общие pytest-фикстуры для всего тест-сьюта.

Фикстуры автоматически доступны во всех тестовых файлах
без явного импорта.

Структура:
    - Лёгкие фикстуры (без NLP-моделей): создаются заново для каждого теста
    - Тяжёлые фикстуры (NLP-модели): scope="session" — загружаются один раз
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Добавляем src в PYTHONPATH чтобы все модули были доступны
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT_DIR, 'src')

for path in (ROOT_DIR, SRC_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_configure(config):
    """Регистрируем кастомные маркеры."""
    config.addinivalue_line(
        "markers",
        "slow: тест выполняется долго (загрузка NLP-модели, обращение к API)"
    )


# NLP-фикстуры (scope=session — модель загружается один раз)

@pytest.fixture(scope="session")
def sentiment_analyzer():
    """
    Анализатор тональности FinBERT/RoBERTa.
    Загружается один раз на всю тестовую сессию (~30 сек при первом запуске).
    Пометьте медленные тесты декоратором @pytest.mark.slow.
    """
    from analysis.sentiment_analyzer import SentimentAnalyzer
    return SentimentAnalyzer(model_name="distilroberta-financial")


# Данные DJIA

@pytest.fixture(scope="session")
def djia_data():
    """
    Датасет DJIA (Combined_News_DJIA.csv).
    Загружается один раз на сессию.
    Требует наличия файла data/stocknews/Combined_News_DJIA.csv.
    """
    from visualization.data_loader import load_djia_data
    df = load_djia_data()
    if df.empty:
        pytest.skip("Датасет DJIA не найден: data/stocknews/Combined_News_DJIA.csv")
    return df


# Influence Index

@pytest.fixture
def influence_calculator():
    """InfluenceIndexCalculator с весами по умолчанию."""
    from analysis.influence_index import InfluenceIndexCalculator
    return InfluenceIndexCalculator()


@pytest.fixture
def sample_influence_items():
    """
    Набор тестовых новостей для тестов InfluenceIndexCalculator.
    Содержит разные комбинации компонентов индекса.
    """
    return [
        {
            'title': 'Tesla announces record quarterly earnings',
            'mentions_count': 150,
            'max_mentions': 200,
            'min_mentions': 1,
            'sentiment_score': 0.85,
            'spread_data': {
                'retweets': 5000, 'likes': 15000,
                'comments': 800, 'time_window': 6
            },
            'source': 'Reuters',
            'date': '2024-02-15',
        },
        {
            'title': 'Apple faces antitrust investigation in EU',
            'mentions_count': 80,
            'max_mentions': 200,
            'min_mentions': 1,
            'sentiment_score': -0.65,
            'spread_data': {
                'retweets': 2000, 'likes': 5000,
                'comments': 1200, 'time_window': 12
            },
            'source': 'Bloomberg',
            'date': '2024-02-15',
        },
        {
            'title': 'Microsoft cloud revenue beats expectations',
            'mentions_count': 60,
            'max_mentions': 200,
            'min_mentions': 1,
            'sentiment_score': 0.45,
            'spread_data': {
                'retweets': 800, 'likes': 2000,
                'comments': 300, 'time_window': 24
            },
            'source': 'CNBC',
            'date': '2024-02-14',
        },
        {
            'title': 'Small cap stock shows unusual activity',
            'mentions_count': 5,
            'max_mentions': 200,
            'min_mentions': 1,
            'sentiment_score': 0.1,
            'spread_data': {
                'retweets': 10, 'likes': 25,
                'comments': 2, 'time_window': 48
            },
            'source': 'Unknown Blog',
            'date': '2024-02-14',
        },
    ]


# Текстовые данные

@pytest.fixture
def sample_news_texts():
    """
    Репрезентативная выборка финансовых новостей для тестов
    SentimentAnalyzer и TextPreprocessor.
    """
    return [
        "Tesla stock surges after record quarterly earnings beat expectations",
        "Apple announces groundbreaking new AI technology for iPhone",
        "Federal Reserve raises interest rates amid inflation concerns",
        "Company faces lawsuit over environmental violations in California",
        "Microsoft cloud revenue exceeds Wall Street forecasts this quarter",
        "Market crash leaves investors with significant losses overnight",
        "Amazon AWS dominates cloud computing services market globally",
        "Nvidia reports strong demand for AI chips, stock hits new high",
        "Regulatory challenges threaten big tech companies in Europe",
        "Bitcoin reaches record high price as institutional adoption grows",
    ]


# Временные ряды и рыночные данные

@pytest.fixture
def sample_stock_data():
    """
    Синтетические данные по акции (100 торговых дней).
    Содержит Close, returns, direction — как у CorrelationAnalyzer.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='B')
    returns = np.random.randn(100) * 0.02
    price = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'Close': price,
        'Open': price * (1 + np.random.randn(100) * 0.005),
        'High': price * (1 + np.abs(np.random.randn(100)) * 0.01),
        'Low': price * (1 - np.abs(np.random.randn(100)) * 0.01),
        'Volume': np.random.randint(1_000_000, 10_000_000, 100),
        'returns': returns,
        'direction': (returns > 0).astype(int),
    }, index=dates)

    return df


@pytest.fixture
def sample_news_features():
    """
    Агрегированные новостные признаки по дням (выход prepare_news_features).
    Используется в тестах CorrelationAnalyzer.
    """
    np.random.seed(0)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='B')

    df = pd.DataFrame({
        'date': dates,
        'news_count': np.random.randint(1, 20, 100),
        'sentiment_score': np.random.randn(100) * 0.3,
    })
    return df


@pytest.fixture
def sample_trend_data():
    """
    Временной ряд упоминаний тренда (100 дней).
    Используется в тестах TemporalAnalyzer.
    Паттерн: быстрый рост — пик — затухание.
    """
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    mentions = np.concatenate([
        np.linspace(5, 100, 30),
        np.linspace(100, 15, 40),
        np.ones(30) * 15,
    ])
    return pd.DataFrame({
        'date': dates,
        'mentions': mentions,
        'sentiment': np.random.randn(100) * 0.2,
    })


# TextPreprocessor

@pytest.fixture
def text_preprocessor():
    """TextPreprocessor в режиме clustering (по умолчанию для большинства тестов)."""
    from nlp_processing.text_preprocessor import get_clustering_preprocessor
    return get_clustering_preprocessor()


@pytest.fixture
def sentiment_preprocessor():
    """TextPreprocessor в режиме sentiment."""
    from nlp_processing.text_preprocessor import get_sentiment_preprocessor
    return get_sentiment_preprocessor()


# Корреляционный анализатор

@pytest.fixture
def correlation_analyzer():
    """CorrelationAnalyzer без зависимостей."""
    from analysis.correlation_analyzer import CorrelationAnalyzer
    return CorrelationAnalyzer()


# Временной анализатор

@pytest.fixture
def temporal_analyzer():
    """TemporalAnalyzer без зависимостей."""
    from analysis.temporal_analyzer import TemporalAnalyzer
    return TemporalAnalyzer()