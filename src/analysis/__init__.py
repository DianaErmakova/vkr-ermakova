"""
analysis — модуль анализа рыночных трендов.

Включает:
    MarketTrendAnalyzer   — главный класс, объединяет все компоненты анализа
    SentimentAnalyzer     — анализ тональности новостей (FinBERT / RoBERTa)
    TrendClusterer        — тематическая кластеризация (BERTopic)
    InfluenceIndexCalculator — композитный индекс влияния новости на рынок
    CorrelationAnalyzer   — корреляция тональности с ценами акций
    TemporalAnalyzer      — временной анализ жизненного цикла трендов

Пример использования:
    from analysis.market_trend_analyzer import MarketTrendAnalyzer

    analyzer = MarketTrendAnalyzer(news_api_key="ваш_ключ")
    results  = analyzer.analyze_with_sentiment(companies=["Tesla", "Apple"])
"""

from .market_trend_analyzer import MarketTrendAnalyzer
from .sentiment_analyzer import SentimentAnalyzer
from .trend_clusterer import TrendClusterer
from .influence_index import InfluenceIndexCalculator
from .correlation_analyzer import CorrelationAnalyzer
from .temporal_analyzer import TemporalAnalyzer

__all__ = [
    'MarketTrendAnalyzer',
    'SentimentAnalyzer',
    'TrendClusterer',
    'InfluenceIndexCalculator',
    'CorrelationAnalyzer',
    'TemporalAnalyzer',
]