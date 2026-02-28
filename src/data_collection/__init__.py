"""
data_collection — модуль сбора данных из внешних источников.

Включает:
    NewsCollector  — сбор новостей через NewsAPI и RSS-ленты
                     (Reuters, BBC, CNBC, Yahoo Finance)
    StockCollector — сбор исторических цен акций через yfinance

Пример использования:
    from data_collection.news_collector import NewsCollector
    from data_collection.stock_collector import StockCollector

    news   = NewsCollector(api_key="ваш_ключ")
    stocks = StockCollector()

    articles   = news.get_news("Tesla", days_back=7)
    price_data = stocks.get_stock_data("TSLA")
"""

from .news_collector import NewsCollector
from .stock_collector import StockCollector

__all__ = [
    'NewsCollector',
    'StockCollector',
]