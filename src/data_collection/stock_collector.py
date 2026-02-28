"""
Сбор исторических данных по акциям через yfinance.

Не требует API-ключа.
"""

import yfinance as yf
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StockCollector:
    """
    Сборщик исторических данных по акциям.

    Использует yfinance — API-ключ не требуется.
    Добавляет колонку returns (дневная доходность в долях)
    для совместимости с CorrelationAnalyzer.
    """

    def get_stock_data(self, ticker, period="3mo"):
        """
        Получить данные по одной акции.

        Args:
            ticker: биржевой тикер, например 'TSLA'
            period: период ('1mo', '3mo', '6mo', '1y', '2y', '5y')

        Returns:
            Словарь с полями:
                ticker        — тикер
                company_name  — полное название компании
                sector        — сектор
                history       — DataFrame с OHLCV и returns
                current_price — последняя цена закрытия
                data_points   — количество торговых дней
            или None при ошибке
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)

            if hist.empty:
                logger.warning(f"Нет данных для {ticker}")
                return None

            # Дневная доходность — для совместимости с CorrelationAnalyzer
            hist['returns'] = hist['Close'].pct_change()

            info = {}
            try:
                info = stock.info
            except Exception as e:
                logger.warning(f"Не удалось получить info для {ticker}: {e}")

            return {
                'ticker':        ticker,
                'company_name':  info.get('longName', ticker),
                'sector':        info.get('sector', 'Unknown'),
                'history':       hist,
                'current_price': float(hist['Close'].iloc[-1]),
                'data_points':   len(hist),
            }

        except Exception as e:
            logger.error(f"Ошибка получения данных для {ticker}: {e}")
            return None

    def get_multiple_stocks(self, tickers, period="3mo"):
        """
        Получить данные по нескольким акциям.

        Args:
            tickers: список тикеров
            period:  период (передаётся в get_stock_data)

        Returns:
            Словарь {ticker: данные}, невалидные тикеры пропускаются
        """
        results = {}
        for ticker in tickers:
            data = self.get_stock_data(ticker, period=period)
            if data is not None:
                results[ticker] = data
            else:
                logger.warning(f"Тикер {ticker} пропущен")
        return results
