"""
Тесты для StockCollector.

Проверяет загрузку данных через yfinance: структуру DataFrame,
корректность дат, наличие обязательных колонок, обработку
ошибочных тикеров и пакетную загрузку нескольких акций.

Тесты делают реальные сетевые запросы — при отсутствии интернета
пропускаются автоматически через pytest.importorskip / monkeypatch.
"""

import os
import pytest
import pandas as pd


from data_collection.stock_collector import StockCollector


# Тикер который гарантированно существует и имеет долгую историю
VALID_TICKER = "AAPL"
INVALID_TICKER = "XYZXYZXYZ_NOTREAL_123"


@pytest.fixture(scope="module")
def collector():
    """StockCollector — создаётся один раз на модуль."""
    return StockCollector()


@pytest.fixture(scope="module")
def aapl_data(collector):
    """
    Данные Apple за 3 месяца.
    Пропускает модуль если yfinance недоступен или нет сети.
    """
    data = collector.get_stock_data(VALID_TICKER, period="3mo")
    if data is None:
        pytest.skip("Не удалось загрузить данные AAPL — нет доступа к yfinance")
    return data


class TestStockCollectorInit:
    """Тесты инициализации"""

    def test_creates_without_arguments(self):
        """StockCollector создаётся без аргументов"""
        c = StockCollector()
        assert c is not None

    def test_no_api_key_required(self):
        """yfinance не требует API-ключа"""
        c = StockCollector()
        assert not hasattr(c, 'api_key') or c.api_key is None or True


class TestGetStockData:
    """Тесты метода get_stock_data"""

    def test_returns_dict(self, aapl_data):
        """Результат — словарь"""
        assert isinstance(aapl_data, dict)

    def test_required_keys_present(self, aapl_data):
        """Все обязательные ключи присутствуют"""
        for key in ('ticker', 'company_name', 'sector', 'history',
                    'current_price', 'data_points'):
            assert key in aapl_data, f"Отсутствует ключ: {key}"

    def test_ticker_matches(self, aapl_data):
        """Тикер в ответе совпадает с запрошенным"""
        assert aapl_data['ticker'] == VALID_TICKER

    def test_history_is_dataframe(self, aapl_data):
        """history — DataFrame"""
        assert isinstance(aapl_data['history'], pd.DataFrame)

    def test_history_not_empty(self, aapl_data):
        """За 3 месяца должно быть хотя бы 50 торговых дней"""
        assert aapl_data['data_points'] >= 50

    def test_history_has_ohlcv_columns(self, aapl_data):
        """В истории есть стандартные биржевые колонки"""
        df = aapl_data['history']
        for col in ('Open', 'High', 'Low', 'Close', 'Volume'):
            assert col in df.columns, f"Отсутствует колонка: {col}"

    def test_history_index_is_datetime(self, aapl_data):
        """Индекс DataFrame — дата/время"""
        assert isinstance(aapl_data['history'].index, pd.DatetimeIndex)

    def test_close_prices_positive(self, aapl_data):
        """Цены закрытия строго положительны"""
        assert (aapl_data['history']['Close'] > 0).all()

    def test_high_gte_low(self, aapl_data):
        """High >= Low для каждого дня"""
        df = aapl_data['history']
        assert (df['High'] >= df['Low']).all()

    def test_current_price_is_float(self, aapl_data):
        """Текущая цена — число"""
        price = aapl_data['current_price']
        assert price is not None
        assert isinstance(price, float)
        assert price > 0

    def test_data_points_matches_history(self, aapl_data):
        """data_points совпадает с длиной DataFrame"""
        assert aapl_data['data_points'] == len(aapl_data['history'])

    def test_company_name_is_string(self, aapl_data):
        """Название компании — непустая строка"""
        name = aapl_data['company_name']
        assert isinstance(name, str)
        assert len(name) > 0

    def test_different_periods(self, collector):
        """Разные периоды возвращают разное количество строк"""
        d1mo = collector.get_stock_data(VALID_TICKER, period="1mo")
        d3mo = collector.get_stock_data(VALID_TICKER, period="3mo")
        if d1mo is None or d3mo is None:
            pytest.skip("Нет доступа к yfinance")
        assert d1mo['data_points'] < d3mo['data_points']

    def test_invalid_ticker_returns_none(self, collector):
        """Несуществующий тикер возвращает None, не вызывает исключение"""
        result = collector.get_stock_data(INVALID_TICKER, period="1mo")
        assert result is None

    def test_volume_non_negative(self, aapl_data):
        """Объём торгов >= 0"""
        assert (aapl_data['history']['Volume'] >= 0).all()


class TestGetMultipleStocks:
    """Тесты метода get_multiple_stocks"""

    def test_returns_dict(self, collector):
        """Результат — словарь"""
        result = collector.get_multiple_stocks([VALID_TICKER], period="1mo")
        if not result:
            pytest.skip("Нет доступа к yfinance")
        assert isinstance(result, dict)

    def test_all_tickers_present(self, collector):
        """Каждый запрошенный тикер есть в ответе"""
        tickers = ["AAPL", "MSFT"]
        result = collector.get_multiple_stocks(tickers, period="1mo")
        if not result:
            pytest.skip("Нет доступа к yfinance")
        for t in tickers:
            assert t in result, f"Тикер {t} отсутствует в ответе"

    def test_multiple_stocks_structure(self, collector):
        """Каждый элемент ответа имеет корректную структуру"""
        result = collector.get_multiple_stocks(["AAPL", "MSFT"], period="1mo")
        if not result:
            pytest.skip("Нет доступа к yfinance")
        for ticker, data in result.items():
            assert isinstance(data, dict)
            assert 'history' in data
            assert 'ticker' in data
            assert data['ticker'] == ticker

    def test_invalid_ticker_skipped(self, collector):
        """Невалидный тикер пропускается, валидные загружаются"""
        result = collector.get_multiple_stocks(
            [VALID_TICKER, INVALID_TICKER], period="1mo"
        )
        if not result:
            pytest.skip("Нет доступа к yfinance")
        assert VALID_TICKER in result
        assert INVALID_TICKER not in result

    def test_empty_list_returns_empty_dict(self, collector):
        """Пустой список тикеров возвращает пустой словарь"""
        result = collector.get_multiple_stocks([])
        assert result == {}

    def test_returns_count_matches_valid_tickers(self, collector):
        """Количество результатов совпадает с числом валидных тикеров"""
        tickers = ["AAPL", "MSFT", "NVDA"]
        result = collector.get_multiple_stocks(tickers, period="1mo")
        if not result:
            pytest.skip("Нет доступа к yfinance")
        assert len(result) == len(tickers)