import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))


def test_stock_collector():
    """Тестируем сбор данных по акциям"""
    from data_collection.stock_collector import StockCollector

    collector = StockCollector()
    data = collector.get_stock_data("TSLA")

    assert data is not None
    assert data['company_name'] == "Tesla, Inc."
    assert data['current_price'] > 0
    print(f"Тест пройден: {data['company_name']} - ${data['current_price']:.2f}")
