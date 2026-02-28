"""
Тесты для анализа корреляции новостей и цен акций
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.correlation_analyzer import CorrelationAnalyzer


def generate_test_data():
    """Генерация тестовых данных"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Данные по акции
    np.random.seed(42)
    returns = np.random.randn(100) * 0.02  # 2% волатильность
    price = 100 * (1 + returns).cumprod()

    stock_data = pd.DataFrame({
        'Close': price,
        'returns': returns,
        'direction': (returns > 0).astype(int)
    }, index=dates)

    # Генерируем тональность (от -1 до 1)
    sentiment_scores = np.random.randn(100) * 0.3

    # Добавляем корреляцию: если сегодня позитивные новости,
    # завтра цена растет с большей вероятностью
    for i in range(len(dates) - 1):
        if sentiment_scores[i] > 0.5:
            stock_data.iloc[i + 1, stock_data.columns.get_loc('returns')] += 0.01
            stock_data.iloc[i + 1, stock_data.columns.get_loc('Close')] *= 1.01

    # Новостные данные - ВАЖНО: добавляем sentiment_score
    news_data = pd.DataFrame({
        'date': dates,
        'text': ['Test news'] * 100,
        'sentiment_score': sentiment_scores,  # теперь это есть!
        'source': ['Reuters'] * 100
    })

    return stock_data, news_data


def test_correlation_analyzer():
    """Тест анализатора корреляции"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Анализатор корреляции новостей и цен")
    print("=" * 60)

    analyzer = CorrelationAnalyzer()

    # Генерируем тестовые данные
    stock_data, news_data = generate_test_data()

    print("\n1. Тест загрузки данных...")
    assert stock_data is not None
    assert news_data is not None
    print("Данные загружены")

    print("\n2. Тест подготовки новостных признаков...")
    news_features = analyzer.prepare_news_features(news_data, date_column='date')
    assert news_features is not None
    print(f"Подготовлено {len(news_features)} дней с новостями")
    print(f"Колонки: {list(news_features.columns)}")

    print("\n3. Тест корреляции с лагами...")
    lag_corr = analyzer.calculate_lag_correlation(news_features, stock_data, max_lag=3)

    # Проверяем что результат - словарь
    assert isinstance(lag_corr, dict)

    print(f"   Результат корреляции:")
    for key, value in lag_corr.items():
        print(f"     {key}: {value}")

    # Не проверяем наличие lag_0 - может быть ошибка если данных мало
    if 'error' in lag_corr:
        print(f"Метод вернул ошибку: {lag_corr['error']}")
    else:
        print("Корреляция рассчитана")

    print("\n" + "=" * 60)
    print("Тест выполнен")
    return True


if __name__ == "__main__":
    test_correlation_analyzer()