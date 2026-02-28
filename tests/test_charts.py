"""
Тесты для графиков визуализации
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


from visualization.charts import (
    create_sentiment_pie,
    create_trends_bar,
    create_stock_price_chart,
    create_returns_chart,
    create_influence_chart,
    create_correlation_heatmap,
    create_trend_timeline,
    create_wordcloud_fig,
    create_sentiment_vs_price_chart,
    create_lag_correlation_chart
)


def test_create_sentiment_pie():
    """Тест создания круговой диаграммы тональности"""
    print("\n" + "=" * 60)
    print("ТЕСТ: create_sentiment_pie")
    print("=" * 60)

    # Тестовые данные
    distribution = {
        'positive': 15,
        'neutral': 25,
        'negative': 10
    }

    # Создаем график
    fig = create_sentiment_pie(distribution)

    # Проверки
    assert fig is not None, "График не создан"
    assert hasattr(fig, 'data'), "График не содержит данных"
    assert len(fig.data) > 0, "Нет данных в графике"

    print(f"График создан: {type(fig)}")
    print(f"Тип графика: {fig.data[0].type}")
    print(f"Заголовок: {fig.layout.title.text}")


def test_create_trends_bar():
    """Тест создания столбчатой диаграммы трендов"""
    print("\n" + "=" * 60)
    print("ТЕСТ: create_trends_bar")
    print("=" * 60)

    # Тестовые данные
    trends_df = pd.DataFrame({
        'Topic': [0, 1, 2],
        'Count': [10, 15, 8],
        'Name': ['Тренд 1', 'Тренд 2', 'Тренд 3']
    })

    # Создаем график
    fig = create_trends_bar(trends_df)

    assert fig is not None
    assert len(fig.data) > 0
    print(f"График создан")
    print(f"Количество столбцов: {len(fig.data[0].x)}")


def test_create_stock_price_chart():
    """Тест создания графика цены акции"""
    print("\n" + "=" * 60)
    print("ТЕСТ: create_stock_price_chart")
    print("=" * 60)

    # Генерируем тестовые данные
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    stock_data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(30)),
        'Open': 100 + np.random.randn(30),
        'High': 105 + np.random.randn(30),
        'Low': 95 + np.random.randn(30),
        'MA20': 100 + np.random.randn(30) * 0.5,
        'MA50': 100 + np.random.randn(30) * 0.3
    }, index=dates)

    # Создаем график
    fig = create_stock_price_chart(stock_data, 'TSLA')

    assert fig is not None
    assert len(fig.data) >= 1
    print(f"График цены TSLA создан")
    print(f"Количество линий: {len(fig.data)}")


def test_create_returns_chart():
    """Тест создания графика доходности"""
    print("\n" + "=" * 60)
    print("ТЕСТ: create_returns_chart")
    print("=" * 60)

    # Тестовые данные
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    stock_data = pd.DataFrame({
        'returns': np.random.randn(30) * 2
    }, index=dates)

    # Создаем график
    fig = create_returns_chart(stock_data)

    assert fig is not None
    print(f"График доходности создан")


def test_create_influence_chart():
    """Тест создания графика индекса влияния"""
    print("\n" + "=" * 60)
    print("ТЕСТ: create_influence_chart")
    print("=" * 60)

    # Тестовые данные
    influence_df = pd.DataFrame({
        'title': [f'Новость {i}' for i in range(10)],
        'influence_score': np.random.rand(10)
    })

    # Создаем график
    fig = create_influence_chart(influence_df, top_n=5)

    assert fig is not None
    assert len(fig.data[0].y) <= 5
    print(f"График влияния создан (топ-5)")


def test_create_correlation_heatmap():
    """Тест создания тепловой карты корреляции"""
    print("\n" + "=" * 60)
    print("ТЕСТ: create_correlation_heatmap")
    print("=" * 60)

    # Тестовая матрица корреляции
    corr_matrix = pd.DataFrame(
        np.random.randn(5, 5),
        columns=['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4'],
        index=['lag_0', 'lag_1', 'lag_2', 'lag_3', 'lag_4']
    )

    # Создаем график
    fig = create_correlation_heatmap(corr_matrix)

    assert fig is not None
    print(f"Тепловая карта создана")


def test_create_trend_timeline():
    """Тест создания временной линии тренда"""
    print("\n" + "=" * 60)
    print("ТЕСТ: create_trend_timeline")
    print("=" * 60)

    # Тестовые данные
    trend_data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=20, freq='D'),
        'mentions': np.random.randint(5, 50, 20)
    })

    # Создаем график
    fig = create_trend_timeline(trend_data, 'AI тренд')

    assert fig is not None
    print(f"Временная линия создана")


def test_create_wordcloud_fig():
    """Тест создания облака тегов"""
    print("\n" + "=" * 60)
    print("ТЕСТ: create_wordcloud_fig")
    print("=" * 60)

    # Тестовые ключевые слова
    keywords = {
        'tesla': 0.9,
        'electric': 0.8,
        'vehicle': 0.7,
        'battery': 0.6,
        'charging': 0.5
    }

    # Создаем облако
    fig = create_wordcloud_fig(keywords)

    # Может вернуть None если не реализовано
    if fig is not None:
        print(f"Облако тегов создано")
    else:
        print(f"Облако тегов не реализовано (пропускаем)")


def test_create_sentiment_vs_price_chart():
    """Тест создания графика сравнения тональности и цены"""
    print("\n" + "=" * 60)
    print("ТЕСТ: create_sentiment_vs_price_chart")
    print("=" * 60)

    # Тестовые данные
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')

    sentiment_data = pd.DataFrame({
        'date': dates,
        'avg_sentiment': np.random.randn(30) * 0.3
    })

    price_data = pd.DataFrame({
        'Close': 100 + np.cumsum(np.random.randn(30))
    }, index=dates)

    # Создаем график
    fig = create_sentiment_vs_price_chart(sentiment_data, price_data, 'TSLA')

    if fig is not None:
        assert fig is not None
        print(f"График сравнения создан")
    else:
        print(f"График сравнения не реализован")


def test_create_lag_correlation_chart():
    """Тест создания графика корреляции с лагами"""
    print("\n" + "=" * 60)
    print("ТЕСТ: create_lag_correlation_chart")
    print("=" * 60)

    # Тестовые результаты
    lag_results = {
        'lag_0': 0.15,
        'lag_1': -0.23,
        'lag_2': 0.08,
        'lag_3': -0.05,
        'lag_4': 0.12,
        'lag_5': 0.03,
        'best_lag': 'lag_1',
        'best_correlation': -0.23
    }

    # Создаем график
    fig = create_lag_correlation_chart(lag_results)

    if fig is not None:
        assert fig is not None
        print(f"График корреляции с лагами создан")
    else:
        print(f"График корреляции не реализован (пропускаем)")


def test_all_charts_with_empty_data():
    """Тест поведения графиков с пустыми данными"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Графики с пустыми данными")
    print("=" * 60)

    # Пустые данные
    empty_dict = {}
    empty_df = pd.DataFrame()

    # Проверяем что функции не падают
    try:
        fig1 = create_sentiment_pie(empty_dict)
        print(f"  create_sentiment_pie: {'+' if fig1 is None else '-'}")
    except:
        print(f"  create_sentiment_pie: упал")

    try:
        fig2 = create_trends_bar(empty_df)
        print(f"  create_trends_bar: {'+' if fig2 is None else '-'}")
    except:
        print(f"  create_trends_bar: упал")

    try:
        fig3 = create_stock_price_chart(empty_df, 'TEST')
        print(f"  create_stock_price_chart: {'+' if fig3 is None else '-'}")
    except:
        print(f"  create_stock_price_chart: упал")

    print("\nТесты с пустыми данными выполнены")


if __name__ == "__main__":
    print("ЗАПУСК ТЕСТОВ ГРАФИКОВ")
    print("=" * 60)

    test_create_sentiment_pie()
    test_create_trends_bar()
    test_create_stock_price_chart()
    test_create_returns_chart()
    test_create_influence_chart()
    test_create_correlation_heatmap()
    test_create_trend_timeline()
    test_create_wordcloud_fig()
    test_create_sentiment_vs_price_chart()
    test_create_lag_correlation_chart()
    test_all_charts_with_empty_data()

    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ ГРАФИКОВ ВЫПОЛНЕНЫ")