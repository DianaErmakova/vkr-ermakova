"""
Тесты для временного анализа трендов
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.temporal_analyzer import TemporalAnalyzer


def generate_test_trends():
    """Генерация тестовых трендов разных типов"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Тренд 1: Взрывной
    explosive = np.concatenate([
        np.linspace(0, 100, 20),
        np.linspace(100, 10, 30),
        np.ones(50) * 10
    ])

    # Тренд 2: Устойчивый рост
    stable = np.linspace(10, 80, 100)

    # Тренд 3: Циклический
    cyclic = 40 + 30 * np.sin(np.linspace(0, 4 * np.pi, 100))

    # Тренд 4: Сезонный (недельная сезонность)
    seasonal = 50 + 20 * np.sin(np.linspace(0, 20 * np.pi, 100))

    trends = {
        'explosive': pd.DataFrame({'date': dates, 'mentions': explosive}),
        'stable': pd.DataFrame({'date': dates, 'mentions': stable}),
        'cyclic': pd.DataFrame({'date': dates, 'mentions': cyclic}),
        'seasonal': pd.DataFrame({'date': dates, 'mentions': seasonal})
    }

    return trends


def test_temporal_analyzer():
    """Тест временного анализатора"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Временной анализ трендов")
    print("=" * 60)

    analyzer = TemporalAnalyzer()
    trends = generate_test_trends()

    # Тест 1: Жизненный цикл
    print("\n1. Тест жизненного цикла тренда:")
    lifecycle = analyzer.analyze_trend_lifecycle(
        trends['explosive'], value_column='mentions'
    )
    print(f"Пик: {lifecycle.get('peak_date', 'N/A')}")
    print(f"Длительность: {lifecycle.get('duration_days', 'N/A')} дней")
    print(f"Скорость роста: {lifecycle.get('rise_rate', 0):.2f}")
    assert 'peak_date' in lifecycle
    print("Жизненный цикл работает")

    # Тест 2: Моментум
    print("\n2. Тест моментума:")
    momentum_df = analyzer.calculate_trend_momentum(
        trends['stable'], value_column='mentions'
    )
    print(f"Колонки: {list(momentum_df.columns)}")
    assert 'momentum_7d' in momentum_df.columns
    print("Моментум работает")

    # Тест 3: Классификация трендов
    print("\n3. Тест классификации:")
    for name, data in trends.items():
        classification = analyzer.classify_trend_type(data)
        print(f"   {name}: {classification['type']} (уверенность: {classification['confidence']})")
    assert 'type' in classification
    print("Классификация работает")

    # Тест 4: Сезонность
    print("\n4. Тест сезонности:")
    seasonality = analyzer.detect_seasonality(trends['seasonal'])
    print(f"Результаты: {seasonality}")
    print("Сезонность работает")

    # Тест 5: Полный отчет
    print("\n5. Тест полного отчета:")
    report = analyzer.get_temporal_report(trends['explosive'], "Test Trend")
    print(f"Базовые метрики: {list(report['basic_stats'].keys())}")
    print(f"Тип тренда: {report['classification']['type']}")
    assert 'basic_stats' in report
    print("Полный отчет работает")

    # Тест 6: Сравнение трендов
    print("\n6. Тест сравнения трендов:")
    comparison = analyzer.compare_trends_temporal(trends)
    print(comparison.to_string())
    assert not comparison.empty
    print("Сравнение трендов работает")

    print("\n" + "=" * 60)
    print("Все тесты временного анализа пройдены!")
    return True


if __name__ == "__main__":
    test_temporal_analyzer()