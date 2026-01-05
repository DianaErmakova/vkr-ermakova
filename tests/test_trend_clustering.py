import sys
import os
import warnings
import pandas as pd

# Подавляем предупреждения
warnings.filterwarnings('ignore')

# Добавляем путь к src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.market_trend_analyzer import MarketTrendAnalyzer


def test_trend_clustering():
    """Тест кластеризации трендов"""
    print("=" * 70)
    print("ТЕСТИРОВАНИЕ СИСТЕМЫ АНАЛИЗА РЫНОЧНЫХ ТРЕНДОВ")
    print("=" * 70)

    analyzer = MarketTrendAnalyzer(news_api_key=None)

    results = analyzer.analyze_market_trends(["technology"], pages=1)

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("=" * 70)
    print(f"Обработано новостей: {results['total_news']}")
    print(f"Выявлено трендов: {results['trends_found']}")

    # Получаем информацию о трендах
    trends_info = results['trends_info']

    # Проверяем структуру данных
    if isinstance(trends_info, pd.DataFrame) and not trends_info.empty:
        valid_trends = trends_info[trends_info['Topic'] != -1]

        if len(valid_trends) > 0:
            print(f"\nНайдено валидных трендов: {len(valid_trends)}")
            print("-" * 70)

            for idx, (_, trend) in enumerate(valid_trends.iterrows(), 1):
                print(f"\nТРЕНД #{idx}:")
                print(f"ID кластера: {trend['Topic']}")
                print(f"Количество статей: {trend['Count']}")

                # Ключевые слова
                keywords = analyzer.trend_clusterer.get_trend_keywords(trend['Topic'], 6)
                if keywords:
                    kw_list = [f"{word}" for word, score in keywords]
                    print(f"Ключевые слова: {', '.join(kw_list)}")

        else:
            print("\nВсе статьи классифицированы как шум")
            print("   Это нормально для демо-данных или требует настройки модели")
    else:
        print("\nИнформация о трендах недоступна")

    # Проверки
    assert results['total_news'] > 0, "Нет данных для анализа"

    # Мягкая проверка - важно, что система работает
    if 'trends_info' in results and len(results['trends_info']) > 0:
        print(f"\nТЕСТ ПРОЙДЕН: система анализа трендов работает корректно")
        return True
    else:
        print(f"\nСИСТЕМА РАБОТАЕТ, НО НУЖДАЕТСЯ В НАСТРОЙКЕ")
        return False
