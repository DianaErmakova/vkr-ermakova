"""
Тест интеграции анализа тональности с основной системой
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.market_trend_analyzer import MarketTrendAnalyzer


def test_sentiment_integration():
    """Тест интеграции анализа тональности"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Интеграция анализа тональности с MarketTrendAnalyzer")
    print("=" * 60)

    try:
        # Создаем анализатор с включенным анализом тональности
        analyzer = MarketTrendAnalyzer(enable_sentiment=True)

        # Тестируем на небольшом наборе компаний
        companies = ["Tesla", "Apple"]

        print(f"\nАнализируем компании: {', '.join(companies)}")
        print("Используем демо-режим (тестовые данные)...")

        # Запускаем анализ с тональностью
        results = analyzer.analyze_with_sentiment(companies, pages=1)

        # Проверяем базовые результаты
        assert 'total_news' in results, "Нет total_news в результатах"
        assert 'trends_found' in results, "Нет trends_found в результатах"

        print(f"\nБазовый анализ успешен:")
        print(f"Обработано новостей: {results['total_news']}")
        print(f"Найдено трендов: {results['trends_found']}")

        # Проверяем анализ тональности
        if 'sentiment_analysis' in results:
            sentiment_data = results['sentiment_analysis']

            if 'error' in sentiment_data:
                print(f"Анализ тональности вернул ошибку: {sentiment_data['error']}")
                return False

            if 'summary' in sentiment_data:
                summary = sentiment_data['summary']
                print(f"\nАнализ тональности успешен:")
                print(f"Средняя оценка: {summary.get('average_score', 0):.2f}")
                print(f"Индекс настроения: {summary.get('sentiment_index', 0)}")

                # Проверяем что значения в допустимых диапазонах
                assert -1 <= summary.get('average_score', 0) <= 1, "Средняя оценка вне диапазона [-1, 1]"
                assert summary.get('total_texts', 0) > 0, "Нет проанализированных текстов"

                return True
            else:
                print("В sentiment_analysis нет summary")
                return False
        else:
            print("В результатах нет sentiment_analysis")
            return False

    except Exception as e:
        print(f"\nОшибка теста: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_without_sentiment():
    """Тест работы без анализа тональности"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Работа без анализа тональности")
    print("=" * 60)

    try:
        # Создаем анализатор с отключенным анализом тональности
        analyzer = MarketTrendAnalyzer(enable_sentiment=False)

        companies = ["Microsoft"]
        results = analyzer.analyze_market_trends(companies, pages=1)

        assert 'total_news' in results
        assert 'trends_found' in results

        print(f"\nАнализ без тональности успешен:")
        print(f"Обработано новостей: {results['total_news']}")
        print(f"Найдено трендов: {results['trends_found']}")

        # Проверяем что sentiment_analysis нет в результатах
        assert 'sentiment_analysis' not in results, "sentiment_analysis не должен быть при enable_sentiment=False"

        return True

    except Exception as e:
        print(f"\nОшибка теста: {e}")
        return False


if __name__ == "__main__":
    print("\nЗАПУСК ТЕСТОВ ИНТЕГРАЦИИ АНАЛИЗА ТОНАЛЬНОСТИ")
    print("=" * 60)

    test1_passed = test_sentiment_integration()
    test2_passed = test_without_sentiment()

    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ:")

    if test1_passed:
        print("Тест интеграции с анализом тональности: ПРОЙДЕН")
    else:
        print("Тест интеграции с анализом тональности: НЕ ПРОЙДЕН")

    if test2_passed:
        print("Тест работы без анализа тональности: ПРОЙДЕН")
    else:
        print("Тест работы без анализа тональности: НЕ ПРОЙДЕН")

    all_passed = test1_passed and test2_passed
    print("\n" + "=" * 60)

    if all_passed:
        print("ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
    else:
        print("Некоторые тесты не пройдены")

    sys.exit(0 if all_passed else 1)