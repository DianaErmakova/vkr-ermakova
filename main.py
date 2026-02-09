# main.py (исправленная версия)
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.market_trend_analyzer import MarketTrendAnalyzer


def main():
    print("=== Система анализа рыночных трендов с ML ===")
    print("Используется машинное обучение (BERTopic) для выявления трендов\n")

    # Создаем анализатор
    # Если есть API ключ NewsAPI - передайте его:
    # analyzer = MarketTrendAnalyzer(news_api_key="ваш_ключ")

    # Без ключа - использует тестовые данные
    analyzer = MarketTrendAnalyzer(news_api_key=None)

    # Компании для анализа
    companies = ["Tesla", "Apple", "Microsoft", "Nvidia", "Amazon", "Google"]

    print(f"Анализируем компании: {', '.join(companies)}")
    print("Обрабатываю новости и применяю ML-модель...\n")

    # Запускаем ML-анализ
    results = analyzer.analyze_market_trends(companies, pages=1)

    # Выводим результаты
    print("=" * 50)
    print("РЕЗУЛЬТАТЫ ML-АНАЛИЗА")
    print("=" * 50)

    print(f"Обработано новостей: {results['total_news']}")
    print(f"Выявлено рыночных трендов: {results['trends_found']}")

    print("\nДЕТАЛИ ТРЕНДОВ:")
    print("-" * 30)

    # Показываем каждый тренд
    trends_info = results['trends_info']
    for idx, row in trends_info.iterrows():
        if row['Topic'] != -1:  # Пропускаем "шум"
            print(f"\nТренд #{row['Topic']}:")
            print(f"   Упоминаний в новостях: {row['Count']}")

            # Получаем ключевые слова
            keywords = analyzer.trend_clusterer.get_trend_keywords(row['Topic'], top_n=5)
            keyword_list = [word[0] for word in keywords]
            print(f"   Ключевые слова: {', '.join(keyword_list)}")

    # Анализ тональности
    if 'sentiment_summary' in results:
        print("\n📊 АНАЛИЗ ТОНАЛЬНОСТИ:")
        print(f"  Средний показатель: {results['sentiment_summary']['average_score']:.2f}")
        print(f"  Индекс настроения: {results['sentiment_summary']['sentiment_index']}")
        print(f"  Распределение: {results['sentiment_summary']['distribution_percentage']}")

    print("\n" + "=" * 50)
    print("Анализ завершен! Система успешно обнаружила рыночные тренды.")
    print("=" * 50)


if __name__ == "__main__":
    main()
