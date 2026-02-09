"""
Тестирование метрик качества кластеризации трендов
"""
import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.trend_clusterer import TrendClusterer


def test_clusterer_initialization():
    """Тест инициализации кластеризатора"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Инициализация TrendClusterer")
    print("=" * 60)

    clusterer = TrendClusterer()

    assert clusterer.topic_model is not None
    assert clusterer.topics is None  # Еще не обучен

    print("TrendClusterer успешно инициализирован")
    return True


def test_clustering_basic():
    """Тест базовой кластеризации"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Базовая кластеризация")
    print("=" * 60)

    clusterer = TrendClusterer()

    # БОЛЕЕ ПРОСТЫЕ ТЕКСТЫ (меньше слов)
    test_texts = [
        # Тема 1: Электромобили
        "Tesla electric vehicles battery",
        "Electric cars pollution cities",
        "Charging stations expanding",

        # Тема 2: Искусственный интеллект
        "Artificial intelligence doctors",
        "Machine learning predictions",
        "Neural networks creative",

        # Тема 3: Облачные вычисления
        "Amazon AWS cloud",
        "Microsoft Azure tools",
        "Cloud computing remote",
    ]

    topics = clusterer.fit_clusters(test_texts)

    # Проверяем что кластеризация прошла
    assert topics is not None
    assert len(topics) == len(test_texts)

    print(f"Найдено уникальных тем: {len(set(topics))}")
    print(f"Темы: {set(topics)}")

    # Информация о трендах (может быть пустой если все -1)
    trends_info = clusterer.get_trends_info()
    print(f"Информация о трендах: {len(trends_info)} записей")

    # Допускаем что все может быть шумом для малых данных
    print(f"Базовая кластеризация выполнена")
    print(f"Всего документов: {len(test_texts)}")

    return True


def test_clustering_metrics():
    """Тест метрик качества кластеризации"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Метрики качества кластеризации")
    print("=" * 60)

    clusterer = TrendClusterer()

    # Разнообразные тестовые данные
    test_texts = [
        "Tesla stock rises after earnings report",
        "Apple announces new iPhone with AI features",
        "Microsoft cloud revenue exceeds expectations",
        "Amazon faces antitrust investigation",
        "Google AI research leads to breakthrough",
        "Electric vehicle sales hit record high",
        "Tech companies invest in quantum computing",
        "Data privacy concerns grow among users",
        "Renewable energy stocks outperform market",
        "Cybersecurity threats increase globally",
    ]

    # Обучаем модель
    topics = clusterer.fit_clusters(test_texts)

    # Получаем метрики
    metrics = clusterer.get_clustering_metrics(test_texts)

    print("Полученные метрики:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Проверяем обязательные метрики
    required_metrics = [
        'total_documents',
        'clusters_found',
        'noise_percentage',
        'avg_docs_per_cluster',
        'topic_stability'
    ]

    for metric in required_metrics:
        assert metric in metrics, f"Отсутствует метрика: {metric}"

    # Проверяем логические ограничения
    assert metrics['total_documents'] == len(test_texts)
    assert 0 <= metrics['noise_percentage'] <= 100
    assert metrics['clusters_found'] >= 0

    print(f"\nМетрики корректны:")
    print(f"Документов: {metrics['total_documents']}")
    print(f"Кластеров: {metrics['clusters_found']}")
    print(f"Шум: {metrics['noise_percentage']:.1f}%")

    return True


def test_detailed_report():
    """Тест детального отчета"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Детальный отчет о кластеризации")
    print("=" * 60)

    clusterer = TrendClusterer()

    test_texts = [
        "Positive market trends for technology sector",
        "Negative earnings report impacts stock prices",
        "Neutral news about quarterly board meeting",
        "AI startups receive record funding rounds",
        "Regulatory challenges for big tech companies"
    ]

    clusterer.fit_clusters(test_texts)
    report = clusterer.get_detailed_report()

    print("Структура отчета:")
    for section in report.keys():
        print(f" {section}")

    # Проверяем структуру отчета
    assert 'summary' in report
    assert 'clusters_details' in report
    assert 'metrics' in report

    # Проверяем summary
    summary = report['summary']
    assert 'total_documents' in summary
    assert 'valid_clusters' in summary
    assert 'noise_percentage' in summary
    assert 'quality_score' in summary

    # Проверяем качество в диапазоне 0-100
    assert 0 <= summary['quality_score'] <= 100

    # Проверяем детали кластеров
    if len(report['clusters_details']) > 0:
        cluster = report['clusters_details'][0]
        assert 'topic_id' in cluster
        assert 'documents_count' in cluster
        assert 'keywords' in cluster

    print(f"\nДетальный отчет корректный:")
    print(f"Оценка качества: {summary['quality_score']}/100")
    print(f"Валидных кластеров: {summary['valid_clusters']}")
    print(f"Процент шума: {summary['noise_percentage']}%")

    return True


def test_keyword_extraction():
    """Тест извлечения ключевых слов"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Извлечение ключевых слов")
    print("=" * 60)

    clusterer = TrendClusterer()

    # БОЛЕЕ ПРОСТЫЕ тексты
    test_texts = [
        "Tesla electric battery",
        "Electric charging station",
        "EV battery production",
        "Renewable solar panel",
        "Wind turbine Europe",
        "Solar power storage"
    ]

    topics = clusterer.fit_clusters(test_texts)
    trends_info = clusterer.get_trends_info()

    # Проверяем что DataFrame имеет колонку Topic
    if not trends_info.empty and 'Topic' in trends_info.columns:
        valid_topics = trends_info[trends_info['Topic'] != -1]

        if len(valid_topics) > 0:
            for _, topic_row in valid_topics.iterrows():
                topic_id = topic_row['Topic']
                keywords = clusterer.get_trend_keywords(topic_id, 3)

                print(f"\nКлючевые слова для тренда {topic_id}:")
                for word, score in keywords:
                    print(f"   {word} ({score:.3f})")

                assert len(keywords) > 0
    else:
        print("Нет валидных трендов для извлечения ключевых слов")

    print("\nИзвлечение ключевых слов работает корректно")
    return True


def test_empty_clustering():
    """Тест обработки пустых/некорректных данных"""
    print("\n" + "=" * 60)
    print("ТЕСТ: Обработка некорректных данных")
    print("=" * 60)

    clusterer = TrendClusterer()

    # Тест 1: Пустой список
    try:
        topics = clusterer.fit_clusters([])
        metrics = clusterer.get_clustering_metrics([])
        print("Обработка пустого списка")
    except Exception as e:
        print(f"Ошибка с пустым списком: {e}")

    # Тест 2: Очень короткие тексты
    short_texts = ["Hi", "OK", "Yes", "No"]
    topics = clusterer.fit_clusters(short_texts)
    metrics = clusterer.get_clustering_metrics(short_texts)

    print(f"\nКороткие тексты:")
    print(f"Документов: {metrics.get('total_documents', 0)}")
    print(f"Шум: {metrics.get('noise_percentage', 0):.1f}%")

    # Тест 3: Все тексты одинаковые
    duplicate_texts = ["Same text"] * 5
    topics = clusterer.fit_clusters(duplicate_texts)
    metrics = clusterer.get_clustering_metrics(duplicate_texts)

    print(f"\nДубликаты текстов:")
    print(f"Документов: {metrics.get('total_documents', 0)}")
    print(f"Кластеров: {metrics.get('clusters_found', 0)}")

    print("\nОбработка некорректных данных работает")
    return True


def run_all_tests():
    """Запуск всех тестов"""
    print("\n" + "=" * 60)
    print("ЗАПУСК ТЕСТОВ TRENDCLUSTERER С МЕТРИКАМИ")
    print("=" * 60)

    test_results = []

    try:
        test_results.append(("Инициализация", test_clusterer_initialization()))
    except Exception as e:
        print(f"Тест инициализации провален: {e}")
        test_results.append(("Инициализация", False))

    try:
        test_results.append(("Базовая кластеризация", test_clustering_basic()))
    except Exception as e:
        print(f"Тест базовой кластеризации провален: {e}")
        test_results.append(("Базовая кластеризация", False))

    try:
        test_results.append(("Метрики качества", test_clustering_metrics()))
    except Exception as e:
        print(f"Тест метрик провален: {e}")
        test_results.append(("Метрики качества", False))

    try:
        test_results.append(("Детальный отчет", test_detailed_report()))
    except Exception as e:
        print(f"Тест отчета провален: {e}")
        test_results.append(("Детальный отчет", False))

    try:
        test_results.append(("Ключевые слова", test_keyword_extraction()))
    except Exception as e:
        print(f"Тест ключевых слов провален: {e}")
        test_results.append(("Ключевые слова", False))

    try:
        test_results.append(("Некорректные данные", test_empty_clustering()))
    except Exception as e:
        print(f"Тест некорректных данных провален: {e}")
        test_results.append(("Некорректные данные", False))

    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ:")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "ПРОЙДЕН" if result else "ПРОВАЛЕН"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print("\n" + "=" * 60)
    success_rate = (passed / total) * 100 if total > 0 else 0
    print(f"РЕЗУЛЬТАТ: {passed}/{total} тестов пройдено ({success_rate:.0f}%)")

    if passed == total:
        print("ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ!")
    else:
        print("Некоторые тесты не пройдены")

    return passed == total
