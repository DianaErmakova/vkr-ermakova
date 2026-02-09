import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.sentiment_analyzer import SentimentAnalyzer


def test_sentiment_analyzer_initialization():
    """Тест инициализации анализатора"""
    analyzer = SentimentAnalyzer(model_name="distilroberta-financial")

    # Проверяем что модель загружена
    assert analyzer.model is not None
    assert analyzer.tokenizer is not None

    # Проверяем информацию о модели
    model_info = analyzer.get_model_info()
    assert "name" in model_info
    assert "parameters" in model_info

    print("✓ Тест инициализации пройден")


def test_single_text_analysis():
    """Тест анализа одиночного текста"""
    analyzer = SentimentAnalyzer(model_name="distilroberta-financial")

    # Позитивный текст
    positive_text = "Company reports record profits and strong growth outlook"
    pos_result = analyzer.analyze_text(positive_text)

    assert "sentiment" in pos_result
    assert "score" in pos_result
    assert "confidence" in pos_result
    assert pos_result["score"] > -1 and pos_result["score"] < 1

    # Негативный текст
    negative_text = "Stock plummets after disappointing earnings report"
    neg_result = analyzer.analyze_text(negative_text)

    # Проверяем что негативный текст имеет меньшую оценку
    if pos_result["sentiment"] == "positive" and neg_result["sentiment"] == "negative":
        assert pos_result["score"] > neg_result["score"]

    print("✓ Тест анализа одиночного текста пройден")


def test_batch_analysis():
    """Тест пакетного анализа"""
    analyzer = SentimentAnalyzer(model_name="distilroberta-financial")

    texts = [
        "Positive news about market recovery",
        "Negative reports on company performance",
        "Neutral announcement about meeting"
    ]

    results = analyzer.analyze_batch(texts, batch_size=2)

    assert len(results) == 3
    assert all("sentiment" in r for r in results)
    assert all("confidence" in r for r in results)

    print("✓ Тест пакетного анализа пройден")


def test_sentiment_summary():
    """Тест сводной статистики"""
    analyzer = SentimentAnalyzer(model_name="distilroberta-financial")

    texts = [
        "Great earnings report from Tesla",
        "Apple faces regulatory challenges",
        "Market shows mixed signals today",
        "Microsoft announces innovative partnership"
    ]

    summary = analyzer.get_sentiment_summary(texts)

    assert summary["total_texts"] == 4
    assert "average_score" in summary
    assert "distribution_percentage" in summary
    assert "dominant_sentiment" in summary

    # Проверяем что сумма распределения ≈ 100%
    distribution = summary["distribution_percentage"]
    total_percentage = sum(distribution.values())
    assert 99 <= total_percentage <= 101  # Допускаем погрешность округления

    print("✓ Тест сводной статистики пройден")


def test_empty_text_handling():
    """Тест обработки пустых текстов"""
    analyzer = SentimentAnalyzer(model_name="distilroberta-financial")

    # Пустой текст
    empty_result = analyzer.analyze_text("")
    assert empty_result["sentiment"] == "neutral"
    assert empty_result["error"] == True

    # Пустой список
    empty_summary = analyzer.get_sentiment_summary([])
    assert empty_summary["total_texts"] == 0

    print("✓ Тест обработки пустых текстов пройден")


def test_different_models():
    """Тест работы с разными моделями"""
    # Тестируем несколько моделей (если есть время/ресурсы)
    test_models = ["distilroberta-financial"]  # Можно добавить другие

    for model_name in test_models:
        analyzer = SentimentAnalyzer(model_name=model_name)
        result = analyzer.analyze_text("Test sentence for model validation")

        assert result["model"] == model_name
        assert result["sentiment"] in ["positive", "neutral", "negative"]

    print("✓ Тест разных моделей пройден")


if __name__ == "__main__":
    print("Запуск тестов анализатора тональности...")
    print("=" * 60)

    test_sentiment_analyzer_initialization()
    test_single_text_analysis()
    test_batch_analysis()
    test_sentiment_summary()
    test_empty_text_handling()
    test_different_models()

    print("=" * 60)
    print("Все тесты успешно пройдены!")