"""
Тесты для SentimentAnalyzer
"""
import pytest


def test_initialization(sentiment_analyzer):
    assert sentiment_analyzer.model is not None
    assert sentiment_analyzer.tokenizer is not None
    info = sentiment_analyzer.get_model_info()
    assert "name" in info
    assert "parameters" in info
    assert info["parameters"] > 0


def test_analyze_positive_text(sentiment_analyzer):
    result = sentiment_analyzer.analyze_text(
        "Company reports record profits and strong growth outlook"
    )
    assert result["sentiment"] in ("positive", "neutral", "negative")
    assert -1.0 <= result["score"] <= 1.0
    assert 0.0 <= result["confidence"] <= 1.0
    assert result.get("error") is not True


def test_analyze_negative_text(sentiment_analyzer):
    result = sentiment_analyzer.analyze_text(
        "Stock plummets after catastrophic earnings miss"
    )
    assert result["sentiment"] in ("positive", "neutral", "negative")
    assert -1.0 <= result["score"] <= 1.0


def test_positive_score_higher_than_negative(sentiment_analyzer):
    pos = sentiment_analyzer.analyze_text("Record profits, strong growth, bullish market")
    neg = sentiment_analyzer.analyze_text("Bankruptcy, massive losses, market crash")
    # Позитивный текст должен иметь score выше негативного
    assert pos["score"] > neg["score"]


def test_empty_text_returns_neutral(sentiment_analyzer):
    result = sentiment_analyzer.analyze_text("")
    assert result["sentiment"] == "neutral"
    assert result["error"] is True


def test_none_handling(sentiment_analyzer):
    result = sentiment_analyzer.analyze_text(None)
    assert result["sentiment"] == "neutral"
    assert result["error"] is True


def test_batch_analysis(sentiment_analyzer, sample_news_texts):
    results = sentiment_analyzer.analyze_batch(sample_news_texts, batch_size=4)
    assert len(results) == len(sample_news_texts)
    assert all("sentiment" in r for r in results)
    assert all("confidence" in r for r in results)
    assert all(-1.0 <= r["score"] <= 1.0 for r in results)


def test_sentiment_summary(sentiment_analyzer, sample_news_texts):
    summary = sentiment_analyzer.get_sentiment_summary(sample_news_texts)

    assert summary["total_texts"] == len(sample_news_texts)
    assert "average_score" in summary
    assert "dominant_sentiment" in summary
    assert "distribution_percentage" in summary

    # Сумма распределения ≈ 100%
    total_pct = sum(summary["distribution_percentage"].values())
    assert 99.0 <= total_pct <= 101.0

    # Средняя оценка в допустимом диапазоне
    assert -1.0 <= summary["average_score"] <= 1.0


def test_empty_list_summary(sentiment_analyzer):
    summary = sentiment_analyzer.get_sentiment_summary([])
    assert summary["total_texts"] == 0
    assert summary["average_score"] == 0.0


def test_long_text_truncation(sentiment_analyzer):
    """Длинный текст не должен вызывать ошибку (BERT ограничен 512 токенами)"""
    long_text = "The market is volatile. " * 200
    result = sentiment_analyzer.analyze_text(long_text)
    assert result["sentiment"] in ("positive", "neutral", "negative")
    assert result.get("error") is not True