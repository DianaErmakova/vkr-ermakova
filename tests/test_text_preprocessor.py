"""
Тесты для TextPreprocessor (nlp_processing)
"""
import pytest
from nlp_processing.text_preprocessor import (
    TextPreprocessor,
    get_clustering_preprocessor,
    get_sentiment_preprocessor,
)


# Инициализация

def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        TextPreprocessor(mode='invalid')


def test_clustering_mode_creates():
    p = get_clustering_preprocessor()
    assert p.mode == 'clustering'


def test_sentiment_mode_creates():
    p = get_sentiment_preprocessor()
    assert p.mode == 'sentiment'


# Очистка byte-prefix (DJIA)

def test_removes_byte_prefix_single(text_preprocessor):
    result = text_preprocessor.clean("b'Fed raises interest rates'")
    assert not result.startswith("b'")
    assert 'fed raises interest rates' in result


def test_removes_byte_prefix_double(text_preprocessor):
    result = text_preprocessor.clean('b"Apple earnings beat expectations"')
    assert not result.startswith('b"')


def test_empty_string_returns_empty(text_preprocessor):
    assert text_preprocessor.clean("") == ""


def test_none_returns_empty(text_preprocessor):
    assert text_preprocessor.clean(None) == ""


# Режим clustering

def test_clustering_lowercases():
    p = get_clustering_preprocessor()
    result = p.clean("Tesla STOCK Surges")
    assert result == result.lower()


def test_clustering_removes_digits():
    p = get_clustering_preprocessor()
    result = p.clean("Apple reported $123 billion revenue in Q3 2024")
    assert '123' not in result
    assert '2024' not in result


def test_clustering_removes_special_chars():
    p = get_clustering_preprocessor()
    result = p.clean("Stock up 15%! Record #earnings @Tesla")
    assert '%' not in result
    assert '#' not in result
    assert '@' not in result


def test_clustering_removes_urls():
    p = get_clustering_preprocessor()
    result = p.clean("Read more at https://reuters.com/article/123")
    assert 'https' not in result
    assert 'reuters' not in result


# Режим sentiment

def test_sentiment_preserves_case():
    p = get_sentiment_preprocessor()
    result = p.clean("Tesla Stock SURGES after earnings")
    assert 'Tesla' in result
    assert 'SURGES' in result


def test_sentiment_removes_mentions():
    p = get_sentiment_preprocessor()
    result = p.clean("Great news @elonmusk #Tesla today")
    assert '@elonmusk' not in result
    assert '#Tesla' not in result


def test_sentiment_removes_urls():
    p = get_sentiment_preprocessor()
    result = p.clean("See https://example.com for details")
    assert 'https' not in result


def test_sentiment_truncates_long_text():
    p = get_sentiment_preprocessor()
    long_text = "market is volatile " * 200
    result = p.clean(long_text)
    assert len(result.split()) <= 400


# Пакетная обработка

def test_clean_batch_removes_empty(text_preprocessor):
    texts = ["Valid news text here", "   ", "", "Another valid text"]
    result = text_preprocessor.clean_batch(texts)
    assert len(result) == 2
    assert all(t.strip() for t in result)


def test_clean_batch_all_valid(text_preprocessor, sample_news_texts):
    result = text_preprocessor.clean_batch(sample_news_texts)
    assert len(result) == len(sample_news_texts)


# Статистика

def test_get_stats(text_preprocessor, sample_news_texts):
    stats = text_preprocessor.get_stats(sample_news_texts)
    assert stats['total_texts'] == len(sample_news_texts)
    assert stats['valid_texts'] <= stats['total_texts']
    assert stats['avg_words'] > 0


def test_get_stats_empty(text_preprocessor):
    stats = text_preprocessor.get_stats([])
    assert stats == {}


# Ключевые слова

def test_extract_keywords_returns_list(text_preprocessor):
    text = "Federal Reserve raises interest rates amid inflation concerns"
    keywords = text_preprocessor.extract_keywords(text, top_n=5)
    assert isinstance(keywords, list)
    assert len(keywords) <= 5
    assert all(isinstance(k, str) for k in keywords)


def test_extract_keywords_no_stopwords(text_preprocessor):
    text = "the and or but in on at to for of with by from"
    keywords = text_preprocessor.extract_keywords(text, top_n=10)
    stopwords = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from'}
    assert not any(k in stopwords for k in keywords)