"""
Тесты для загрузки и обработки датасета DJIA
"""
import pytest
import pandas as pd
from visualization.data_loader import load_djia_data, extract_day_news, analyze_djia_week


def test_djia_loads_successfully(djia_data):
    assert djia_data is not None
    assert not djia_data.empty


def test_djia_expected_columns(djia_data):
    assert 'Label' in djia_data.columns
    assert 'Date' in djia_data.columns
    assert 'Top1' in djia_data.columns
    assert 'Top25' in djia_data.columns


def test_djia_row_count(djia_data):
    assert len(djia_data) == 1989


def test_djia_label_values(djia_data):
    """Метки должны быть только 0 или 1"""
    assert set(djia_data['Label'].unique()).issubset({0, 1})


def test_djia_label_distribution(djia_data):
    """Примерно половина дней — рост, половина — падение"""
    rise_pct = djia_data['Label'].mean()
    assert 0.4 <= rise_pct <= 0.6, f"Неожиданное распределение меток: {rise_pct:.2f}"


def test_extract_day_news_returns_list(djia_data):
    news = extract_day_news(djia_data.iloc[0])
    assert isinstance(news, list)
    assert len(news) > 0


def test_extract_day_news_no_byte_prefix(djia_data):
    """После очистки не должно быть префикса b'...'"""
    news = extract_day_news(djia_data.iloc[0])
    for text in news:
        assert not text.startswith("b'"), f"Не очищен byte-prefix: {text[:30]}"
        assert not text.startswith('b"'), f"Не очищен byte-prefix: {text[:30]}"


def test_extract_day_news_min_length(djia_data):
    """Все новости должны быть длиннее 10 символов"""
    news = extract_day_news(djia_data.iloc[0])
    assert all(len(t) > 10 for t in news)


def test_analyze_djia_week_returns_dataframe(djia_data, sentiment_analyzer):
    result = analyze_djia_week(djia_data, sentiment_analyzer, week_offset=0, days=3)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_analyze_djia_week_columns(djia_data, sentiment_analyzer):
    result = analyze_djia_week(djia_data, sentiment_analyzer, week_offset=0, days=3)
    for col in ('date', 'label', 'avg_sentiment', 'news_count'):
        assert col in result.columns, f"Отсутствует колонка: {col}"


def test_analyze_djia_week_sentiment_range(djia_data, sentiment_analyzer):
    result = analyze_djia_week(djia_data, sentiment_analyzer, week_offset=0, days=3)
    assert result['avg_sentiment'].between(-1.0, 1.0).all()


def test_analyze_djia_week_invalid_offset(djia_data, sentiment_analyzer):
    """Слишком большое смещение должно вернуть пустой DataFrame, не упасть"""
    result = analyze_djia_week(djia_data, sentiment_analyzer, week_offset=9999)
    assert isinstance(result, pd.DataFrame)
    assert result.empty