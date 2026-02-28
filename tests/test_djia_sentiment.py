"""
Тесты для анализа тональности датасета DJIA.

Фикстуры sentiment_analyzer и djia_data берутся из conftest.py.
Очистка текста делегирована в TextPreprocessor (режим sentiment).
"""

import os
import pandas as pd
import pytest


from nlp_processing.text_preprocessor import get_sentiment_preprocessor

_preprocessor = get_sentiment_preprocessor()


def extract_day_news(row, max_news=25):
    """Извлечение всех новостей за день с очисткой через TextPreprocessor."""
    news_list = []
    for i in range(1, max_news + 1):
        col = f'Top{i}'
        if col in row and pd.notna(row[col]):
            clean = _preprocessor.clean(str(row[col]))
            if clean and len(clean) > 10:
                news_list.append(clean)
    return news_list


def test_djia_data_loading(djia_data):
    """Тест загрузки данных"""
    assert djia_data is not None
    assert len(djia_data) == 1989
    assert 'Label' in djia_data.columns
    print(f"\nДанные загружены: {len(djia_data)} дней")


def test_news_cleaning(djia_data):
    """Тест очистки новостей через TextPreprocessor"""
    sample = djia_data.iloc[0]['Top1']
    cleaned = _preprocessor.clean(str(sample))

    assert cleaned is not None
    assert len(cleaned) > 10
    assert not cleaned.startswith("b'")
    assert not cleaned.startswith('b"')
    print(f"\nОчистка работает:")
    print(f"Оригинал: {str(sample)[:80]}...")
    print(f"Очищено: {cleaned[:80]}...")


def test_extract_day_news(djia_data):
    """Тест извлечения новостей за день"""
    day_news = extract_day_news(djia_data.iloc[0])

    assert len(day_news) > 0
    assert all(isinstance(news, str) for news in day_news)
    print(f"\nИзвлечено {len(day_news)} новостей за первый день")


@pytest.mark.slow
def test_sentiment_on_first_week(djia_data, sentiment_analyzer):
    """
    Тест анализа тональности на первой неделе данных.
    Помечен как slow — выполняется долго.
    """
    print("\nТЕСТ: Анализ тональности на первой неделе DJIA")

    sample_days = djia_data.head(7)
    results = []

    for idx, row in sample_days.iterrows():
        day_news = extract_day_news(row)
        if not day_news:
            continue

        day_scores = []
        day_sentiments = []

        for news in day_news[:5]:
            result = sentiment_analyzer.analyze_text(news)
            day_scores.append(result['score'])
            day_sentiments.append(result['sentiment'])

        avg_score = sum(day_scores) / len(day_scores) if day_scores else 0

        results.append({
            'day': idx + 1,
            'date': row['Date'],
            'label': row['Label'],
            'avg_sentiment': round(avg_score, 3),
            'positive_count': day_sentiments.count('positive'),
            'neutral_count': day_sentiments.count('neutral'),
            'negative_count': day_sentiments.count('negative'),
            'news_count': len(day_news),
        })

        print(
            f"День {idx + 1} ({row['Date']}, Label={row['Label']}): "
            f"avg_sentiment={avg_score:.3f} "
            f"P:{day_sentiments.count('positive')} "
            f"N:{day_sentiments.count('neutral')} "
            f"Neg:{day_sentiments.count('negative')}"
        )

    results_df = pd.DataFrame(results)

    assert len(results) > 0
    assert all(-1 <= r['avg_sentiment'] <= 1 for r in results)

    output_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'processed',
        'djia_first_week_sentiment.csv'
    )
    results_df.to_csv(output_path, index=False)
    print(f"Результаты сохранены: {output_path}")

    return results_df


@pytest.mark.slow
@pytest.mark.parametrize("day_index", [0, 1, 2])
def test_specific_day_sentiment(djia_data, sentiment_analyzer, day_index):
    """Тест тональности для конкретного дня (параметризованный)"""
    row = djia_data.iloc[day_index]
    day_news = extract_day_news(row)[:3]

    print(f"\nАнализ дня {day_index + 1} (Label: {row['Label']})")

    for i, news in enumerate(day_news):
        result = sentiment_analyzer.analyze_text(news)
        print(f"  Новость {i + 1}: {result['sentiment']} ({result['score']:.2f})")
        assert result['score'] is not None
        assert result['sentiment'] in ['positive', 'neutral', 'negative']