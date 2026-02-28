"""
Тесты для InfluenceIndexCalculator
"""
import pytest
from analysis.influence_index import InfluenceIndexCalculator


def test_default_weights_sum_to_one(influence_calculator):
    total = sum(influence_calculator.weights.values())
    assert abs(total - 1.0) < 0.01


def test_custom_weights_normalized():
    """Если веса не суммируются в 1 — они должны быть нормированы"""
    calc = InfluenceIndexCalculator(weights={
        'intensity': 2.0, 'sentiment': 2.0,
        'virality': 1.0, 'authority': 1.0
    })
    total = sum(calc.weights.values())
    assert abs(total - 1.0) < 0.01


def test_intensity_range(influence_calculator):
    assert influence_calculator.calculate_intensity(0)   == 0.0
    assert influence_calculator.calculate_intensity(50)  <= 1.0
    assert influence_calculator.calculate_intensity(200) <= 1.0


def test_intensity_normalized(influence_calculator):
    score = influence_calculator.calculate_intensity(
        50, max_mentions=100, min_mentions=0
    )
    assert abs(score - 0.5) < 0.01


def test_sentiment_positive_higher_than_negative(influence_calculator):
    pos = influence_calculator.calculate_sentiment_component(0.8)
    neg = influence_calculator.calculate_sentiment_component(-0.5)
    assert pos > neg


def test_sentiment_neutral(influence_calculator):
    score = influence_calculator.calculate_sentiment_component(0.0)
    assert score == 0.0


def test_authority_reuters_higher_than_blog(influence_calculator):
    assert (influence_calculator.calculate_authority('Reuters') >
            influence_calculator.calculate_authority('Unknown Blog'))


def test_authority_empty_source(influence_calculator):
    score = influence_calculator.calculate_authority('')
    assert 0.0 <= score <= 1.0


def test_virality_high_vs_low(influence_calculator):
    high = influence_calculator.calculate_virality(
        {'retweets': 5000, 'likes': 10000, 'time_window': 3}
    )
    low = influence_calculator.calculate_virality(
        {'retweets': 5, 'likes': 10, 'time_window': 72}
    )
    assert high > low


def test_virality_empty(influence_calculator):
    assert influence_calculator.calculate_virality({}) == 0.0


def test_influence_score_range(influence_calculator, sample_influence_items):
    for item in sample_influence_items:
        result = influence_calculator.calculate_influence(item)
        assert 0.0 <= result['influence_score'] <= 1.0
        assert 'components' in result
        for v in result['components'].values():
            assert 0.0 <= v <= 1.0


def test_high_sentiment_higher_score(influence_calculator):
    """Новость с позитивным тональностью должна иметь выше индекс чем нейтральная"""
    base = {'mentions_count': 50, 'spread_data': {}, 'source': 'Reuters'}
    positive = influence_calculator.calculate_influence({**base, 'sentiment_score': 0.9})
    neutral  = influence_calculator.calculate_influence({**base, 'sentiment_score': 0.0})
    assert positive['influence_score'] > neutral['influence_score']


def test_batch_influence_sorted(influence_calculator, sample_influence_items):
    df = influence_calculator.calculate_batch_influence(sample_influence_items)
    assert len(df) == len(sample_influence_items)
    scores = df['influence_score'].tolist()
    assert scores == sorted(scores, reverse=True)


def test_top_influencers_count(influence_calculator, sample_influence_items):
    top = influence_calculator.identify_top_influencers(sample_influence_items, top_n=2)
    assert len(top) == 2


def test_trend_influence_keys(influence_calculator, sample_influence_items):
    result = influence_calculator.calculate_trend_influence(sample_influence_items)
    assert 'trend_influence_score' in result
    assert 'influence_distribution' in result
    assert 'total_news' in result
    assert result['total_news'] == len(sample_influence_items)


def test_trend_influence_distribution(influence_calculator, sample_influence_items):
    result = influence_calculator.calculate_trend_influence(sample_influence_items)
    dist = result['influence_distribution']
    total = dist['high'] + dist['medium'] + dist['low']
    assert total == len(sample_influence_items)