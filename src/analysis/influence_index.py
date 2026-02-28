"""
Композитный индекс влияния медиасобытий на рынок.

Формула из литературного обзора:
    Influence = w1 * Интенсивность + w2 * Тональность + w3 * Виральность + w4 * Авторитетность

Веса компонентов фиксированы и обоснованы в литературном обзоре (раздел 2.3).
Их изменение возможно только при наличии размеченных исторических данных
для валидации, что выходит за рамки данной работы.

Ограничение: компонент виральности в текущей реализации равен нулю
ввиду отсутствия доступа к данным социальных сетей (Twitter/Reddit API).
Интеграция с этими источниками обозначена как направление дальнейшего развития.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class InfluenceIndexCalculator:
    """
    Калькулятор композитного индекса влияния медиасобытий на рынок.

    Веса компонентов зафиксированы согласно литературному обзору
    и не подлежат автоматической оптимизации в рамках данной работы.
    Для изменения весов передайте словарь в параметре weights.

    Компоненты индекса:
        intensity  (0.3) — нормированное число упоминаний
        sentiment  (0.4) — тональность FinBERT/RoBERTa, диапазон [-1, 1]
        virality   (0.2) — скорость распространения (ретвиты, лайки и т.д.)
        authority  (0.1) — авторитетность источника

    Примечание по виральности:
        При работе с NewsAPI и датасетом DJIA данные о распространении
        в социальных сетях недоступны, поэтому virality = 0.0.
        Это задокументированное ограничение системы; реальные данные
        можно получить через Twitter API или Reddit API.
    """

    DEFAULT_WEIGHTS = {
        'intensity': 0.3,
        'sentiment': 0.4,
        'virality':  0.2,
        'authority': 0.1,
    }

    SOURCE_AUTHORITY = {
        'reuters':          1.0,
        'bloomberg':        1.0,
        'wsj':              1.0,
        'ft':               1.0,
        'cnbc':             0.9,
        'bbc':              0.8,
        'apnews':           0.85,
        'marketwatch':      0.75,
        'yahoo finance':    0.7,
        'seeking alpha':    0.6,
        'businessinsider':  0.6,
        'reddit':           0.4,
        'twitter':          0.3,
        'blog':             0.2,
        'unknown':          0.5,
    }

    def __init__(self, weights=None):
        """
        Args:
            weights: словарь с весами компонентов.
                     Если None — используются DEFAULT_WEIGHTS из литобзора.
                     Сумма весов нормируется к 1 автоматически.
        """
        self.weights = (weights or self.DEFAULT_WEIGHTS).copy()

        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Сумма весов {total:.3f} != 1.0, выполняется нормировка")
            for key in self.weights:
                self.weights[key] /= total

    def calculate_intensity(self, mentions_count, max_mentions=None,
                            min_mentions=None, method='minmax'):
        """
        Нормированная интенсивность упоминаний.

        Args:
            mentions_count: количество упоминаний
            max_mentions:   максимум для min-max нормировки
            min_mentions:   минимум для min-max нормировки
            method:         'minmax' или 'log'

        Returns:
            float в диапазоне [0, 1]
        """
        if mentions_count <= 0:
            return 0.0

        if method == 'minmax':
            if max_mentions is not None and min_mentions is not None:
                if max_mentions > min_mentions:
                    return (mentions_count - min_mentions) / (max_mentions - min_mentions)
                return 1.0
            return min(1.0, mentions_count / 100)

        if method == 'log':
            return min(1.0, np.log1p(mentions_count) / np.log1p(100))

        return min(1.0, mentions_count / 100)

    def calculate_sentiment_component(self, sentiment_score, method='positive_only'):
        """
        Компонент тональности.

        Args:
            sentiment_score: оценка тональности в диапазоне [-1, 1]
            method:
                'positive_only' — только позитивные новости дают вклад
                'absolute'      — важна сила эмоции, знак не учитывается
                'bipolar'       — преобразует [-1, 1] в [0, 1]

        Returns:
            float в диапазоне [0, 1]
        """
        if pd.isna(sentiment_score):
            return 0.5

        if method == 'positive_only':
            return max(0.0, float(sentiment_score))
        if method == 'absolute':
            return abs(float(sentiment_score))
        if method == 'bipolar':
            return (float(sentiment_score) + 1) / 2

        return max(0.0, float(sentiment_score))

    def calculate_virality(self, spread_data):
        """
        Компонент виральности на основе данных о распространении.

        Принимает словарь с метриками социальных сетей.
        Если словарь пуст или не передан — возвращает 0.0.
        Это штатное поведение при работе с NewsAPI и DJIA,
        где данные соцсетей недоступны.

        Args:
            spread_data: словарь с ключами:
                retweets, likes, comments, shares, time_window (часы)

        Returns:
            float в диапазоне [0, 1], 0.0 если данных нет
        """
        if not spread_data:
            return 0.0

        components = []

        if 'retweets' in spread_data:
            components.append(min(1.0, spread_data['retweets'] / 1000))
        if 'likes' in spread_data:
            components.append(min(1.0, spread_data['likes'] / 5000))
        if 'comments' in spread_data:
            components.append(min(1.0, spread_data['comments'] / 500))
        if 'shares' in spread_data:
            components.append(min(1.0, spread_data['shares'] / 1000))
        if 'time_window' in spread_data:
            hours = spread_data['time_window']
            components.append(max(0.0, min(1.0, 24 / hours if hours > 0 else 1.0)))

        return float(np.mean(components)) if components else 0.0

    def calculate_authority(self, source):
        """
        Авторитетность источника.

        Args:
            source: название или URL источника

        Returns:
            float в диапазоне [0, 1]
        """
        if not source or pd.isna(source):
            return self.SOURCE_AUTHORITY['unknown']

        source_lower = str(source).lower().strip()

        for key, value in self.SOURCE_AUTHORITY.items():
            if key in source_lower:
                return value

        return self.SOURCE_AUTHORITY['unknown']

    def calculate_influence(self, news_item):
        """
        Расчёт композитного индекса влияния для одной новости.

        Args:
            news_item: словарь с полями:
                mentions_count  — количество упоминаний
                sentiment_score — оценка тональности [-1, 1]
                spread_data     — данные о распространении (может быть {})
                source          — источник
                max_mentions    — (опционально) максимум для нормировки
                min_mentions    — (опционально) минимум для нормировки

        Returns:
            Словарь с полями influence_score, components, weights, raw_sentiment
        """
        intensity = self.calculate_intensity(
            news_item.get('mentions_count', 1),
            max_mentions=news_item.get('max_mentions'),
            min_mentions=news_item.get('min_mentions'),
        )
        sentiment = self.calculate_sentiment_component(
            news_item.get('sentiment_score', 0)
        )
        virality = self.calculate_virality(
            news_item.get('spread_data', {})
        )
        authority = self.calculate_authority(
            news_item.get('source', 'unknown')
        )

        influence_score = (
            self.weights['intensity'] * intensity +
            self.weights['sentiment'] * sentiment +
            self.weights['virality']  * virality +
            self.weights['authority'] * authority
        )

        return {
            'influence_score': round(influence_score, 3),
            'components': {
                'intensity': round(intensity, 3),
                'sentiment': round(sentiment, 3),
                'virality':  round(virality, 3),
                'authority': round(authority, 3),
            },
            'weights':       self.weights.copy(),
            'raw_sentiment': news_item.get('sentiment_score', 0),
        }

    def calculate_batch_influence(self, news_items):
        """
        Расчёт индекса влияния для списка новостей.

        Returns:
            DataFrame, отсортированный по influence_score убыванию
        """
        results = []
        for item in news_items:
            result = self.calculate_influence(item)
            result['title'] = item.get('title', '')
            result['date']  = item.get('date', '')
            results.append(result)

        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('influence_score', ascending=False)
        return df

    def identify_top_influencers(self, news_items, top_n=10):
        """
        Топ-N наиболее влиятельных новостей.

        Returns:
            DataFrame с top_n строками
        """
        df = self.calculate_batch_influence(news_items)
        return df.head(top_n)

    def calculate_trend_influence(self, trend_news):
        """
        Агрегированный индекс влияния для группы новостей одного тренда.

        Returns:
            Словарь с метриками: trend_influence_score, max_influence,
            min_influence, std_influence, top_news, total_news,
            influence_distribution
        """
        if not trend_news:
            return {'error': 'Нет новостей для анализа'}

        df = self.calculate_batch_influence(trend_news)

        return {
            'trend_influence_score': round(df['influence_score'].mean(), 3),
            'max_influence':         round(df['influence_score'].max(), 3),
            'min_influence':         round(df['influence_score'].min(), 3),
            'std_influence':         round(df['influence_score'].std(), 3),
            'top_news':              df.head(3).to_dict('records'),
            'total_news':            len(trend_news),
            'influence_distribution': {
                'high':   int(len(df[df['influence_score'] > 0.7])),
                'medium': int(len(df[(df['influence_score'] > 0.4) &
                                     (df['influence_score'] <= 0.7)])),
                'low':    int(len(df[df['influence_score'] <= 0.4])),
            },
        }


def create_influence_analyzer(weights=None):
    """Фабричная функция для создания InfluenceIndexCalculator."""
    return InfluenceIndexCalculator(weights=weights)