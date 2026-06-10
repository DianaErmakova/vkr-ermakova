"""
Композитный индекс влияния медиасобытий на рынок.

Формула из литературного обзора:
    Influence = w1 * Интенсивность + w2 * Тональность + w3 * Виральность + w4 * Авторитетность

Веса компонентов фиксированы и обоснованы в литературном обзоре (раздел 2.3).
Их изменение возможно только при наличии размеченных исторических данных
для валидации, что выходит за рамки данной работы.

Примечание по виральности:
    В текущей реализации используется прокси-метрика — количество уникальных источников.
    Чем больше изданий опубликовало новость, тем выше значение виральности.
    Это замена данным соцсетей (Twitter/Reddit API), доступ к которым ограничен.
    Интеграция с реальными соцсетями — направление дальнейшего развития.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class InfluenceIndexCalculator:
    """
    Калькулятор композитного индекса влияния медиасобытий на рынок.

    Веса компонентов зафиксированы согласно литературному обзору.
    Для изменения весов передайте словарь в параметре weights.

    Компоненты индекса:
        intensity  (30%) — нормированное число упоминаний
        sentiment  (40%) — тональность (FinBERT/RuBERT), диапазон [-1, 1]
        virality   (20%) — скорость распространения (прокси: количество источников)
        authority  (10%) — авторитетность источника (Reuters=1.0, блог=0.2)

    Виральность рассчитывается по формуле:
        virality = min(1.0, (sources - 1) / 4)
        где sources — количество уникальных источников, опубликовавших новость.
        1 источник → 0, 5+ источников → 1.0
    """

    # Вариант A (финальный)
    DEFAULT_WEIGHTS_A = {
        'intensity': 0.3,
        'sentiment': 0.4,
        'virality': 0.2,
        'authority': 0.1,
    }

    # Вариант B
    DEFAULT_WEIGHTS_B = {
        'intensity': 0.1,
        'sentiment': 0.4,
        'virality': 0.2,
        'authority': 0.3,
    }

    # Вариант C
    DEFAULT_WEIGHTS_C = {
        'intensity': 0.15,
        'sentiment': 0.6,
        'virality': 0.15,
        'authority': 0.10,
    }

    # Вариант D
    DEFAULT_WEIGHTS_D = {
        'intensity': 0.25,
        'sentiment': 0.25,
        'virality': 0.25,
        'authority': 0.25,
    }

    DEFAULT_WEIGHTS = DEFAULT_WEIGHTS_A

    # Авторитетность источников (экспертная оценка)
    SOURCE_AUTHORITY = {
        # Ведущие мировые агентства
        'reuters':          1.0,
        'bloomberg':        1.0,
        'wsj':              1.0,
        'ft':               1.0,
        # Крупные деловые издания
        'cnbc':             0.9,
        'bbc':              0.8,
        'apnews':           0.85,
        'marketwatch':      0.75,
        # Финансовые порталы
        'yahoo finance':    0.7,
        'seeking alpha':    0.6,
        'businessinsider':  0.6,
        # Соцсети и блоги
        'reddit':           0.4,
        'twitter':          0.3,
        'blog':             0.2,
        'unknown':          0.5,
    }

    def __init__(self, weights=None):
        """
        Инициализация калькулятора.

        Args:
            weights: словарь с весами компонентов.
                     Если None — используются DEFAULT_WEIGHTS.
                     Сумма весов автоматически нормируется к 1.
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
            # Если нет min/max, используем порог 100 упоминаний
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
                'positive_only' — только позитивные новости дают вклад (0 для негатива)
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

        Поддерживает:
            - Социальные метрики (retweets, likes, comments, shares, time_window)
            - Прокси-метрику source_count (количество уникальных источников)

        Args:
            spread_data: словарь с метриками (может быть пустым)

        Returns:
            float в диапазоне [0, 1]
        """
        if not spread_data:
            return 0.0

        components = []

        # Социальные метрики (для будущего расширения)
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

        # Прокси-метрика: количество уникальных источников
        if 'source_count' in spread_data:
            sources = spread_data['source_count']
            # 1 источник = 0, 5+ источников = 1.0
            val = min(1.0, (sources - 1) / 4) if sources > 1 else 0.0
            components.append(val)

        # Готовая прокси-метрика (если передана)
        if 'virality_proxy' in spread_data:
            components.append(spread_data['virality_proxy'])

        return float(np.mean(components)) if components else 0.0

    def calculate_authority(self, source):
        """
        Авторитетность источника на основе домена или названия.

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
            news_item: dict с полями:
                mentions_count  — количество упоминаний
                sentiment_score — оценка тональности [-1, 1]
                spread_data     — данные о распространении (может быть {})
                source          — источник
                max_mentions    — (опционально) максимум для нормировки
                min_mentions    — (опционально) минимум для нормировки

        Returns:
            dict с influence_score, components, weights, raw_sentiment
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
            result['date'] = item.get('date', '')
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
            dict с метриками: trend_influence_score, max_influence,
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