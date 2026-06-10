"""
Главный модуль анализа рыночных трендов.

Объединяет сбор новостей, кластеризацию, анализ тональности
и расчёт индекса влияния в единый пайплайн.

Архитектура:
    NewsCollector (данные) → TextPreprocessor (очистка) → TrendClusterer (темы)
                                                       ↓
                                              SentimentAnalyzer (тональность)
                                                       ↓
                                            InfluenceIndexCalculator (индекс)
                                                       ↓
                                                   дашборд (Streamlit)

Импорты абсолютные — src/ добавляется в sys.path при запуске
через app.py, eda_djia.py и pytest.ini (pythonpath = src .).
"""

import logging
import pandas as pd
from collections import defaultdict

from data_collection.news_collector import NewsCollector
from data_collection.stock_collector import StockCollector
from analysis.trend_clusterer import TrendClusterer

logger = logging.getLogger(__name__)

# Опциональные модули (если недоступны — отключаем функциональность)
try:
    from analysis.sentiment_analyzer import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SentimentAnalyzer недоступен: {e}")
    SENTIMENT_AVAILABLE = False

try:
    from analysis.influence_index import InfluenceIndexCalculator
    INFLUENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"InfluenceIndexCalculator недоступен: {e}")
    INFLUENCE_AVAILABLE = False

try:
    from .industry_classifier import create_industry_classifier
    INDUSTRY_AVAILABLE = True
except ImportError:
    INDUSTRY_AVAILABLE = False
    logger.warning("IndustryClassifier недоступен")

# Демо-тексты — используются когда нет API-ключа или API не вернул статей
# Содержат 15 новостей по 4 темам: электромобили, AI/ML, криптовалюты, облачные технологии
_DEMO_TEXTS = [
    "Tesla electric vehicles have advanced battery technology and long range",
    "Electric cars reduce air pollution and carbon emissions in cities",
    "Charging stations for electric vehicles are expanding across Europe",
    "Artificial intelligence helps doctors diagnose diseases more accurately",
    "Machine learning algorithms improve stock market prediction models",
    "Neural networks can generate realistic images and creative content",
    "Bitcoin cryptocurrency reaches record high prices this year",
    "Ethereum blockchain transitions to energy efficient proof of stake",
    "NFT digital collectibles create new markets for artists",
    "Amazon AWS dominates the cloud computing services market",
    "Microsoft Azure offers powerful machine learning tools for developers",
    "Cloud computing enables remote work and digital transformation",
    "TSMC manufactures advanced semiconductor chips for electronics",
    "Global chip shortage impacts car production and consumer electronics",
    "Intel builds new semiconductor factories to increase production",
]


class MarketTrendAnalyzer:
    """
    Главный класс анализа рыночных трендов.

    Оркестрирует весь пайплайн:
        1. Сбор новостей (NewsCollector)
        2. Кластеризация тем (TrendClusterer / BERTopic)
        3. Анализ тональности (SentimentAnalyzer / FinBERT/RuBERT)
        4. Расчёт индекса влияния (InfluenceIndexCalculator)
        5. Отраслевая классификация (эвристическая)

    Args:
        news_api_key:     ключ NewsAPI (если None — используются демо-данные или RSS)
        enable_sentiment: включить анализ тональности (требует загрузки модели)
        language:         язык новостей ('english' или 'russian')
    """

    def __init__(self, news_api_key=None, enable_sentiment=True, language='english'):
        self.news_collector = NewsCollector(api_key=news_api_key)
        self.stock_collector = StockCollector()
        self.trend_clusterer = TrendClusterer(language=language)
        self.enable_sentiment = enable_sentiment
        self.sentiment_analyzer = None
        self.language = language
        self.industry_classifier = None

        # Инициализация отраслевого классификатора (эвристика)
        if INDUSTRY_AVAILABLE:
            try:
                self.industry_classifier = create_industry_classifier()
                logger.info("Отраслевой классификатор инициализирован")
            except Exception as e:
                logger.warning(f"Ошибка инициализации IndustryClassifier: {e}")

        # Инициализация анализатора тональности
        if self.enable_sentiment and SENTIMENT_AVAILABLE:
            try:
                # Выбор модели в зависимости от языка
                if language == 'russian':
                    model_name = "rubert-sentiment"
                else:
                    model_name = "distilroberta-financial"

                self.sentiment_analyzer = SentimentAnalyzer(model_name=model_name)
                logger.info(f"Анализатор тональности инициализирован (модель: {model_name})")
            except Exception as e:
                logger.error(f"Ошибка создания SentimentAnalyzer: {e}")
                self.enable_sentiment = False
        elif self.enable_sentiment and not SENTIMENT_AVAILABLE:
            logger.warning("Анализатор тональности недоступен")
            self.enable_sentiment = False

    def analyze_market_trends(self, companies, pages=1):
        """
        Базовый анализ: сбор новостей + кластеризация (BERTopic).

        Args:
            companies: список названий компаний для поиска
            pages:     количество страниц NewsAPI (1 стр. = 20 статей)

        Returns:
            dict: total_news, trends_found, trends_info, news_samples
        """
        # Если нет API-ключа — сразу демо-режим
        if not self.news_collector.api_key:
            logger.info("Демо-режим: используем тестовые данные")
            all_news = _DEMO_TEXTS
        else:
            all_news = []
            for company in companies:
                try:
                    news = self.news_collector.get_news(company, pages=pages, language=self.language)
                    all_news.extend([article['title'] for article in news])
                except Exception as e:
                    logger.error(f"Ошибка при сборе новостей для {company}: {e}")

            # Если ничего не собрали — падаем на демо
            if not all_news:
                logger.info("Новости не собраны, используем демо-данные")
                all_news = _DEMO_TEXTS

        # Кластеризация (требуется минимум 2 текста)
        if len(all_news) >= 2:
            self.trend_clusterer.fit_clusters(all_news)
            trends_info = self.trend_clusterer.get_trends_info()
        else:
            logger.warning(f"Недостаточно текстов для кластеризации: {len(all_news)}")
            trends_info = pd.DataFrame()

        return {
            'total_news': len(all_news),
            'trends_found': len(trends_info),
            'trends_info': trends_info,
            'news_samples': all_news[:15],
        }

    def analyze_with_sentiment(self, companies, pages=1):
        """
        Анализ трендов с тональностью новостей.

        Returns:
            Результат analyze_market_trends, дополненный sentiment_analysis
        """
        logger.info("Анализ рыночных трендов...")
        results = self.analyze_market_trends(companies, pages)

        if not (self.enable_sentiment and self.sentiment_analyzer):
            results['sentiment_analysis'] = {
                'error': 'Анализатор тональности не инициализирован'
            }
            return results

        logger.info("Анализ тональности новостей...")
        try:
            news_texts = results['news_samples']
            if not news_texts:
                results['sentiment_analysis'] = {'error': 'Нет новостей для анализа'}
                return results

            sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(news_texts)

            individual_scores = []
            for text in news_texts:
                analysis = self.sentiment_analyzer.analyze_text(text)
                individual_scores.append({
                    'text':       text[:80] + "..." if len(text) > 80 else text,
                    'sentiment':  analysis['sentiment'],
                    'score':      analysis['score'],
                    'confidence': analysis['confidence'],
                })

            results['sentiment_analysis'] = {
                'summary':            sentiment_summary,
                'individual_samples': individual_scores,
                'model_used':         self.sentiment_analyzer.get_model_info()['name'],
            }
            # Сохраняем для influence-пайплайна
            results['_individual_scores'] = individual_scores

            logger.info(f"Средняя тональность: {sentiment_summary['average_score']:.2f}")

        except Exception as e:
            logger.error(f"Ошибка при анализе тональности: {e}")
            results['sentiment_analysis'] = {'error': str(e)}

        return results

    def analyze_with_influence(self, companies, pages=1):
        """
        Полный анализ: тренды + тональность + индекс влияния + отрасли.

        Виральность рассчитывается как прокси-метрика:
            количество уникальных источников (чем больше — тем выше виральность)

        Returns:
            Результат analyze_with_sentiment, дополненный influence_analysis
            и industry_classification
        """
        logger.info("Полный анализ с индексом влияния...")
        results = self.analyze_with_sentiment(companies, pages)

        if not INFLUENCE_AVAILABLE:
            results['influence_analysis'] = {
                'error': 'InfluenceIndexCalculator недоступен'
            }
            return results

        logger.info("Расчет композитного индекса влияния...")
        influence_calc = InfluenceIndexCalculator()

        individual_scores = results.pop('_individual_scores', [])
        news_samples = results.get('news_samples', [])

        # Группируем похожие новости по началу заголовка
        source_count = defaultdict(int)
        for text in news_samples:
            title_key = text[:80].strip()
            source_count[title_key] += 1

        influence_items = []
        for i, text in enumerate(news_samples):
            real_score = individual_scores[i]['score'] if i < len(individual_scores) else 0.0
            title_key = text[:80].strip()
            sources = source_count[title_key]

            # Прокси-виральность: 1 источник = 0, 5+ = 1.0
            virality = min(1.0, (sources - 1) / 4) if sources > 1 else 0.0

            influence_items.append({
                'title': text[:100],
                'mentions_count': sources,
                'max_mentions': max(source_count.values()) if source_count else 1,
                'min_mentions': 1,
                'sentiment_score': real_score,
                'spread_data': {
                    'source_count': sources,
                    'virality_proxy': virality
                },
                'source': 'multiple_sources' if sources > 1 else 'single_source',
                'date': '',
            })

        if influence_items:
            top_influencers = influence_calc.identify_top_influencers(influence_items, top_n=5)
            trend_influence = influence_calc.calculate_trend_influence(influence_items)

            results['influence_analysis'] = {
                'trend_score': trend_influence,
                'top_influencers': top_influencers.to_dict('records') if not top_influencers.empty else [],
                'total_items': len(influence_items),
                'note': (
                    'Виральность рассчитана по количеству уникальных источников '
                    '(прокси-метрика). Чем больше изданий опубликовало новость, '
                    'тем выше значение виральности.'
                ),
            }

            logger.info(f"Индекс влияния тренда: {trend_influence.get('trend_influence_score', 'N/A')}")

        # Отраслевая классификация
        if self.industry_classifier:
            try:
                industries = []
                for text in news_samples:
                    industries.append(self.industry_classifier.classify(text))
                results['industry_classification'] = industries
            except Exception as e:
                logger.warning(f"Ошибка классификации отраслей: {e}")

        return results

    def analyze_with_metrics(self, companies):
        """
        Анализ с метриками (для обратной совместимости).
        Возвращает метрики кластеризации и тональности.
        """
        results = self.analyze_market_trends(companies)

        # Метрики кластеризации
        if hasattr(self.trend_clusterer.topic_model, 'get_topic_info'):
            topic_info = self.trend_clusterer.get_trends_info()
            results['clustering_metrics'] = {
                'topics_identified':  len(topic_info[topic_info['Topic'] != -1]),
                'avg_docs_per_topic': topic_info['Count'].mean(),
            }

        # Метрики тональности
        if (self.enable_sentiment and self.sentiment_analyzer
                and results.get('news_samples')):
            try:
                sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(
                    results['news_samples']
                )
                results['sentiment_metrics'] = {
                    'average_sentiment':  sentiment_summary['average_score'],
                    'sentiment_index':    sentiment_summary['sentiment_index'],
                    'dominant_sentiment': sentiment_summary['dominant_sentiment'],
                }
            except Exception as e:
                logger.warning(f"Ошибка расчёта sentiment_metrics: {e}")

        return results


def create_market_trend_analyzer(news_api_key=None):
    """Фабричная функция для создания MarketTrendAnalyzer."""
    return MarketTrendAnalyzer(news_api_key=news_api_key, enable_sentiment=True)