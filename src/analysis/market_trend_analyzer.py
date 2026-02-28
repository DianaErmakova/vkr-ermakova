"""
Главный модуль анализа рыночных трендов.

Объединяет сбор новостей, кластеризацию, анализ тональности
и расчёт индекса влияния в единый пайплайн.

Расположение файла: src/analysis/market_trend_analyzer.py

Импорты абсолютные — src/ добавляется в sys.path при запуске
через app.py, eda_djia.py и pytest.ini (pythonpath = src .).
Это стандартный подход для проектов где src/ не является
установленным пакетом (нет setup.py / pyproject.toml).
"""

import logging
import pandas as pd

from data_collection.news_collector import NewsCollector
from data_collection.stock_collector import StockCollector
from analysis.trend_clusterer import TrendClusterer

logger = logging.getLogger(__name__)

try:
    from analysis.sentiment_analyzer import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"SentimentAnalyzer недоступен: {e}")
    SENTIMENT_AVAILABLE = False
except Exception as e:
    logger.warning(f"Ошибка импорта SentimentAnalyzer: {e}")
    SENTIMENT_AVAILABLE = False

try:
    from analysis.influence_index import InfluenceIndexCalculator
    INFLUENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"InfluenceIndexCalculator недоступен: {e}")
    INFLUENCE_AVAILABLE = False


# Демо-тексты используются когда NewsAPI недоступен или не вернул статей
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

    Оркестрирует: NewsCollector -> TextPreprocessor -> TrendClusterer
                               -> SentimentAnalyzer -> InfluenceIndexCalculator

    Args:
        news_api_key:     ключ NewsAPI (если None — используются демо-данные)
        enable_sentiment: включить анализ тональности (требует загрузки модели)
    """

    def __init__(self, news_api_key=None, enable_sentiment=True):
        self.news_collector = NewsCollector(api_key=news_api_key)
        self.stock_collector = StockCollector()
        self.trend_clusterer = TrendClusterer()
        self.enable_sentiment = enable_sentiment
        self.sentiment_analyzer = None

        if self.enable_sentiment and SENTIMENT_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                logger.info("Анализатор тональности инициализирован")
            except Exception as e:
                logger.error(f"Ошибка создания SentimentAnalyzer: {e}")
                self.enable_sentiment = False
        elif self.enable_sentiment and not SENTIMENT_AVAILABLE:
            logger.warning("Анализатор тональности недоступен")
            self.enable_sentiment = False

    def analyze_market_trends(self, companies, pages=1):
        """
        Базовый анализ трендов: сбор новостей + BERTopic кластеризация.

        Args:
            companies: список названий компаний для поиска новостей
            pages:     количество страниц NewsAPI (1 стр. = 20 статей)

        Returns:
            Словарь с total_news, trends_found, trends_info, news_samples
        """
        # Если нет API-ключа и компании не указаны — сразу демо-режим
        if not self.news_collector.api_key:
            logger.info("Демо-режим: используем тестовые данные")
            all_news = _DEMO_TEXTS
        else:
            # Пытаемся собрать реальные новости через NewsAPI/RSS
            all_news = []
            for company in companies:
                try:
                    news = self.news_collector.get_news(company, pages=pages)
                    all_news.extend([article['title'] for article in news])
                except Exception as e:
                    logger.error(f"Ошибка при сборе новостей для {company}: {e}")

            # Если ничего не собрали — падаем на демо
            if not all_news:
                logger.info("Новости не собраны, используем демо-данные")
                all_news = _DEMO_TEXTS

        # Кластеризация (если текстов достаточно)
        if len(all_news) >= 2:
            topics = self.trend_clusterer.fit_clusters(all_news)
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
            # Внутренний ключ для передачи в influence-пайплайн
            results['_individual_scores'] = individual_scores

            logger.info(f"Средняя тональность: {sentiment_summary['average_score']:.2f}")

        except Exception as e:
            logger.error(f"Ошибка при анализе тональности: {e}")
            results['sentiment_analysis'] = {'error': str(e)}

        return results

    def analyze_with_influence(self, companies, pages=1):
        """
        Полный анализ: тренды + тональность + индекс влияния.

        Индекс влияния использует реальные данные тональности из SentimentAnalyzer.
        Виральность = 0 (ограничение: данные соцсетей недоступны через NewsAPI).

        Returns:
            Результат analyze_with_sentiment, дополненный influence_analysis
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

        influence_items = []
        for i, text in enumerate(news_samples):
            real_score = individual_scores[i]['score'] if i < len(individual_scores) else 0.0
            influence_items.append({
                'title':           text[:100],
                'mentions_count':  1,
                'max_mentions':    len(news_samples),
                'min_mentions':    1,
                'sentiment_score': real_score,
                'spread_data':     {},
                'source':          'newsapi',
                'date':            '',
            })

        if influence_items:
            top_influencers = influence_calc.identify_top_influencers(
                influence_items, top_n=5
            )
            trend_influence = influence_calc.calculate_trend_influence(influence_items)

            results['influence_analysis'] = {
                'trend_score':     trend_influence,
                'top_influencers': (
                    top_influencers.to_dict('records')
                    if not top_influencers.empty else []
                ),
                'total_items': len(influence_items),
                'note': (
                    'Виральность = 0: данные социальных сетей недоступны через NewsAPI. '
                    'Тональность рассчитана моделью FinBERT/RoBERTa.'
                ),
            }

            logger.info(
                f"Индекс влияния тренда: "
                f"{trend_influence.get('trend_influence_score', 'N/A')}"
            )

        return results

    def analyze_with_metrics(self, companies):
        """Для обратной совместимости."""
        results = self.analyze_market_trends(companies)

        if hasattr(self.trend_clusterer.topic_model, 'get_topic_info'):
            topic_info = self.trend_clusterer.get_trends_info()
            results['clustering_metrics'] = {
                'topics_identified':  len(topic_info[topic_info['Topic'] != -1]),
                'avg_docs_per_topic': topic_info['Count'].mean(),
            }

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