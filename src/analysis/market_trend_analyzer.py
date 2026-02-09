# market_trend_analyzer.py
from data_collection.news_collector import NewsCollector
from data_collection.stock_collector import StockCollector
from analysis.trend_clusterer import TrendClusterer

# Попробуем импортировать анализатор тональности (с обработкой ошибок)
try:
    from analysis.sentiment_analyzer import SentimentAnalyzer

    SENTIMENT_AVAILABLE = True
except ImportError as e:
    print(f"Не удалось импортировать SentimentAnalyzer: {e}")
    SENTIMENT_AVAILABLE = False
except Exception as e:
    print(f"Ошибка импорта SentimentAnalyzer: {e}")
    SENTIMENT_AVAILABLE = False


class MarketTrendAnalyzer:
    def __init__(self, news_api_key=None, enable_sentiment=True):
        """
        Инициализация анализатора рыночных трендов.

        Args:
            news_api_key: Ключ API для NewsAPI (если None - демо-режим)
            enable_sentiment: Включить анализ тональности (по умолчанию True)
        """
        # Передаем API ключ в NewsCollector
        self.news_collector = NewsCollector(api_key=news_api_key)
        self.stock_collector = StockCollector()
        self.trend_clusterer = TrendClusterer()

        # Настройка анализа тональности
        self.enable_sentiment = enable_sentiment

        # Инициализация анализатора тональности
        self.sentiment_analyzer = None
        if self.enable_sentiment and SENTIMENT_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                print("Анализатор тональности инициализирован")
            except Exception as e:
                print(f"Ошибка создания SentimentAnalyzer: {e}")
                self.enable_sentiment = False
        elif self.enable_sentiment and not SENTIMENT_AVAILABLE:
            print("Анализатор тональности недоступен (модуль не найден)")
            self.enable_sentiment = False

    def analyze_market_trends(self, companies, pages=1):
        """Основной метод анализа трендов (без тональности)"""
        all_news = []
        for company in companies:
            try:
                news = self.news_collector.get_news(company, pages=pages)
                all_news.extend([article['title'] for article in news])
            except Exception as e:
                print(f"Ошибка при сборе новостей для {company}: {e}")

        # Если нет новостей, используем тестовые данные
        if not all_news:
            print("Новости не собраны, используем тестовые данные")
            all_news = [
                # Тема 1: ELECTRIC VEHICLES
                "Tesla electric vehicles have advanced battery technology and long range",
                "Electric cars reduce air pollution and carbon emissions in cities",
                "Charging stations for electric vehicles are expanding across Europe",
                # Тема 2: ARTIFICIAL INTELLIGENCE
                "Artificial intelligence helps doctors diagnose diseases more accurately",
                "Machine learning algorithms improve stock market prediction models",
                "Neural networks can generate realistic images and creative content",
                # Тема 3: CRYPTOCURRENCY
                "Bitcoin cryptocurrency reaches record high prices this year",
                "Ethereum blockchain transitions to energy efficient proof of stake",
                "NFT digital collectibles create new markets for artists",
                # Тема 4: CLOUD COMPUTING
                "Amazon AWS dominates the cloud computing services market",
                "Microsoft Azure offers powerful machine learning tools for developers",
                "Cloud computing enables remote work and digital transformation",
                # Тема 5: SEMICONDUCTORS
                "TSMC manufactures advanced semiconductor chips for electronics",
                "Global chip shortage impacts car production and consumer electronics",
                "Intel builds new semiconductor factories to increase production",
            ]

        topics = self.trend_clusterer.fit_clusters(all_news)
        trends_info = self.trend_clusterer.get_trends_info()

        return {
            'total_news': len(all_news),
            'trends_found': len(trends_info),
            'trends_info': trends_info,
            'news_samples': all_news[:10]  # первые 10 новостей для анализа тональности
        }

    def analyze_with_sentiment(self, companies, pages=1):
        """
        Расширенный анализ с тональностью.

        Args:
            companies: Список компаний для анализа
            pages: Количество страниц новостей

        Returns:
            Результаты анализа с тональностью
        """
        # Получаем базовые результаты
        print("Анализ рыночных трендов...")
        results = self.analyze_market_trends(companies, pages)

        # Добавляем анализ тональности если доступен
        if self.enable_sentiment and self.sentiment_analyzer:
            print("Анализ тональности новостей...")

            try:
                # Анализируем все новости
                news_texts = results['news_samples']

                if not news_texts:
                    print("Нет новостей для анализа тональности")
                    results['sentiment_analysis'] = {'error': 'Нет новостей для анализа'}
                    return results

                # Получаем сводку по тональности
                sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(news_texts)

                # Анализируем отдельные новости (первые 5 для примера)
                individual_scores = []
                for text in news_texts[:5]:
                    analysis = self.sentiment_analyzer.analyze_text(text)
                    individual_scores.append({
                        'text': text[:80] + "..." if len(text) > 80 else text,
                        'sentiment': analysis['sentiment'],
                        'score': analysis['score'],
                        'confidence': analysis['confidence']
                    })

                # Сохраняем результаты
                results['sentiment_analysis'] = {
                    'summary': sentiment_summary,
                    'individual_samples': individual_scores,
                    'model_used': self.sentiment_analyzer.get_model_info()['name']
                }

                # Выводим краткую сводку
                print(f"Средняя тональность: {sentiment_summary['average_score']:.2f}")
                print(f"Индекс настроения: {sentiment_summary['sentiment_index']}")

            except Exception as e:
                print(f"Ошибка при анализе тональности: {e}")
                results['sentiment_analysis'] = {'error': str(e)}

        elif self.enable_sentiment and not self.sentiment_analyzer:
            print("Анализатор тональности недоступен")
            results['sentiment_analysis'] = {'error': 'Анализатор тональности не инициализирован'}

        return results

    def analyze_with_metrics(self, companies):
        """Расширенный анализ с метриками (старая функция - оставляем для совместимости)"""
        results = self.analyze_market_trends(companies)

        # Добавляем метрики качества
        if hasattr(self.trend_clusterer.topic_model, 'get_topic_info'):
            topic_info = self.trend_clusterer.get_trends_info()
            results['clustering_metrics'] = {
                'topics_identified': len(topic_info[topic_info['Topic'] != -1]),
                'avg_docs_per_topic': topic_info['Count'].mean(),
            }

        # Добавляем тональность если доступно
        if self.enable_sentiment and self.sentiment_analyzer and results.get('news_samples'):
            try:
                sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(results['news_samples'])
                results['sentiment_metrics'] = {
                    'average_sentiment': sentiment_summary['average_score'],
                    'sentiment_index': sentiment_summary['sentiment_index'],
                    'dominant_sentiment': sentiment_summary['dominant_sentiment']
                }
            except:
                pass  # Игнорируем ошибки тональности в этом методе

        return results


# Для обратной совместимости - создаем алиас старому конструктору
def create_market_trend_analyzer(news_api_key=None):
    """Старая функция для обратной совместимости"""
    return MarketTrendAnalyzer(news_api_key=news_api_key, enable_sentiment=True)