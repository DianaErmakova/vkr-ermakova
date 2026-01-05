# analysis/market_trend_analyzer.py
from data_collection.news_collector import NewsCollector
from data_collection.stock_collector import StockCollector
from analysis.trend_clusterer import TrendClusterer


class MarketTrendAnalyzer:
    def __init__(self, news_api_key=None):
        # Передаем API ключ в NewsCollector
        self.news_collector = NewsCollector(api_key=news_api_key)
        self.stock_collector = StockCollector()
        self.trend_clusterer = TrendClusterer()

    def analyze_market_trends(self, companies, pages=1):
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
                # Тема 1: ELECTRIC VEHICLES - полные предложения
                "Tesla electric vehicles have advanced battery technology and long range",
                "Electric cars reduce air pollution and carbon emissions in cities",
                "Charging stations for electric vehicles are expanding across Europe",
                "Government subsidies make electric cars more affordable for consumers",
                "Battery technology improvements increase electric vehicle performance",

                # Тема 2: ARTIFICIAL INTELLIGENCE
                "Artificial intelligence helps doctors diagnose diseases more accurately",
                "Machine learning algorithms improve stock market prediction models",
                "Neural networks can generate realistic images and creative content",
                "AI automation transforms manufacturing and logistics industries",
                "Deep learning enables breakthroughs in natural language processing",

                # Тема 3: CRYPTOCURRENCY
                "Bitcoin cryptocurrency reaches record high prices this year",
                "Ethereum blockchain transitions to energy efficient proof of stake",
                "NFT digital collectibles create new markets for artists",
                "Cryptocurrency regulations aim to prevent fraud and money laundering",
                "Blockchain technology provides secure and transparent transactions",

                # Тема 4: CLOUD COMPUTING
                "Amazon AWS dominates the cloud computing services market",
                "Microsoft Azure offers powerful machine learning tools for developers",
                "Cloud computing enables remote work and digital transformation",
                "Google Cloud provides data analytics and storage solutions",
                "Hybrid cloud architectures combine public and private infrastructure",

                # Тема 5: SEMICONDUCTORS
                "TSMC manufactures advanced semiconductor chips for electronics",
                "Global chip shortage impacts car production and consumer electronics",
                "Intel builds new semiconductor factories to increase production",
                "NVIDIA graphics processors are essential for gaming and AI",
                "Semiconductor industry faces challenges with supply chain logistics"
            ]

        topics = self.trend_clusterer.fit_clusters(all_news)
        trends_info = self.trend_clusterer.get_trends_info()

        return {
            'total_news': len(all_news),
            'trends_found': len(trends_info),
            'trends_info': trends_info,
            'news_samples': all_news[:5]  # первые 5 новостей для примера
        }

    def analyze_with_metrics(self, companies):
        """Расширенный анализ с метриками"""
        results = self.analyze_market_trends(companies)

        # Добавляем метрики качества
        if hasattr(self.trend_clusterer.topic_model, 'get_topic_info'):
            topic_info = self.trend_clusterer.get_trends_info()
            results['clustering_metrics'] = {
                'topics_identified': len(topic_info[topic_info['Topic'] != -1]),
                'avg_docs_per_topic': topic_info['Count'].mean(),
                'coherence_score': self._calculate_topic_coherence()
            }
        return results
