"""
Сборщик новостей из нескольких источников.

Источники:
  - NewsAPI       (требует API-ключ, бесплатный план: 100 запросов/день)
  - RSS-ленты     (без ключа: Reuters, BBC, FT, CNBC, Yahoo Finance)

Использование:
    collector = NewsCollector(api_key="ваш_ключ")   # с NewsAPI
    collector = NewsCollector()                       # только RSS

    news = collector.get_news("Tesla", days_back=7)
    news = collector.get_news_by_date("Apple", "2024-01-01", "2024-01-31")
    news = collector.get_multi_source("Nvidia", days_back=3)
"""

import requests
import logging
import time
from datetime import datetime, timedelta
from urllib.parse import quote

logger = logging.getLogger(__name__)


# Авторитетность источников (используется InfluenceIndexCalculator)
SOURCE_AUTHORITY = {
    'reuters.com':      1.0,
    'bloomberg.com':    1.0,
    'wsj.com':          1.0,
    'ft.com':           1.0,
    'cnbc.com':         0.9,
    'bbc.com':          0.8,
    'bbc.co.uk':        0.8,
    'apnews.com':       0.85,
    'marketwatch.com':  0.75,
    'finance.yahoo.com':0.7,
    'seekingalpha.com': 0.6,
    'businessinsider.com': 0.6,
    'reddit.com':       0.4,
    'unknown':          0.5,
}

# RSS-ленты деловых изданий (без API-ключа)
RSS_FEEDS = {
    # существующие
    'reuters_business': 'https://feeds.reuters.com/reuters/businessNews',
    'bbc_business': 'https://feeds.bbci.co.uk/news/business/rss.xml',
    'cnbc_top': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
    'yahoo_finance': 'https://finance.yahoo.com/rss/topstories',

    # русские реальные новости
    'rbc': 'https://www.rbc.ru/rss',
    'tass': 'http://tass.ru/rss/v2.xml',
    'rt_russian': 'https://russian.rt.com/rss',
    'kommersant': 'https://www.kommersant.ru/RSS/news.xml',
    'vedomosti': 'https://vedomosti.ru/rss/rss.xml',
}


class NewsCollector:
    """
    Сборщик новостей с поддержкой NewsAPI и RSS.

    Args:
        api_key: ключ NewsAPI (если None — только RSS)
        timeout: таймаут HTTP-запросов в секундах
        max_retries: количество повторных попыток при ошибке сети
    """

    NEWSAPI_BASE = "https://newsapi.org/v2/"

    def __init__(self, api_key=None, timeout=10, max_retries=3):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': 'MarketTrendAnalyzer/1.0'})

    # Публичный интерфейс

    def get_news(self, query, pages=1, days_back=7, language='en'):
        """
        Получить новости по запросу.

        Args:
            query:     поисковый запрос (например, "Tesla earnings")
            pages:     количество страниц NewsAPI (1 стр. = 20 статей)
            days_back: глубина поиска в днях от сегодня
            language:  язык ('en', 'ru', ...)

        Returns:
            Список словарей с ключами: title, description, url,
            source, published_at, authority_score
        """
        if not self.api_key:
            logger.info("API-ключ не задан — используем RSS")
            return self._get_rss_news(query, language=language)

        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        return self._fetch_newsapi(
            query=query,
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d'),
            pages=pages,
            language=language
        )

    def get_news_by_date(self, query, from_date, to_date,
                         pages=1, language='en'):
        """
        Получить новости за конкретный период.

        Args:
            query:     поисковый запрос
            from_date: начало периода 'YYYY-MM-DD'
            to_date:   конец периода  'YYYY-MM-DD'
            pages:     количество страниц
            language:  язык

        Returns:
            Список статей
        """
        if not self.api_key:
            logger.warning("get_news_by_date требует API-ключ NewsAPI")
            return []

        # Валидация дат
        try:
            dt_from = datetime.strptime(from_date, '%Y-%m-%d')
            dt_to   = datetime.strptime(to_date,   '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Неверный формат даты: {e}. Ожидается YYYY-MM-DD")
            return []

        if dt_from > dt_to:
            logger.error("from_date не может быть позже to_date")
            return []

        # Бесплатный план NewsAPI ограничен 30 днями назад
        oldest_allowed = datetime.utcnow() - timedelta(days=29)
        if dt_from < oldest_allowed:
            logger.warning(
                f"NewsAPI (бесплатный план) даёт данные только за последние 30 дней. "
                f"Запрошена дата {from_date} — будет скорректирована."
            )
            dt_from = oldest_allowed

        return self._fetch_newsapi(
            query=query,
            from_date=dt_from.strftime('%Y-%m-%d'),
            to_date=dt_to.strftime('%Y-%m-%d'),
            pages=pages,
            language=language
        )

    def get_multi_source(self, query, days_back=7, language='en'):
        """
        Собрать новости из всех доступных источников (NewsAPI + RSS).

        Args:
            query:     поисковый запрос
            days_back: глубина поиска в днях

        Returns:
            Объединённый список статей без дублей (по URL)
        """
        all_articles = []
        seen_urls = set()

        # NewsAPI
        if self.api_key:
            api_articles = self.get_news(query, pages=2, days_back=days_back)
            for art in api_articles:
                if art['url'] not in seen_urls:
                    seen_urls.add(art['url'])
                    all_articles.append(art)
            logger.info(f"NewsAPI: {len(api_articles)} статей")

        # RSS
        rss_articles = self._get_rss_news(query, language=language)
        for art in rss_articles:
            if art['url'] not in seen_urls:
                seen_urls.add(art['url'])
                all_articles.append(art)
        logger.info(f"RSS: {len(rss_articles)} статей")

        logger.info(f"Итого уникальных статей: {len(all_articles)}")
        return all_articles

    # NewsAPI

    def _fetch_newsapi(self, query, from_date, to_date, pages, language):
        """Запросы к NewsAPI /everything с повторными попытками"""
        articles = []

        for page in range(1, pages + 1):
            url = (
                f"{self.NEWSAPI_BASE}everything"
                f"?q={quote(query)}"
                f"&from={from_date}"
                f"&to={to_date}"
                f"&language={language}"
                f"&sortBy=relevancy"
                f"&pageSize=20"
                f"&page={page}"
                f"&apiKey={self.api_key}"
            )

            data = self._safe_get(url)
            if data is None:
                break

            status = data.get('status', '')
            if status != 'ok':
                code = data.get('code', 'unknown')
                msg  = data.get('message', '')
                logger.error(f"NewsAPI ошибка: [{code}] {msg}")

                if code == 'rateLimited':
                    logger.warning("Лимит запросов NewsAPI исчерпан. Подождите.")
                elif code == 'apiKeyInvalid':
                    logger.error("Неверный API-ключ NewsAPI.")
                break

            raw = data.get('articles', [])
            if not raw:
                logger.info(f"Страница {page}: статей нет, завершаем.")
                break

            for art in raw:
                normalized = self._normalize_article(art, source='newsapi')
                if normalized:
                    articles.append(normalized)

            logger.info(f"Страница {page}: получено {len(raw)} статей")

            # Пауза между запросами чтобы не бить по rate limit
            if page < pages:
                time.sleep(0.5)

        return articles

    # RSS

    def _get_rss_news(self, query, language='en'):
        """
        Загружает RSS-ленты и фильтрует статьи по ключевым словам.
        Не требует API-ключа.
        """
        try:
            import feedparser
        except ImportError:
            logger.warning(
                "feedparser не установлен. Установите: pip install feedparser. "
                "RSS будет пропущен."
            )
            return []

        query_words = query.lower().split()
        articles = []

        for feed_name, feed_url in RSS_FEEDS.items():
            # фильтруем ленты по языку
            if language == 'ru' and not any(r in feed_name for r in ['rbc', 'tass', 'rt', 'kommersant', 'vedomosti']):
                continue
            if language == 'en' and any(r in feed_name for r in ['rbc', 'tass', 'rt', 'kommersant', 'vedomosti']):
                continue

            try:
                feed = feedparser.parse(feed_url)
                count = 0

                for entry in feed.entries:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    combined = (title + ' ' + summary).lower()

                    if not any(word in combined for word in query_words):
                        continue

                    published = self._parse_rss_date(entry)
                    source_url = entry.get('link', '')
                    source_name = self._extract_domain(source_url) or feed_name

                    articles.append({
                        'title': title,
                        'description': summary[:300] if summary else '',
                        'url': source_url,
                        'source': source_name,
                        'published_at': published,
                        'authority_score': self._get_authority(source_url),
                        'data_source': 'rss',
                    })
                    count += 1

                logger.info(f"RSS [{feed_name}]: {count} релевантных статей")

            except Exception as e:
                logger.warning(f"Ошибка RSS [{feed_name}]: {e}")

        return articles

    # HTTP с повторными попытками

    def _safe_get(self, url):
        """
        GET-запрос с повторными попытками и обработкой ошибок.

        Returns:
            dict (JSON-ответ) или None при неудаче
        """
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.get(url, timeout=self.timeout)

                # HTTP-ошибки
                if response.status_code == 401:
                    logger.error("NewsAPI: неверный API-ключ (401)")
                    return None
                if response.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limit (429). Ждём {wait}с...")
                    time.sleep(wait)
                    continue
                if response.status_code == 426:
                    logger.error(
                        "NewsAPI: требуется платный план (426). "
                        "Запрос по датам доступен только в платной версии."
                    )
                    return None
                if response.status_code >= 500:
                    logger.warning(f"Сервер недоступен ({response.status_code}), попытка {attempt}")
                    time.sleep(2 ** attempt)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning(f"Таймаут (попытка {attempt}/{self.max_retries})")
                last_error = "timeout"
            except requests.exceptions.ConnectionError:
                logger.warning(f"Нет соединения (попытка {attempt}/{self.max_retries})")
                last_error = "connection"
                time.sleep(2)
            except requests.exceptions.JSONDecodeError:
                logger.error("Ответ не является валидным JSON")
                return None
            except requests.exceptions.RequestException as e:
                logger.error(f"Неожиданная ошибка запроса: {e}")
                return None

        logger.error(f"Все {self.max_retries} попытки исчерпаны. Последняя ошибка: {last_error}")
        return None

    # Вспомогательные методы

    def _normalize_article(self, raw, source='newsapi'):
        """
        Приводит сырую статью NewsAPI к единому формату.
        Возвращает None если статья невалидна.
        """
        title = (raw.get('title') or '').strip()
        url   = (raw.get('url')   or '').strip()

        # Пропускаем статьи без заголовка или URL
        if not title or not url:
            return None

        # Пропускаем удалённые статьи
        if title == '[Removed]':
            return None

        source_info = raw.get('source', {})
        source_name = source_info.get('name', '') or self._extract_domain(url)

        return {
            'title':           title,
            'description':     (raw.get('description') or '')[:300],
            'url':             url,
            'source':          source_name,
            'published_at':    raw.get('publishedAt', ''),
            'authority_score': self._get_authority(url),
            'data_source':     source,
        }

    def _get_authority(self, url):
        """Определяет авторитетность источника по URL"""
        if not url:
            return SOURCE_AUTHORITY['unknown']
        url_lower = url.lower()
        for domain, score in SOURCE_AUTHORITY.items():
            if domain in url_lower:
                return score
        return SOURCE_AUTHORITY['unknown']

    def _extract_domain(self, url):
        """Извлекает домен из URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace('www.', '')
        except Exception:
            return 'unknown'

    def _parse_rss_date(self, entry):
        """Парсит дату из RSS-записи"""
        for field in ('published', 'updated', 'created'):
            raw = entry.get(field, '')
            if raw:
                try:
                    import email.utils
                    parsed = email.utils.parsedate_to_datetime(raw)
                    return parsed.strftime('%Y-%m-%dT%H:%M:%SZ')
                except Exception:
                    return raw
        return ''

# Демо

if __name__ == "__main__":
    import os

    print("=" * 55)
    print("ДЕМО: NewsCollector")
    print("=" * 55)

    api_key = os.getenv("NEWS_API_KEY")  # или вставьте ключ строкой

    collector = NewsCollector(api_key=api_key)

    if api_key:
        print("\n1. Новости за последние 7 дней (Tesla):")
        news = collector.get_news("Tesla", pages=1, days_back=7)
        print(f"   Получено: {len(news)} статей")
        for art in news[:3]:
            print(f"   [{art['source']}] {art['title'][:70]}...")
            print(f"   Авторитетность: {art['authority_score']}")

        print("\n2. Новости за конкретный период:")
        news2 = collector.get_news_by_date("Apple", "2024-12-01", "2024-12-07")
        print(f"   Получено: {len(news2)} статей")

    print("\n3. Мультиисточниковый сбор (RSS без ключа):")
    news3 = collector.get_multi_source("AI technology", days_back=3)
    print(f"   Получено: {len(news3)} статей")
    for art in news3[:3]:
        print(f"   [{art['source']}] {art['title'][:70]}...")