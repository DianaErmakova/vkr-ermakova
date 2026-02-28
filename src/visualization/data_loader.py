"""
Загрузка и подготовка данных для дашборда.
Предобработка текста делегирована в nlp_processing.TextPreprocessor.
"""

import pandas as pd
import os
import sys
import logging
import yfinance as yf
from datetime import datetime, timedelta

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

# Используем централизованный препроцессор вместо своей clean_news_text
try:
    from nlp_processing.text_preprocessor import get_sentiment_preprocessor
    _preprocessor = get_sentiment_preprocessor()
except ImportError:
    _preprocessor = None
    logger.warning("nlp_processing недоступен, используем встроенную очистку")


def _clean_text(text) -> str:
    """
    Очистка текста новости.
    Использует TextPreprocessor если доступен, иначе fallback.
    """
    if pd.isna(text):
        return ""
    if _preprocessor:
        return _preprocessor.clean(str(text))
    # Fallback: убираем только b'...'
    s = str(text).strip()
    if s.startswith("b'") or s.startswith('b"'):
        s = s[2:].rstrip("'\"")
    return s


def load_djia_data() -> pd.DataFrame:
    """
    Загрузка датасета DJIA.

    Returns:
        DataFrame с колонками Date, Label, Top1-Top25
        или пустой DataFrame при ошибке
    """
    from config import DATA_PATH
    file_path = DATA_PATH

    if not os.path.exists(file_path):
        logger.error(f"Файл не найден: {file_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        logger.info(
            f"DJIA загружен: {len(df)} строк, "
            f"период {df['Date'].min().date()} — {df['Date'].max().date()}"
        )
        return df
    except Exception as e:
        logger.error(f"Ошибка загрузки DJIA: {e}")
        return pd.DataFrame()


def extract_day_news(row, max_news: int = 25) -> list:
    """
    Извлечение новостей за один торговый день.

    Args:
        row:      строка DataFrame DJIA
        max_news: максимальное количество новостей (Top1-Top25)

    Returns:
        Список очищенных строк новостей
    """
    news_list = []
    for i in range(1, max_news + 1):
        col = f'Top{i}'
        if col in row and pd.notna(row[col]):
            text = _clean_text(row[col])
            if text and len(text) > 10:
                news_list.append(text)
    return news_list


def analyze_djia_week(df, analyzer, week_offset: int = 0, days: int = 7) -> pd.DataFrame:
    """
    Анализ тональности для выбранного периода датасета DJIA.

    Args:
        df:          DataFrame DJIA
        analyzer:    SentimentAnalyzer
        week_offset: смещение в неделях от начала датасета (0 = август 2008)
        days:        количество дней в периоде

    Returns:
        DataFrame с колонками:
            date, label, avg_sentiment, sentiment_std,
            positive_count, neutral_count, negative_count,
            news_count, top_news
    """
    if df.empty:
        logger.error("Передан пустой DataFrame")
        return pd.DataFrame()

    start_idx = week_offset * days
    end_idx = min(start_idx + days, len(df))

    if start_idx >= len(df):
        logger.warning(
            f"week_offset={week_offset} выходит за пределы датасета ({len(df)} строк)"
        )
        return pd.DataFrame()

    sample_days = df.iloc[start_idx:end_idx]
    results = []

    for _, row in sample_days.iterrows():
        day_news = extract_day_news(row)

        if not day_news:
            logger.debug(f"Нет новостей для {row['Date']}")
            continue

        # Анализируем первые 5 новостей для скорости
        day_scores = []
        day_sentiments = []
        analyzed_news = day_news[:5]

        for news_text in analyzed_news:
            try:
                result = analyzer.analyze_text(news_text)
                day_scores.append(result['score'])
                day_sentiments.append(result['sentiment'])
            except Exception as e:
                logger.warning(f"Ошибка анализа новости: {e}")

        if not day_scores:
            continue

        avg_score = sum(day_scores) / len(day_scores)

        # Стандартное отклонение показывает разброс тональности за день
        std_score = (
            (sum((s - avg_score) ** 2 for s in day_scores) / len(day_scores)) ** 0.5
            if len(day_scores) > 1 else 0.0
        )

        results.append({
            'date': row['Date'],
            'label': int(row['Label']),
            'avg_sentiment': round(avg_score, 4),
            'sentiment_std': round(std_score, 4),
            'positive_count': day_sentiments.count('positive'),
            'neutral_count': day_sentiments.count('neutral'),
            'negative_count': day_sentiments.count('negative'),
            'news_count': len(day_news),
            # Первая новость дня для отображения в дашборде
            'top_news': analyzed_news[0][:120] if analyzed_news else '',
        })

    result_df = pd.DataFrame(results)

    if not result_df.empty:
        logger.info(
            f"Проанализировано {len(result_df)} дней | "
            f"avg_sentiment={result_df['avg_sentiment'].mean():.3f}"
        )

    return result_df


def load_stock_prices(ticker: str, start_date: str, end_date: str):
    """
    Загрузка цен акций через yfinance.

    Args:
        ticker:     тикер (например 'AAPL')
        start_date: начало периода 'YYYY-MM-DD'
        end_date:   конец периода  'YYYY-MM-DD'

    Returns:
        DataFrame с ценами и индикаторами или None при ошибке
    """
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)

        if data.empty:
            logger.warning(f"Нет данных для {ticker} за период {start_date}-{end_date}")
            return None

        data['returns'] = data['Close'].pct_change() * 100
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()

        logger.info(f"{ticker}: загружено {len(data)} торговых дней")
        return data

    except Exception as e:
        logger.error(f"Ошибка загрузки {ticker}: {e}")
        return None


def get_ticker_from_company(company_name: str) -> str:
    """Преобразование названия компании в биржевой тикер"""
    ticker_map = {
        "Tesla": "TSLA",
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Nvidia": "NVDA",
        "Amazon": "AMZN",
        "Google": "GOOGL",
        "Meta": "META",
        "Netflix": "NFLX",
        "Intel": "INTC",
        "AMD": "AMD",
    }
    return ticker_map.get(company_name, company_name)