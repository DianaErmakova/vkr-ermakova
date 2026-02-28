"""
Анализ корреляции между новостными трендами и движением цен акций
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging

logger = logging.getLogger(__name__)


class CorrelationAnalyzer:
    """
    Анализатор корреляции между медиа-трендами и рыночными показателями
    """

    def __init__(self):
        self.correlation_results = {}

    def load_stock_data(self, ticker, start_date, end_date):
        """
        Загрузка исторических данных по акции

        Args:
            ticker: тикер (например, 'TSLA')
            start_date: начальная дата
            end_date: конечная дата

        Returns:
            DataFrame с ценами и доходностью
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date)

            if data.empty:
                logger.warning(f"Нет данных для {ticker}")
                return None

            data['returns'] = data['Close'].pct_change()
            data['direction'] = (data['returns'] > 0).astype(int)

            logger.info(f"Загружено {len(data)} дней данных для {ticker}")
            return data

        except Exception as e:
            logger.error(f"Ошибка загрузки данных для {ticker}: {e}")
            return None

    def prepare_news_features(self, news_data, date_column='date'):
        """
        Подготовка новостных признаков для корреляции.

        Агрегирует данные по дням: считает количество новостей
        и среднюю тональность (если колонка присутствует).

        Args:
            news_data: DataFrame с новостями (дата, текст, тональность и т.д.)
            date_column: название колонки с датой

        Returns:
            DataFrame с агрегированными по дням признаками,
            гарантированно с плоскими именами колонок (без MultiIndex)
        """
        if date_column not in news_data.columns:
            logger.error(f"Колонка {date_column} не найдена")
            return None

        df = news_data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df['_date'] = df[date_column].dt.normalize()

        # Ищем колонку с тональностью
        sentiment_col = None
        candidates = ['sentiment_score', 'sentiment', 'score',
                      'sentiment_analysis', 'sentiment_value', 'avg_sentiment']
        for col in candidates:
            if col in df.columns:
                sentiment_col = col
                logger.debug(f"Найдена колонка тональности: {col}")
                break

        # Считаем количество новостей за день
        result = df.groupby('_date').size().reset_index()
        result.columns = ['date', 'news_count']

        # Добавляем агрегаты по тональности если колонка есть
        if sentiment_col is not None:
            sentiment_agg = (
                df.groupby('_date')[sentiment_col]
                .agg(sentiment_mean='mean', sentiment_std='std')
                .reset_index()
            )
            sentiment_agg.columns = ['date', 'sentiment_mean', 'sentiment_std']
            result = result.merge(sentiment_agg, on='date', how='left')
        else:
            logger.warning("Колонка тональности не найдена, агрегат не добавлен")

        return result

    def calculate_lag_correlation(self, news_features, stock_data, max_lag=5):
        """
        Расчет корреляции с учетом лагов (новости влияют на цену с задержкой)

        Args:
            news_features: DataFrame с новостными признаками по дням
                           (обязательные колонки: date, sentiment_mean)
            stock_data: DataFrame с данными по акции
            max_lag: максимальный лаг в днях

        Returns:
            Словарь с корреляциями для разных лагов
        """
        if 'date' not in news_features.columns:
            return {'error': 'Нет колонки date в новостных данных'}

        if 'sentiment_mean' not in news_features.columns:
            return {'error': 'Нет колонки sentiment_mean. Запустите prepare_news_features() сначала'}

        news = news_features.copy()
        news['date'] = pd.to_datetime(news['date'])

        # Снимаем часовой пояс с индекса stock_data если есть
        stock = stock_data.copy()
        if not isinstance(stock.index, pd.DatetimeIndex):
            try:
                stock.index = pd.to_datetime(stock.index)
            except Exception:
                return {'error': 'Индекс stock_data не является датой'}

        if stock.index.tz is not None:
            stock.index = stock.index.tz_localize(None)

        if news['date'].dt.tz is not None:
            news['date'] = news['date'].dt.tz_localize(None)

        # Объединяем по датам
        merged = pd.merge(
            news[['date', 'sentiment_mean']],
            stock[['returns']],
            left_on='date',
            right_index=True,
            how='inner'
        )

        if merged.empty:
            return {'error': 'Нет общих дат между новостями и ценами акций'}

        results = {}

        # Корреляция без лага
        valid = merged[['sentiment_mean', 'returns']].dropna()
        if len(valid) > 5:
            results['lag_0'] = round(valid['sentiment_mean'].corr(valid['returns']), 3)
        else:
            results['lag_0'] = None

        # Корреляция с лагами: сдвигаем тональность вперёд
        for lag in range(1, max_lag + 1):
            lagged = merged['sentiment_mean'].shift(-lag)
            lag_df = pd.DataFrame({
                'sentiment': lagged,
                'returns': merged['returns']
            }).dropna()

            if len(lag_df) > 5:
                corr = lag_df['sentiment'].corr(lag_df['returns'])
                results[f'lag_{lag}'] = round(corr, 3) if not pd.isna(corr) else None
            else:
                results[f'lag_{lag}'] = None

        # Находим оптимальный лаг по максимуму абсолютной корреляции
        valid_lags = {k: v for k, v in results.items() if v is not None}
        if valid_lags:
            best_lag = max(valid_lags, key=lambda k: abs(valid_lags[k]))
            results['best_lag'] = best_lag
            results['best_correlation'] = valid_lags[best_lag]
        else:
            results['best_lag'] = None
            results['best_correlation'] = None

        return results

    def analyze_event_study(self, news_events, stock_data, window=(-5, 10)):
        """
        Event study анализ: как цена ведёт себя вокруг события

        Args:
            news_events: список дат с сильными новостями
            stock_data: данные по акции
            window: окно вокруг события (дни до, дни после)

        Returns:
            Список словарей с аномальной доходностью вокруг событий
        """
        results = []

        for event_date in news_events:
            if event_date not in stock_data.index:
                continue

            event_pos = stock_data.index.get_loc(event_date)

            start_pos = max(0, event_pos + window[0])
            end_pos = min(len(stock_data) - 1, event_pos + window[1])

            window_returns = stock_data['returns'].iloc[start_pos:end_pos + 1]

            # Нормальная доходность — среднее за 30 дней до события
            normal_window = stock_data['returns'].iloc[max(0, event_pos - 30):event_pos]
            normal_return = normal_window.mean() if len(normal_window) > 0 else 0

            abnormal_returns = window_returns - normal_return
            cumulative_abnormal = abnormal_returns.cumsum()

            results.append({
                'event_date': event_date,
                'window_dates': stock_data.index[start_pos:end_pos + 1].tolist(),
                'abnormal_returns': abnormal_returns.tolist(),
                'cumulative_abnormal': cumulative_abnormal.tolist(),
                'max_cumulative': cumulative_abnormal.max() if len(cumulative_abnormal) > 0 else 0,
                'min_cumulative': cumulative_abnormal.min() if len(cumulative_abnormal) > 0 else 0,
            })

        return results

    def get_correlation_report(self, ticker, news_data, stock_data=None,
                               start_date=None, end_date=None):
        """
        Полный отчёт о корреляции новостей и цены акции

        Args:
            ticker: тикер
            news_data: новостные данные (дата, текст, sentiment)
            stock_data: данные по акции (если None, загружаем автоматически)
            start_date: начальная дата
            end_date: конечная дата

        Returns:
            Словарь с результатами корреляции
        """
        report = {
            'ticker': ticker,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results': {}
        }

        if stock_data is None and start_date and end_date:
            stock_data = self.load_stock_data(ticker, start_date, end_date)

        if stock_data is None:
            report['error'] = 'Не удалось загрузить данные по акции'
            return report

        news_features = self.prepare_news_features(news_data)

        if news_features is None:
            report['error'] = 'Не удалось подготовить новостные признаки'
            return report

        lag_corr = self.calculate_lag_correlation(news_features, stock_data)
        report['results']['lag_correlation'] = lag_corr

        # Event study по дням с сильной тональностью
        if 'sentiment_mean' in news_features.columns and 'news_count' in news_features.columns:
            strong_events = news_features[
                (news_features['sentiment_mean'].abs() > 0.5) &
                (news_features['news_count'] > 5)
            ]['date'].tolist()

            if strong_events:
                event_study = self.analyze_event_study(strong_events[:10], stock_data)
                report['results']['event_study'] = event_study

        report['results']['summary'] = {
            'total_news_days': len(news_features),
            'total_trading_days': len(stock_data),
            'avg_daily_news': (
                news_features['news_count'].mean()
                if 'news_count' in news_features.columns else 0
            ),
            'avg_daily_return': stock_data['returns'].mean() * 100,
            'volatility': stock_data['returns'].std() * 100,
        }

        return report