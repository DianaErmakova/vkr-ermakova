"""
Анализ временной динамики трендов
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class TemporalAnalyzer:
    """
    Анализатор временной динамики трендов

    Позволяет отслеживать:
    - Жизненный цикл трендов (появление, пик, затухание)
    - Скорость изменения популярности
    - Устойчивость трендов
    - Сезонные паттерны
    """

    def __init__(self):
        self.trend_history = {}
        self.temporal_metrics = {}

    def analyze_trend_lifecycle(self, trend_data, date_column='date',
                                value_column='mentions', window=7):
        """
        Анализ жизненного цикла тренда

        Args:
            trend_data: DataFrame с данными по тренду (даты, метрики)
            date_column: колонка с датами
            value_column: колонка со значениями (упоминания, тональность и т.д.)
            window: окно для сглаживания (дни)

        Returns:
            Словарь с метриками жизненного цикла
        """
        df = trend_data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)

        # Сглаживаем временной ряд (скользящее среднее)
        df['smoothed'] = df[value_column].rolling(window=window, min_periods=1).mean()

        # Находим ключевые точки
        peak_idx = df['smoothed'].idxmax()
        peak_date = df.loc[peak_idx, date_column]
        peak_value = df.loc[peak_idx, 'smoothed']

        # Порог для определения начала и конца (10% от пика)
        threshold = peak_value * 0.1

        # Ищем начало тренда (первое превышение порога)
        start_idx = None
        for idx, value in df['smoothed'].items():
            if value >= threshold:
                start_idx = idx
                break

        # Ищем конец тренда (последнее превышение порога)
        end_idx = None
        for idx in reversed(df.index):
            if df.loc[idx, 'smoothed'] >= threshold:
                end_idx = idx
                break

        # Получаем даты для индексов
        start_date = df.loc[start_idx, date_column] if start_idx is not None else None
        end_date = df.loc[end_idx, date_column] if end_idx is not None else None

        # Рассчитываем длительность в днях ИСПОЛЬЗУЯ ДАТЫ, а не индексы
        if start_date and end_date:
            duration_days = (end_date - start_date).days
        else:
            duration_days = None

        if start_date and peak_date:
            rise_days = (peak_date - start_date).days
        else:
            rise_days = None

        if peak_date and end_date:
            decline_days = (end_date - peak_date).days
        else:
            decline_days = None

        lifecycle = {
            'peak_date': peak_date,
            'peak_value': peak_value,
            'start_date': start_date,
            'end_date': end_date,
            'duration_days': duration_days,
            'rise_days': rise_days,
            'decline_days': decline_days
        }

        # Добавляем фазы
        if rise_days and rise_days > 0:
            lifecycle['rise_rate'] = peak_value / rise_days
        else:
            lifecycle['rise_rate'] = 0

        if decline_days and decline_days > 0:
            lifecycle['decline_rate'] = peak_value / decline_days
        else:
            lifecycle['decline_rate'] = 0

        return lifecycle

    def calculate_trend_momentum(self, trend_data, date_column='date',
                                 value_column='mentions', periods=[3, 7, 14]):
        """
        Расчет моментума (скорости изменения) тренда

        Args:
            trend_data: DataFrame с данными
            date_column: колонка с датами
            value_column: колонка со значениями
            periods: периоды для расчета моментума (дни)

        Returns:
            DataFrame с добавленными метриками моментума
        """
        df = trend_data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        df = df.reset_index(drop=True)  # Сбрасываем индекс для надежности

        for period in periods:
            # Процентное изменение за период
            df[f'momentum_{period}d'] = df[value_column].pct_change(periods=period) * 100

            # Абсолютное изменение
            df[f'change_{period}d'] = df[value_column].diff(periods=period)

            # Ускорение (изменение моментума)
            if period > 1:
                df[f'acceleration_{period}d'] = df[f'momentum_{period}d'].diff()

        return df

    def classify_trend_type(self, trend_data, value_column='mentions', window=30):
        """
        Классификация типа тренда

        Типы:
        - Взрывной: быстрый рост и падение
        - Устойчивый: медленный рост, долгое плато
        - Циклический: повторяющиеся пики
        - Шум: случайные колебания

        Returns:
            Словарь с классификацией
        """
        df = trend_data.copy()

        if len(df) < window:
            return {'type': 'insufficient_data', 'confidence': 0}

        # Рассчитываем метрики
        mean_value = df[value_column].mean()
        std_value = df[value_column].std()
        cv = std_value / mean_value if mean_value > 0 else 0  # коэффициент вариации

        # Тренд (линейная регрессия)
        x = np.arange(len(df))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, df[value_column])

        # Автокорреляция (для поиска цикличности)
        autocorr = df[value_column].autocorr(lag=1)

        # Определяем тип
        if abs(slope) < 0.01 and cv < 0.3:
            trend_type = 'stable'
            confidence = 0.7
        elif abs(slope) > 0.1 and cv > 0.5:
            trend_type = 'volatile'
            confidence = 0.8
        elif abs(autocorr) > 0.5:
            trend_type = 'cyclical'
            confidence = 0.6
        else:
            trend_type = 'noise'
            confidence = 0.4

        # Детальный анализ для взрывных трендов
        if len(df) > 10:
            # Ищем резкие скачки
            changes = df[value_column].pct_change().abs()
            max_change = changes.max()
            if max_change > 0.5:  # скачок > 50%
                if trend_type != 'cyclical':
                    trend_type = 'explosive'
                    confidence = 0.9

        return {
            'type': trend_type,
            'confidence': confidence,
            'metrics': {
                'slope': slope,
                'r_squared': r_value ** 2,
                'cv': cv,
                'autocorrelation': autocorr,
                'max_daily_change': changes.max() if 'changes' in locals() else 0
            }
        }

    def detect_seasonality(self, trend_data, value_column='mentions',
                           periods=[7, 30, 90]):
        """
        Обнаружение сезонности в трендах

        Args:
            trend_data: DataFrame с данными
            value_column: колонка со значениями
            periods: периоды для проверки (дни)

        Returns:
            Словарь с результатами сезонности
        """
        df = trend_data.copy()

        if len(df) < max(periods):
            return {'error': 'Недостаточно данных для анализа сезонности'}

        seasonality_results = {}

        for period in periods:
            if len(df) < period * 2:
                continue

            # Создаем лаги для проверки сезонности
            lagged = df[value_column].shift(period)
            valid_data = pd.DataFrame({
                'original': df[value_column],
                'lagged': lagged
            }).dropna()

            if len(valid_data) > period:
                # Корреляция с лагом
                corr = valid_data['original'].corr(valid_data['lagged'])

                # Если корреляция > 0.3, возможно есть сезонность
                if abs(corr) > 0.3:
                    seasonality_results[f'period_{period}d'] = {
                        'correlation': round(corr, 3),
                        'strength': 'strong' if abs(corr) > 0.5 else 'moderate',
                        'samples': len(valid_data)
                    }

        return seasonality_results

    def get_temporal_report(self, trend_data, trend_name=None):
        """
        Полный отчет по временной динамике тренда
        """
        report = {
            'trend_name': trend_name or 'unknown',
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'basic_stats': {},
            'lifecycle': {},
            'momentum': {},
            'classification': {},
            'seasonality': {}
        }

        # Базовая статистика
        report['basic_stats'] = {
            'total_days': len(trend_data),
            'mean_mentions': trend_data['mentions'].mean() if 'mentions' in trend_data.columns else 0,
            'max_mentions': trend_data['mentions'].max() if 'mentions' in trend_data.columns else 0,
            'min_mentions': trend_data['mentions'].min() if 'mentions' in trend_data.columns else 0,
            'std_mentions': trend_data['mentions'].std() if 'mentions' in trend_data.columns else 0
        }

        # Жизненный цикл
        if 'mentions' in trend_data.columns:
            report['lifecycle'] = self.analyze_trend_lifecycle(
                trend_data, value_column='mentions'
            )

        # Моментум
        if 'mentions' in trend_data.columns:
            momentum_df = self.calculate_trend_momentum(trend_data)
            report['momentum'] = {
                'latest_momentum_3d': momentum_df['momentum_3d'].iloc[
                    -1] if 'momentum_3d' in momentum_df.columns else None,
                'latest_momentum_7d': momentum_df['momentum_7d'].iloc[
                    -1] if 'momentum_7d' in momentum_df.columns else None,
                'max_momentum_3d': momentum_df['momentum_3d'].max() if 'momentum_3d' in momentum_df.columns else None,
                'min_momentum_3d': momentum_df['momentum_3d'].min() if 'momentum_3d' in momentum_df.columns else None
            }

        # Классификация
        if 'mentions' in trend_data.columns:
            report['classification'] = self.classify_trend_type(trend_data)

        # Сезонность
        if 'mentions' in trend_data.columns:
            report['seasonality'] = self.detect_seasonality(trend_data)

        return report

    def compare_trends_temporal(self, trends_dict, top_n=5):
        """
        Сравнение нескольких трендов по временным характеристикам

        Args:
            trends_dict: словарь {trend_name: trend_data}
            top_n: сколько лучших показать

        Returns:
            DataFrame с рейтингом трендов
        """
        comparison = []

        for trend_name, trend_data in trends_dict.items():
            if 'mentions' not in trend_data.columns:
                continue

            # Рассчитываем ключевые метрики
            momentum = self.calculate_trend_momentum(trend_data)

            trend_metrics = {
                'trend_name': trend_name,
                'total_mentions': trend_data['mentions'].sum(),
                'peak_mentions': trend_data['mentions'].max(),
                'avg_mentions': trend_data['mentions'].mean(),
                'growth_rate': momentum['momentum_7d'].iloc[-1] if 'momentum_7d' in momentum.columns else 0,
                'volatility': trend_data['mentions'].std() / trend_data['mentions'].mean() if trend_data[
                                                                                                  'mentions'].mean() > 0 else 0,
                'duration': len(trend_data)
            }

            # Рассчитываем составной балл
            trend_metrics['temporal_score'] = (
                trend_metrics['growth_rate'] * 0.4 +
                trend_metrics['peak_mentions'] * 0.3 +
                (1 / trend_metrics['volatility']) * 0.3 if trend_metrics['volatility'] > 0 else 0
            )

            comparison.append(trend_metrics)

        # Сортируем по баллу
        comparison_df = pd.DataFrame(comparison)
        if not comparison_df.empty:
            comparison_df = comparison_df.sort_values('temporal_score', ascending=False)

        return comparison_df.head(top_n)


# Пример использования
if __name__ == "__main__":
    # Создаем тестовые данные
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')

    # Тренд 1: Взрывной (быстрый рост и падение)
    explosive = np.concatenate([
        np.linspace(0, 100, 30),  # рост
        np.linspace(100, 20, 30),  # падение
        np.ones(40) * 20  # плато
    ])

    # Тренд 2: Устойчивый (медленный рост)
    stable = np.linspace(10, 50, 100)

    # Тренд 3: Циклический
    cyclic = 30 + 20 * np.sin(np.linspace(0, 4 * np.pi, 100))

    trend1_data = pd.DataFrame({
        'date': dates,
        'mentions': explosive[:100],
        'sentiment': np.random.randn(100) * 0.2
    })

    trend2_data = pd.DataFrame({
        'date': dates,
        'mentions': stable,
        'sentiment': np.random.randn(100) * 0.1
    })

    trend3_data = pd.DataFrame({
        'date': dates,
        'mentions': cyclic,
        'sentiment': np.random.randn(100) * 0.15
    })

    analyzer = TemporalAnalyzer()

    print("=" * 60)
    print("ДЕМО: Временной анализ трендов")
    print("=" * 60)

    # Анализ взрывного тренда
    print("\n1. Анализ взрывного тренда:")
    report1 = analyzer.get_temporal_report(trend1_data, "Explosive Trend")
    print(f"   Тип: {report1['classification']['type']}")
    print(f"   Пик: {report1['lifecycle'].get('peak_date', 'N/A')}")
    print(f"   Длительность: {report1['lifecycle'].get('duration_days', 'N/A')} дней")

    # Анализ устойчивого тренда
    print("\n2. Анализ устойчивого тренда:")
    report2 = analyzer.get_temporal_report(trend2_data, "Stable Trend")
    print(f"   Тип: {report2['classification']['type']}")
    print(f"   Скорость роста: {report2['momentum'].get('latest_momentum_7d', 0):.2f}%")

    # Анализ циклического тренда
    print("\n3. Анализ циклического тренда:")
    report3 = analyzer.get_temporal_report(trend3_data, "Cyclic Trend")
    print(f"   Тип: {report3['classification']['type']}")
    print(f"   Сезонность: {report3['seasonality']}")

    # Сравнение трендов
    print("\n4. Сравнение трендов:")
    trends_dict = {
        'Explosive': trend1_data,
        'Stable': trend2_data,
        'Cyclic': trend3_data
    }
    comparison = analyzer.compare_trends_temporal(trends_dict)
    print(comparison.to_string())

    print("\n" + "=" * 60)
    print("Демо завершено")
