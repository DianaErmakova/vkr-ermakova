"""
Компоненты дашборда (боковая панель, вкладки)
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from . import config
from .charts import (
    create_sentiment_pie, create_trends_bar,
    create_stock_price_chart, create_returns_chart,
    create_lag_correlation_chart,
)
from .data_loader import (
    load_stock_prices, get_ticker_from_company,
    load_djia_data, analyze_djia_week,
)


def render_sidebar():
    with st.sidebar:
        st.header("Параметры анализа")

        data_source = st.radio(
            "Источник данных",
            ["Демо-данные", "Исторические (DJIA)", "Реальные (NewsAPI)"],
            key="data_source"
        )

        companies = st.multiselect(
            "Выберите компании",
            list(config.COMPANY_TICKERS.keys()),
            default=config.DEFAULT_COMPANIES,
            key="selected_companies"
        )

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Начало",
                datetime(2008, 8, 1) if data_source == "Исторические (DJIA)"
                else datetime.now() - timedelta(days=90),
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "Конец",
                datetime(2008, 9, 1) if data_source == "Исторические (DJIA)"
                else datetime.now(),
                key="end_date"
            )

        st.subheader("Настройки")
        enable_sentiment = st.checkbox("Анализ тональности", value=True, key="enable_sentiment")

        week_offset = 0
        if data_source == "Исторические (DJIA)":
            week_offset = st.slider(
                "Неделя (от начала)", 0, 280, 0,
                help="0 = первая неделя (август 2008)",
                key="week_offset"
            )

        api_key = None
        if data_source == "Реальные (NewsAPI)":
            api_key = st.text_input("API ключ NewsAPI", type="password", key="api_key")

        analyze_btn = st.button(
            "Запустить анализ",
            type="primary",
            use_container_width=True,
            key="analyze_btn"
        )

        st.divider()
        with st.expander("Инструкция"):
            st.markdown("""
            1. Выберите источник данных
            2. Укажите компании и период
            3. Нажмите «Запустить анализ»
            4. Исследуйте результаты по вкладкам
            """)

        return {
            'data_source':      data_source,
            'companies':        companies,
            'start_date':       start_date,
            'end_date':         end_date,
            'enable_sentiment': enable_sentiment,
            'week_offset':      week_offset,
            'api_key':          api_key,
            'analyze_btn':      analyze_btn,
        }


def render_metrics_row(results):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Всего новостей", results.get('total_news', 0))

    with col2:
        st.metric("Найдено трендов", results.get('trends_found', 0))

    with col3:
        sentiment = results.get('sentiment_analysis', {})
        summary   = sentiment.get('summary', {})
        avg_sent  = summary.get('average_score', None)
        if avg_sent is not None:
            st.metric(
                "Средняя тональность",
                f"{avg_sent:.2f}",
                delta=f"{avg_sent * 100:.0f}%"
            )
        else:
            st.metric("Средняя тональность", "—")

    with col4:
        trends_info = results.get('trends_info')
        if trends_info is not None and not trends_info.empty:
            valid = trends_info[trends_info['Topic'] != -1]
            st.metric("Валидные тренды", len(valid))
        else:
            st.metric("Валидные тренды", "—")


def render_tab_overview(results, companies, data_source):
    st.header("Обзор анализа")
    render_metrics_row(results)
    st.divider()

    st.subheader("Анализируемые компании")
    st.write(", ".join(companies) if companies else "—")

    st.subheader("Источник данных")
    st.info(f"{data_source}")

    samples = results.get('news_samples', [])
    if samples:
        st.subheader("Примеры новостей")
        for i, news in enumerate(samples[:5]):
            with st.expander(f"Новость {i + 1}"):
                st.write(news)


def render_tab_trends(results, analyzer=None):
    """
    Вкладка «Тренды».

    Работает в двух режимах:
      - С analyzer  (демо / NewsAPI): показывает BERTopic-кластеры
      - Без analyzer (DJIA-режим):    показывает динамику тональности по дням
    """
    st.header("Обнаруженные тренды")

    trends_info = results.get('trends_info')

    if analyzer is not None and trends_info is not None and not trends_info.empty:
        valid_trends = trends_info[trends_info['Topic'] != -1]

        if not valid_trends.empty:
            st.subheader("Тематические кластеры (BERTopic)")

            trend_data = []
            for _, row in valid_trends.iterrows():
                topic_id = row['Topic']
                keywords = analyzer.trend_clusterer.get_trend_keywords(topic_id, 5)
                kw_str   = ", ".join([kw[0] for kw in keywords]) if keywords else "—"
                trend_data.append({
                    'ID':             topic_id,
                    'Новостей':       int(row['Count']),
                    'Ключевые слова': kw_str,
                })

            st.dataframe(pd.DataFrame(trend_data), use_container_width=True)

            st.subheader("Размер кластеров")
            fig = create_trends_bar(valid_trends)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("BERTopic не выявил валидных кластеров. Попробуйте увеличить число новостей.")

        return

    sentiment_results = st.session_state.get('sentiment_results')

    if sentiment_results is not None and not sentiment_results.empty:
        st.info(
            "В режиме исторических данных (DJIA) тренды определяются "
            "по динамике тональности новостей, а не BERTopic-кластеризацией."
        )

        st.subheader("Тональность по дням")

        display_df = sentiment_results[['date', 'label', 'avg_sentiment',
                                        'positive_count', 'neutral_count',
                                        'negative_count', 'news_count']].copy()
        display_df['label'] = display_df['label'].map({1: 'Рост', 0: 'Падение'})
        display_df.columns = ['Дата', 'Рынок', 'Ср. тональность',
                               'Позитив', 'Нейтраль', 'Негатив', 'Новостей']
        st.dataframe(display_df, use_container_width=True)

        import plotly.graph_objects as go
        fig = go.Figure()

        colors = [
            '#00CC96' if s > 0.1 else '#EF553B' if s < -0.1 else '#B6B6B6'
            for s in sentiment_results['avg_sentiment']
        ]

        fig.add_trace(go.Bar(
            x=pd.to_datetime(sentiment_results['date']),
            y=sentiment_results['avg_sentiment'],
            marker_color=colors,
            name='Тональность',
            hovertemplate='%{x}<br>Тональность: %{y:.3f}<extra></extra>'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
        fig.update_layout(
            title="Динамика тональности новостей DJIA",
            xaxis_title="Дата",
            yaxis_title="Средний sentiment score",
            template='plotly_white',
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        if 'top_news' in sentiment_results.columns:
            st.subheader("Ведущая новость дня")
            for _, row in sentiment_results.iterrows():
                if row.get('top_news'):
                    label_text = 'Рост рынка' if row['label'] == 1 else 'Падение рынка'
                    st.markdown(
                        f"**{pd.to_datetime(row['date']).strftime('%d.%m.%Y')}** "
                        f"({label_text}) "
                        f"`{row['avg_sentiment']:+.3f}` — {row['top_news']}"
                    )
    else:
        st.warning(
            "Нет данных для отображения трендов. "
            "Запустите анализ через боковую панель."
        )


def render_tab_sentiment(results):
    st.header("Анализ тональности")

    sentiment_data = results.get('sentiment_analysis', {})

    if not sentiment_data or 'error' in sentiment_data:
        err = sentiment_data.get('error', 'Анализ тональности не выполнялся')
        st.warning(err)
        return

    summary = sentiment_data.get('summary', {})
    if not summary:
        st.warning("Нет данных тональности")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Средняя тональность", f"{summary.get('average_score', 0):.2f}")
    with col2:
        st.metric("Индекс настроения", summary.get('sentiment_index', 0))
    with col3:
        dominant = summary.get('dominant_sentiment', 'neutral')
        st.metric("Доминирующая", dominant.capitalize())

    st.subheader("Распределение")
    dist = summary.get('distribution_counts', {})
    if dist:
        fig = create_sentiment_pie(dist)
        st.plotly_chart(fig, use_container_width=True)

    samples = sentiment_data.get('individual_samples', [])
    if samples:
        st.subheader("Примеры анализа")
        st.dataframe(pd.DataFrame(samples), use_container_width=True)

    model = sentiment_data.get('model_used')
    if model:
        st.caption(f"Модель: {model}")


def render_tab_correlation(results, companies):
    st.header("Корреляция с ценами акций")

    if not companies:
        st.warning("Выберите хотя бы одну компанию")
        return

    selected_company = st.selectbox(
        "Выберите компанию", companies, key="correlation_company"
    )
    ticker = get_ticker_from_company(selected_company)

    # Даты по умолчанию — период анализа DJIA если есть, иначе последние 90 дней
    default_start = datetime.now() - timedelta(days=90)
    default_end   = datetime.now()
    if 'analysis_period' in st.session_state:
        default_start = st.session_state['analysis_period']['start']
        default_end   = st.session_state['analysis_period']['end']

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Начальная дата", default_start, key="correlation_start"
        )
    with col2:
        end_date = st.date_input(
            "Конечная дата", default_end, key="correlation_end"
        )

    # Подсказка если период слишком короткий
    period_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
    if period_days < 30:
        st.warning(
            f"Выбранный период — {period_days} дней. "
            "Для расчёта корреляции рекомендуется минимум 30 дней. "
            "Расширьте диапазон дат."
        )

    if st.button(f"Загрузить данные {ticker}", key="load_stock_btn"):
        with st.spinner(f"Загружаем данные для {ticker}..."):
            stock_data = load_stock_prices(
                ticker,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            if stock_data is not None and not stock_data.empty:
                st.session_state['stock_data']     = stock_data
                st.session_state['current_ticker'] = ticker
                st.success(f"Данные {ticker} загружены: {len(stock_data)} торговых дней")
                st.rerun()
            else:
                st.error(f"Не удалось загрузить данные для {ticker}")

    if 'stock_data' in st.session_state:
        stock_data = st.session_state['stock_data']
        ticker     = st.session_state.get('current_ticker', 'Акция')

        col1, col2, col3, col4 = st.columns(4)
        current_price = stock_data['Close'].iloc[-1]
        prev_price    = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
        change        = ((current_price - prev_price) / prev_price) * 100

        with col1:
            st.metric("Текущая цена", f"${current_price:.2f}", delta=f"{change:.2f}%")
        with col2:
            st.metric("Максимум", f"${stock_data['High'].max():.2f}")
        with col3:
            st.metric("Минимум", f"${stock_data['Low'].min():.2f}")
        with col4:
            st.metric("Ср. объём", f"{stock_data['Volume'].mean() / 1e6:.1f}M")

        st.subheader(f"{ticker} — динамика цены")
        st.plotly_chart(create_stock_price_chart(stock_data, ticker), use_container_width=True)

        if 'returns' in stock_data.columns:
            st.subheader("Дневная доходность")
            st.plotly_chart(create_returns_chart(stock_data), use_container_width=True)

    st.markdown("---")
    st.subheader("Корреляция тональности с доходностью")

    # Проверяем наличие данных до показа кнопки
    has_sentiment = 'sentiment_results' in st.session_state
    has_stock     = 'stock_data' in st.session_state

    if not has_sentiment:
        st.info("Для расчёта корреляции сначала запустите анализ исторических данных (DJIA).")
    if not has_stock:
        st.info("Для расчёта корреляции сначала загрузите данные по акции.")

    if has_sentiment and has_stock:
        if st.button("Рассчитать корреляцию", key="calc_correlation_btn"):
            with st.spinner("Рассчитываем корреляцию..."):
                _calculate_and_store_correlation()
                st.rerun()

    if 'lag_results' in st.session_state:
        _render_correlation_results()


def _calculate_and_store_correlation():
    """
    Рассчитывает лаговую корреляцию и сохраняет в session_state.
    Вынесено отдельно чтобы не смешивать логику с рендерингом.
    """
    from analysis.correlation_analyzer import CorrelationAnalyzer

    sentiment_df = st.session_state['sentiment_results'].copy()
    stock_data   = st.session_state['stock_data'].copy()

    # Снимаем часовой пояс
    if hasattr(stock_data.index, 'tz') and stock_data.index.tz is not None:
        stock_data.index = stock_data.index.tz_localize(None)

    sentiment_df['date'] = pd.to_datetime(sentiment_df['date']).dt.normalize()
    stock_data.index     = stock_data.index.normalize()

    # Ищем общие даты
    stock_dates  = set(stock_data.index)
    common_mask  = sentiment_df['date'].isin(stock_dates)
    common_dates = sentiment_df[common_mask]

    if common_dates.empty:
        # Периоды не пересекаются — сообщаем пользователю
        sentiment_min = sentiment_df['date'].min().strftime('%Y-%m-%d')
        sentiment_max = sentiment_df['date'].max().strftime('%Y-%m-%d')
        stock_min     = stock_data.index.min().strftime('%Y-%m-%d')
        stock_max     = stock_data.index.max().strftime('%Y-%m-%d')

        st.session_state['lag_results'] = {
            'error': (
                f"Нет общих дат между тональностью ({sentiment_min} — {sentiment_max}) "
                f"и ценами акции ({stock_min} — {stock_max}). "
                f"Выберите период акции совпадающий с периодом анализа DJIA."
            )
        }
        return

    # Готовим news_features в формате который ожидает calculate_lag_correlation
    news_features = pd.DataFrame({
        'date':           common_dates['date'].values,
        'sentiment_mean': common_dates['avg_sentiment'].values,
    })

    corr_analyzer = CorrelationAnalyzer()
    lag_results   = corr_analyzer.calculate_lag_correlation(
        news_features, stock_data, max_lag=5
    )

    # Добавляем диагностику для отображения пользователю
    lag_results['_meta'] = {
        'common_days':    len(common_dates),
        'sentiment_days': len(sentiment_df),
        'stock_days':     len(stock_data),
    }

    st.session_state['lag_results'] = lag_results


def _render_correlation_results():
    """Отображает сохранённые результаты корреляции."""
    lag_results = st.session_state['lag_results']

    if 'error' in lag_results:
        st.warning(lag_results['error'])
        return

    # Диагностика
    meta = lag_results.get('_meta', {})
    if meta:
        common = meta.get('common_days', 0)
        if common < 6:
            st.warning(
                f"Найдено только {common} общих дней — недостаточно для надёжной корреляции "
                f"(нужно минимум 6). Расширьте период загрузки акции."
            )
        else:
            st.success(f"Корреляция рассчитана по {common} общим торговым дням.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Таблица корреляций")
        lag_data = []
        for key, value in lag_results.items():
            if key.startswith('lag_') and value is not None:
                lag_data.append({
                    "Лаг (дней)": key.replace('lag_', ''),
                    "Корреляция": f"{value:+.3f}",
                })
        if lag_data:
            st.table(pd.DataFrame(lag_data))
        else:
            st.info("Недостаточно данных для расчёта корреляции по лагам.")

        best_lag  = lag_results.get('best_lag')
        best_corr = lag_results.get('best_correlation')
        if best_lag and best_corr is not None:
            lag_num = best_lag.replace('lag_', '')
            st.metric(
                "Оптимальный лаг",
                f"{lag_num} дн.",
                f"{best_corr:+.3f}",
                help="Лаг с максимальной абсолютной корреляцией"
            )

        st.caption(
            "Корреляция показывает связь между тональностью новостей "
            "и доходностью акции со смещением на N дней вперёд."
        )

    with col2:
        st.subheader("График по лагам")
        # Передаём только lag_* ключи без _meta
        chart_data = {k: v for k, v in lag_results.items() if not k.startswith('_')}
        fig = create_lag_correlation_chart(chart_data)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("График недоступен — нет данных для отображения.")