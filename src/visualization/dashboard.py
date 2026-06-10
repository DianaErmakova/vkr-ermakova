"""
Главный файл дашборда
"""
import streamlit as st
import pandas as pd
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from analysis.market_trend_analyzer import MarketTrendAnalyzer
from . import components
from . import charts
from . import config


def render_tab_influence(results):
    """
    Вкладка 'Влияние' — композитный индекс влияния медиасобытий.
    Использует реальные данные тональности из SentimentAnalyzer.
    """
    st.header("Индекс влияния медиасобытий")

    st.markdown("""
    **Композитный индекс** агрегирует четыре компонента:
    - **Интенсивность** (30%) — нормированное число упоминаний
    - **Тональность** (40%) — сентимент из FinBERT/RoBERTa
    - **Виральность** (20%) — скорость распространения
    - **Авторитетность** (10%) — вес источника
    """)

    if 'influence_analysis' not in results:
        st.warning("Индекс влияния не рассчитан. Запустите анализ с включённой тональностью.")
        return

    influence_data = results['influence_analysis']

    if 'error' in influence_data:
        st.error(f"Ошибка: {influence_data['error']}")
        return

    # Метрики верхнего уровня
    trend_score = influence_data.get('trend_score', {})

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        score = trend_score.get('trend_influence_score', 0)
        st.metric("Индекс влияния тренда", f"{score:.3f}", help="0-1, чем выше — тем влиятельнее")
    with col2:
        st.metric("Макс. влияние", f"{trend_score.get('max_influence', 0):.3f}")
    with col3:
        st.metric("Новостей проанализировано", influence_data.get('total_items', 0))
    with col4:
        dist = trend_score.get('influence_distribution', {})
        high = dist.get('high', 0)
        st.metric("Высокое влияние (>0.7)", high)

    st.divider()

    # Распределение влияния
    dist = trend_score.get('influence_distribution', {})
    if dist:
        st.subheader("Распределение новостей по уровню влияния")
        dist_df = pd.DataFrame({
            'Уровень': ['Высокое (>0.7)', 'Среднее (0.4-0.7)', 'Низкое (<0.4)'],
            'Количество': [dist.get('high', 0), dist.get('medium', 0), dist.get('low', 0)]
        })

        import plotly.express as px
        fig = px.bar(
            dist_df, x='Уровень', y='Количество',
            color='Уровень',
            color_discrete_map={
                'Высокое (>0.7)': '#00CC96',
                'Среднее (0.4-0.7)': '#FFA15A',
                'Низкое (<0.4)': '#EF553B'
            },
            title="Распределение новостей по уровню влияния",
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Топ влиятельных новостей
    top_influencers = influence_data.get('top_influencers', [])
    if top_influencers:
        st.subheader("Топ-5 наиболее влиятельных новостей")

        rows = []
        for item in top_influencers:
            components_data = item.get('components', {})
            rows.append({
                'Новость': item.get('title', '')[:80] + '...',
                'Индекс влияния': round(item.get('influence_score', 0), 3),
                'Тональность': round(components_data.get('sentiment', 0), 3),
                'Интенсивность': round(components_data.get('intensity', 0), 3),
                'Авторитетность': round(components_data.get('authority', 0), 3),
            })

        top_df = pd.DataFrame(rows)
        st.dataframe(
            top_df,
            use_container_width=True,
            column_config={
                'Индекс влияния': st.column_config.ProgressColumn(
                    min_value=0, max_value=1, format="%.3f"
                )
            }
        )

        import plotly.express as px
        fig2 = px.bar(
            top_df,
            x='Индекс влияния',
            y='Новость',
            orientation='h',
            color='Тональность',
            color_continuous_scale='RdYlGn',
            color_continuous_midpoint=0,
            title="Топ новостей по индексу влияния (цвет = тональность)",
            template='plotly_white'
        )
        fig2.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

    # Примечание
    note = influence_data.get('note', '')
    if note:
        st.info(note)

    # Веса компонентов
    with st.expander("Веса компонентов индекса"):
        st.markdown("""
        | Компонент | Вес | Описание |
        |-----------|-----|----------|
        | Тональность | 40% | Оценка FinBERT/RoBERTa (-1 до +1) |
        | Интенсивность | 30% | Нормированное число упоминаний |
        | Виральность | 20% | Лайки, репосты, комментарии |
        | Авторитетность | 10% | Вес источника (Reuters=1.0, блог=0.2) |
        """)
        st.caption("Веса можно настроить в analysis/influence_index.py -> DEFAULT_WEIGHTS")


def _get_week_offset(params):
    """
    Единый источник week_offset для режима DJIA.

    Читает значение из params (виджет sidebar с key='week_offset').
    Запись обратно в session_state не производится — Streamlit запрещает
    изменять session_state по ключу, который уже привязан к виджету.
    """
    return params.get('week_offset', 0)


def run_dashboard():
    """Запуск дашборда"""
    st.set_page_config(
        page_title="Анализ рыночных трендов",
        page_icon="",
        layout="wide"
    )

    st.title("Анализ рыночных трендов")
    st.markdown("Автоматическое выявление трендов и анализ влияния на основе NLP")

    params = components.render_sidebar()

    # Обработка кнопки
    if params['analyze_btn']:
        with st.spinner("Анализируем данные... Это может занять несколько минут"):

            if params['data_source'] == "Исторические (DJIA)":
                from .data_loader import load_djia_data, analyze_djia_week, extract_day_news
                from analysis.sentiment_analyzer import SentimentAnalyzer

                df = load_djia_data()
                if df.empty:
                    st.error("Не удалось загрузить данные DJIA")
                    return

                # Единый источник week_offset
                week_offset = _get_week_offset(params)

                sentiment_analyzer = SentimentAnalyzer()
                sentiment_results = analyze_djia_week(
                    df, sentiment_analyzer, week_offset=week_offset, days=7
                )

                if sentiment_results.empty:
                    st.warning(f"Нет данных для недели {week_offset}")
                    return

                start_idx = week_offset * 7
                sample_news = (
                    extract_day_news(df.iloc[start_idx])[:5]
                    if start_idx < len(df) else []
                )

                # Расчёт influence на исторических данных
                influence_items = []
                for _, row in sentiment_results.iterrows():
                    influence_items.append({
                        'title':           f"DJIA {row['date']} (avg sentiment)",
                        'mentions_count':  row.get('news_count', 5),
                        'max_mentions':    25,
                        'min_mentions':    1,
                        'sentiment_score': float(row['avg_sentiment']),
                        'spread_data':     {},
                        'source':          'djia_dataset',
                    })

                influence_analysis = {}
                if influence_items:
                    try:
                        from analysis.influence_index import InfluenceIndexCalculator
                        calc = InfluenceIndexCalculator()
                        top5 = calc.identify_top_influencers(influence_items, top_n=5)
                        trend_inf = calc.calculate_trend_influence(influence_items)
                        influence_analysis = {
                            'trend_score':     trend_inf,
                            'top_influencers': top5.to_dict('records'),
                            'total_items':     len(influence_items),
                            'note': (
                                'Данные из датасета DJIA. '
                                'Виральность = 0: данные социальных сетей недоступны.'
                            ),
                        }
                    except Exception as e:
                        influence_analysis = {'error': str(e)}

                results = {
                    'total_news':    len(sentiment_results) * 5,
                    'trends_found':  3,
                    'news_samples':  sample_news,
                    'sentiment_analysis': {
                        'summary': {
                            'average_score': round(
                                sentiment_results['avg_sentiment'].mean(), 3
                            ),
                            'sentiment_index': round(
                                sentiment_results['avg_sentiment'].mean() * 100, 2
                            ),
                            'dominant_sentiment': (
                                'positive'
                                if sentiment_results['avg_sentiment'].mean() > 0.1
                                else 'negative'
                                if sentiment_results['avg_sentiment'].mean() < -0.1
                                else 'neutral'
                            ),
                            'distribution_counts': {
                                'positive': int(sentiment_results['positive_count'].sum()),
                                'neutral':  int(sentiment_results['neutral_count'].sum()),
                                'negative': int(sentiment_results['negative_count'].sum()),
                            },
                        }
                    },
                    'influence_analysis': influence_analysis,
                }

                st.session_state['sentiment_results'] = sentiment_results
                st.session_state['analysis_period'] = {
                    'start': pd.to_datetime(sentiment_results['date'].min()),
                    'end':   pd.to_datetime(sentiment_results['date'].max()),
                }

            else:
                api_key = (
                    params.get('api_key')
                    if params['data_source'] == "Реальные (NewsAPI)" else None
                )

                if params['data_source'] == "Реальные (NewsAPI)":
                    pass

                analyzer = MarketTrendAnalyzer(
                    news_api_key=api_key,
                    enable_sentiment=params['enable_sentiment'],
                    language=params.get('language', 'english')
                )

                results = analyzer.analyze_with_influence(
                    companies=params['companies'],
                    pages=1
                )

                if params['data_source'] == "Реальные (NewsAPI)":
                    total = results.get('total_news', 0)
                    if total == 0:
                        st.warning(
                            "NewsAPI не вернул статей по выбранным компаниям. "
                            "Возможные причины:\n"
                            "- исчерпан лимит запросов (100/день на бесплатном плане)\n"
                            "- неверный API-ключ\n"
                            "- нет новостей за последние 24 часа по данным компаниям\n\n"
                            "Попробуйте Демо-данные или Исторические (DJIA)."
                        )
                        return

                st.session_state['analyzer'] = analyzer

            st.session_state['results']    = results
            st.session_state['companies']  = params['companies']

            st.success("Анализ завершен!")
            st.rerun()

    # Отображение результатов
    if 'results' in st.session_state:
        results = st.session_state['results']
        companies = st.session_state.get('companies', [])
        data_source = params.get('data_source', '') if 'params' in locals() else ''
        analyzer = st.session_state.get('analyzer', None)

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Обзор", "Тренды", "Тональность", "Корреляция", "Влияние"
        ])

        with tab1:
            components.render_tab_overview(results, companies, data_source)
        with tab2:
            components.render_tab_trends(results, analyzer)
        with tab3:
            components.render_tab_sentiment(results)
        with tab4:
            components.render_tab_correlation(results, companies)
        with tab5:
            render_tab_influence(results)

    st.divider()
    st.caption(f"© {datetime.now().year} | Система анализа рыночных трендов для ВКР")