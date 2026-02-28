"""
Функции для создания графиков Plotly
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from . import config


def create_sentiment_pie(distribution_counts):
    """Круговая диаграмма распределения тональности"""
    if not distribution_counts:
        return None

    fig = px.pie(
        values=list(distribution_counts.values()),
        names=list(distribution_counts.keys()),
        title="Распределение тональности",
        color=list(distribution_counts.keys()),
        color_discrete_map=config.SENTIMENT_COLORS,
        template=config.CHART_TEMPLATE
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


def create_trends_bar(trends_df):
    """Столбчатая диаграмма распределения трендов"""
    if trends_df is None or trends_df.empty:
        return None

    fig = px.bar(
        trends_df,
        x='Topic',
        y='Count',
        title="Количество документов по трендам",
        labels={'Topic': 'ID тренда', 'Count': 'Количество новостей'},
        color='Count',
        color_continuous_scale='Viridis',
        template=config.CHART_TEMPLATE
    )
    return fig


def create_stock_price_chart(stock_data, ticker):
    """
    График цены акции с дополнительными индикаторами

    Args:
        stock_data: DataFrame с ценами
        ticker: тикер акции

    Returns:
        Plotly Figure или None
    """
    if stock_data is None or stock_data.empty or 'Close' not in stock_data.columns:
        return None

    fig = go.Figure()

    # Основная линия цены
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=stock_data['Close'],
        mode='lines',
        name='Close',
        line=dict(color='#1f77b4', width=2)
    ))

    # Скользящие средние
    if 'MA20' in stock_data.columns:
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['MA20'],
            mode='lines',
            name='MA20',
            line=dict(color='orange', width=1, dash='dash')
        ))

    if 'MA50' in stock_data.columns:
        fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['MA50'],
            mode='lines',
            name='MA50',
            line=dict(color='red', width=1, dash='dash')
        ))

    # Свечи (скрыты по умолчанию)
    if all(col in stock_data.columns for col in ['Open', 'High', 'Low']):
        fig.add_trace(go.Candlestick(
            x=stock_data.index,
            open=stock_data['Open'],
            high=stock_data['High'],
            low=stock_data['Low'],
            close=stock_data['Close'],
            name='Candles',
            visible='legendonly'
        ))

    fig.update_layout(
        title=f'{ticker} - динамика цены',
        xaxis_title='Дата',
        yaxis_title='Цена ($)',
        template=config.CHART_TEMPLATE,
        hovermode='x unified',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_xaxes(rangeslider_visible=True)
    return fig


def create_returns_chart(stock_data):
    """График дневной доходности"""
    if stock_data is None or stock_data.empty or 'returns' not in stock_data.columns:
        return None

    colors = ['#00CC96' if x > 0 else '#EF553B' for x in stock_data['returns']]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=stock_data.index,
        y=stock_data['returns'],
        name='Доходность',
        marker_color=colors,
        showlegend=False
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        title="Дневная доходность (%)",
        xaxis_title='Дата',
        yaxis_title='Доходность %',
        template=config.CHART_TEMPLATE,
        height=300,
        hovermode='x unified'
    )
    return fig


def create_influence_chart(influence_df, top_n=10):
    """График топ влиятельных новостей"""
    if influence_df is None or influence_df.empty:
        return None

    top_df = influence_df.head(top_n)
    fig = px.bar(
        top_df,
        x='influence_score',
        y='title',
        orientation='h',
        title=f"Топ-{top_n} новостей по индексу влияния",
        labels={'influence_score': 'Индекс влияния', 'title': 'Новость'},
        color='influence_score',
        color_continuous_scale='Viridis',
        template=config.CHART_TEMPLATE
    )
    fig.update_layout(height=config.DEFAULT_HEIGHT)
    return fig


def create_correlation_heatmap(correlation_matrix):
    """Тепловая карта корреляций между новостями и ценой"""
    if correlation_matrix is None or (hasattr(correlation_matrix, 'empty') and correlation_matrix.empty):
        return None

    fig = px.imshow(
        correlation_matrix,
        text_auto='.2f',
        aspect="auto",
        title="Матрица корреляций (новости → цена)",
        color_continuous_scale='RdBu_r',
        template=config.CHART_TEMPLATE,
        zmin=-1, zmax=1
    )
    fig.update_layout(
        height=400,
        xaxis_title="Метрики",
        yaxis_title="Лаг (дни)"
    )
    return fig


def create_trend_timeline(trend_data, trend_name):
    """Временная динамика тренда"""
    if trend_data is None or trend_data.empty:
        return None

    fig = px.line(
        trend_data,
        x='date',
        y='mentions',
        title=f"Динамика тренда: {trend_name}",
        labels={'date': 'Дата', 'mentions': 'Количество упоминаний'},
        template=config.CHART_TEMPLATE
    )
    fig.update_layout(height=config.DEFAULT_HEIGHT)
    return fig


def create_wordcloud_fig(keywords_dict):
    """Облако тегов (упрощённое через scatter)"""
    if not keywords_dict:
        return None

    words = list(keywords_dict.keys())
    sizes = list(keywords_dict.values())
    sizes = [20 + (s / max(sizes)) * 50 for s in sizes]

    np.random.seed(42)
    x = np.random.randn(len(words))
    y = np.random.randn(len(words))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='text',
        text=words,
        textfont=dict(size=sizes, color='darkblue'),
        hoverinfo='text',
        textposition='middle center'
    ))
    fig.update_layout(
        title="Ключевые слова тренда",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        template=config.CHART_TEMPLATE
    )
    return fig


def create_sentiment_vs_price_chart(sentiment_data, price_data, ticker):
    """
    График сравнения тональности и цены.

    Args:
        sentiment_data: DataFrame с колонками 'date' и 'avg_sentiment'
        price_data:     DataFrame с индексом-датой и колонкой 'Close'
        ticker:         тикер акции

    Returns:
        Plotly Figure или None
    """
    if (sentiment_data is None or price_data is None
            or sentiment_data.empty or price_data.empty):
        return None

    # Объединяем по датам
    combined = pd.DataFrame(index=price_data.index)
    combined['price'] = price_data['Close']

    if 'avg_sentiment' in sentiment_data.columns:
        sentiment_series = sentiment_data.set_index('date')['avg_sentiment']
        combined['sentiment'] = sentiment_series
    else:
        return None

    combined = combined.dropna()
    if combined.empty:
        return None

    fig = go.Figure()

    # Цена — левая ось
    fig.add_trace(go.Scatter(
        x=combined.index,
        y=combined['price'],
        mode='lines',
        name=f'{ticker} цена',
        line=dict(color='#1f77b4', width=2),
        yaxis='y'
    ))

    # Тональность — правая ось
    fig.add_trace(go.Scatter(
        x=combined.index,
        y=combined['sentiment'],
        mode='lines+markers',
        name='Тональность',
        line=dict(color='#EF553B', width=2, dash='dash'),
        marker=dict(size=6),
        yaxis='y2'
    ))

    fig.update_layout(
        title=f"Сравнение тональности и цены {ticker}",
        xaxis=dict(title="Дата"),
        # ── FIX: titlefont → title=dict(font=dict(...)) ──
        yaxis=dict(
            title=dict(text="Цена ($)", font=dict(color="#1f77b4")),
            tickfont=dict(color="#1f77b4")
        ),
        yaxis2=dict(
            title=dict(text="Тональность", font=dict(color="#EF553B")),
            tickfont=dict(color="#EF553B"),
            anchor="x",
            overlaying="y",
            side="right",
            range=[-1, 1]
        ),
        template=config.CHART_TEMPLATE,
        height=500,
        hovermode='x unified'
    )
    return fig


def create_lag_correlation_chart(lag_results):
    """График корреляции с разными лагами"""
    if not lag_results or 'error' in lag_results:
        return None

    lags, correlations = [], []
    for key, value in lag_results.items():
        if key.startswith('lag_') and value is not None:
            lags.append(int(key.split('_')[1]))
            correlations.append(value)

    if not lags:
        return None

    colors = ['#00CC96' if c > 0 else '#EF553B' for c in correlations]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=lags,
        y=correlations,
        marker_color=colors,
        text=[f'{c:.3f}' for c in correlations],
        textposition='outside',
        name='Корреляция'
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    if lag_results.get('best_lag') and lag_results.get('best_correlation') is not None:
        lag_num = int(lag_results['best_lag'].split('_')[1])
        fig.add_annotation(
            x=lag_num,
            y=lag_results['best_correlation'],
            text=f"Оптимальный лаг: {lag_num} дн.",
            showarrow=True,
            arrowhead=2,
            ax=0, ay=-40
        )

    fig.update_layout(
        title="Корреляция тональности с доходностью (по лагам)",
        xaxis_title="Лаг (дни)",
        yaxis_title="Корреляция",
        template=config.CHART_TEMPLATE,
        height=400,
        yaxis_range=[-1, 1]
    )
    return fig