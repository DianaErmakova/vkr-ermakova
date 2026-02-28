"""
Точка входа для разведочного анализа данных DJIA.

Использование:
    python eda_djia.py

Графики сохраняются в reports/figures/.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.eda import (
    load_and_inspect,
    analyze_news_length,
    analyze_sentiment_sample,
    analyze_correlation,
    plot_temporal_dynamics,
    analyze_word_frequency,
    print_conclusions,
)
from config import DATA_PATH, REPORTS_DIR

if __name__ == "__main__":
    print("Разведочный анализ данных DJIA")
    print("Система анализа рыночных трендов\n")

    df = load_and_inspect(DATA_PATH)
    all_lengths, lengths_by_label = analyze_news_length(df)
    sentiment_df = analyze_sentiment_sample(df, n_days=30)
    corr, accuracy = analyze_correlation(sentiment_df)
    plot_temporal_dynamics(sentiment_df)
    top_rise, top_fall = analyze_word_frequency(df)
    print_conclusions(df, all_lengths, corr, accuracy, top_rise, top_fall)

    print(f"\nВсе графики сохранены в: {REPORTS_DIR}")
    print("EDA завершён.")