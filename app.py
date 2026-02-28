"""
Запуск веб-дашборда для анализа рыночных трендов

Использование:
    streamlit run app.py
"""
import sys
import os

# Путь к src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from visualization.dashboard import run_dashboard

if __name__ == "__main__":
    run_dashboard()