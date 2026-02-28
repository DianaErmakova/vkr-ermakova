"""
nlp_processing — модуль предобработки текста.

Экспортирует:
    TextPreprocessor          — основной класс
    get_clustering_preprocessor — препроцессор для BERTopic
    get_sentiment_preprocessor  — препроцессор для FinBERT/RoBERTa
"""

from .text_preprocessor import (
    TextPreprocessor,
    get_clustering_preprocessor,
    get_sentiment_preprocessor,
)

__all__ = [
    'TextPreprocessor',
    'get_clustering_preprocessor',
    'get_sentiment_preprocessor',
]