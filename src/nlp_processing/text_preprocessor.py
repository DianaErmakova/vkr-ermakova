"""
Предобработка текста для NLP-пайплайна.

Централизует всю логику очистки и нормализации текста,
которая ранее дублировалась в TrendClusterer и data_loader.

Место в архитектуре:
    NewsCollector → TextPreprocessor → SentimentAnalyzer
                                     → TrendClusterer
                                     → InfluenceIndexCalculator
"""

import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Стоп-слова (расширяемый список)
STOPWORDS_EN = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
    'for', 'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was',
    'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
    'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'shall', 'that', 'this', 'it', 'its', 'not', 'no',
    'he', 'she', 'they', 'we', 'i', 'you', 'after', 'over',
    'into', 'up', 'than', 'more', 'also', 'about', 'said',
}


class TextPreprocessor:
    """
    Единая точка предобработки текста для всего NLP-пайплайна.

    Поддерживает два режима:
      - 'clustering'  : агрессивная очистка для BERTopic
                        (нижний регистр, удаление цифр и спецсимволов)
      - 'sentiment'   : мягкая очистка для BERT/FinBERT
                        (сохраняет заглавные буквы и пунктуацию,
                         важные для тональности)
    """

    def __init__(self, mode: str = 'clustering',
                 remove_stopwords: bool = False,
                 min_word_length: int = 3):
        """
        Args:
            mode:               'clustering' или 'sentiment'
            remove_stopwords:   удалять ли стоп-слова (для кластеризации)
            min_word_length:    минимальная длина слова после фильтрации
        """
        if mode not in ('clustering', 'sentiment'):
            raise ValueError("mode должен быть 'clustering' или 'sentiment'")

        self.mode = mode
        self.remove_stopwords = remove_stopwords
        self.min_word_length = min_word_length

    # Основные методы

    def clean(self, text: str) -> str:
        """
        Очистка одного текста в зависимости от режима.

        Args:
            text: сырой текст

        Returns:
            Очищенный текст
        """
        if not text or not isinstance(text, str):
            return ''

        # Убираем префикс b'...' из датасета DJIA
        text = self._remove_byte_prefix(text)

        if self.mode == 'sentiment':
            return self._clean_for_sentiment(text)
        else:
            return self._clean_for_clustering(text)

    def clean_batch(self, texts: List[str]) -> List[str]:
        """
        Очистка списка текстов.

        Args:
            texts: список сырых текстов

        Returns:
            Список очищенных текстов (пустые строки исключаются)
        """
        cleaned = []
        skipped = 0

        for text in texts:
            result = self.clean(text)
            if result:
                cleaned.append(result)
            else:
                skipped += 1

        if skipped:
            logger.debug(f"Пропущено пустых текстов: {skipped}")

        return cleaned

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Простое извлечение ключевых слов по частоте
        (без ML — для быстрого анализа).

        Args:
            text:  входной текст
            top_n: количество ключевых слов

        Returns:
            Список ключевых слов по убыванию частоты
        """
        cleaned = self._clean_for_clustering(text)
        words = cleaned.split()

        # Фильтруем стоп-слова и короткие слова
        words = [w for w in words
                 if w not in STOPWORDS_EN and len(w) >= self.min_word_length]

        # Частотный словарь
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1

        # Сортируем по убыванию частоты
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:top_n]]

    def get_stats(self, texts: List[str]) -> dict:
        """
        Базовая статистика по корпусу текстов.

        Args:
            texts: список текстов

        Returns:
            Словарь со статистикой
        """
        if not texts:
            return {}

        cleaned = self.clean_batch(texts)
        lengths = [len(t.split()) for t in cleaned]

        return {
            'total_texts':    len(texts),
            'valid_texts':    len(cleaned),
            'empty_texts':    len(texts) - len(cleaned),
            'avg_words':      round(sum(lengths) / len(lengths), 1) if lengths else 0,
            'min_words':      min(lengths) if lengths else 0,
            'max_words':      max(lengths) if lengths else 0,
            'total_words':    sum(lengths),
        }

    # Внутренние методы очистки

    def _clean_for_clustering(self, text: str) -> str:
        """
        Агрессивная очистка для BERTopic:
        нижний регистр, только буквы, без цифр.
        """
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', ' ', text)      # ссылки
        text = re.sub(r'[^\w\s]', ' ', text)              # спецсимволы
        text = re.sub(r'\d+', '', text)                   # цифры
        text = re.sub(r'\s+', ' ', text).strip()          # лишние пробелы

        if self.remove_stopwords:
            words = [w for w in text.split()
                     if w not in STOPWORDS_EN
                     and len(w) >= self.min_word_length]
            text = ' '.join(words)

        return text

    def _clean_for_sentiment(self, text: str) -> str:
        """
        Мягкая очистка для BERT/FinBERT:
        сохраняет регистр и пунктуацию — они важны для тональности.
        Только убирает мусор: ссылки, спецсимволы соцсетей, лишние пробелы.
        """
        text = re.sub(r'http\S+|www\S+', '', text)        # ссылки
        text = re.sub(r'@\w+', '', text)                   # @упоминания
        text = re.sub(r'#\w+', '', text)                   # #хэштеги
        text = re.sub(r'\s+', ' ', text).strip()           # лишние пробелы

        # Обрезаем до 512 токенов (лимит BERT) — грубая оценка: 1 слово ≈ 1.3 токена
        words = text.split()
        if len(words) > 390:
            text = ' '.join(words[:390])

        return text

    @staticmethod
    def _remove_byte_prefix(text: str) -> str:
        """
        Убирает префикс b'...' из сырых строк датасета DJIA.
        Было:  b'Fed raises interest rates amid inflation concerns'
        Стало: Fed raises interest rates amid inflation concerns
        """
        text = text.strip()
        if text.startswith("b'") or text.startswith('b"'):
            text = text[2:].rstrip("'\"")
        return text

# Фабричные функции для удобства


def get_clustering_preprocessor(language='english') -> TextPreprocessor:
    """Препроцессор для BERTopic (агрессивная очистка)"""
    if language == 'russian':
        return TextPreprocessor(mode='clustering', remove_stopwords=True, min_word_length=3, language='russian')
    return TextPreprocessor(mode='clustering', remove_stopwords=True, min_word_length=3)


def get_sentiment_preprocessor() -> TextPreprocessor:
    """Препроцессор для FinBERT/RoBERTa (мягкая очистка)"""
    return TextPreprocessor(
        mode='sentiment',
        remove_stopwords=False
    )


def _clean_for_clustering_russian(self, text: str) -> str:
    """Очистка для русских текстов"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # русские стоп-слова
    stopwords_ru = {'и', 'в', 'во', 'не', 'что', 'на', 'я', 'с', 'со', 'как', 'а', 'но', 'это', 'так', 'же', 'бы', 'по',
                    'только', 'еще', 'уже', 'вот', 'да', 'нет', 'все', 'было', 'если', 'или', 'без', 'до', 'для', 'за',
                    'из', 'к', 'о', 'об', 'от', 'при', 'через'}

    if self.remove_stopwords:
        words = [w for w in text.split() if w not in stopwords_ru and len(w) >= 3]
        text = ' '.join(words)

    return text

# Демо


if __name__ == "__main__":
    print("=" * 55)
    print("ДЕМО: TextPreprocessor")
    print("=" * 55)

    test_texts = [
        "b'Fed raises interest rates amid inflation concerns'",
        "Tesla $TSLA surges 15% after record Q4 earnings! #Tesla @elonmusk",
        "Apple faces EU antitrust probe over App Store practices",
        "   ",   # пустая строка
        "AI and machine learning transform financial markets in 2024",
    ]

    print("\n── Режим CLUSTERING (для BERTopic) ──")
    cp = get_clustering_preprocessor()
    for text in test_texts:
        result = cp.clean(text)
        print(f"  Вход:  {text[:60]}")
        print(f"  Выход: {result}")
        print()

    print("── Режим SENTIMENT (для FinBERT) ──")
    sp = get_sentiment_preprocessor()
    for text in test_texts[:3]:
        result = sp.clean(text)
        print(f"  Вход:  {text[:60]}")
        print(f"  Выход: {result}")
        print()

    print("── Статистика корпуса ──")
    stats = cp.get_stats(test_texts)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\n── Ключевые слова ──")
    sample = "Federal Reserve raises interest rates amid rising inflation concerns in financial markets"
    keywords = cp.extract_keywords(sample, top_n=5)
    print(f"  Текст: {sample}")
    print(f"  Ключевые слова: {keywords}")