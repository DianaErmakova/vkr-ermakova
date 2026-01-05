from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import warnings

# Подавляем предупреждения
warnings.filterwarnings('ignore')


class TrendClusterer:
    def __init__(self):
        # УЛУЧШЕННАЯ КОНФИГУРАЦИЯ
        self.topic_model = None
        self.topics = None

        # Настраиваем модель
        self._setup_model()

    def _setup_model(self):
        """Настройка модели для лучшего качества"""

        # Улучшенный векторизатор
        vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,  # Можно снизить для малых данных
            max_df=0.8,
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b',  # Слова от 3 букв (было 4)
            max_features=100  # Ограничиваем количество фич
        )

        # НАСТРОЙКА BERTopic - только существующие параметры
        self.topic_model = BERTopic(
            # Основные настройки
            language="english",
            min_topic_size=3,  # Уменьшаем для малых данных (было 5)
            nr_topics="auto",  # Автоматическое определение

            # Качество кластеризации
            vectorizer_model=vectorizer_model,

            # Производительность
            calculate_probabilities=False,
            verbose=False  # Отключаем для чистоты вывода
        )

    def fit_clusters(self, texts):
        print("Обучение модели кластеризации трендов...")
        print(f"Количество текстов: {len(texts)}")

        # Препроцессинг
        processed_texts = self._preprocess_texts(texts)

        # Проверяем, что тексты достаточно разнообразны
        unique_words = len(set(" ".join(processed_texts).split()))
        print(f"Уникальных слов: {unique_words}")

        # Обучаем модель
        try:
            self.topics, self.probabilities = self.topic_model.fit_transform(processed_texts)

            # Дополнительная информация
            topic_info = self.get_trends_info()
            valid_topics = topic_info[topic_info['Topic'] != -1]

            print(f"Найдено кластеров: {len(topic_info)}")
            print(f"Валидные тренды: {len(valid_topics)}")

            # Покажем названия трендов, если они есть
            if len(valid_topics) > 0:
                print("Обнаруженные тренды:")
                for _, topic_row in valid_topics.iterrows():
                    topic_id = topic_row['Topic']
                    keywords = self.get_trend_keywords(topic_id, 3)
                    if keywords:
                        kw_str = ", ".join([kw[0] for kw in keywords])
                        print(f"  Тренд {topic_id}: {kw_str}")

        except Exception as e:
            print(f"Ошибка при обучении модели: {e}")
            # Возвращаем все как шум
            self.topics = [-1] * len(texts)

        return self.topics

    def _preprocess_texts(self, texts):
        """Улучшенная предобработка"""
        processed = []
        for text in texts:
            # Приводим к нижнему регистру
            text = text.lower()

            # Убираем специальные символы, оставляем буквы и пробелы
            text = re.sub(r'[^\w\s]', ' ', text)

            # Убираем числа
            text = re.sub(r'\d+', '', text)

            # Убираем лишние пробелы
            text = re.sub(r'\s+', ' ', text).strip()

            processed.append(text)

        return processed

    def get_trends_info(self):
        if self.topic_model:
            try:
                return self.topic_model.get_topic_info()
            except:
                return pd.DataFrame()
        return pd.DataFrame()

    def get_trend_keywords(self, trend_id, top_n=10):
        """Получение ключевых слов с улучшенной фильтрацией"""
        if not self.topic_model:
            return []

        try:
            keywords = self.topic_model.get_topic(trend_id)
            if not keywords:
                return []

            # Фильтруем
            filtered = []
            seen_words = set()

            for word, score in keywords:
                word = str(word).strip()
                if (len(word) > 2 and  # Слова от 3 букв
                        word not in seen_words and
                        not word.isdigit()):

                    # Исключаем слишком общие слова
                    common_words = {'the', 'and', 'for', 'with', 'that',
                                    'this', 'have', 'from', 'their', 'they'}
                    if word not in common_words:
                        filtered.append((word, score))
                        seen_words.add(word)

                if len(filtered) >= top_n:
                    break

            return filtered
        except Exception as e:
            print(f"Ошибка при получении ключевых слов для тренда {trend_id}: {e}")
            return []