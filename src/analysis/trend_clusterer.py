import numpy as np
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
            max_df=0.95,  # ИЗМЕНИТЬ: было 0.8, стало 0.95
            token_pattern=r'(?u)\b[a-zA-Z]{3,}\b',
            max_features=100
        )

        # НАСТРОЙКА BERTopic
        self.topic_model = BERTopic(
            language="english",
            min_topic_size=2,  # ИЗМЕНИТЬ: было 3, стало 2 (для малых данных)
            nr_topics="auto",
            vectorizer_model=vectorizer_model,
            calculate_probabilities=False,
            verbose=False
        )

    def fit_clusters(self, texts):
        print("Обучение модели кластеризации трендов...")
        print(f"Количество текстов: {len(texts)}")

        # Препроцессинг
        processed_texts = self._preprocess_texts(texts)
        self.processed_texts = processed_texts  # Сохраняем для метрик

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
            self.topics = [-1] * len(texts)  # ГАРАНТИРУЕМ что это список
            # Создаем пустой DataFrame для совместимости
            import pandas as pd
            self._empty_topic_info = pd.DataFrame(columns=['Topic', 'Count', 'Name'])

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
                result = self.topic_model.get_topic_info()
                # Проверяем что результат валидный
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result
                else:
                    # Возвращаем пустой DataFrame с правильными колонками
                    return pd.DataFrame(columns=['Topic', 'Count', 'Name'])
            except:
                # Если есть сохраненный пустой DataFrame
                if hasattr(self, '_empty_topic_info'):
                    return self._empty_topic_info
                return pd.DataFrame(columns=['Topic', 'Count', 'Name'])
        # Возвращаем пустой DataFrame с правильными колонками
        return pd.DataFrame(columns=['Topic', 'Count', 'Name'])

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

    def _calculate_topic_coherence(self, texts):
        """Вычисление когерентности тем"""
        # Упрощенная метрика
        if not hasattr(self, 'topics') or -1 not in self.topics:
            return 0

        # Подсчет пересечения ключевых слов
        coherence_scores = []
        for topic_id in set(self.topics):
            if topic_id != -1:
                keywords = [kw[0] for kw in self.get_trend_keywords(topic_id, 5)]
                # Простая проверка: сколько ключевых слов встречаются вместе
                score = self._check_keyword_cooccurrence(texts, keywords)
                coherence_scores.append(score)

        return np.mean(coherence_scores) if coherence_scores else 0

    def get_clustering_metrics(self, texts=None):
        """Метрики качества кластеризации"""
        # ГАРАНТИРУЕМ что self.topics это список
        if not hasattr(self, 'topics') or self.topics is None:
            return {'error': 'Модель еще не обучена'}

        # Преобразуем к списку если нужно
        if isinstance(self.topics, (bool, int, float)):
            # Если что-то пошло не так, создаем список шума
            if hasattr(self, 'processed_texts'):
                self.topics = [-1] * len(self.processed_texts)
            else:
                return {'error': 'Некорректные данные topics'}

        # Используем переданные тексты или сохраненные
        if texts is None and hasattr(self, 'processed_texts'):
            texts = self.processed_texts

        topic_info = self.get_trends_info()

        # Проверяем что topic_info не пустой и имеет нужные колонки
        if topic_info.empty or 'Topic' not in topic_info.columns:
            total_docs = len(self.topics) if isinstance(self.topics, (list, np.ndarray)) else 0
            return {
                'total_documents': total_docs,
                'clusters_found': 0,
                'noise_documents': total_docs,
                'noise_percentage': 100.0,
                'avg_docs_per_cluster': 0,
                'topic_stability': 0
            }

        # Убеждаемся что self.topics это массив
        topics_array = np.array(self.topics) if not isinstance(self.topics, np.ndarray) else self.topics

        valid_topics = topic_info[topic_info['Topic'] != -1]
        total_docs = len(topics_array)

        # Безопасный расчет
        noise_count = np.sum(topics_array == -1) if total_docs > 0 else 0

        # Базовые метрики
        metrics = {
            'total_documents': total_docs,
            'clusters_found': len(valid_topics),
            'noise_documents': int(noise_count),
            'noise_percentage': (noise_count / total_docs * 100) if total_docs > 0 else 0,
            'avg_docs_per_cluster': valid_topics['Count'].mean() if not valid_topics.empty else 0,
            'topic_stability': len(valid_topics) / len(topic_info) if len(topic_info) > 0 else 0,
        }

        return metrics

    def get_detailed_report(self):
        """Детальный отчет о кластеризации"""
        metrics = self.get_clustering_metrics()

        report = {
            'summary': {
                'total_documents': metrics['total_documents'],
                'valid_clusters': metrics['clusters_found'],
                'noise_percentage': round(metrics['noise_percentage'], 2),
                'quality_score': self._calculate_quality_score(metrics)
            },
            'clusters_details': [],
            'metrics': metrics
        }

        # Информация о каждом кластере
        topic_info = self.get_trends_info()
        if not topic_info.empty:
            valid_topics = topic_info[topic_info['Topic'] != -1]

            for _, topic_row in valid_topics.iterrows():
                topic_id = topic_row['Topic']
                keywords = self.get_trend_keywords(topic_id, 5)

                report['clusters_details'].append({
                    'topic_id': topic_id,
                    'documents_count': int(topic_row['Count']),
                    'percentage': round(topic_row['Count'] / metrics['total_documents'] * 100, 2) if metrics[
                                                                                                         'total_documents'] > 0 else 0,
                    'keywords': [kw[0] for kw in keywords],
                    'keyword_scores': [kw[1] for kw in keywords]
                })

        return report

    def _calculate_quality_score(self, metrics):
        """Общий балл качества кластеризации (0-100)"""
        try:
            score = 0

            # Меньше шума = лучше (максимум 40 баллов)
            noise_penalty = min(40, 40 - (metrics['noise_percentage'] * 0.4))
            score += noise_penalty

            # Больше кластеров (но не слишком) = лучше (максимум 30 баллов)
            if metrics['clusters_found'] > 0:
                cluster_score = min(30, metrics['clusters_found'] * 3)
                score += cluster_score

            # Размер кластеров сбалансирован = лучше (максимум 30 баллов)
            if metrics['avg_docs_per_cluster'] > 0 and metrics['total_documents'] > 0:
                ideal_size = metrics['total_documents'] / max(metrics['clusters_found'], 1)
                size_ratio = metrics['avg_docs_per_cluster'] / ideal_size
                size_score = 30 * min(1, 2 - size_ratio) if size_ratio > 0 else 0
                score += size_score

            return round(min(100, score), 2)
        except:
            return 0