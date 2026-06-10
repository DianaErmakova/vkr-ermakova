"""
Кластеризация новостей по темам (трендам) с помощью BERTopic.
Предобработка текста вынесена в nlp_processing.text_preprocessor.
"""

import numpy as np
import pandas as pd
import warnings
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

warnings.filterwarnings('ignore')

# Импортируем централизованный препроцессор
try:
    from nlp_processing.text_preprocessor import get_clustering_preprocessor
    _PREPROCESSOR = get_clustering_preprocessor()
except ImportError:
    # Fallback если модуль ещё не в PYTHONPATH
    import re
    class _FallbackPreprocessor:
        def clean_batch(self, texts):
            result = []
            for t in texts:
                t = t.lower()
                t = re.sub(r'[^\w\s]', ' ', t)
                t = re.sub(r'\d+', '', t)
                t = re.sub(r'\s+', ' ', t).strip()
                if t:
                    result.append(t)
            return result
        def get_stats(self, texts):
            return {'total_texts': len(texts)}
    _PREPROCESSOR = _FallbackPreprocessor()


class TrendClusterer:
    def __init__(self, language='english'):
        self.topic_model = None
        self.topics = None
        self.processed_texts = None
        self.language = language
        self._empty_topic_info = pd.DataFrame(columns=['Topic', 'Count', 'Name'])
        self._setup_model()

    def _setup_model(self):
        if self.language == 'russian':
            # русские настройки
            embedding_model = SentenceTransformer('DeepPavlov/rubert-base-cased')
            stop_words = "russian"
            token_pattern = r'(?u)\b[а-яё]{3,}\b'
            language = "russian"
        else:
            # английские настройки
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            stop_words = "english"
            token_pattern = r'(?u)\b[a-zA-Z]{3,}\b'
            language = "english"

        vectorizer_model = CountVectorizer(
            stop_words=stop_words,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            token_pattern=token_pattern,
            max_features=100
        )

        self.topic_model = BERTopic(
            language=language,
            embedding_model=embedding_model,
            min_topic_size=2,
            nr_topics="auto",
            vectorizer_model=vectorizer_model,
            calculate_probabilities=False,
            verbose=False
        )

    def fit_clusters(self, texts):
        print("Обучение модели кластеризации трендов...")
        print(f"Количество текстов: {len(texts)}")

        # Используем централизованный препроцессор
        processed = _PREPROCESSOR.clean_batch(texts)
        self.processed_texts = processed

        stats = _PREPROCESSOR.get_stats(texts)
        unique_words = len(set(" ".join(processed).split()))
        print(f"Уникальных слов после очистки: {unique_words}")

        try:
            self.topics, self.probabilities = self.topic_model.fit_transform(processed)

            topic_info = self.get_trends_info()
            valid_topics = topic_info[topic_info['Topic'] != -1]

            print(f"Найдено кластеров: {len(topic_info)}")
            print(f"Валидные тренды: {len(valid_topics)}")

            if len(valid_topics) > 0:
                print("Обнаруженные тренды:")
                for _, row in valid_topics.iterrows():
                    topic_id = row['Topic']
                    keywords = self.get_trend_keywords(topic_id, 3)
                    if keywords:
                        kw_str = ", ".join([kw[0] for kw in keywords])
                        print(f"  Тренд {topic_id}: {kw_str}")

        except Exception as e:
            print(f"Ошибка при обучении модели: {e}")
            self.topics = [-1] * len(texts)

        return self.topics

    def get_trends_info(self):
        if self.topic_model:
            try:
                result = self.topic_model.get_topic_info()
                if isinstance(result, pd.DataFrame) and not result.empty:
                    return result
            except Exception:
                pass
        return self._empty_topic_info.copy()

    def get_trend_keywords(self, trend_id, top_n=10):
        if not self.topic_model:
            return []
        try:
            keywords = self.topic_model.get_topic(trend_id)
            if not keywords:
                return []

            common_words = {'the', 'and', 'for', 'with', 'that',
                            'this', 'have', 'from', 'their', 'they'}
            filtered = []
            seen = set()
            for word, score in keywords:
                word = str(word).strip()
                if (len(word) > 2 and word not in seen
                        and not word.isdigit()
                        and word not in common_words):
                    filtered.append((word, score))
                    seen.add(word)
                if len(filtered) >= top_n:
                    break
            return filtered
        except Exception as e:
            print(f"Ошибка при получении ключевых слов для тренда {trend_id}: {e}")
            return []

    def get_clustering_metrics(self):
        if not hasattr(self, 'topics') or self.topics is None:
            return {'error': 'Модель ещё не обучена'}

        topics_array = np.array(self.topics)
        topic_info = self.get_trends_info()
        valid_topics = topic_info[topic_info['Topic'] != -1] if not topic_info.empty else pd.DataFrame()
        total_docs = len(topics_array)
        noise_count = int(np.sum(topics_array == -1))

        return {
            'total_documents':    total_docs,
            'clusters_found':     len(valid_topics),
            'noise_documents':    noise_count,
            'noise_percentage':   round(noise_count / total_docs * 100, 2) if total_docs else 0,
            'avg_docs_per_cluster': valid_topics['Count'].mean() if not valid_topics.empty else 0,
            'topic_stability':    (len(valid_topics) / len(topic_info)
                                   if not topic_info.empty else 0),
        }

    def get_detailed_report(self):
        metrics = self.get_clustering_metrics()
        report = {
            'summary': {
                'total_documents': metrics.get('total_documents', 0),
                'valid_clusters':  metrics.get('clusters_found', 0),
                'noise_percentage': metrics.get('noise_percentage', 0),
                'quality_score':   self._calculate_quality_score(metrics)
            },
            'clusters_details': [],
            'metrics': metrics
        }

        topic_info = self.get_trends_info()
        if not topic_info.empty:
            for _, row in topic_info[topic_info['Topic'] != -1].iterrows():
                topic_id = row['Topic']
                keywords = self.get_trend_keywords(topic_id, 5)
                total = metrics.get('total_documents', 1)
                report['clusters_details'].append({
                    'topic_id':        topic_id,
                    'documents_count': int(row['Count']),
                    'percentage':      round(row['Count'] / total * 100, 2),
                    'keywords':        [kw[0] for kw in keywords],
                    'keyword_scores':  [kw[1] for kw in keywords]
                })

        return report

    def _calculate_quality_score(self, metrics):
        try:
            score = 0
            score += min(40, 40 - (metrics.get('noise_percentage', 100) * 0.4))
            clusters = metrics.get('clusters_found', 0)
            if clusters > 0:
                score += min(30, clusters * 3)
            avg = metrics.get('avg_docs_per_cluster', 0)
            total = metrics.get('total_documents', 1)
            if avg > 0 and total > 0:
                ideal = total / max(clusters, 1)
                ratio = avg / ideal
                score += 30 * min(1, 2 - ratio) if ratio > 0 else 0
            return round(min(100, score), 2)
        except Exception:
            return 0