import os
import sys
import logging

# ========== НАСТРОЙКА ДЛЯ WINDOWS ==========
# Решаем проблему с symlinks на Windows
os.environ['HF_HOME'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Создаем папку для кэша
cache_dir = os.environ['HF_HOME']
os.makedirs(cache_dir, exist_ok=True)
print(f"Кэш моделей: {cache_dir}")

# ========== ИМПОРТЫ ==========
# Теперь безопасно импортировать
from typing import List, Dict, Union, Any  # ← ДОБАВЬТЕ ЭТУ СТРОЧКУ!
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Анализатор тональности текстов на основе предобученных моделей.

    Поддерживает различные модели для разных типов текстов:
    - Финансовые новости (FinBERT)
    - Социальные медиа (Twitter-RoBERTa)
    - Универсальные тексты
    """

    # Доступные модели с их конфигурацией
    MODEL_CONFIGS = {
        # Финансовые модели (оптимизированы для новостей)
        "finbert": {
            "path": "ProsusAI/finbert",
            "labels": {0: "negative", 1: "neutral", 2: "positive"},
            "description": "FinBERT - специализирован для финансовых текстов"
        },
        "distilroberta-financial": {
            "path": "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            "labels": {0: "negative", 1: "neutral", 2: "positive"},
            "description": "Дистиллированная модель для финансовых новостей (быстрее)"
        },

        # Модели для социальных медиа
        "twitter-roberta": {
            "path": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "labels": {0: "negative", 1: "neutral", 2: "positive"},
            "description": "Оптимизирована для коротких текстов из соцсетей"
        },

        # Универсальные модели
        "bert-base": {
            "path": "bert-base-uncased",
            "labels": {0: "negative", 1: "neutral", 2: "positive"},
            "description": "Базовая модель BERT"
        }
    }

    def __init__(self, model_name: str = "distilroberta-financial",
                 device: str = None):
        """
        Инициализация анализатора тональности.

        Args:
            model_name: Название модели из MODEL_CONFIGS
            device: 'cuda', 'cpu', или None (автоопределение)
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Модель {model_name} не найдена. Доступные: {list(self.MODEL_CONFIGS.keys())}")

        self.model_config = self.MODEL_CONFIGS[model_name]
        self.model_name = model_name

        # Определяем устройство
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Инициализация анализатора тональности: {model_name}")
        logger.info(f"Используется устройство: {self.device}")
        logger.info(f"Описание модели: {self.model_config['description']}")

        # Загружаем модель и токенизатор
        self._load_model()

    def _load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            logger.info(f"Загрузка модели: {self.model_config['path']}")

            # Загружаем токенизатор
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['path'],
                use_fast=True  # Используем быстрый токенизатор
            )

            # Загружаем модель
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config['path']
            )

            # Перемещаем модель на нужное устройство
            self.model.to(self.device)
            self.model.eval()  # Переключаем в режим оценки

            logger.info("Модель успешно загружена")

        except Exception as e:
            logger.error(f"Ошибка при загрузке модели: {e}")
            raise

    def analyze_text(self, text: str, return_raw: bool = False) -> Dict:
        """
        Анализ тональности одного текста.

        Args:
            text: Текст для анализа
            return_raw: Возвращать ли сырые предсказания

        Returns:
            Словарь с результатами анализа
        """
        if not text or not isinstance(text, str):
            return self._get_empty_result()

        try:
            # Токенизация текста
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512  # Ограничение длины
            )

            # Перемещаем на устройство
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Получаем предсказания (без вычисления градиентов)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # Определяем предсказанную категорию
            predicted_class = torch.argmax(predictions).item()
            confidence = predictions[0][predicted_class].item()

            # Получаем метку класса
            label = self.model_config['labels'].get(
                predicted_class,
                "neutral"
            )

            # Нормализуем оценку в диапазон [-1, 1]
            # где -1 = негатив, 0 = нейтрально, 1 = позитив
            normalized_score = self._normalize_score(predicted_class, confidence)

            # Формируем результат
            result = {
                "text": text[:100] + "..." if len(text) > 100 else text,  # Обрезаем для читаемости
                "sentiment": label,
                "score": normalized_score,  # -1 to 1
                "confidence": confidence,
                "predicted_class": predicted_class,
                "model": self.model_name
            }

            # Добавляем сырые предсказания если нужно
            if return_raw:
                result["raw_predictions"] = predictions.cpu().numpy().tolist()[0]

            return result

        except Exception as e:
            logger.error(f"Ошибка при анализе текста: {e}")
            return self._get_empty_result(text)

    def _normalize_score(self, predicted_class: int, confidence: float) -> float:
        """
        Нормализация оценки в диапазон [-1, 1].

        -1: максимально негативный
         0: нейтральный
         1: максимально позитивный
        """
        # Маппинг классов на числовые значения
        class_mapping = {
            0: -1.0,  # negative
            1: 0.0,  # neutral
            2: 1.0  # positive
        }

        base_score = class_mapping.get(predicted_class, 0.0)

        # Учитываем уверенность модели
        # Если уверенность низкая, приближаем к нейтральному
        adjusted_score = base_score * confidence

        return round(adjusted_score, 4)

    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[Dict]:
        """
        Пакетный анализ тональности.

        Args:
            texts: Список текстов
            batch_size: Размер батча для обработки

        Returns:
            Список результатов анализа
        """
        if not texts:
            return []

        results = []
        total_texts = len(texts)

        logger.info(f"Начало пакетного анализа {total_texts} текстов")

        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            logger.info(f"Обработка батча {batch_num} ({len(batch)} текстов)")

            for text in batch:
                result = self.analyze_text(text)
                results.append(result)

        logger.info(f"Пакетный анализ завершен. Обработано: {len(results)} текстов")
        return results

    def get_sentiment_summary(self, texts: List[str]) -> Dict:
        """
        Сводная статистика по тональности массива текстов.

        Args:
            texts: Список текстов

        Returns:
            Словарь со сводной статистикой
        """
        if not texts:
            return self._get_empty_summary()

        # Анализируем все тексты
        analyses = self.analyze_batch(texts)

        # Статистика по категориям
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        scores = []
        confidences = []

        for analysis in analyses:
            sentiment = analysis["sentiment"]
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1

            scores.append(analysis["score"])
            confidences.append(analysis["confidence"])

        # Рассчитываем метрики
        total = len(analyses)

        # Распределение в процентах
        distribution = {
            sentiment: round(count / total * 100, 2)
            for sentiment, count in sentiment_counts.items()
        }

        # Определяем доминирующую тональность
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)

        # Средняя оценка тональности
        avg_score = np.mean(scores) if scores else 0
        avg_confidence = np.mean(confidences) if confidences else 0

        # Индекс настроения (от -100 до 100)
        sentiment_index = round(avg_score * 100, 2)

        return {
            "total_texts": total,
            "average_score": round(avg_score, 4),
            "average_confidence": round(avg_confidence, 4),
            "sentiment_index": sentiment_index,
            "distribution_percentage": distribution,
            "distribution_counts": sentiment_counts,
            "dominant_sentiment": dominant_sentiment,
            "score_range": {
                "min": round(min(scores), 4) if scores else 0,
                "max": round(max(scores), 4) if scores else 0,
                "std": round(np.std(scores), 4) if scores else 0
            }
        }

    def _get_empty_result(self, text: str = "") -> Dict:
        """Пустой результат при ошибке"""
        return {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "predicted_class": 1,
            "model": self.model_name,
            "error": True
        }

    def _get_empty_summary(self) -> Dict:
        """Пустая сводка при отсутствии текстов"""
        return {
            "total_texts": 0,
            "average_score": 0.0,
            "average_confidence": 0.0,
            "sentiment_index": 0.0,
            "distribution_percentage": {"positive": 0, "neutral": 0, "negative": 0},
            "distribution_counts": {"positive": 0, "neutral": 0, "negative": 0},
            "dominant_sentiment": "neutral",
            "score_range": {"min": 0, "max": 0, "std": 0}
        }

    def get_model_info(self) -> Dict:
        """Информация о загруженной модели"""
        return {
            "name": self.model_name,
            "path": self.model_config["path"],
            "description": self.model_config["description"],
            "labels": self.model_config["labels"],
            "device": str(self.device),
            "parameters": sum(p.numel() for p in self.model.parameters())
        }


# Фабричная функция для удобства
def create_sentiment_analyzer(model_name="distilroberta-financial", **kwargs):
    """
    Создание анализатора тональности.

    Args:
        model_name: Имя модели из SentimentAnalyzer.MODEL_CONFIGS
        **kwargs: Дополнительные аргументы для конструктора

    Returns:
        SentimentAnalyzer instance
    """
    return SentimentAnalyzer(model_name=model_name, **kwargs)


if __name__ == "__main__":
    # Демонстрация работы
    print("=" * 60)
    print("Демонстрация анализатора тональности")
    print("=" * 60)

    # Создаем анализатор
    analyzer = SentimentAnalyzer(model_name="distilroberta-financial")

    # Тестовые тексты
    test_texts = [
        "Tesla stock surges after record quarterly earnings",
        "Company faces lawsuit over environmental violations",
        "The board meeting will be held next Tuesday",
        "Apple announces groundbreaking new AI technology",
        "Market crash leaves investors with huge losses"
    ]

    print("\nИнформация о модели:")
    model_info = analyzer.get_model_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    print("\nАнализ отдельных текстов:")
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_text(text)
        print(f"\n{i}. {result['text']}")
        print(f"   Тональность: {result['sentiment']} (оценка: {result['score']:.2f})")
        print(f"   Уверенность: {result['confidence']:.2%}")

    print("\nСводная статистика:")
    summary = analyzer.get_sentiment_summary(test_texts)
    print(f"  Всего текстов: {summary['total_texts']}")
    print(f"  Средняя оценка: {summary['average_score']:.2f}")
    print(f"  Индекс настроения: {summary['sentiment_index']}")
    print(f"  Распределение: {summary['distribution_percentage']}")
    print(f"  Доминирующая тональность: {summary['dominant_sentiment']}")

    print("\n" + "=" * 60)
    print("Демонстрация завершена")