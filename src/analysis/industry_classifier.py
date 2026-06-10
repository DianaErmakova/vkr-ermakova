"""
Отраслевой классификатор на основе ключевых слов.
(Простая эвристика для демонстрации в ВКР)
"""

import logging

logger = logging.getLogger(__name__)


class IndustryClassifier:
    """
    Простой классификатор отраслей по ключевым словам.
    Не требует загрузки моделей, работает быстро.

    Принцип работы:
        - Для каждой категории (Финансы, Технологии, Спорт) задан список ключевых слов
        - Текст разбивается на слова, подсчитывается количество совпадений
        - Выбирается категория с максимальным числом совпадений
        - Уверенность = количество совпадений / 10 (кап 0.95)
        - Если совпадений нет — по умолчанию 'Финансы' с уверенностью 0.5

    Ограничения:
        - Не учитывает контекст (например, "apple" может быть фруктом или компанией)
        - Для ВКР этого достаточно, так как демонстрирует принцип классификации
        - В реальной системе можно заменить на fine-tuned BERT
    """

    # Ключевые слова для каждой категории
    # Поддерживаются английский и русский языки
    KEYWORDS = {
        'Business': [
            # английские
            'stock', 'share', 'market', 'earnings', 'profit', 'revenue',
            'investment', 'fund', 'bank', 'finance', 'economy', 'trade',
            'business', 'company', 'corporation', 'ceo', 'merger', 'acquisition',
            # русские
            'акции', 'рынок', 'прибыль', 'инвестиции', 'банк', 'финансы',
            'экономика', 'бизнес', 'компания'
        ],
        'Sci/Tech': [
            # английские
            'tech', 'software', 'hardware', 'ai', 'artificial intelligence',
            'chip', 'semiconductor', 'digital', 'cloud', 'data', 'app',
            'technology', 'innovation', 'startup',
            # русские
            'технологии', 'софт', 'чип', 'ии', 'искусственный интеллект', 'цифровой'
        ],
        'Sports': [
            # английские
            'sport', 'game', 'player', 'team', 'match', 'tournament',
            'olympic', 'football', 'basketball',
            # русские
            'спорт', 'игра', 'матч', 'футбол', 'олимпиада'
        ]
    }

    def __init__(self):
        logger.info("Отраслевой классификатор (эвристический) инициализирован")

    def classify(self, text):
        """
        Классифицирует текст по ключевым словам.

        Args:
            text: текст новости (строка)

        Returns:
            dict: {
                'id': int,           # 0-World, 1-Sports, 2-Business, 3-Sci/Tech
                'name': str,         # английское название категории
                'industry_ru': str,  # русское название для дашборда
                'confidence': float  # уверенность от 0 до 1
            }
        """
        if not text:
            return {'id': 2, 'name': 'Business', 'industry_ru': 'Финансы', 'confidence': 0.5}

        text_lower = text.lower()

        # Считаем количество совпадений для каждой категории
        scores = {}
        for category, keywords in self.KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[category] = count

        # Определяем категорию с максимальным числом совпадений
        max_score = max(scores.values())

        if max_score == 0:
            # Нет совпадений — по умолчанию Business
            best_category = 'Business'
            confidence = 0.5
        else:
            best_category = max(scores, key=scores.get)
            # Уверенность растёт с количеством совпадений, максимум 0.95
            confidence = min(0.95, max_score / 10)

        # Маппинг для отображения в дашборде
        industry_ru = {
            'Business': 'Финансы',
            'Sci/Tech': 'Технологии',
            'Sports': 'Спорт',
            'World': 'Макроэкономика'
        }.get(best_category, 'Финансы')

        # ID для совместимости с исходным кодом (4 категории AG News)
        id_map = {'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}

        return {
            'id': id_map.get(best_category, 2),
            'name': best_category,
            'industry_ru': industry_ru,
            'confidence': round(confidence, 3)
        }

    def classify_batch(self, texts):
        """Пакетная классификация списка текстов"""
        return [self.classify(text) for text in texts]


def create_industry_classifier():
    """Фабричная функция для создания классификатора"""
    return IndustryClassifier()