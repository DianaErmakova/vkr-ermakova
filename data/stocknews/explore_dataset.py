# explore_dataset.py
import pandas as pd
import os


def explore_stocknews():
    # Путь к данным
    base_dir = "C:/Users/racco/PycharmProjects/vkr-ermakova"
    data_path = os.path.join(base_dir, 'data', 'stocknews', 'Combined_News_DJIA.csv')

    # Проверяем существует ли файл
    if not os.path.exists(data_path):
        print(f"Файл не найден: {data_path}")
        print("Скачай датасет с Kaggle и положи в data/stocknews/")
        print("Ссылка: https://www.kaggle.com/datasets/aaron7sun/stocknews")
        return None

    # Загружаем данные
    print(f"Загрузка данных из: {data_path}")
    df = pd.read_csv(data_path)

    # Основная информация
    print("\n" + "=" * 50)
    print("ОСНОВНАЯ ИНФОРМАЦИЯ О ДАТАСЕТЕ")
    print("=" * 50)
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} колонок")

    print(f"\nКолонки:")
    for i, col in enumerate(df.columns):
        print(f"  {i + 1}. {col}")

    # Информация о данных
    print(f"\nТИПЫ ДАННЫХ:")
    print(df.dtypes)

    print(f"\nПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    # Диапазон дат
    print(f"\nДИАПАЗОН ДАТ:")
    print(f"  Начало: {df['Date'].min()}")
    print(f"  Конец:  {df['Date'].max()}")
    print(f"  Всего дней: {df['Date'].nunique()}")

    # Распределение меток
    print(f"\nРАСПРЕДЕЛЕНИЕ МЕТОК (Label):")
    print(df['Label'].value_counts())
    print(f"  1 = рост индекса DJIA")
    print(f"  0 = падение индекса DJIA")

    # Пример новостей
    print(f"\n" + "=" * 50)
    print("ПРИМЕРЫ НОВОСТЕЙ (первый день)")
    print("=" * 50)

    # Покажем первую строку с новостями
    first_row = df.iloc[0]
    print(f"Дата: {first_row['Date']}")
    print(f"Индекс DJIA: {first_row['Label']} (1=рост, 0=падение)")

    print(f"\nТоп-5 новостей этого дня:")
    for i in range(1, 6):
        col_name = f'Top{i}'
        if col_name in df.columns:
            news_text = str(first_row[col_name])
            print(f"\n{i}. {col_name}:")
            print(f"   {news_text[:150]}...")

    # Статистика по новостям
    print(f"\n" + "=" * 50)
    print("СТАТИСТИКА ПО НОВОСТЯМ")
    print("=" * 50)

    # Считаем среднюю длину новостей
    news_columns = [col for col in df.columns if 'Top' in col]
    avg_lengths = []

    for col in news_columns[:3]:  # Проверим только первые 3 колонки для скорости
        avg_len = df[col].astype(str).str.len().mean()
        avg_lengths.append(avg_len)
        print(f"  {col}: средняя длина {avg_len:.0f} символов")

    # Информация о дубликатах
    print(f"\nДУБЛИКАТЫ:")
    duplicates = df.duplicated(subset=['Date']).sum()
    print(f"  Дубликатов дат: {duplicates}")

    return df


if __name__ == "__main__":
    df = explore_stocknews()

    if df is not None:
        print(f"\n" + "=" * 50)
        print("КРАТКАЯ СВОДКА")
        print("=" * 50)
        print(f"• Файл загружен: {len(df)} записей")
        print(f"• Период: {df['Date'].min()} - {df['Date'].max()}")
        print(f"• Новостных колонок: {len([c for c in df.columns if 'Top' in c])}")
        print(f"• Всего новостей: {len(df) * 25} (25 в день)")
        print(f"• Ростов индекса: {(df['Label'] == 1).sum()} дней")
        print(f"• Падений индекса: {(df['Label'] == 0).sum()} дней")