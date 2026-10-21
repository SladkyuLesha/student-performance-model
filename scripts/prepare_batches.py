from pathlib import Path

import numpy as np
import pandas as pd


def prepare_the_division(df: pd.DataFrame):
    # Перемешиваем строки, чтобы разделение было случайным
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Определяем размер исходной части (60%)
    initial_size = int(len(shuffled) * 0.6)

    # Выделяем исходные данные (60%)
    initial = shuffled.iloc[:initial_size].copy()

    # Оставшиеся строки станут новыми порциями
    incoming = shuffled.iloc[initial_size:].copy()

    # Получаем четыре DataFrame по индексам
    indices = np.array_split(np.arange(len(incoming)), 4)
    four_parts = [incoming.iloc[idx].copy() for idx in indices]
    batch_01, batch_02, batch_03, batch_04 = four_parts

    return (
        initial,
        incoming,
        batch_01,
        batch_02,
        batch_03,
        batch_04,
    )



def main():
    # Загружаем исходные данные
    df = pd.read_csv("data/raw/student_data.csv")

    # Получаем результаты разбиения
    initial, incoming, batch_01, batch_02, batch_03, batch_04 = prepare_the_division(df)

    # Создаём каталоги, если их нет
    initial_dir = Path("data/initial")
    incoming_dir = Path("data/incoming")

    initial_dir.mkdir(parents=True, exist_ok=True)
    incoming_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем исходную часть
    initial.to_csv(initial_dir / "initial.csv", index=False)

    # Собираем четыре порции в список
    batches = [batch_01, batch_02, batch_03, batch_04]

    # Объединяем все части обратно
    all_parts = [initial, *batches]
    reconstructed = pd.concat(all_parts).sort_index()

    # Проверяем, что ни одна позиция не повторилась
    assert not reconstructed.index.has_duplicates

    # Проверяем, что получен исходный перемешанный набор
    shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    assert reconstructed.equals(shuffled)

    print("Проверка пройдена: все 395 записей распределены правильно!")

    # Сохраняем порции
    for i, batch in enumerate(batches, start=1):
        path = incoming_dir / f"batch_{i:02d}.csv"
        batch.to_csv(path, index=False)
        print(f"{path}: {len(batch)} строк")

    # Проверяем количество строк
    total = len(initial) + sum(len(batch) for batch in batches)

    assert total == len(df), "Количество строк не совпадает!"

    print(f"\nИсходных записей: {len(df)}")
    print(f"Начальная часть: {len(initial)}")
    print(f"Новые данные: {len(incoming)}")
    print(f"Всего после разбиения: {total}")


if __name__ == "__main__":
    main()
