from itertools import combinations
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def count_matches(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> tuple[int, int, int]:
    left_profiles = left[columns].drop_duplicates()
    right_profiles = right[columns].drop_duplicates()

    shared_profiles = left_profiles.merge(right_profiles, on=columns, how="inner")
    left_rows = left[columns].merge(shared_profiles, on=columns, how="inner")
    right_rows = right[columns].merge(shared_profiles, on=columns, how="inner")

    return len(shared_profiles), len(left_rows), len(right_rows)


def main() -> None:
    paths = {
        "initial": PROJECT_ROOT / "data/initial/initial.csv",
        **{
            f"batch_{number:02d}": PROJECT_ROOT / f"data/incoming/batch_{number:02d}.csv"
            for number in range(1, 5)
        },
    }
    datasets = {name: pd.read_csv(path) for name, path in paths.items()}

    columns = datasets["initial"].columns.tolist()
    missing_grades = {"G1", "G2", "G3"} - set(columns)
    if missing_grades:
        raise ValueError(f"В исходной части отсутствуют столбцы: {sorted(missing_grades)}")
    for name, frame in datasets.items():
        if set(frame.columns) != set(columns):
            raise ValueError(f"Набор столбцов {name} отличается от initial")

    checks = {
        "Полные строки": columns,
        "Признаки без G3": [column for column in columns if column != "G3"],
        "Признаки без G1, G2 и G3": [
            column for column in columns if column not in {"G1", "G2", "G3"}
        ],
    }

    print("Размеры частей:")
    for name, frame in datasets.items():
        print(f"  {name}: {len(frame)} строк")

    for description, check_columns in checks.items():
        print(f"\n{description}:")
        print("Пара | Общих профилей | Строк в первой части | Строк во второй части")
        for (left_name, left), (right_name, right) in combinations(datasets.items(), 2):
            profiles, left_count, right_count = count_matches(left, right, check_columns)
            print(f"{left_name} / {right_name} | {profiles} | {left_count} | {right_count}")


if __name__ == "__main__":
    main()
