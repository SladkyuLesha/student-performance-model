import hashlib
import io
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/initial/initial.csv"
SPLIT_PATH = PROJECT_ROOT / "reports/tables/model_split.json"


def create_split(data_path: Path) -> dict:
    data_bytes = data_path.read_bytes()
    data = pd.read_csv(io.BytesIO(data_bytes))
    grades = {"G1", "G2", "G3"}
    missing = grades - set(data.columns)
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {sorted(missing)}")

    early_features = data.drop(columns=list(grades))
    if early_features.duplicated().any():
        raise ValueError("Есть повторяющиеся ранние профили: необходимо пересмотреть разбиение")

    development_positions, test_positions = train_test_split(
        list(range(len(data))),
        test_size=0.2,
        random_state=42,
    )
    train_positions, validation_positions = train_test_split(
        sorted(development_positions),
        test_size=0.2,
        random_state=42,
    )

    return {
        "source": "data/initial/initial.csv",
        "source_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "row_count": len(data),
        "test_size": 0.2,
        "validation_fraction_of_development": 0.2,
        "random_state": 42,
        "train_positions": sorted(train_positions),
        "validation_positions": sorted(validation_positions),
        "test_positions": sorted(test_positions),
    }


def main() -> None:
    split = create_split(DATA_PATH)
    if SPLIT_PATH.exists():
        existing = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
        if existing != split:
            raise ValueError(f"Существующее разбиение отличается: {SPLIT_PATH}")
        print("Существующее разбиение совпадает с повторным расчётом.")
    else:
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SPLIT_PATH.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Разбиение сохранено.")

    print(f"Train: {len(split['train_positions'])} строк")
    print(f"Validation: {len(split['validation_positions'])} строк")
    print(f"Test: {len(split['test_positions'])} строк")
    print(f"Файл: {SPLIT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
