import hashlib
import io
import json
from pathlib import Path

import pandas as pd

from src1.split_data import DATA_PATH, SPLIT_PATH


def load_split(
    data_path: Path = DATA_PATH,
    split_path: Path = SPLIT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_bytes = Path(data_path).read_bytes()
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))

    actual_hash = hashlib.sha256(data_bytes).hexdigest()
    if actual_hash != split["source_sha256"]:
        raise ValueError("Контрольная сумма CSV не совпадает с сохранённым разбиением")

    data = pd.read_csv(io.BytesIO(data_bytes))
    if len(data) != split["row_count"]:
        raise ValueError("Количество строк CSV не совпадает с сохранённым разбиением")

    keys = ("train_positions", "validation_positions", "test_positions")
    positions = [split[key] for key in keys]
    if any(not isinstance(part, list) or not part for part in positions):
        raise ValueError("Каждая выборка должна содержать список позиций строк")

    all_positions = [position for part in positions for position in part]
    if any(type(position) is not int for position in all_positions):
        raise ValueError("Позиции строк должны быть целыми числами")
    if len(all_positions) != len(data) or set(all_positions) != set(range(len(data))):
        raise ValueError("Позиции строк повторяются, пропущены или выходят за границы CSV")

    train, validation, test = (data.iloc[part].copy() for part in positions)
    return train, validation, test
