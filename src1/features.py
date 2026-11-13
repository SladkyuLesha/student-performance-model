import pandas as pd


def make_feature_set(
    data: pd.DataFrame,
    *,
    use_grades: bool,
) -> tuple[pd.DataFrame, pd.Series]:
    required = {"G1", "G2", "G3"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Отсутствуют столбцы: {sorted(missing)}")

    columns_to_drop = ["G3"] if use_grades else ["G1", "G2", "G3"]
    features = data.drop(columns=columns_to_drop).copy()
    target = data["G3"].copy()

    return features, target
