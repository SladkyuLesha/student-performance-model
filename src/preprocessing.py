import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=["record_id"], errors="ignore")
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    return df


def make_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    X_with_grades = df.drop(columns=["G3"])
    X_without_grades = df.drop(columns=["G1", "G2", "G3"])
    y = df["G3"]
    return X_with_grades, X_without_grades, y