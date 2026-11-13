from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Добавляет признаки без оценки статистик на новых данных."""

    def __init__(self, add_features: bool = True):
        self.add_features = add_features

    def fit(self, X: pd.DataFrame, y=None):
        self.input_columns_ = X.columns.tolist()
        if self.add_features:
            missing = {"failures", "studytime", "Medu", "Fedu"} - set(X.columns)
            if missing:
                raise ValueError(f"Отсутствуют признаки: {sorted(missing)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if set(X.columns) != set(self.input_columns_):
            raise ValueError("Столбцы входных данных отличаются от обучающих")
        result = X.loc[:, self.input_columns_].copy()
        if self.add_features:
            result["has_failures"] = (result["failures"] > 0).astype(float).where(result["failures"].notna())
            result["studytime_high"] = (result["studytime"] >= 3).astype(float).where(result["studytime"].notna())
            result["parent_education_mean"] = (result["Medu"] + result["Fedu"]) / 2
        return result


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, make_column_selector(dtype_include=np.number)),
        ("categorical", categorical, make_column_selector(dtype_exclude=np.number)),
    ])


def make_model_pipeline(model, *, add_features: bool) -> Pipeline:
    return Pipeline([
        ("features", FeatureBuilder(add_features=add_features)),
        ("preprocess", make_preprocessor()),
        ("model", model),
    ])


def prepare_new_data(data: pd.DataFrame, pipeline: Pipeline, *, use_grades: bool):
    """Применяет сохранённую обработку к новым данным без повторного fit."""
    excluded = ["G3"] if use_grades else ["G1", "G2", "G3"]
    features = data.drop(columns=excluded, errors="ignore")
    return pipeline[:-1].transform(features)


def prepare_csv_for_prediction(path: str | Path, pipeline: Pipeline, *, use_grades: bool):
    """Читает CSV и применяет к нему ранее обученные преобразования."""
    return prepare_new_data(pd.read_csv(path), pipeline, use_grades=use_grades)
