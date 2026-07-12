from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def build_metrics_row(model_name:str, features: str, metrics: dict[str, float],
                      n_iter: int | None = None, params: dict | None = None) -> dict[str, float]:
    row = {
        "model": model_name,
        "features": features,
        **metrics
        }
    if n_iter is not None:
        row["n_iter"] = n_iter
    if params is not None:
        for key, value in params.items():
            row[key] = value

    return row

def build_metrics_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)