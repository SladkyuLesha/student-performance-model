from pathlib import Path
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import preprocess_data, make_feature_sets
from src.splitting import honest_split_data
from src.metrics import calculate_metrics
from src.train_models import train_dummy_regressor, train_linear_regression, train_gradient_boosting_model

def main():
    data_path = Path("data/raw/student_data_500k.csv")

    df = load_data(data_path)
    df_prepared = preprocess_data(df)

    X_with_grades, X_without_grades, y = make_feature_sets(df_prepared)

    X_train_with_grades, X_test_with_grades, y_train_with, y_test_with = honest_split_data(X_with_grades, y)
    X_train_without_grades, X_test_without_grades, y_train_without, y_test_without = honest_split_data(X_without_grades, y)

    # Train models
    dummy_model_with_grades = train_dummy_regressor(X_train_with_grades, y_train_with)
    y_pred = dummy_model_with_grades.predict(X_test_with_grades)
    dummy_model_without_grades = train_dummy_regressor(X_train_without_grades, y_train_without)
    y_pred_without_grades = dummy_model_without_grades.predict(X_test_without_grades)

    linear_model_with_grades = train_linear_regression(X_train_with_grades, y_train_with)
    y_pred_linear = linear_model_with_grades.predict(X_test_with_grades)
    linear_model_without_grades = train_linear_regression(X_train_without_grades, y_train_without)
    y_pred_linear_without_grades = linear_model_without_grades.predict(X_test_without_grades)

    gradient_boosting_model_with_grades = train_gradient_boosting_model(X_train_with_grades, y_train_with)
    y_pred_gb = gradient_boosting_model_with_grades.predict(X_test_with_grades)
    gradient_boosting_model_without_grades = train_gradient_boosting_model(X_train_without_grades, y_train_without)
    y_pred_gb_without_grades = gradient_boosting_model_without_grades.predict(X_test_without_grades)

    # Calculate metrics
    metrics_dummy_with_grades = calculate_metrics(y_test_with, y_pred)
    metrics_dummy_without_grades = calculate_metrics(y_test_without, y_pred_without_grades)

    metrics_linear_with_grades = calculate_metrics(y_test_with, y_pred_linear)
    metrics_linear_without_grades = calculate_metrics(y_test_without, y_pred_linear_without_grades)
    
    metrics_gb_with_grades = calculate_metrics(y_test_with, y_pred_gb)
    metrics_gb_without_grades = calculate_metrics(y_test_without, y_pred_gb_without_grades)
    
    # DataFrame to store metrics
    metrics_df = pd.DataFrame({
        "Model": ["Dummy Regressor (with grades)", "Dummy Regressor (without grades)",
                  "Linear Regression (with grades)", "Linear Regression (without grades)",
                  "Gradient Boosting (with grades)", "Gradient Boosting (without grades)"],
        "MAE": [metrics_dummy_with_grades["mae"], metrics_dummy_without_grades["mae"],
                metrics_linear_with_grades["mae"], metrics_linear_without_grades["mae"],
                metrics_gb_with_grades["mae"], metrics_gb_without_grades["mae"]],
        "MSE": [metrics_dummy_with_grades["mse"], metrics_dummy_without_grades["mse"],
                metrics_linear_with_grades["mse"], metrics_linear_without_grades["mse"],
                metrics_gb_with_grades["mse"], metrics_gb_without_grades["mse"]],
        "RMSE": [metrics_dummy_with_grades["rmse"], metrics_dummy_without_grades["rmse"],
                 metrics_linear_with_grades["rmse"], metrics_linear_without_grades["rmse"],
                 metrics_gb_with_grades["rmse"], metrics_gb_without_grades["rmse"]],
        "R2": [metrics_dummy_with_grades["r2"], metrics_dummy_without_grades["r2"],
               metrics_linear_with_grades["r2"], metrics_linear_without_grades["r2"],
               metrics_gb_with_grades["r2"], metrics_gb_without_grades["r2"]]
    })

    print(metrics_df)


if __name__ == "__main__":
    main()
    