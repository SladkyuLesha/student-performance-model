from pathlib import Path
import pandas as pd

from src.data_loader import load_data
from src.preprocessing import preprocess_data, make_feature_sets
from src.splitting import honest_split_data
from src.metrics import calculate_metrics, build_metrics_row, build_metrics_dataframe
from src.visualization import (
    plot_r2_comparison,
    plot_mae_comparison,
    plot_actual_vs_predicted,
)
from src.train_models import train_dummy_regressor, train_linear_regression, train_gradient_boosting_model

def main():
    data_path = Path("data/raw/student_data_500k.csv")

    df = load_data(data_path)
    df_prepared = preprocess_data(df)

    X_with_grades, X_without_grades, y = make_feature_sets(df_prepared)

    X_train_with_grades, X_test_with_grades, y_train_with, y_test_with = honest_split_data(X_with_grades, y)
    X_train_without_grades, X_test_without_grades, y_train_without, y_test_without = honest_split_data(X_without_grades, y)

    # Train models
    print("[1/6] Training Dummy Regressor with grades...")
    dummy_model_with_grades = train_dummy_regressor(X_train_with_grades, y_train_with)
    y_pred = dummy_model_with_grades.predict(X_test_with_grades)
    print("[2/6] Training Dummy Regressor without grades...")
    dummy_model_without_grades = train_dummy_regressor(X_train_without_grades, y_train_without)
    y_pred_without_grades = dummy_model_without_grades.predict(X_test_without_grades)

    print("[3/6] Training Linear Regression with grades...")
    linear_model_with_grades = train_linear_regression(X_train_with_grades, y_train_with)
    y_pred_linear = linear_model_with_grades.predict(X_test_with_grades)
    print("[4/6] Training Linear Regression without grades...")
    linear_model_without_grades = train_linear_regression(X_train_without_grades, y_train_without)
    y_pred_linear_without_grades = linear_model_without_grades.predict(X_test_without_grades)

    print("[5/6] Training Gradient Boosting with grades...")
    gradient_boosting_model_with_grades = train_gradient_boosting_model(X_train_with_grades, y_train_with, max_iter=20000, learning_rate=0.03, min_samples_leaf=70, l2_regularization=2.0, n_iter_no_change=60)
    y_pred_gb = gradient_boosting_model_with_grades.predict(X_test_with_grades)
    print("[6/6] Training Gradient Boosting without grades...")
    gradient_boosting_model_without_grades = train_gradient_boosting_model(X_train_without_grades, y_train_without, max_iter=20000, learning_rate=0.03, min_samples_leaf=70, l2_regularization=2.0, n_iter_no_change=60)
    y_pred_gb_without_grades = gradient_boosting_model_without_grades.predict(X_test_without_grades)

    # Calculate metrics
    metrics_rows = []
    metrics_dummy_with_grades = calculate_metrics(y_test_with, y_pred)
    metrics_rows.append(build_metrics_row("Dummy Regressor", "with grades", metrics_dummy_with_grades))
    metrics_dummy_without_grades = calculate_metrics(y_test_without, y_pred_without_grades)
    metrics_rows.append(build_metrics_row("Dummy Regressor", "without grades", metrics_dummy_without_grades))

    metrics_linear_with_grades = calculate_metrics(y_test_with, y_pred_linear)
    metrics_rows.append(build_metrics_row("Linear Regression", "with grades", metrics_linear_with_grades))
    metrics_linear_without_grades = calculate_metrics(y_test_without, y_pred_linear_without_grades)
    metrics_rows.append(build_metrics_row("Linear Regression", "without grades", metrics_linear_without_grades))
    
    metrics_gb_with_grades = calculate_metrics(y_test_with, y_pred_gb)
    metrics_rows.append(build_metrics_row("Gradient Boosting", "with grades", metrics_gb_with_grades))
    metrics_gb_without_grades = calculate_metrics(y_test_without, y_pred_gb_without_grades)
    metrics_rows.append(build_metrics_row("Gradient Boosting", "without grades", metrics_gb_without_grades))
    
    metrics_df = build_metrics_dataframe(metrics_rows)
    reports_table_dir = Path("reports/tables/model_comparison/legacy")
    reports_table_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = reports_table_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

    # Visualizations
    figures_dir = Path("reports/figures/model_comparison/legacy")
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_r2_comparison(metrics_df, figures_dir / "r2_comparison.png")
    plot_mae_comparison(metrics_df, figures_dir / "mae_comparison.png")
    predictions_dir = Path("reports/figures/predictions/legacy")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    plot_actual_vs_predicted(
        y_test_with,
        y_pred_gb,
        predictions_dir / "actual_vs_predicted_gb_with_grades.png",
        "Gradient Boosting",
        "with grades"
        )
    print(f"R2 comparison plot saved to {figures_dir / 'r2_comparison.png'}")
    print(f"MAE comparison plot saved to {figures_dir / 'mae_comparison.png'}")
    print(f"Actual vs Predicted plot saved to {predictions_dir / 'actual_vs_predicted_gb_with_grades.png'}")

    # DataFrame to store metrics
    print(metrics_df)
    print("GB with grades iterations:", gradient_boosting_model_with_grades.n_iter_)
    print("GB without grades iterations:", gradient_boosting_model_without_grades.n_iter_)

if __name__ == "__main__":
    main()
