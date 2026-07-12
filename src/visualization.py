import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_r2_comparison(metrics_df, output_path):
    output_path = Path(output_path)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=metrics_df,
        x="r2",
        y='model',
        hue='features'
    )
    plt.title("R2 by Model")
    plt.xlabel("R2")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_mae_comparison(metrics_df, output_path):
    output_path = Path(output_path)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=metrics_df,
        x="mae",
        y='model',
        hue='features'
    )
    plt.title("MAE by Model")
    plt.xlabel("MAE")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_actual_vs_predicted(y_true, y_pred, output_path: str | Path, model_name: str, features: str):
    output_path = Path(output_path)

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.title(f"Actual vs Predicted - {model_name} ({features})")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()