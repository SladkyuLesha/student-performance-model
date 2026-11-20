import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src1.features import make_feature_set
from src1.load_split import load_split
from src1.preprocessing import make_model_pipeline, prepare_new_data
from src1.split_data import PROJECT_ROOT, SPLIT_PATH


TABLE_DIR = PROJECT_ROOT / "reports/tables/model_comparison/experiments"
FIGURE_DIR = PROJECT_ROOT / "reports/figures/model_comparison/experiments"
MODEL_DIR = PROJECT_ROOT / "models"


def candidates():
    """Небольшая заранее заданная сетка для обоих сценариев."""
    result = [("Dummy median", DummyRegressor(strategy="median"), False, 0)]
    for engineered in (False, True):
        result.append(("Linear", LinearRegression(), engineered, 1))
        for alpha in (0.01, 0.1, 1.0):
            result.append((f"Lasso alpha={alpha}", Lasso(alpha=alpha, max_iter=20000), engineered, 2))
            result.append((f"Ridge alpha={alpha}", Ridge(alpha=alpha), engineered, 2))
        result.append(("KNN k=5", KNeighborsRegressor(n_neighbors=5), engineered, 3))
        for depth in (2, 4, 8, None):
            result.append((f"Tree depth={depth}", DecisionTreeRegressor(max_depth=depth, random_state=42), engineered, 4))
        result.append(("Random forest", RandomForestRegressor(n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=1), engineered, 5))
        for leaves, rate in ((7, 0.05), (15, 0.05), (15, 0.1)):
            result.append((f"Boosting leaves={leaves} rate={rate}", HistGradientBoostingRegressor(
                max_iter=200, learning_rate=rate, max_leaf_nodes=leaves,
                min_samples_leaf=10, early_stopping=False, random_state=42,
            ), engineered, 6))
    return result


def choose_candidate(results: pd.DataFrame) -> int:
    """При разнице MAE до 0.05 балла предпочитаем более простую модель."""
    threshold = results["cv_mae_mean"].min() + 0.05
    eligible = results[results["cv_mae_mean"] <= threshold]
    return int(eligible.sort_values(["complexity", "engineered", "cv_mae_mean", "candidate_id"]).iloc[0]["candidate_id"])


def regression_metrics(y_true, predictions) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, predictions)),
        "mse": float(mean_squared_error(y_true, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "r2": float(r2_score(y_true, predictions)),
    }


def save_plots(results: pd.DataFrame, coefficients: pd.DataFrame) -> None:
    for scenario in ("early", "late"):
        subset = results[(results["scenario"] == scenario) & (~results["engineered"])]
        subset = subset[subset["model"].str.startswith("Tree depth=")].copy()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(subset["model"], subset["cv_mae_mean"], yerr=subset["cv_mae_std"], marker="o", capsize=4)
        ax.set(title=f"Decision tree: CV MAE ({scenario})", xlabel="Maximum depth", ylabel="MAE (grade points)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"tree_depth_{scenario}.png", dpi=160)
        plt.close(fig)

        subset = coefficients[(coefficients["scenario"] == scenario) & (~coefficients["engineered"])]
        fig, ax = plt.subplots(figsize=(7, 4))
        for family in ("Lasso", "Ridge"):
            line = subset[subset["family"] == family].sort_values("alpha")
            ax.plot(line["alpha"], line["nonzero_coefficients"], marker="o", label=family)
        ax.set_xscale("log")
        ax.set(title=f"Regularization and coefficients ({scenario})", xlabel="alpha", ylabel="Nonzero coefficients")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"regularization_{scenario}.png", dpi=160)
        plt.close(fig)

        subset = results[(results["scenario"] == scenario) & (~results["engineered"])]
        fig, ax = plt.subplots(figsize=(7, 4))
        for family in ("Lasso", "Ridge"):
            line = subset[subset["model"].str.startswith(family + " alpha=")].copy()
            line["alpha"] = line["model"].str.split("=").str[-1].astype(float)
            line = line.sort_values("alpha")
            ax.errorbar(line["alpha"], line["cv_mae_mean"], yerr=line["cv_mae_std"], marker="o", capsize=3, label=family)
        ax.set_xscale("log")
        ax.set(title=f"Regularization and CV MAE ({scenario})", xlabel="alpha", ylabel="MAE (grade points)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / f"regularization_mae_{scenario}.png", dpi=160)
        plt.close(fig)

        pair = results[(results["scenario"] == scenario) & results["model"].isin(["Linear", "Random forest"])]
        pivot = pair.pivot(index="model", columns="engineered", values="cv_mae_mean")
        pivot.columns = ["Original features", "With engineered features"]
        ax = pivot.plot.bar(figsize=(7, 4), rot=0)
        ax.set(title=f"Feature comparison ({scenario})", xlabel="Model", ylabel="Mean CV MAE (grade points)")
        ax.figure.tight_layout()
        ax.figure.savefig(FIGURE_DIR / f"feature_comparison_{scenario}.png", dpi=160)
        plt.close(ax.figure)


def main() -> None:
    for directory in (TABLE_DIR, FIGURE_DIR, MODEL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    train, validation, _test = load_split()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"}
    configurations = candidates()
    results, fold_rows, coefficient_rows, coefficient_details, validations = [], [], [], [], []
    split_info = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    run_utc = datetime.now(timezone.utc).isoformat()

    for scenario, use_grades in (("early", False), ("late", True)):
        X_train, y_train = make_feature_set(train, use_grades=use_grades)
        X_validation, y_validation = make_feature_set(validation, use_grades=use_grades)
        scenario_results = []
        for index, (name, estimator, engineered, complexity) in enumerate(configurations):
            print(f"{scenario}: {index + 1}/{len(configurations)} {name}, engineered={engineered}", flush=True)
            pipeline = make_model_pipeline(clone(estimator), add_features=engineered)
            start = time.perf_counter()
            scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
            elapsed = time.perf_counter() - start
            row = {"scenario": scenario, "candidate_id": index, "model": name, "engineered": engineered, "run_utc": run_utc,
                   "complexity": complexity, "cv_mae_mean": float(-scores["test_mae"].mean()),
                   "cv_mae_std": float(scores["test_mae"].std(ddof=0)),
                   "cv_rmse_mean": float(-scores["test_rmse"].mean()), "cv_rmse_std": float(scores["test_rmse"].std(ddof=0)),
                   "cv_r2_mean": float(scores["test_r2"].mean()), "cv_r2_std": float(scores["test_r2"].std(ddof=0)),
                   "cv_seconds": elapsed, "estimator_parameters": json.dumps(estimator.get_params(), sort_keys=True)}
            scenario_results.append(row)
            results.append(row)
            for fold in range(5):
                fold_rows.append({"scenario": scenario, "candidate_id": index, "fold": fold + 1,
                                  "mae": float(-scores["test_mae"][fold]), "rmse": float(-scores["test_rmse"][fold]),
                                  "r2": float(scores["test_r2"][fold])})
            if isinstance(estimator, (Lasso, Ridge)):
                pipeline.fit(X_train, y_train)
                names = pipeline.named_steps["preprocess"].get_feature_names_out()
                coefs = pipeline.named_steps["model"].coef_
                coefficient_rows.append({"scenario": scenario, "engineered": engineered, "family": type(estimator).__name__,
                                         "alpha": estimator.alpha, "nonzero_coefficients": int(np.count_nonzero(np.abs(coefs) > 1e-8))})
                for feature, coef in zip(names, coefs):
                    coefficient_details.append({"scenario": scenario, "engineered": engineered, "family": type(estimator).__name__,
                                                "alpha": estimator.alpha, "feature": feature, "coefficient": float(coef)})

        winner_id = choose_candidate(pd.DataFrame(scenario_results))
        name, estimator, engineered, _ = configurations[winner_id]
        winner = make_model_pipeline(clone(estimator), add_features=engineered)
        winner.fit(X_train, y_train)
        predicted = winner.predict(X_validation)
        metrics = regression_metrics(y_validation, predicted)
        validations.append({"scenario": scenario, "candidate_id": winner_id, "model": name,
                            "engineered": engineered, **metrics})
        prepared_first = prepare_new_data(validation, winner, use_grades=use_grades)
        prepared_second = prepare_new_data(validation, winner, use_grades=use_grades)
        if not np.array_equal(prepared_first, prepared_second, equal_nan=True):
            raise ValueError("Повторное преобразование validation дало разные результаты")

        X_development = pd.concat([X_train, X_validation])
        y_development = pd.concat([y_train, y_validation])
        final_model = make_model_pipeline(clone(estimator), add_features=engineered)
        final_model.fit(X_development, y_development)
        model_path = MODEL_DIR / f"{scenario}_model.joblib"
        joblib.dump(final_model, model_path)
        metadata = {"scenario": scenario, "model": name, "engineered": engineered,
                    "parameters": estimator.get_params(), "train_rows": len(X_train), "validation_rows": len(X_validation),
                    "final_training_rows": len(X_development), "validation_metrics_before_refit": metrics,
                    "selection_rule": "minimum CV MAE with 0.05 tolerance for simpler models",
                    "source_sha256": split_info["source_sha256"], "random_state": 42,
                    "python_version": platform.python_version(), "numpy_version": np.__version__,
                    "sklearn_version": sklearn.__version__, "pandas_version": pd.__version__, "joblib_version": joblib.__version__,
                    "created_utc": run_utc}
        (MODEL_DIR / f"{scenario}_model.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reloaded = joblib.load(model_path)
        if not np.allclose(final_model.predict(X_validation), reloaded.predict(X_validation)):
            raise ValueError("Прогнозы сохранённой и исходной модели отличаются")
        print(f"Selected {scenario}: {name}, engineered={engineered}, validation MAE={metrics['mae']:.3f}", flush=True)

    results_df = pd.DataFrame(results)
    coefficients_df = pd.DataFrame(coefficient_rows)
    results_df.to_csv(TABLE_DIR / "cv_results.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(TABLE_DIR / "cv_folds.csv", index=False, encoding="utf-8-sig")
    coefficients_df.to_csv(TABLE_DIR / "coefficient_counts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coefficient_details).to_csv(TABLE_DIR / "coefficients.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(validations).to_csv(TABLE_DIR / "validation_metrics.csv", index=False, encoding="utf-8-sig")
    save_plots(results_df, coefficients_df)
    print("Experiment tables, figures and models saved.", flush=True)


if __name__ == "__main__":
    main()
