"""Оценка моделей и качества данных без обучения на тестовых строках."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.metrics import mean_absolute_error

from src1.features import make_feature_set
from src1.preprocessing import make_model_pipeline
from src1.run_experiments import candidates, regression_metrics
from src1.split_data import PROJECT_ROOT


def development_models(train, validation):
    """Загружает выбранные модели, обучает baseline и бустинг по прежнему CV."""
    development = pd.concat([train, validation])
    cv = pd.read_csv(PROJECT_ROOT / "reports/tables/model_comparison/experiments/cv_results.csv")
    boost_rows = cv[(cv["scenario"] == "late") & cv["model"].str.startswith("Boosting")]
    best = boost_rows.sort_values(["cv_mae_mean", "candidate_id"]).iloc[0]
    config = candidates()[int(best["candidate_id"])]
    if config[0] != best["model"] or bool(config[2]) != bool(best["engineered"]):
        raise ValueError("Конфигурация бустинга не совпадает с сохранёнными результатами CV")
    models = {}
    for scenario, use_grades in (("early", False), ("late", True)):
        X, y = make_feature_set(development, use_grades=use_grades)
        model = joblib.load(PROJECT_ROOT / f"models/{scenario}_model.joblib")
        baseline = DummyRegressor(strategy="median").fit(X, y)
        models[scenario] = {"selected": model, "baseline": baseline}
        if use_grades:
            boosting = make_model_pipeline(clone(config[1]), add_features=config[2])
            models[scenario]["boosting"] = boosting.fit(X, y)
    return models, best


def bootstrap_mae(y, prediction, *, other=None, n_resamples=1000, random_state=42):
    """Интервал MAE или парной разности MAE(prediction) - MAE(other)."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    if len(y) == 0 or len(y) != len(prediction) or n_resamples < 1:
        raise ValueError("Нужны непустые массивы одинаковой длины и хотя бы один повтор")
    errors = np.abs(y - prediction)
    if other is not None:
        other = np.asarray(other)
        if len(other) != len(y):
            raise ValueError("Размер второго прогноза отличается")
        errors = errors - np.abs(y - other)
    rng = np.random.default_rng(random_state)
    samples = rng.integers(0, len(y), size=(n_resamples, len(y)))
    values = errors[samples].mean(axis=1)
    return tuple(np.quantile(values, [0.025, 0.975]))


def evaluate_models(models, data):
    rows, predictions = [], {}
    for scenario, entries in models.items():
        X, y = make_feature_set(data, use_grades=scenario == "late")
        for name, model in entries.items():
            pred = model.predict(X)
            low, high = bootstrap_mae(y, pred)
            rows.append({"scenario": scenario, "model": name, **regression_metrics(y, pred),
                         "mae_ci_low": low, "mae_ci_high": high, "n": len(y)})
            predictions[(scenario, name)] = pred
    return pd.DataFrame(rows), predictions


def data_quality(data, reference):
    """Возвращает диагностические проверки, не изменяя данные."""
    required = list(reference.columns)
    missing = sorted(set(required) - set(data.columns))
    rows = [{"check": "required_columns", "issues": len(missing), "details": ", ".join(missing), "blocking": bool(missing)}]
    def add(name, mask, blocking=True):
        count = int(mask.sum())
        rows.append({"check": name, "issues": count, "details": "", "blocking": blocking and count > 0})

    rows.append({"check": "empty_batch", "issues": int(data.empty), "details": "", "blocking": data.empty})
    for name in ("age", "studytime", "failures"):
        if name in data:
            add(f"missing_{name}", data[name].isna())
    for name, low, high in (("G1", 0, 20), ("G2", 0, 20), ("G3", 0, 20),
                            ("studytime", 1, 4), ("failures", 0, 3), ("absences", 0, np.inf)):
        if name in data:
            numeric = pd.to_numeric(data[name], errors="coerce")
            invalid = data[name].notna() & (numeric.isna() | (numeric < low) | (numeric > high) | (numeric % 1 != 0))
            add(f"range_{name}", invalid)
    for name in reference.select_dtypes(exclude="number").columns:
        if name in data:
            add(f"unknown_{name}", data[name].notna() & ~data[name].isin(reference[name].dropna().unique()), blocking=False)
    for name in reference.select_dtypes(include="number").columns:
        if name in data:
            add(f"numeric_{name}", data[name].notna() & pd.to_numeric(data[name], errors="coerce").isna())
    if "G3" in data:
        add("missing_G3", data["G3"].isna())
    add("duplicate_rows", data.duplicated(keep=False), blocking=False)
    return pd.DataFrame(rows)


def psi_numeric(reference, current, *, bins=5):
    """PSI по квантилям обучающей выборки; пропуски учитываются отдельно."""
    ref = pd.to_numeric(reference, errors="coerce")
    cur = pd.to_numeric(current, errors="coerce")
    if ref.notna().sum() == 0 or len(cur) == 0:
        return np.nan
    unique = np.sort(ref.dropna().unique())
    if len(unique) <= 10:
        cuts = (unique[:-1] + unique[1:]) / 2
    else:
        cuts = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))[1:-1]
    edges = np.r_[-np.inf, cuts, np.inf]
    ref_counts = np.r_[np.histogram(ref.dropna(), bins=edges)[0], ref.isna().sum()].astype(float)
    cur_counts = np.r_[np.histogram(cur.dropna(), bins=edges)[0], cur.isna().sum()].astype(float)
    ref_dist = (ref_counts + 0.5) / (len(ref) + 0.5 * len(ref_counts))
    cur_dist = (cur_counts + 0.5) / (len(cur) + 0.5 * len(cur_counts))
    return float(np.sum((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)))


def group_metrics(data, predictions, *, groups=("sex", "school")):
    rows = []
    for group in groups:
        for value, positions in data.reset_index(drop=True).groupby(group, dropna=False).indices.items():
            y = data["G3"].to_numpy()[positions]
            pred = np.asarray(predictions)[positions]
            rows.append({"group": group, "value": value, "n": len(y), "mae": mean_absolute_error(y, pred)})
    return pd.DataFrame(rows)


def importance(model, X, y):
    result = permutation_importance(model, X, y, scoring="neg_mean_absolute_error", n_repeats=10, random_state=42, n_jobs=1)
    return pd.DataFrame({"feature": X.columns, "importance": result.importances_mean,
                         "std": result.importances_std}).sort_values("importance", ascending=False)


def local_sensitivity(model, X, reference, index, features):
    row = X.iloc[[index]].copy()
    original = float(model.predict(row)[0])
    results = []
    for feature in features:
        replacement = reference[feature].median() if pd.api.types.is_numeric_dtype(reference[feature]) else reference[feature].mode().iloc[0]
        modified = row.copy()
        modified.loc[:, feature] = replacement
        results.append({"feature": feature, "original": row.iloc[0][feature], "replacement": replacement,
                        "prediction": original, "changed_prediction": float(model.predict(modified)[0])})
    return pd.DataFrame(results)


def robustness(model, X, y, *, feature, reference=None, random_state=42):
    """Оценивает отдельные искусственные изменения на копиях входных данных."""
    rng = np.random.default_rng(random_state)
    noisy, missing = X.copy(), X.copy()
    reference_values = X[feature] if reference is None else reference[feature]
    scale = float(pd.to_numeric(reference_values, errors="raise").std()) * 0.1
    if not np.isfinite(scale):
        raise ValueError("Недостаточно значений для расчёта шума")
    perturbed = pd.to_numeric(noisy[feature], errors="raise") + rng.normal(0, scale, len(X))
    if feature in ("G1", "G2", "failures", "absences"):
        lower, upper = {"G1": (0, 20), "G2": (0, 20), "failures": (0, 3), "absences": (0, np.inf)}[feature]
        perturbed = perturbed.round().clip(lower=lower, upper=upper)
    noisy[feature] = perturbed
    positions = rng.choice(len(X), size=max(1, int(len(X) * 0.1)), replace=False)
    missing.iloc[positions, missing.columns.get_loc(feature)] = np.nan
    results = [
        {"feature": feature, "condition": name, "changed_rows": int(count),
         "mae": mean_absolute_error(y, model.predict(frame))}
        for name, frame, count in (
            ("original", X, 0),
            ("noise_10pct_std", noisy, (~noisy[feature].eq(X[feature]) & ~(noisy[feature].isna() & X[feature].isna())).sum()),
            ("missing_10pct_rows", missing, (X[feature].notna() & missing[feature].isna()).sum()),
        )
    ]
    if feature == "failures":
        discrete = X.copy()
        values = pd.to_numeric(discrete[feature], errors="raise")
        available = np.flatnonzero(values.notna().to_numpy())
        count = 0
        if len(available):
            count = max(1, int(len(available) * 0.1))
            selected = rng.choice(available, size=count, replace=False)
            original = values.iloc[selected].to_numpy()
            direction = rng.choice([-1, 1], size=count)
            direction = np.where(original <= 0, 1, direction)
            direction = np.where(original >= 3, -1, direction)
            discrete.iloc[selected, discrete.columns.get_loc(feature)] = original + direction
        results.append({"feature": feature, "condition": "discrete_step_10pct_rows",
                        "changed_rows": count,
                        "mae": mean_absolute_error(y, model.predict(discrete))})
    return pd.DataFrame(results)


def zero_grade_errors(y, prediction):
    """Сравнивает ошибки при нулевых и ненулевых оценках без идентификаторов."""
    actual = np.asarray(y)
    pred = np.asarray(prediction)
    if len(actual) != len(pred):
        raise ValueError("Размеры фактических значений и прогнозов отличаются")
    rows = []
    for label, mask in (("G3 = 0", actual == 0), ("G3 > 0", actual > 0)):
        errors = np.abs(actual[mask] - pred[mask])
        rows.append({"group": label, "n": int(mask.sum()),
                     "mae": float(errors.mean()) if len(errors) else np.nan,
                     "mean_prediction": float(pred[mask].mean()) if len(errors) else np.nan})
    return pd.DataFrame(rows)


def save_table(frame, category, filename):
    path = PROJECT_ROOT / "reports/tables" / category / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_figure(fig, category, filename):
    path = PROJECT_ROOT / "reports/figures" / category / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def diagnostic_plots(y, prediction, scenario):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y, prediction, alpha=0.7)
    axes[0].plot([0, 20], [0, 20], "--", color="black")
    axes[0].set(xlabel="Фактическая G3", ylabel="Прогноз G3", title="Факт и прогноз")
    axes[1].hist(np.asarray(y) - prediction, bins=12, edgecolor="white")
    axes[1].set(xlabel="Остаток: факт − прогноз", ylabel="Количество", title="Распределение остатков")
    save_figure(fig, "predictions", f"{scenario}_diagnostics.png")
    return fig


def plot_importance(table, scenario):
    subset = table.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(subset["feature"], subset["importance"], xerr=subset["std"])
    ax.set(xlabel="Рост MAE после перестановки", title=f"Важность признаков: {scenario}")
    save_figure(fig, "feature_importance", f"{scenario}_importance.png")
    return fig


def partial_dependence(model, X, features, scenario):
    X = X.copy()
    for feature in features:
        X[feature] = X[feature].astype(float)
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 4), squeeze=False)
    PartialDependenceDisplay.from_estimator(model, X, features, ax=axes[0])
    save_figure(fig, "feature_importance", f"{scenario}_partial_dependence.png")
    return fig
