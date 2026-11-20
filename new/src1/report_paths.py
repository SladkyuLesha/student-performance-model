"""Единая карта расположения результатов по содержанию."""
from pathlib import Path

TABLE_CATEGORIES = {
    "g3_correlations.csv": "correlation",
    "quality_checks_initial.csv": "data_quality", "category_validation.csv": "data_quality",
    "value_validation.csv": "data_quality", "missing_by_column.csv": "data_quality",
    "quality_requirements.csv": "data_quality",
    "feature_dictionary.csv": "data_dictionary", "feature_structure.csv": "data_dictionary",
    "feature_availability.csv": "data_dictionary",
    "g3_frequency.csv": "distributions", "g3_statistics.csv": "distributions",
    "numeric_descriptives.csv": "distributions", "iqr_outliers.csv": "distributions",
    "hypotheses_results.csv": "hypotheses", "absences_grouped_g3.csv": "relationships",
    "conclusion_2_2.txt": "analysis", "kpi_plan.csv": "architecture",
    "system_requirements.csv": "architecture", "pipeline_steps.csv": "architecture",
    "new_feature_plan.csv": "feature_engineering", "engineered_features_example.csv": "feature_engineering",
}
FIGURE_CATEGORIES = {
    "correlation_heatmap.png": "correlation",
    "absences_outliers.png": "distributions", "categorical_distributions.png": "distributions",
    "numeric_distributions.png": "distributions", "g3_distribution.png": "distributions",
    "g3_by_absences_groups.png": "relationships", "g3_by_groups.png": "relationships",
    "grade_and_absences_relation.png": "relationships",
    "c4_containers.png": "architecture", "c4_context.png": "architecture",
    "uml_activity.png": "architecture", "uml_sequence.png": "architecture", "uml_use_cases.png": "architecture",
}


class ReportDirectory:
    """Разрешает путь из имени сохранённого результата и создаёт его каталог."""

    def __init__(self, root, kind):
        if kind not in ("tables", "figures"):
            raise ValueError("Неизвестный тип результатов")
        self.root = Path(root)
        self.kind = kind

    def __truediv__(self, filename):
        categories = TABLE_CATEGORIES if self.kind == "tables" else FIGURE_CATEGORIES
        if filename not in categories:
            raise ValueError(f"Категория результата не определена: {filename}")
        path = self.root / "reports" / self.kind / categories[filename] / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def mkdir(self, **kwargs):
        (self.root / "reports" / self.kind).mkdir(**kwargs)
