import unittest
import contextlib
import io
import json

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

from src1.evaluation import bootstrap_mae, data_quality, group_metrics, psi_numeric, robustness, zero_grade_errors
from src1.features import make_feature_set
from src1.split_data import PROJECT_ROOT


class EvaluationTests(unittest.TestCase):
    def test_bootstrap_is_reproducible_and_paired(self):
        first = bootstrap_mae([1, 2, 3], [2, 2, 4], other=[1, 2, 3])
        self.assertEqual(first, bootstrap_mae([1, 2, 3], [2, 2, 4], other=[1, 2, 3]))
        self.assertGreaterEqual(first[0], 0)

    def test_bootstrap_rejects_length_mismatch(self):
        with self.assertRaises(ValueError):
            bootstrap_mae([1], [1, 2])

    def test_psi_identical_and_shifted(self):
        reference = pd.Series([0, 1, 2, 3, np.nan] * 20)
        self.assertAlmostEqual(psi_numeric(reference, reference), 0)
        self.assertGreater(psi_numeric(reference, pd.Series([100] * 100)), 0)

    def test_psi_distinguishes_zero_and_positive_discrete_values(self):
        reference = pd.Series([0] * 90 + [1] * 10)
        changed = pd.Series([0] * 10 + [1] * 90)
        self.assertGreater(psi_numeric(reference, changed), 0.25)

    def test_zero_grade_errors_reports_separate_groups(self):
        result = zero_grade_errors([0, 0, 10], [3, 5, 9]).set_index("group")
        self.assertEqual(result.loc["G3 = 0", "n"], 2)
        self.assertEqual(result.loc["G3 = 0", "mae"], 4)
        self.assertEqual(result.loc["G3 > 0", "mae"], 1)

    def test_robustness_reports_effective_changes(self):
        X = pd.DataFrame({"G2": [0., 5., 10., 15., 20.] * 4})
        model = DummyRegressor().fit(X, range(len(X)))
        result = robustness(model, X, range(len(X)), feature="G2", reference=X).set_index("condition")
        self.assertEqual(result.loc["original", "changed_rows"], 0)
        self.assertEqual(result.loc["missing_10pct_rows", "changed_rows"], 2)

    def test_failures_discrete_noise_changes_valid_values_reproducibly(self):
        X = pd.DataFrame({"failures": [0, 1, 2, 3] * 10})
        original = X.copy(deep=True)

        class RecordingModel:
            def __init__(self):
                self.inputs = []

            def predict(self, data):
                self.inputs.append(data.copy(deep=True))
                return np.zeros(len(data))

        first = RecordingModel()
        result = robustness(first, X, np.zeros(len(X)), feature="failures", reference=X)
        second = RecordingModel()
        repeated = robustness(second, X, np.zeros(len(X)), feature="failures", reference=X)
        changed = first.inputs[-1]["failures"] - X["failures"]

        self.assertEqual(result.set_index("condition").loc["discrete_step_10pct_rows", "changed_rows"], 4)
        self.assertEqual((changed != 0).sum(), 4)
        self.assertTrue(changed[changed != 0].abs().eq(1).all())
        self.assertTrue(first.inputs[-1]["failures"].between(0, 3).all())
        pd.testing.assert_frame_equal(X, original)
        pd.testing.assert_frame_equal(result, repeated)
        pd.testing.assert_frame_equal(first.inputs[-1], second.inputs[-1])

    def test_quality_detects_invalid_values_without_mutation(self):
        reference = pd.DataFrame({"age": [17, 18], "studytime": [1, 2], "failures": [0, 1],
                                  "absences": [0, 1], "sex": ["F", "M"], "G3": [10, 12]})
        damaged = reference.copy()
        damaged.loc[0, "studytime"] = 5
        original = damaged.copy()
        result = data_quality(damaged, reference).set_index("check")
        self.assertTrue(result.loc["range_studytime", "blocking"])
        pd.testing.assert_frame_equal(damaged, original)

    def test_group_metrics_and_robustness(self):
        data = pd.DataFrame({"sex": ["F", "M"], "school": ["GP", "GP"], "G3": [10, 12]})
        scores = group_metrics(data, [9, 10])
        self.assertEqual(len(scores), 3)
        X = pd.DataFrame({"age": [16., 17., 18.]})
        model = DummyRegressor().fit(X, [1, 2, 3])
        self.assertEqual(len(robustness(model, X, [1, 2, 3], feature="age")), 3)

    def test_new_notebook_cells_on_synthetic_data(self):
        data = pd.DataFrame({"failures": [0, 1] * 10, "absences": list(range(20)),
                             "G1": list(range(20)), "G2": list(range(20)), "G3": list(range(20))})
        models, predictions, saved = {}, {}, []
        for scenario in ("early", "late"):
            X, y = make_feature_set(data, use_grades=scenario == "late")
            model = DummyRegressor().fit(X, y)
            models[scenario] = {"selected": model}
            predictions[(scenario, "selected")] = model.predict(X)
        scope = {"pd": pd, "test": data, "development": data, "models": models,
                 "predictions": predictions, "make_feature_set": make_feature_set,
                 "robustness": robustness, "zero_grade_errors": zero_grade_errors,
                 "display": lambda *args: None, "save_table": lambda *args: saved.append(args[2])}
        notebook = json.loads((PROJECT_ROOT / "notebooks/03_model_evaluation.ipynb").read_text(encoding="utf-8"))
        with contextlib.redirect_stdout(io.StringIO()):
            for index in (17, 19):
                exec(compile("".join(notebook["cells"][index]["source"]), f"cell {index}", "exec"), scope)
        self.assertEqual(len(saved), 4)


if __name__ == "__main__":
    unittest.main()
