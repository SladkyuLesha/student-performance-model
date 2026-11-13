import unittest

import pandas as pd

from src1.run_experiments import choose_candidate, regression_metrics


class ExperimentTests(unittest.TestCase):
    def test_chooses_simpler_candidate_within_tolerance(self):
        results = pd.DataFrame([
            {"candidate_id": 1, "cv_mae_mean": 1.00, "complexity": 5, "engineered": False},
            {"candidate_id": 2, "cv_mae_mean": 1.04, "complexity": 1, "engineered": False},
            {"candidate_id": 3, "cv_mae_mean": 1.06, "complexity": 0, "engineered": False},
        ])
        self.assertEqual(choose_candidate(results), 2)

    def test_metrics(self):
        metrics = regression_metrics([1, 3], [2, 3])
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["mse"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], 0.5 ** 0.5)
        self.assertAlmostEqual(metrics["r2"], 0.5)


if __name__ == "__main__":
    unittest.main()
