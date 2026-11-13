import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src1.preprocessing import FeatureBuilder, make_model_pipeline, prepare_csv_for_prediction, prepare_new_data


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            "school": ["GP", "MS", "GP", "MS"],
            "failures": [0, 1, 0, 2],
            "studytime": [1, 3, 2, 4],
            "Medu": [1, 2, 3, 4],
            "Fedu": [2, 2, 4, 3],
            "G1": [8, 10, 12, 14],
            "G2": [9, 11, 13, 15],
        })
        self.y = pd.Series([9, 11, 13, 15])

    def test_new_features_and_input_unchanged(self):
        original = self.X.copy(deep=True)
        result = FeatureBuilder().fit_transform(self.X)
        self.assertEqual(result["has_failures"].tolist(), [0, 1, 0, 1])
        self.assertEqual(result["studytime_high"].tolist(), [0, 1, 0, 1])
        self.assertEqual(result["parent_education_mean"].tolist(), [1.5, 2, 3.5, 3.5])
        pd.testing.assert_frame_equal(self.X, original)

    def test_unseen_category_and_missing_value_transform(self):
        pipeline = make_model_pipeline(Ridge(), add_features=True).fit(self.X, self.y)
        incoming = self.X.iloc[:1].copy()
        incoming.loc[incoming.index[0], "school"] = "new category"
        incoming.loc[incoming.index[0], "Medu"] = np.nan
        transformed = prepare_new_data(incoming, pipeline, use_grades=True)
        self.assertEqual(transformed.shape[0], 1)
        self.assertTrue(np.isfinite(transformed).all())

    def test_new_data_does_not_require_target_and_is_deterministic(self):
        pipeline = make_model_pipeline(Ridge(), add_features=True).fit(self.X, self.y)
        incoming = self.X.assign(G3=self.y)
        first = prepare_new_data(incoming, pipeline, use_grades=True)
        second = prepare_new_data(incoming.drop(columns="G3"), pipeline, use_grades=True)
        np.testing.assert_array_equal(first, second)

    def test_missing_column_is_rejected(self):
        pipeline = make_model_pipeline(Ridge(), add_features=False).fit(self.X, self.y)
        with self.assertRaisesRegex(ValueError, "отличаются"):
            prepare_new_data(self.X.drop(columns="school"), pipeline, use_grades=True)

    def test_csv_uses_the_same_saved_transforms(self):
        pipeline = make_model_pipeline(Ridge(), add_features=True).fit(self.X, self.y)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incoming.csv"
            self.X.to_csv(path, index=False)
            actual = prepare_csv_for_prediction(path, pipeline, use_grades=True)
            expected = prepare_new_data(self.X, pipeline, use_grades=True)
            np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
