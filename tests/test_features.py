import unittest

import pandas as pd
from pandas.testing import assert_frame_equal

from src1.features import make_feature_set


class MakeFeatureSetTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "school": ["GP", "MS"],
            "absences": [2, 4],
            "G1": [10, 12],
            "G2": [11, 13],
            "G3": [12, 14],
        })

    def test_with_grades_keeps_g1_and_g2_but_not_g3(self):
        features, target = make_feature_set(self.data, use_grades=True)

        self.assertEqual(features.columns.tolist(), ["school", "absences", "G1", "G2"])
        self.assertEqual(features["school"].tolist(), ["GP", "MS"])
        self.assertEqual(target.tolist(), [12, 14])

    def test_without_grades_removes_all_grades_from_features(self):
        features, target = make_feature_set(self.data, use_grades=False)

        self.assertEqual(features.columns.tolist(), ["school", "absences"])
        self.assertEqual(target.tolist(), [12, 14])

    def test_does_not_change_input(self):
        original = self.data.copy(deep=True)
        features, target = make_feature_set(self.data, use_grades=True)
        features.loc[0, "school"] = "changed"
        target.loc[0] = 0

        assert_frame_equal(self.data, original)

    def test_rejects_missing_grade_columns(self):
        for column in ("G1", "G2", "G3"):
            with self.subTest(column=column):
                with self.assertRaisesRegex(ValueError, column):
                    make_feature_set(self.data.drop(columns=column), use_grades=False)


if __name__ == "__main__":
    unittest.main()
