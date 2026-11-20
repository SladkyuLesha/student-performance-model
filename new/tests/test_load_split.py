import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src1.load_split import load_split


class LoadSplitTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.data_path = root / "data.csv"
        self.split_path = root / "split.json"
        self.data_path.write_text("feature,G3\na,10\nb,12\nc,14\nd,16\n", encoding="utf-8")
        self.split = {
            "source_sha256": hashlib.sha256(self.data_path.read_bytes()).hexdigest(),
            "row_count": 4,
            "train_positions": [0, 2],
            "validation_positions": [1],
            "test_positions": [3],
        }
        self.write_split()

    def write_split(self):
        self.split_path.write_text(json.dumps(self.split), encoding="utf-8")

    def test_returns_expected_rows_without_transforming_columns(self):
        train, validation, test = load_split(self.data_path, self.split_path)
        self.assertEqual(train["feature"].tolist(), ["a", "c"])
        self.assertEqual(validation["feature"].tolist(), ["b"])
        self.assertEqual(test["feature"].tolist(), ["d"])
        self.assertEqual(train.columns.tolist(), ["feature", "G3"])

    def test_rejects_modified_csv(self):
        self.data_path.write_text("feature,G3\na,11\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Контрольная сумма"):
            load_split(self.data_path, self.split_path)

    def test_rejects_overlapping_positions(self):
        self.split["test_positions"] = [1]
        self.write_split()
        with self.assertRaisesRegex(ValueError, "Позиции строк"):
            load_split(self.data_path, self.split_path)

    def test_rejects_non_integer_positions(self):
        self.split["test_positions"] = [True]
        self.write_split()
        with self.assertRaisesRegex(ValueError, "целыми числами"):
            load_split(self.data_path, self.split_path)


if __name__ == "__main__":
    unittest.main()
