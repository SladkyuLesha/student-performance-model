import unittest

from scripts.measure_api import summarize


class MeasureApiTests(unittest.TestCase):
    def test_summary(self):
        result = summarize([1, 2, 3, 4, 5])
        self.assertEqual(result["requests"], 5)
        self.assertEqual(result["mean_ms"], 3)
        self.assertEqual(result["median_ms"], 3)
        self.assertAlmostEqual(result["p95_ms"], 4.8)

    def test_empty_samples_rejected(self):
        with self.assertRaises(ValueError):
            summarize([])
