import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from src1.api import app
from src1.service import reference_data


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patcher = patch('src1.api.DB_PATH', Path(self.tmp.name) / 'test.sqlite3')
        self.patcher.start()
        self.client = TestClient(app)
        self.student = reference_data().drop(columns=['G3']).iloc[0].to_dict()

    def tearDown(self):
        self.client.close()
        self.patcher.stop()
        self.tmp.cleanup()

    def test_health(self):
        self.assertEqual(self.client.get('/health').json()['status'], 'ok')

    def test_valid_prediction(self):
        response = self.client.post('/predict', json={'scenario': 'late', 'student': self.student})
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json()['prediction'], float)

    def test_invalid_value_is_rejected(self):
        self.student['studytime'] = 99
        response = self.client.post('/predict', json={'scenario': 'late', 'student': self.student})
        self.assertEqual(response.status_code, 422)

    def test_metrics_initially_empty(self):
        self.assertEqual(self.client.get('/metrics').json(), [])
