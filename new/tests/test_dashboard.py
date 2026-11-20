import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src1.service import connect, process_batch


class DashboardTests(unittest.TestCase):
    def test_pages_render_with_a_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / 'dashboard.sqlite3'
            with closing(connect(database)) as conn:
                path = Path('data/incoming/batch_01.csv')
                process_batch(conn, path.read_bytes(), path.name)
            with patch('src1.service.connect', side_effect=lambda: connect(database)):
                app = AppTest.from_file(str(Path('new/app/dashboard.py').resolve()), default_timeout=30).run()
                self.assertFalse(app.exception)
                for page in ('Прогноз студента', 'Порции данных', 'Модели', 'Обзор'):
                    app.radio[0].set_value(page).run()
                    self.assertFalse(app.exception, f'{page}: {app.exception}')

    def test_history_selector_uses_only_earlier_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / 'history.sqlite3'
            with closing(connect(database)) as conn:
                for path in sorted(Path('data/incoming').glob('batch_*.csv')):
                    process_batch(conn, path.read_bytes(), path.name)
            with patch('src1.service.connect', side_effect=lambda: connect(database)):
                app = AppTest.from_file(str(Path('new/app/dashboard.py').resolve()), default_timeout=30).run()
                app.selectbox[0].select('1 · batch_01.csv').run()
                self.assertFalse(app.exception)
                self.assertEqual(app.metric[0].value, '1')
                self.assertEqual(len(app.dataframe[0].value), 2)
