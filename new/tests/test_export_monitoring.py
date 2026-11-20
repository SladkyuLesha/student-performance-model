import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import pandas as pd

from scripts.export_monitoring import export_history
from src1.service import connect, get_batches, process_batch


class MonitoringExportTests(unittest.TestCase):
    def test_export_contains_aggregates_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with closing(connect(root / 'data' / 'test.sqlite3')) as conn:
                source = Path('data/incoming/batch_01.csv')
                process_batch(conn, source.read_bytes(), source.name)
                paths = export_history(get_batches(conn), root / 'figures', root / 'tables')
                self.assertEqual(len(paths), 4)
                self.assertTrue(all(path.is_file() for path in paths))
                frame = pd.read_csv(root / 'tables/panel_kpi.csv')
                self.assertEqual(len(frame), 2)
                self.assertNotIn('sex', frame.columns)
                with self.assertRaises(FileExistsError):
                    export_history(get_batches(conn), root / 'figures', root / 'tables')
