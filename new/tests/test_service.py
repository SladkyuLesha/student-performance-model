import json
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src1.service import (activate, connect, get_batches, metrics, predict, process_batch,
                          reference_data, train_candidate, validate_frame, version_list)


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.conn = connect(Path(self.tmp.name) / 'test.sqlite3')
        self.data = reference_data().head(10).copy()

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_predict_does_not_require_target(self):
        features = self.data.drop(columns=['G3'])
        late, _ = predict(self.conn, features, 'late')
        early, _ = predict(self.conn, features.drop(columns=['G1', 'G2']), 'early')
        self.assertEqual(len(late), 10)
        self.assertEqual(len(early), 10)

    def test_validation_rejects_invalid_and_extra(self):
        features = self.data.drop(columns=['G3']).copy()
        features.loc[features.index[0], 'studytime'] = 99
        with self.assertRaises(ValueError):
            validate_frame(features, 'late')
        with self.assertRaises(ValueError):
            validate_frame(self.data.assign(person_name='private'), 'late')

    def test_mixed_batch_quarantines_and_logs_each_bad_row(self):
        data = self.data.head(4).copy()
        data.iloc[0, data.columns.get_loc('studytime')] = 99
        data.iloc[1, data.columns.get_loc('failures')] = 99
        process_batch(self.conn, data.to_csv(index=False).encode(), 'mixed.csv')
        quality = json.loads(get_batches(self.conn)[0]['quality_json'])
        self.assertEqual((quality['total_rows'], quality['rows'], quality['rejected']), (4, 2, 2))
        self.assertEqual(quality['valid_fraction'], 0.5)
        log = self.conn.execute('SELECT row_number,reasons_json FROM rejected_rows ORDER BY row_number').fetchall()
        self.assertEqual([row['row_number'] for row in log], [2, 3])
        self.assertIn('studytime', log[0]['reasons_json'])
        self.assertIn('failures', log[1]['reasons_json'])

    def test_unbounded_absences_allowed(self):
        data = self.data.copy()
        data.loc[data.index[0], 'absences'] = 200
        self.assertEqual(len(validate_frame(data, 'late')), len(data))

    def test_automatic_candidate_does_not_activate(self):
        for path in sorted(Path('data/incoming').glob('batch_*.csv')):
            process_batch(self.conn, path.read_bytes(), path.name)
        events = self.conn.execute("SELECT * FROM events WHERE action='automatic_candidate'").fetchall()
        self.assertTrue(events)
        for scenario in ('early', 'late'):
            active = [v for v in version_list(self.conn, scenario) if v['active']]
            self.assertEqual(active[0]['version'], 'original')

    def test_candidate_failure_keeps_processed_batch(self):
        with patch('src1.service.PSI_LIMIT', -1), patch(
            'src1.service.train_candidate', side_effect=RuntimeError('private details')
        ):
            process_batch(self.conn, self.data.to_csv(index=False).encode(), 'data.csv')
        self.assertEqual(len(get_batches(self.conn)), 2)
        events = self.conn.execute(
            "SELECT details FROM events WHERE action='automatic_candidate_failed'"
        ).fetchall()
        self.assertEqual(len(events), 2)
        self.assertTrue(all('RuntimeError' in row['details'] for row in events))
        self.assertTrue(all('private details' not in row['details'] for row in events))

    def test_batch_is_idempotent_and_atomic(self):
        content = self.data.to_csv(index=False).encode('utf-8')
        process_batch(self.conn, content, 'first.csv')
        with self.assertRaises(ValueError):
            process_batch(self.conn, content, 'renamed.csv')
        self.assertEqual(len(get_batches(self.conn)), 2)
        broken = self.data.assign(studytime=99).to_csv(index=False).encode('utf-8')
        with self.assertRaises(ValueError):
            process_batch(self.conn, broken, 'bad.csv')
        self.assertEqual(len(get_batches(self.conn)), 2)

    def test_unlabelled_batch_has_no_quality_metric(self):
        process_batch(self.conn, self.data.drop(columns=['G3']).to_csv(index=False).encode(), 'no_target.csv')
        self.assertTrue(all(json.loads(row['metrics_json'])['mae'] is None for row in get_batches(self.conn)))

    def test_candidate_requires_separate_labeled_batches(self):
        with self.assertRaises(ValueError):
            train_candidate(self.conn, 'late', Path(self.tmp.name))
        self.assertEqual(version_list(self.conn, 'late')[0]['version'], 'original')
        activate(self.conn, 'late', 'original')

    def test_candidate_and_rollback_on_existing_batches(self):
        for path in sorted(Path('data/incoming').glob('batch_*.csv')):
            process_batch(self.conn, path.read_bytes(), path.name)
        version, info = train_candidate(self.conn, 'late', Path(self.tmp.name) / 'models')
        self.assertEqual(info['training_batches'], [1, 2, 3])
        self.assertEqual(info['comparison_batch'], 4)
        if info['eligible']:
            activate(self.conn, 'late', version)
            self.assertEqual(next(x for x in version_list(self.conn, 'late') if x['active'])['version'], version)
            activate(self.conn, 'late', 'original', rollback=True)
            self.assertEqual(next(x for x in version_list(self.conn, 'late') if x['active'])['version'], 'original')
        else:
            with self.assertRaises(ValueError):
                activate(self.conn, 'late', version)

    def test_original_test_profiles_are_excluded_from_training(self):
        from src1.load_split import load_split
        _, _, test = load_split()
        process_batch(self.conn, self.data.to_csv(index=False).encode(), 'train.csv')
        process_batch(self.conn, test.to_csv(index=False).encode(), 'test.csv')
        with self.assertRaisesRegex(ValueError, 'test'):
            train_candidate(self.conn, 'late', Path(self.tmp.name) / 'models')

    def test_metrics_ignore_missing_labels(self):
        result = metrics(pd.Series([10., float('nan')]), [12., 100.])
        self.assertEqual(result['n'], 1)
        self.assertEqual(result['mae'], 2)
        self.assertIsNone(result['r2'])
