"""Reproduce batch monitoring in an isolated directory and check processing time."""

import argparse
import hashlib
import json
import tempfile
import time
from contextlib import closing
from pathlib import Path

import pandas as pd

from src1.service import PROJECT_ROOT, connect, get_batches, process_batch


def fingerprint(rows):
    comparable = []
    for row in rows:
        comparable.append({key: row[key] for key in ('name', 'scenario', 'model_version', 'predictions_json', 'metrics_json', 'psi_json')})
    return hashlib.sha256(json.dumps(comparable, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'reports/tables/monitoring/verification.csv')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.exit(1, 'Таблица уже существует; используйте --overwrite для её замены.\n')
    batches = sorted((PROJECT_ROOT / 'data/incoming').glob('batch_*.csv'))
    if len(batches) != 4:
        parser.exit(1, 'Ожидались четыре учебные порции.\n')
    with tempfile.TemporaryDirectory() as tmp:
        results = []
        for run in range(2):
            with closing(connect(Path(tmp) / f'run{run}' / 'demo.sqlite3')) as conn:
                for path in batches:
                    process_batch(conn, path.read_bytes(), path.name)
                results.append(fingerprint(get_batches(conn)))
        source = pd.read_csv(batches[0])
        repeated = pd.concat([source] * 25, ignore_index=True)
        with closing(connect(Path(tmp) / 'speed' / 'speed.sqlite3')) as conn:
            start = time.perf_counter()
            process_batch(conn, repeated.to_csv(index=False).encode('utf-8'), 'repeated_1000.csv')
            seconds = time.perf_counter() - start
        records = [
            {'check': 'repeatability', 'result': results[0] == results[1], 'value': results[0], 'units': 'sha256', 'note': 'two isolated runs; predictions metrics and PSI'},
            {'check': '1000_rows', 'result': seconds <= 30, 'value': seconds, 'units': 'seconds', 'note': '40 source rows repeated 25 times; synthetic load only'},
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(args.output, index=False)
        print(pd.DataFrame(records).to_string(index=False))
        print(f'Результаты: {args.output}')


if __name__ == '__main__':
    main()
