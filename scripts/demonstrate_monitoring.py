"""Exercise candidate creation, activation, rollback and row quarantine in isolation."""

import argparse
import json
import tempfile
from contextlib import closing
from pathlib import Path

from src1.service import PROJECT_ROOT, activate, connect, model_info, process_batch, reference_data, version_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'reports/tables/monitoring/lifecycle_demo.json')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        parser.exit(1, 'Файл существует; для повторного сохранения используйте --overwrite.\n')
    batches = sorted((PROJECT_ROOT / 'data/incoming').glob('batch_*.csv'))
    if len(batches) != 4:
        parser.exit(1, 'Ожидались четыре учебные порции.\n')
    with tempfile.TemporaryDirectory() as tmp:
        with closing(connect(Path(tmp) / 'lifecycle' / 'demo.sqlite3')) as conn:
            for path in batches:
                process_batch(conn, path.read_bytes(), path.name)
            events = [dict(row) for row in conn.execute("SELECT action, details FROM events WHERE action LIKE 'automatic_candidate%' ORDER BY id")]
            candidates = [row for row in version_list(conn, 'late') if row['version'] != 'original' and json.loads(row['metadata_json']).get('eligible')]
            if not candidates:
                raise RuntimeError('Подходящего позднего кандидата для демонстрации нет')
            candidate = candidates[0]
            details = json.loads(candidate['metadata_json'])
            activate(conn, 'late', candidate['version'])
            activated = model_info(conn, 'late')['version'] == candidate['version']
            activate(conn, 'late', 'original', rollback=True)
            rollback = model_info(conn, 'late')['version'] == 'original'
        with closing(connect(Path(tmp) / 'quality' / 'demo.sqlite3')) as conn:
            damaged = reference_data().head(3).copy()
            damaged.iloc[0, damaged.columns.get_loc('studytime')] = 99
            damaged.iloc[1, damaged.columns.get_loc('failures')] = 99
            process_batch(conn, damaged.to_csv(index=False).encode('utf-8'), 'quality_demo.csv')
            rejected = [dict(row) for row in conn.execute('SELECT row_number,reasons_json FROM rejected_rows ORDER BY row_number')]
            accepted = conn.execute('SELECT rows_count FROM batches').fetchone()[0]
        result = {'source': 'four existing random batches; independent temporary database',
                  'automatic_events': events, 'candidate_scenario': 'late', 'candidate': details,
                  'candidate_activated': activated, 'original_restored': rollback,
                  'quality_demo': {'input_rows': 3, 'accepted_rows': accepted, 'rejected_rows': rejected}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f'Сохранено: {args.output}')


if __name__ == '__main__':
    main()
