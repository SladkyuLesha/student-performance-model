"""Export aggregate monitoring history without exporting individual student data."""

import argparse
import io
import json
from contextlib import closing
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src1.service import LIMITS, PROJECT_ROOT, PSI_LIMIT, connect, get_batches


def export_history(rows, figures, tables):
    if not rows:
        raise ValueError('История пуста')
    metrics_rows, kpi_rows, psi_rows = [], [], []
    for row in rows:
        scores = json.loads(row['metrics_json'])
        quality = json.loads(row['quality_json'])
        pred = np.asarray(json.loads(row['predictions_json']))
        data = pd.read_json(io.StringIO(row['data_json']), orient='records')
        actual = data['G3'].dropna() if 'G3' in data else pd.Series(dtype=float)
        base = {'batch_id': row['id'], 'batch': row['name'], 'scenario': row['scenario'], 'model_version': row['model_version']}
        metrics_rows.append({**base, **{key: scores.get(key) for key in ('n', 'mae', 'rmse', 'r2')}})
        kpi_rows.append({**base, 'mean_prediction': float(pred.mean()), 'predicted_below_10': float((pred < 10).mean()),
                         'rows_received': quality.get('total_rows', quality['rows']),
                         'actual_below_10': float((actual < 10).mean()) if len(actual) else None,
                         'valid_fraction': quality.get('valid_fraction', 1.0)})
        if row['scenario'] == 'late':
            psi_rows.extend({'batch_id': row['id'], 'batch': row['name'], 'feature': k, 'psi': v}
                            for k, v in json.loads(row['psi_json']).items())
    metric_data, kpi_data, psi_data = map(pd.DataFrame, (metrics_rows, kpi_rows, psi_rows))
    figures, tables = Path(figures), Path(tables)
    ids = sorted(metric_data['batch_id'].unique())
    outputs = [figures / f'history_{i:02d}.png' for i in ids]
    outputs += [tables / x for x in ('panel_metrics.csv', 'panel_kpi.csv', 'panel_psi.csv')]
    if any(path.exists() for path in outputs):
        raise FileExistsError('Файлы экспорта уже существуют; сохраните их или укажите другие каталоги')
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    metric_data.to_csv(outputs[-3], index=False)
    kpi_data.to_csv(outputs[-2], index=False)
    psi_data.to_csv(outputs[-1], index=False)
    for batch_id, path in zip(ids, outputs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        subset = metric_data[metric_data['batch_id'] <= batch_id]
        for scenario, group in subset.groupby('scenario'):
            axes[0].plot(group['batch_id'], group['mae'], marker='o', label=scenario)
            axes[0].axhline(LIMITS[scenario], linestyle='--', alpha=.4)
        for feature, group in psi_data[psi_data['batch_id'] <= batch_id].groupby('feature'):
            axes[1].plot(group['batch_id'], group['psi'], marker='o', label=feature)
        axes[1].axhline(PSI_LIMIT, color='firebrick', linestyle='--', label='Порог 0,25')
        axes[0].set(title='MAE по порциям', xlabel='Номер порции', ylabel='MAE, баллы')
        axes[1].set(title='PSI по порциям', xlabel='Номер порции', ylabel='PSI')
        for ax in axes:
            ax.set_xticks([x for x in ids if x <= batch_id])
            ax.grid(alpha=.2)
            ax.legend()
        fig.suptitle(f'Сохранённая история до порции {batch_id}')
        fig.savefig(path, dpi=160)
        plt.close(fig)
    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures', type=Path, default=PROJECT_ROOT / 'reports/figures/monitoring')
    parser.add_argument('--tables', type=Path, default=PROJECT_ROOT / 'reports/tables/monitoring')
    args = parser.parse_args()
    with closing(connect()) as conn:
        outputs = export_history(get_batches(conn), args.figures, args.tables)
    for path in outputs:
        print(path)


if __name__ == '__main__':
    main()
