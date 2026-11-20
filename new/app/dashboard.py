"""Локальная панель мониторинга моделей академической успеваемости."""

import io
import json
import math
import sys
from contextlib import closing
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src1.service import (LIMITS, PSI_LIMIT, RANGES, activate, connect, get_batches,
                          model_info, predict, process_batch, reference_data,
                          train_candidate, version_list)

LABELS = {
    'school': 'Школа', 'sex': 'Пол', 'age': 'Возраст', 'address': 'Место проживания',
    'famsize': 'Размер семьи', 'Pstatus': 'Совместное проживание родителей',
    'Medu': 'Образование матери', 'Fedu': 'Образование отца',
    'Mjob': 'Работа матери', 'Fjob': 'Работа отца', 'reason': 'Причина выбора школы',
    'guardian': 'Опекун', 'traveltime': 'Время до школы', 'studytime': 'Время учёбы',
    'failures': 'Прежние неудачи', 'schoolsup': 'Доп. поддержка школы',
    'famsup': 'Поддержка семьи', 'paid': 'Платные занятия',
    'activities': 'Внеучебные занятия', 'nursery': 'Дошкольное образование',
    'higher': 'Планирует высшее образование', 'internet': 'Доступ к интернету',
    'romantic': 'Романтические отношения', 'famrel': 'Отношения в семье',
    'freetime': 'Свободное время', 'goout': 'Встречи с друзьями',
    'Dalc': 'Алкоголь в будни', 'Walc': 'Алкоголь в выходные',
    'health': 'Здоровье', 'absences': 'Пропуски занятий',
    'G1': 'Оценка за первый период', 'G2': 'Оценка за второй период',
}

st.set_page_config(page_title="Академическая успеваемость", page_icon="📊", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1400px; padding-top: 2rem;}
div[data-testid="stMetric"] {border: 1px solid #e0e5e7; padding: 14px; border-radius: 5px;}
</style>""", unsafe_allow_html=True)
st.title("Академическая успеваемость")
st.caption("Локальная исследовательская система · прогноз итоговой оценки G3")
st.warning("Прогноз не является решением о студенте. Ранняя модель не прошла критерии приёмки; поздняя применима только с оговорками.")

reference = reference_data()
pages = ["Обзор", "Прогноз студента", "Порции данных", "Модели"]
page = st.sidebar.radio("Раздел", pages)
st.sidebar.caption("Данные и результаты хранятся локально в runtime. Не используйте реальные персональные данные без разрешения.")

with closing(connect()) as conn:
    rows = get_batches(conn)
    if page == "Обзор":
        st.subheader("Мониторинг")
        if rows:
            batch_ids = sorted({row['id'] for row in rows})
            batch_names = {row['id']: row['name'] for row in rows}
            cutoff = st.selectbox('История до порции', batch_ids, index=len(batch_ids)-1,
                                  format_func=lambda value: f"{value} · {batch_names[value]}")
            rows = [row for row in rows if row['id'] <= cutoff]
        batch_count = len({row['id'] for row in rows})
        latest = next((r for r in reversed(rows) if r['scenario'] == 'late'), None)
        cols = st.columns(4)
        cols[0].metric("Обработано порций", batch_count)
        cols[1].metric("Поздняя модель", latest['model_version'] if latest else model_info(conn, "late")["version"])
        cols[2].metric("Последняя MAE · поздняя", f"{json.loads(latest['metrics_json'])['mae']:.3f}" if latest and json.loads(latest['metrics_json'])['mae'] is not None else "—")
        cols[3].metric("Порог PSI", f"{PSI_LIMIT:.2f}")
        if not rows:
            st.info("История пока пуста. Загрузите обезличенную порцию данных в разделе «Порции данных».")
        else:
            records, shifts = [], []
            for row in rows:
                m = json.loads(row['metrics_json'])
                records.append({"Порция": row['name'], "Номер": row['id'], "Сценарий": row['scenario'],
                                "MAE": m['mae'], "RMSE": m['rmse'], "R²": m['r2'], "Строк с G3": m['n'],
                                "Модель": row['model_version'], "Порог MAE": LIMITS[row['scenario']]})
                if row['scenario'] == 'late':
                    shifts.extend({"Порция": row['name'], "Номер": row['id'], "Признак": k, "PSI": v} for k,v in json.loads(row['psi_json']).items())
            df, psis = pd.DataFrame(records), pd.DataFrame(shifts)
            left, right = st.columns(2)
            with left:
                st.markdown("#### Ошибка модели по порциям")
                chart = alt.Chart(df.dropna(subset=['MAE'])).mark_line(point=True).encode(x=alt.X('Номер:O', title='Порядок порции'), y='MAE:Q', color='Сценарий:N', tooltip=list(df.columns))
                limits = alt.Chart(pd.DataFrame({'Сценарий': list(LIMITS), 'Порог': list(LIMITS.values())})).mark_rule(strokeDash=[4,4]).encode(y='Порог:Q', color='Сценарий:N')
                st.altair_chart(chart + limits, width='stretch')
            with right:
                st.markdown("#### Изменение распределений · PSI")
                base = alt.Chart(psis).mark_line(point=True).encode(x=alt.X('Номер:O', title='Порядок порции'), y='PSI:Q', color='Признак:N', tooltip=list(psis.columns))
                rule = alt.Chart(pd.DataFrame({'Порог':[PSI_LIMIT]})).mark_rule(color='#bf4b3c', strokeDash=[4,4]).encode(y='Порог:Q')
                st.altair_chart(base + rule, width='stretch')
            st.markdown("#### R² по порциям")
            st.altair_chart(alt.Chart(df.dropna(subset=['R²'])).mark_line(point=True).encode(x=alt.X('Номер:O', title='Порядок порции'), y=alt.Y('R²:Q', title='R²'), color='Сценарий:N', tooltip=list(df.columns)), width='stretch')
            st.dataframe(df, hide_index=True, width='stretch')
            warnings = df[df['MAE'].notna() & (df['MAE'] > df['Порог MAE'])]
            drift = psis[psis['PSI'] > PSI_LIMIT]
            if not warnings.empty or not drift.empty:
                st.warning(f"Превышений порога MAE: {len(warnings)}; сигналов PSI: {len(drift)}. Требуется проверка данных и ошибок.")
            else:
                st.success("Пороговых предупреждений в обработанных порциях нет.")
            st.caption("Порции исходного набора сформированы случайно, это не временной поток. PSI и MAE на малых выборках нестабильны.")

    elif page == "Прогноз студента":
        st.subheader("Прогноз одного студента")
        scenario = st.radio("Сценарий", ["early", "late"], format_func=lambda x: "Без G1 и G2" if x == 'early' else "С G1 и G2", horizontal=True)
        columns = [c for c in reference.columns if c != 'G3' and (scenario == 'late' or c not in ('G1', 'G2'))]
        with st.form("student"):
            student = {}
            layout = st.columns(3)
            for index, col in enumerate(columns):
                with layout[index % 3]:
                    label = f"{LABELS.get(col, col)} · {col}"
                    if col in RANGES:
                        lo, hi = RANGES[col]
                        student[col] = st.number_input(label, min_value=lo, max_value=None if math.isinf(hi) else hi, value=int(reference[col].median()), step=1)
                    else:
                        options = sorted(reference[col].dropna().unique().tolist())
                        student[col] = st.selectbox(label, options)
            submitted = st.form_submit_button("Рассчитать прогноз", type="primary")
        if submitted:
            try:
                values, version = predict(conn, pd.DataFrame([student]), scenario)
                st.metric("Прогноз G3", f"{values[0]:.2f} из 20")
                st.caption(f"Версия: {version}. Это непрерывный прогноз; значение не округляется и не заменяет фактическую оценку.")
                if not 0 <= values[0] <= 20:
                    st.warning("Прогноз находится вне шкалы оценок; модель не ограничивает выход автоматически.")
            except ValueError as exc:
                st.error(str(exc))

    elif page == "Порции данных":
        st.subheader("Обработка порций")
        upload = st.file_uploader("CSV с признаками; G3 может отсутствовать", type=['csv'])
        if upload and st.button("Обработать порцию", type="primary"):
            try:
                batch_id = process_batch(conn, upload.getvalue(), upload.name)
                st.success(f"Порция обработана: № {batch_id}")
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(str(exc))
        if rows:
            batches = [r for r in rows if r['scenario'] == 'late']
            names = {r['id']: f"{r['id']} · {r['name']}" for r in batches}
            chosen = st.selectbox("История порций", list(names), format_func=lambda x: names[x])
            current = next(r for r in batches if r['id'] == chosen)
            quality, psi = json.loads(current['quality_json']), json.loads(current['psi_json'])
            st.write("Качество данных", quality)
            st.write("PSI", psi)
            data = pd.read_json(io.StringIO(current['data_json']), orient='records')
            for entry in rows:
                if entry['id'] == chosen:
                    stats = json.loads(entry['metrics_json'])
                    predictions = np.array(json.loads(entry['predictions_json']))
                    actual = data['G3'].dropna() if 'G3' in data else pd.Series(dtype=float)
                    st.write(f"KPI · {entry['scenario']}", {
                        'Средний прогноз G3': stats.get('mean_prediction', float(predictions.mean())),
                        'Доля прогнозов ниже 10': stats.get('predicted_below_10', float((predictions < 10).mean())),
                        'Доля фактических оценок ниже 10': stats.get('actual_below_10', float((actual < 10).mean()) if len(actual) else None),
                        'Размер исходной порции': quality.get('total_rows', quality['rows']),
                        'Доля корректных строк': quality.get('valid_fraction', 1.0)})
            st.caption('Доля прогнозов ниже 10 — учебный индикатор, а не основание для решения о конкретном студенте.')
            detail = pd.DataFrame({'Строка': range(1, len(data)+1), 'Факт G3': data['G3'] if 'G3' in data else None})
            for r in rows:
                if r['id'] == chosen:
                    detail[f"Прогноз · {r['scenario']}"] = json.loads(r['predictions_json'])
            if 'G3' in data:
                detail['Ошибка · late'] = detail['Факт G3'] - detail['Прогноз · late']
                chart_data = detail.dropna(subset=['Факт G3'])
                if not chart_data.empty:
                    st.markdown("#### Факт и прогноз · поздняя модель")
                    st.altair_chart(alt.Chart(chart_data).mark_circle(size=55, opacity=0.65).encode(x='Факт G3:Q', y='Прогноз · late:Q', tooltip=['Строка','Факт G3','Прогноз · late','Ошибка · late']), width='stretch')
            st.dataframe(detail, hide_index=True, width='stretch')
            st.caption("Ошибки — разность фактической оценки и прогноза. Пропуски G3 не используются в метриках.")
        rejected = pd.read_sql_query('SELECT batch_name,row_number,reasons_json,created_at FROM rejected_rows ORDER BY id DESC', conn)
        if not rejected.empty:
            st.markdown('#### Журнал отклонённых строк')
            st.dataframe(rejected, hide_index=True, width='stretch')

    else:
        st.subheader("Версии моделей")
        scenario = st.selectbox("Сценарий", ['early','late'])
        st.caption("Кандидат обучается на исходных train + validation и всех размеченных порциях, кроме последней. Сравнение выполняется на последней порции без её включения в обучение.")
        if st.button("Обучить кандидата"):
            try:
                version, meta = train_candidate(conn, scenario)
                st.success(f"Кандидат {version}: MAE {meta['candidate_mae']:.3f}, текущая модель {meta['old_mae']:.3f}")
                st.json(meta)
            except ValueError as exc:
                st.error(str(exc))
        versions = version_list(conn, scenario)
        for entry in versions:
            meta = json.loads(entry['metadata_json'])
            with st.expander(f"{'Активная · ' if entry['active'] else ''}{entry['version']}"):
                st.json(meta)
                if not entry['active'] and st.button("Сделать активной", key=entry['version']):
                    try:
                        activate(conn, scenario, entry['version'])
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
                if not entry['active'] and st.button("Откатить к этой версии", key=f"rollback-{entry['version']}"):
                    try:
                        activate(conn, scenario, entry['version'], rollback=True)
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        st.caption("Замена выполняется вручную. Исходную версию можно вернуть в любой момент; предупреждение мониторинга само не переобучает модель.")
