"""Локальная обработка обезличенных данных и мониторинг моделей."""

import hashlib
import io
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src1.evaluation import psi_numeric
from src1.load_split import load_split
from src1.split_data import PROJECT_ROOT

RUNTIME = Path(os.environ.get("STUDENT_MONITOR_RUNTIME", PROJECT_ROOT / "new" / "runtime")).resolve()
DB_PATH = RUNTIME / "monitoring.sqlite3"
SCENARIOS = ("early", "late")
LIMITS = {"early": 3.5, "late": 2.0}
PSI_LIMIT = 0.25
RANGES = {
    "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4),
    "traveltime": (1, 4), "studytime": (1, 4), "failures": (0, 3),
    "famrel": (1, 5), "freetime": (1, 5), "goout": (1, 5),
    "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
    "absences": (0, float('inf')), "G1": (0, 20), "G2": (0, 20), "G3": (0, 20),
}


def reference_data():
    train, validation, _ = load_split()
    return pd.concat([train, validation], ignore_index=True)


def connect(db_path=DB_PATH):
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS batches (
        id INTEGER PRIMARY KEY, name TEXT NOT NULL, digest TEXT NOT NULL UNIQUE,
        processed_at TEXT NOT NULL, rows_count INTEGER NOT NULL, elapsed_seconds REAL NOT NULL,
        data_json TEXT NOT NULL, quality_json TEXT NOT NULL, psi_json TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS batch_results (
        batch_id INTEGER NOT NULL REFERENCES batches(id), scenario TEXT NOT NULL,
        model_version TEXT NOT NULL, predictions_json TEXT NOT NULL, metrics_json TEXT NOT NULL,
        PRIMARY KEY (batch_id, scenario)
    );
    CREATE TABLE IF NOT EXISTS model_versions (
        scenario TEXT NOT NULL, version TEXT NOT NULL, artifact TEXT NOT NULL,
        created_at TEXT NOT NULL, metadata_json TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (scenario, version)
    );
    CREATE UNIQUE INDEX IF NOT EXISTS one_active_model ON model_versions(scenario) WHERE active=1;
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, action TEXT NOT NULL, details TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS rejected_rows (
        id INTEGER PRIMARY KEY, digest TEXT NOT NULL, batch_name TEXT NOT NULL,
        row_number INTEGER NOT NULL, reasons_json TEXT NOT NULL, created_at TEXT NOT NULL
    );
    """)
    for scenario in SCENARIOS:
        artifact = PROJECT_ROOT / "models" / f"{scenario}_model.joblib"
        metadata = PROJECT_ROOT / "models" / f"{scenario}_model.json"
        conn.execute("INSERT OR IGNORE INTO model_versions VALUES (?, ?, ?, ?, ?, 0)",
                     (scenario, "original", str(artifact), "initial", metadata.read_text(encoding="utf-8")))
        active = conn.execute("SELECT 1 FROM model_versions WHERE scenario=? AND active=1", (scenario,)).fetchone()
        if not active:
            conn.execute("UPDATE model_versions SET active=1 WHERE scenario=? AND version='original'", (scenario,))
    conn.commit()
    return conn


def validate_frame(frame, scenario, reference=None, *, allow_target=True):
    if scenario not in SCENARIOS:
        raise ValueError("Сценарий должен быть early или late")
    reference = reference_data() if reference is None else reference
    required = [c for c in reference.columns if c != "G3" and (scenario == "late" or c not in ("G1", "G2"))]
    allowed = set(required) | ({"G3"} if allow_target else set())
    missing = sorted(set(required) - set(frame.columns))
    extra = sorted(set(frame.columns) - allowed)
    if missing or extra or frame.empty or frame.columns.duplicated().any():
        raise ValueError(f"Проверьте структуру: отсутствуют {missing}, лишние {extra}; данные должны быть непустыми")
    data = frame.copy()
    for col in data.columns:
        if col in RANGES:
            original = data[col]
            converted = pd.to_numeric(original, errors="coerce")
            low, high = RANGES[col]
            invalid = original.notna() & (converted.isna() | ~np.isfinite(converted) | (converted % 1 != 0) | ~converted.between(low, high))
            if invalid.any():
                raise ValueError(f"{col}: ожидается целое число от {low} до {high}")
            if col != "G3" and converted.isna().any():
                raise ValueError(f"{col}: пропуски недопустимы")
            data[col] = converted
        else:
            if data[col].isna().any() or not data[col].isin(reference[col].dropna().unique()).all():
                raise ValueError(f"{col}: неизвестное значение или пропуск")
    return data


def model_info(conn, scenario):
    if scenario not in SCENARIOS:
        raise ValueError("Неизвестный сценарий")
    row = conn.execute("SELECT * FROM model_versions WHERE scenario=? AND active=1", (scenario,)).fetchone()
    if not row:
        raise ValueError("Активная модель не найдена")
    return dict(row)


def predict(conn, frame, scenario):
    data = validate_frame(frame, scenario)
    info = model_info(conn, scenario)
    model = joblib.load(info["artifact"])
    columns = model.named_steps["features"].input_columns_
    values = model.predict(data.loc[:, columns])
    return values, info["version"]


def metrics(y, predictions):
    valid = y.notna().to_numpy()
    if not valid.any():
        return {"n": 0, "mae": None, "rmse": None, "r2": None}
    actual, predicted = y.to_numpy()[valid], np.asarray(predictions)[valid]
    return {"n": int(valid.sum()), "mae": float(mean_absolute_error(actual, predicted)),
            "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
            "r2": float(r2_score(actual, predicted)) if len(actual) > 1 and np.var(actual) > 0 else None}


def partition_batch(frame, reference):
    """Separate invalid records while retaining only row numbers and reasons."""
    required = set(reference.columns) - {"G3"}
    missing = sorted(required - set(frame.columns))
    extra = sorted(set(frame.columns) - set(reference.columns))
    if missing or extra or frame.empty or frame.columns.duplicated().any():
        raise ValueError(f"Проверьте структуру: отсутствуют {missing}, лишние {extra}; данные должны быть непустыми")
    data = frame.copy().reset_index(drop=True)
    issues = [[] for _ in range(len(data))]
    for col in data.columns:
        if col in RANGES:
            original = data[col]
            numeric = pd.to_numeric(original, errors="coerce")
            low, high = RANGES[col]
            invalid = original.notna() & (numeric.isna() | ~np.isfinite(numeric) | (numeric % 1 != 0) | ~numeric.between(low, high))
            if col != "G3":
                invalid |= numeric.isna()
            data[col] = numeric
        else:
            invalid = data[col].isna() | ~data[col].isin(reference[col].dropna().unique())
        for index in np.flatnonzero(invalid.to_numpy()):
            issues[index].append(f"{col}: недопустимое значение или пропуск")
    valid = np.array([not item for item in issues])
    return data.loc[valid].copy(), [(i + 2, reasons) for i, reasons in enumerate(issues) if reasons]


def process_batch(conn, content, name):
    started = time.perf_counter()
    if len(content) > 10_000_000:
        raise ValueError("Размер CSV превышает 10 МБ")
    digest = hashlib.sha256(content).hexdigest()
    if conn.execute("SELECT id FROM batches WHERE digest=?", (digest,)).fetchone():
        raise ValueError("Такая порция уже обработана")
    try:
        source = pd.read_csv(io.BytesIO(content))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Не удалось прочитать CSV с UTF-8") from exc
    reference = reference_data()
    now = datetime.now(timezone.utc).isoformat()
    try:
        late, rejected = partition_batch(source, reference)
    except ValueError as exc:
        with conn:
            conn.executemany("INSERT INTO rejected_rows(digest,batch_name,row_number,reasons_json,created_at) VALUES (?,?,?,?,?)",
                             [(digest, Path(name).name, i + 2, json.dumps([str(exc)], ensure_ascii=False), now) for i in range(len(source))])
        raise
    if rejected:
        with conn:
            conn.executemany("INSERT INTO rejected_rows(digest,batch_name,row_number,reasons_json,created_at) VALUES (?,?,?,?,?)",
                             [(digest, Path(name).name, number, json.dumps(reasons, ensure_ascii=False), now) for number, reasons in rejected])
    if late.empty:
        raise ValueError("Все строки порции отклонены; причины записаны в журнал")
    early = late.drop(columns=["G1", "G2"])
    predictions = {s: predict(conn, frame, s) for s, frame in (("early", early), ("late", late))}
    psi = {col: psi_numeric(reference[col], late[col]) for col in ("age", "absences", "failures")}
    quality = {"rows": len(late), "total_rows": len(source), "rejected": len(rejected),
               "valid_fraction": len(late) / len(source), "duplicates": int(late.duplicated().sum()),
               "duplicate_rows": [int(i) + 2 for i in late.index[late.duplicated(keep=False)]],
               "unlabeled": int(late["G3"].isna().sum()) if "G3" in late else len(late)}
    with conn:
        cur = conn.execute("INSERT INTO batches (name,digest,processed_at,rows_count,elapsed_seconds,data_json,quality_json,psi_json) VALUES (?,?,?,?,?,?,?,?)",
                           (Path(name).name, digest, now, len(late), time.perf_counter()-started,
                            late.to_json(orient="records"), json.dumps(quality), json.dumps(psi)))
        for scenario, (values, version) in predictions.items():
            y = late["G3"] if "G3" in late else pd.Series(np.nan, index=late.index)
            scores = metrics(y, values)
            scores.update({"mean_prediction": float(np.mean(values)), "predicted_below_10": float(np.mean(values < 10)),
                           "actual_below_10": float((y.dropna() < 10).mean()) if y.notna().any() else None})
            conn.execute("INSERT INTO batch_results VALUES (?,?,?,?,?)", (cur.lastrowid, scenario, version,
                         json.dumps(values.tolist()), json.dumps(scores)))
        conn.execute("INSERT INTO events(created_at,action,details) VALUES (?,?,?)", (now, "batch_processed", Path(name).name))
    batch_id = cur.lastrowid
    # Candidate training never activates or replaces the current model.
    if "G3" in late and late["G3"].notna().all():
        signal_psi = any(value > PSI_LIMIT for value in psi.values())
        for scenario, (values, _) in predictions.items():
            score = metrics(late["G3"], values)["mae"]
            if signal_psi or (score is not None and score > LIMITS[scenario]):
                try:
                    version, _ = train_candidate(conn, scenario)
                    detail = f"{batch_id}:{scenario}:{version}"
                    action = "automatic_candidate"
                except (ValueError, OSError) as exc:
                    detail = f"{batch_id}:{scenario}:{exc}"
                    action = "automatic_candidate_skipped"
                except Exception as exc:
                    detail = f"{batch_id}:{scenario}:{type(exc).__name__}"
                    action = "automatic_candidate_failed"
                with conn:
                    conn.execute("INSERT INTO events(created_at,action,details) VALUES (?,?,?)", (now, action, detail))
    return batch_id


def get_batches(conn):
    rows = conn.execute("SELECT b.*, r.scenario,r.model_version,r.predictions_json,r.metrics_json FROM batches b JOIN batch_results r ON b.id=r.batch_id ORDER BY b.id,r.scenario").fetchall()
    return [dict(row) for row in rows]


def version_list(conn, scenario):
    return [dict(row) for row in conn.execute("SELECT * FROM model_versions WHERE scenario=? ORDER BY created_at DESC", (scenario,))]


def activate(conn, scenario, version, *, rollback=False):
    candidate = conn.execute("SELECT * FROM model_versions WHERE scenario=? AND version=?", (scenario, version)).fetchone()
    if not candidate or not Path(candidate["artifact"]).is_file():
        raise ValueError("Версия модели не найдена")
    metadata = json.loads(candidate["metadata_json"])
    if version != "original" and not metadata.get("eligible", False):
        raise ValueError("Кандидат не прошёл критерий сравнения")
    used_before = version == "original" or conn.execute(
        "SELECT 1 FROM events WHERE action='activate' AND details=?", (f"{scenario}:{version}",)
    ).fetchone() is not None
    if rollback and not used_before:
        raise ValueError("Откат возможен только к ранее активной версии")
    if not rollback and version != "original" and metadata.get("old_version") != model_info(conn, scenario)["version"]:
        raise ValueError("Действующая модель изменилась после сравнения; нужен новый кандидат")
    with conn:
        conn.execute("UPDATE model_versions SET active=0 WHERE scenario=?", (scenario,))
        conn.execute("UPDATE model_versions SET active=1 WHERE scenario=? AND version=?", (scenario, version))
        conn.execute("INSERT INTO events(created_at,action,details) VALUES (?,?,?)", (datetime.now(timezone.utc).isoformat(), "activate", f"{scenario}:{version}"))


def train_candidate(conn, scenario, directory=None):
    if scenario not in SCENARIOS:
        raise ValueError("Неизвестный сценарий")
    batches = conn.execute("SELECT id,data_json FROM batches ORDER BY id").fetchall()
    labeled = [(r["id"], pd.read_json(io.StringIO(r["data_json"]))) for r in batches]
    labeled = [(i, df) for i, df in labeled if "G3" in df and df["G3"].notna().all()]
    if len(labeled) < 2:
        raise ValueError("Нужны минимум две порции с известными G3: для обучения и сравнения")
    reference = reference_data()
    holdout_id, holdout = labeled[-1]
    prior = pd.concat([reference] + [df for _, df in labeled[:-1]], ignore_index=True)
    keys = [c for c in reference.columns if c not in ("G1", "G2", "G3")]
    _, _, original_test = load_split()
    incoming = pd.concat([df for _, df in labeled], ignore_index=True)
    if pd.MultiIndex.from_frame(incoming[keys]).isin(pd.MultiIndex.from_frame(original_test[keys])).any():
        raise ValueError("Порция пересекается с исходной test-выборкой; обучение запрещено")
    if pd.MultiIndex.from_frame(holdout[keys]).isin(pd.MultiIndex.from_frame(prior[keys])).any():
        raise ValueError("Профили сравнения пересекаются с обучающими данными")
    info = model_info(conn, scenario)
    current = joblib.load(info["artifact"])
    columns = current.named_steps["features"].input_columns_
    candidate = clone(current).fit(prior.loc[:, columns], prior["G3"])
    old_mae = mean_absolute_error(holdout["G3"], current.predict(holdout.loc[:, columns]))
    new_mae = mean_absolute_error(holdout["G3"], candidate.predict(holdout.loc[:, columns]))
    eligible = bool(new_mae < old_mae and new_mae <= LIMITS[scenario])
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if directory is None:
        database_path = conn.execute("PRAGMA database_list").fetchone()[2]
        directory = Path(database_path).parent / "models"
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{scenario}_{version}.joblib"
    joblib.dump(candidate, artifact)
    metadata = {"eligible": eligible, "old_version": info["version"], "old_mae": old_mae,
                "candidate_mae": new_mae, "comparison_batch": holdout_id,
                "training_batches": [i for i, _ in labeled[:-1]], "training_rows": len(prior),
                "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "parameters": candidate.named_steps['model'].get_params(),
                "warning": "Случайные порции малы и ранее анализировались; оценка исследовательская."}
    with conn:
        conn.execute("INSERT INTO model_versions VALUES (?,?,?,?,?,0)", (scenario, version, str(artifact), version, json.dumps(metadata, ensure_ascii=False)))
        conn.execute("INSERT INTO events(created_at,action,details) VALUES (?,?,?)", (version, "train_candidate", f"{scenario}:{version}"))
    return version, metadata
