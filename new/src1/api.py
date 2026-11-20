"""Локальный API прогнозирования; предназначен только для доверенного компьютера."""

import json
import time
from contextlib import closing

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from src1.service import DB_PATH, connect, get_batches, model_info, predict

app = FastAPI(title="Student Performance API", version="1.0.0")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str
    student: dict[str, int | str | None]


@app.get("/health")
def health():
    with closing(connect(DB_PATH)) as conn:
        active = {s: model_info(conn, s)["version"] for s in ("early", "late")}
    return {"status": "ok", "models": active}


@app.post("/predict")
def predict_student(request: PredictionRequest):
    started = time.perf_counter()
    try:
        with closing(connect(DB_PATH)) as conn:
            values, version = predict(conn, pd.DataFrame([request.student]), request.scenario)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"scenario": request.scenario, "model_version": version, "prediction": float(values[0]),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "notice": "Исследовательский прогноз, не является решением о студенте."}


@app.get("/metrics")
def monitoring():
    with closing(connect(DB_PATH)) as conn:
        rows = get_batches(conn)
    return [{"batch": row["name"], "scenario": row["scenario"], "model_version": row["model_version"],
             "metrics": json.loads(row["metrics_json"]), "psi": json.loads(row["psi_json"])} for row in rows]
