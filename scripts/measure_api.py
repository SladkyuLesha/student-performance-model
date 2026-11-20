"""Measure local prediction latency without saving student features."""

import argparse
import json
import statistics
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from src1.service import PROJECT_ROOT, reference_data

OUTPUT = PROJECT_ROOT / "reports/tables/monitoring/api_latency.csv"


def summarize(values):
    if not values:
        raise ValueError("Нужен хотя бы один замер")
    series = pd.Series(values, dtype=float)
    return {"requests": len(values), "mean_ms": float(series.mean()),
            "median_ms": float(statistics.median(values)),
            "p95_ms": float(series.quantile(0.95)), "min_ms": float(series.min()),
            "max_ms": float(series.max())}


def request_prediction(student, scenario, port):
    body = json.dumps({"scenario": scenario, "student": student}, ensure_ascii=False).encode("utf-8")
    request = Request(f"http://127.0.0.1:{port}/predict", data=body,
                      headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    return (time.perf_counter() - start) * 1000, result["model_version"]


def main():
    parser = argparse.ArgumentParser(description="Замерить локальный API на учебном примере")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true", help="Разрешить перезапись сохранённой таблицы")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.count <= 1000:
        parser.error("Порт должен быть от 1 до 65535, число запросов от 1 до 1000")
    if OUTPUT.exists() and not args.overwrite:
        parser.exit(1, f"Файл уже существует: {OUTPUT}. Для замены добавьте --overwrite.\n")
    row = json.loads(reference_data().iloc[[0]].to_json(orient="records"))[0]
    row.pop("G3")
    results = []
    try:
        for scenario in ("early", "late"):
            student = {k: v for k, v in row.items() if scenario == "late" or k not in ("G1", "G2")}
            request_prediction(student, scenario, args.port)
            samples, versions = [], set()
            for _ in range(args.count):
                elapsed, version = request_prediction(student, scenario, args.port)
                samples.append(elapsed)
                versions.add(version)
            if len(versions) != 1:
                raise ValueError("Во время замера изменилась активная версия модели")
            results.append({"scenario": scenario, "model_version": versions.pop(), **summarize(samples)})
    except (URLError, OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Замер не выполнен: {exc}. Проверьте запуск локального API.\n")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT, index=False)
    print(pd.DataFrame(results).round(2).to_string(index=False))
    print(f"Сохранено: {OUTPUT}")


if __name__ == "__main__":
    main()
