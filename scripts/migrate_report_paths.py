"""Однократно переносит существующие результаты без перезаписи файлов."""
import hashlib
from pathlib import Path
import shutil

from src1.report_paths import FIGURE_CATEGORIES, TABLE_CATEGORIES
from src1.split_data import PROJECT_ROOT


def planned_moves(root=PROJECT_ROOT):
    root = Path(root)
    moves = []
    for kind, categories in (("tables", TABLE_CATEGORIES), ("figures", FIGURE_CATEGORIES)):
        for name, category in categories.items():
            moves.append((root / "reports" / kind / "ipynb02" / name, root / "reports" / kind / category / name))
        old = root / "reports" / kind / "experiments"
        if old.exists():
            moves.extend((path, root / "reports" / kind / "model_comparison" / "experiments" / path.name)
                         for path in old.iterdir() if path.is_file())
    moves.extend([
        (root / "reports/tables/model_metrics.csv", root / "reports/tables/model_comparison/legacy/model_metrics.csv"),
        *[(root / "reports/figures" / name, root / "reports/figures/model_comparison/legacy" / name)
          for name in ("mae_comparison.png", "r2_comparison.png")],
        (root / "reports/figures/actual_vs_predicted_gb_with_grades.png",
         root / "reports/figures/predictions/legacy/actual_vs_predicted_gb_with_grades.png"),
    ])
    return [(source, target) for source, target in moves if source.exists()]


def main():
    moves = planned_moves()
    targets = [target.resolve() for _, target in moves]
    reports = (PROJECT_ROOT / "reports").resolve()
    if len(targets) != len(set(targets)) or any(not target.is_relative_to(reports) for target in targets):
        raise ValueError("Некорректные или повторяющиеся пути назначения")
    if any(target.exists() for _, target in moves):
        raise FileExistsError("Целевой файл уже существует: перенос остановлен без перезаписи")
    for source, target in moves:
        digest = hashlib.sha256(source.read_bytes()).digest()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        if hashlib.sha256(target.read_bytes()).digest() != digest:
            raise IOError(f"Не совпала контрольная сумма: {target}")
        print(f"{source.relative_to(reports)} -> {target.relative_to(reports)}")
    for kind in ("tables", "figures"):
        for old in (reports / kind / "ipynb02", reports / kind / "experiments"):
            if old.is_dir() and not any(old.iterdir()):
                old.rmdir()
    print(f"Перенесено файлов: {len(moves)}")


if __name__ == "__main__":
    main()
