import ast
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import src1.report_paths as report_paths

from src1.report_paths import FIGURE_CATEGORIES, TABLE_CATEGORIES, ReportDirectory
from src1.split_data import PROJECT_ROOT


class ReportPathTests(unittest.TestCase):
    def test_known_paths_are_grouped(self):
        with tempfile.TemporaryDirectory() as directory:
            tables = ReportDirectory(directory, "tables")
            figures = ReportDirectory(directory, "figures")
            self.assertEqual((tables / "g3_correlations.csv").parent.name, "correlation")
            self.assertEqual((figures / "correlation_heatmap.png").parent.name, "correlation")
            with self.assertRaises(ValueError):
                tables / "unknown.csv"

    def test_notebook_artifacts_have_categories(self):
        notebook = json.loads((PROJECT_ROOT / "notebooks/02_data_analysis.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("save_table", "show_diagram"):
                    if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
                        categories = TABLE_CATEGORIES if node.func.id == "save_table" else FIGURE_CATEGORIES
                        self.assertIn(node.args[1].value, categories)
                if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and isinstance(node.left, ast.Name):
                    if node.left.id in ("tables_dir", "figures_dir") and isinstance(node.right, ast.Constant):
                        categories = TABLE_CATEGORIES if node.left.id == "tables_dir" else FIGURE_CATEGORIES
                        self.assertIn(node.right.value, categories)

    def test_eda_notebook_writes_to_temporary_directories(self):
        notebook = json.loads((PROJECT_ROOT / "notebooks/02_data_analysis.ipynb").read_text(encoding="utf-8"))
        scope = {}
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(report_paths, "ReportDirectory", side_effect=lambda root, kind: ReportDirectory(directory, kind)):
                for index, cell in enumerate(notebook["cells"]):
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]), f"notebook cell {index}", "exec"), scope)
            self.assertTrue((Path(directory) / "reports/figures/correlation/correlation_heatmap.png").is_file())
            self.assertTrue((Path(directory) / "reports/tables/correlation/g3_correlations.csv").is_file())
        plt.close("all")


if __name__ == "__main__":
    unittest.main()
