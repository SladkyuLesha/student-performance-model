"""Keep historical src1 imports working after moving the implementation."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent / "new" / "src1")]
