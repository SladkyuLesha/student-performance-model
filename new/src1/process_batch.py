"""Обработка CSV одной командой из корня проекта."""

import argparse
from contextlib import closing
from pathlib import Path

from src1.service import connect, process_batch


def main():
    parser = argparse.ArgumentParser(description="Проверить и обработать порцию студентов")
    parser.add_argument("path", type=Path, help="Путь к CSV в кодировке UTF-8")
    args = parser.parse_args()
    try:
        with closing(connect()) as conn:
            result = process_batch(conn, args.path.read_bytes(), args.path.name)
        print(f"Порция обработана: {result}")
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
