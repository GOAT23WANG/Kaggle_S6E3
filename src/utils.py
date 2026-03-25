from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd


class ExperimentLogger:
    def __init__(self, run_number: int) -> None:
        self.lines: list[str] = [
            f"Run Number: {run_number}",
            f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

    def add(self, text: str = "") -> None:
        self.lines.append(text)

    def add_section(self, title: str) -> None:
        self.lines.append(title)

    def add_mapping(self, title: str, values: dict) -> None:
        self.lines.append(title)
        for key, value in values.items():
            self.lines.append(f"{key}: {value}")
        self.lines.append("")

    def write(self, output_path: Path) -> None:
        output_path.write_text("\n".join(map(str, self.lines)), encoding="utf-8")


class Timer:
    def __init__(self) -> None:
        self._start = perf_counter()

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self._start


def ensure_directories(paths: dict[str, Path]) -> None:
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)


def get_next_run_number(folder: Path, prefix: str, suffix: str) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(suffix)}$")

    current_max = 0
    for file_path in folder.iterdir():
        match = pattern.match(file_path.name)
        if match:
            current_max = max(current_max, int(match.group(1)))

    return current_max + 1


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def dump_json(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def format_seconds(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:05.2f}"
