from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cleaning import PuneAqiSchema, clean_pune_aqi_dataset


@dataclass(frozen=True)
class DataPaths:
    repo_root: Path

    @property
    def raw_dir(self) -> Path:
        return self.repo_root / "data" / "raw" / "pune_aqi"

    @property
    def processed_dir(self) -> Path:
        return self.repo_root / "data" / "processed"

    @property
    def default_csv(self) -> Path:
        return self.raw_dir / "state_weather_aqi_data_mf2.csv"


def find_repo_root(start: Path) -> Path:
    """Best-effort repo root discovery for local runs."""

    cur = start.resolve()
    for _ in range(8):
        if (cur / "src").exists() and (cur / "app").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def load_default_dataset(*, repo_root: Path | None = None, schema: PuneAqiSchema | None = None) -> pd.DataFrame:
    schema = schema or PuneAqiSchema()
    repo_root = repo_root or find_repo_root(Path.cwd())
    paths = DataPaths(repo_root=repo_root)
    df = pd.read_csv(paths.default_csv)
    return clean_pune_aqi_dataset(df, schema=schema)

