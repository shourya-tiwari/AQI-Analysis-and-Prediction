from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import pandas as pd


_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def slugify_column(name: str) -> str:
    """Convert a column name into a readable snake_case identifier.

    Examples:
    - "PM2.5" -> "pm2_5"
    - "Predominant Parameter" -> "predominant_parameter"
    """

    s = str(name).strip().lower()
    s = s.replace("μ", "u").replace("µ", "u")
    s = s.replace(".", "_")
    s = _NON_ALNUM_RE.sub("_", s)
    s = s.strip("_")
    s = re.sub(r"_+", "_", s)
    return s


@dataclass(frozen=True)
class PuneAqiSchema:
    """Canonical column names used throughout the project."""

    state: str = "state"
    city: str = "city"
    station: str = "station"
    date: str = "date"
    time: str = "time"
    datetime: str = "datetime"
    pm2_5: str = "pm2_5"
    pm10: str = "pm10"
    no2: str = "no2"
    nh3: str = "nh3"
    so2: str = "so2"
    co: str = "co"
    ozone: str = "ozone"
    aqi: str = "aqi"
    predominant_parameter: str = "predominant_parameter"

    def required_columns(self) -> list[str]:
        return [
            self.state,
            self.city,
            self.station,
            self.date,
            self.time,
            self.pm2_5,
            self.pm10,
            self.no2,
            self.nh3,
            self.so2,
            self.co,
            self.ozone,
            self.aqi,
        ]


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical snake_case names.

    This function is intentionally defensive: it slugifies all columns, then
    applies a small set of known aliases for the CPCB-derived dataset.
    """

    df = df.copy()
    slugified = {c: slugify_column(c) for c in df.columns}
    df = df.rename(columns=slugified)

    aliases = {
        "pm2_5": {"pm2_5", "pm2_5_avg", "pm2_5_ug_m3", "pm25", "pm_2_5"},
        "pm10": {"pm10", "pm10_avg", "pm10_ug_m3"},
        "no2": {"no2", "nitrogen_dioxide", "no2_avg"},
        "nh3": {"nh3", "ammonia", "nh3_avg"},
        "so2": {"so2", "sulphur_dioxide", "sulfur_dioxide", "so2_avg"},
        "co": {"co", "carbon_monoxide", "co_avg"},
        "ozone": {"ozone", "o3", "o3_avg"},
        "aqi": {"aqi", "air_quality_index", "air_quality_index_value"},
        "predominant_parameter": {
            "predominant_parameter",
            "predominant_para",
            "predominant_pollutant",
        },
    }

    reverse_alias: dict[str, str] = {}
    for canonical, names in aliases.items():
        for n in names:
            reverse_alias[n] = canonical

    df = df.rename(columns={c: reverse_alias.get(c, c) for c in df.columns})
    return df


def _to_numeric(series: pd.Series) -> pd.Series:
    s = series.replace({"NA": pd.NA, "na": pd.NA, "": pd.NA, "-": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def clean_pune_aqi_dataset(
    df: pd.DataFrame,
    *,
    schema: PuneAqiSchema | None = None,
    drop_rows_without_aqi: bool = True,
) -> pd.DataFrame:
    """Clean the Pune AQI dataset into a consistent, model-ready form."""

    schema = schema or PuneAqiSchema()
    df = standardize_columns(df)

    for col in [
        schema.pm2_5,
        schema.pm10,
        schema.no2,
        schema.nh3,
        schema.so2,
        schema.co,
        schema.ozone,
        schema.aqi,
    ]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    if schema.date in df.columns and schema.time in df.columns:
        dt = (
            df[schema.date].astype(str).str.strip()
            + " "
            + df[schema.time].astype(str).str.strip()
        )
        df[schema.datetime] = pd.to_datetime(dt, errors="coerce")
    elif schema.date in df.columns:
        df[schema.datetime] = pd.to_datetime(df[schema.date], errors="coerce")

    if drop_rows_without_aqi and schema.aqi in df.columns:
        df = df.dropna(subset=[schema.aqi])

    # Gentle de-duplication; keep the newest reading if datetime exists.
    sort_cols: list[str] = [c for c in [schema.datetime] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=True)

    subset: list[str] = [c for c in schema.required_columns() if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last")

    return df.reset_index(drop=True)


def pick_features_and_target(
    df: pd.DataFrame,
    *,
    schema: PuneAqiSchema | None = None,
    categorical: Iterable[str] | None = None,
    numeric: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Return (X, y, categorical_cols, numeric_cols)."""

    schema = schema or PuneAqiSchema()
    categorical_cols = list(categorical or [schema.state, schema.city, schema.station])
    numeric_cols = list(
        numeric
        or [
            schema.pm2_5,
            schema.pm10,
            schema.no2,
            schema.nh3,
            schema.so2,
            schema.co,
            schema.ozone,
        ]
    )

    missing = [c for c in categorical_cols + numeric_cols + [schema.aqi] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    X = df[categorical_cols + numeric_cols].copy()
    y = df[schema.aqi].astype(float)
    return X, y, categorical_cols, numeric_cols

