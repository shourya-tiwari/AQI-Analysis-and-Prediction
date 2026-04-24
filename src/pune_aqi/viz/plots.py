from __future__ import annotations

import pandas as pd


def make_metrics_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a wide metrics frame into a long form for plotting."""

    out = df.reset_index().melt(id_vars=["model"], var_name="metric", value_name="value")
    return out

