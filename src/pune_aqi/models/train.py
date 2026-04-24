from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ..data.cleaning import PuneAqiSchema, pick_features_and_target


@dataclass(frozen=True)
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics_train: dict[str, float]
    metrics_test: dict[str, float]


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = (y_true >= 0) & (y_pred >= 0)
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean((np.log1p(y_pred[mask]) - np.log1p(y_true[mask])) ** 2)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmsle": float(rmsle(y_true, y_pred)),
    }


def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str]) -> ColumnTransformer:
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical, categorical_cols),
            ("num", numeric, numeric_cols),
        ]
    )


def train_and_evaluate_models(
    df: pd.DataFrame,
    *,
    schema: PuneAqiSchema | None = None,
    test_size: float = 0.25,
    random_state: int = 0,
) -> list[ModelResult]:
    """Train multiple regressors and return a comparable set of metrics."""

    schema = schema or PuneAqiSchema()
    X, y, categorical_cols, numeric_cols = pick_features_and_target(df, schema=schema)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    preprocessor = build_preprocessor(categorical_cols, numeric_cols)

    candidates: list[tuple[str, object]] = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor(random_state=random_state)),
        ("Random Forest", RandomForestRegressor(n_estimators=500, random_state=random_state)),
        (
            "SVR (RBF)",
            Pipeline(
                steps=[
                    ("scaler", StandardScaler(with_mean=False)),
                    ("svr", SVR(kernel="rbf")),
                ]
            ),
        ),
    ]

    results: list[ModelResult] = []
    for name, model in candidates:
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        pipe.fit(x_train, y_train)

        pred_train = pipe.predict(x_train)
        pred_test = pipe.predict(x_test)

        results.append(
            ModelResult(
                name=name,
                pipeline=pipe,
                metrics_train=_metrics(y_train, pred_train),
                metrics_test=_metrics(y_test, pred_test),
            )
        )

    return results


def results_to_frame(results: list[ModelResult], *, split: str = "test") -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for r in results:
        m = r.metrics_test if split == "test" else r.metrics_train
        rows.append({"model": r.name, **m})
    df = pd.DataFrame(rows).set_index("model").sort_values("rmse", ascending=True)
    return df

