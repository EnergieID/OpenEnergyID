"""Generic linear regression helpers for analysis modules."""

from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression


def _column_list(frame: pd.DataFrame, names: str | list[str]) -> list[str]:
    """Return `names` as a list, raising when a column is missing from `frame`."""
    columns = [names] if isinstance(names, str) else list(names)
    for name in columns:
        if name not in frame.columns:
            raise ValueError(f"Column '{name}' not found in frame.")
    return columns


def fit_linear_regression(
    frame: pd.DataFrame,
    x_name: str | list[str],
    y_name: str,
    **kwargs: Any,
) -> LinearRegression:
    """Fit a linear regression model on one or more named predictor columns."""
    x_columns = _column_list(frame, x_name)
    _column_list(frame, y_name)

    return LinearRegression(**kwargs).fit(X=frame[x_columns], y=frame[y_name])


def predict_linear_regression(
    model: LinearRegression,
    frame: pd.DataFrame,
    x_name: str | list[str],
) -> pd.Series:
    """Run prediction using a fitted model and the named predictor column(s)."""
    x_columns = _column_list(frame, x_name)

    prediction = model.predict(frame[x_columns])
    return pd.Series(prediction, index=frame.index, name="prediction")


def linear_regression_coefficients(
    model: LinearRegression,
    x_name: str | list[str],
) -> dict[str, float]:
    """Return the fitted coefficient per predictor column, in fit order."""
    x_columns = [x_name] if isinstance(x_name, str) else list(x_name)
    return {name: float(value) for name, value in zip(x_columns, model.coef_, strict=True)}


def linear_regression_diagnostics(
    model: LinearRegression,
    frame: pd.DataFrame,
    x_name: str | list[str],
    y_name: str,
) -> tuple[float, float, float]:
    """Return coefficient, intercept, and R-squared diagnostics.

    With several predictors, the coefficient is that of the first one; use
    `linear_regression_coefficients` for all of them.
    """
    x_columns = _column_list(frame, x_name)
    _column_list(frame, y_name)

    coefficient = float(model.coef_[0])
    intercept = float(model.intercept_)
    r_squared = float(model.score(frame[x_columns], frame[y_name]))

    return coefficient, intercept, r_squared
