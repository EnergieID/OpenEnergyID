"""Generic linear regression helpers for analysis modules."""

from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression


def fit_linear_regression(
    frame: pd.DataFrame,
    x_name: str,
    y_name: str,
    **kwargs: Any,
) -> LinearRegression:
    """Fit a linear regression model on named columns."""
    if x_name not in frame.columns:
        raise ValueError(f"Column '{x_name}' not found in frame.")
    if y_name not in frame.columns:
        raise ValueError(f"Column '{y_name}' not found in frame.")

    return LinearRegression(**kwargs).fit(X=frame[[x_name]], y=frame[y_name])


def predict_linear_regression(
    model: LinearRegression,
    frame: pd.DataFrame,
    x_name: str,
) -> pd.Series:
    """Run prediction using a fitted model and one named input column."""
    if x_name not in frame.columns:
        raise ValueError(f"Column '{x_name}' not found in frame.")

    prediction = model.predict(frame[[x_name]])
    return pd.Series(prediction, index=frame.index, name="prediction")


def linear_regression_diagnostics(
    model: LinearRegression,
    frame: pd.DataFrame,
    x_name: str,
    y_name: str,
) -> tuple[float, float, float]:
    """Return coefficient, intercept, and R-squared diagnostics."""
    if x_name not in frame.columns:
        raise ValueError(f"Column '{x_name}' not found in frame.")
    if y_name not in frame.columns:
        raise ValueError(f"Column '{y_name}' not found in frame.")

    coefficient = float(model.coef_[0])
    intercept = float(model.intercept_)
    r_squared = float(model.score(frame[[x_name]], frame[y_name]))

    return coefficient, intercept, r_squared
