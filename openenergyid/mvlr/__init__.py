"""Multi-variable linear regression (MVLR) module."""

from .main import find_best_mvlr
from .models import (
    IndependentVariableInput,
    IndependentVariableResult,
    MultiVariableRegressionInput,
    MultiVariableRegressionResult,
    OutlierFilteringDiagnostics,
    SourceDataFilteringParameters,
    ValidationParameters,
)
from .source_data_filtering import clean_regression_frame

__all__ = [
    "find_best_mvlr",
    "clean_regression_frame",
    "IndependentVariableInput",
    "MultiVariableRegressionInput",
    "MultiVariableRegressionResult",
    "OutlierFilteringDiagnostics",
    "SourceDataFilteringParameters",
    "ValidationParameters",
    "IndependentVariableResult",
]
