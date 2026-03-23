"""Long-term PV evolution analysis implementation."""

import pandas as pd

from openenergyid import const
from openenergyid.utils import (
    fit_linear_regression,
    linear_regression_diagnostics,
    predict_linear_regression,
)

from .models import (
    PVLongTermAnalysisInput,
    PVLongTermAnalysisOutput,
    PVRegressionDiagnostics,
    PVYearResult,
)


class LongTermPVAnalyzer:
    """Analyze long-term PV production using a 12-month linear reference model."""

    def _reference_dataset(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the first 12 rows as regression reference period."""
        return frame.sort_index().iloc[:12].copy()

    def analyze(self, input_data: PVLongTermAnalysisInput) -> PVLongTermAnalysisOutput:
        """Run the analysis and return typed yearly metrics and diagnostics."""
        frame = input_data.to_pandas(timezone=input_data.timezone).sort_index()
        year_index = frame.index.year  # type: ignore

        reference = self._reference_dataset(frame)
        regression = fit_linear_regression(
            reference,
            x_name=const.SOLAR_RADIATION,
            y_name=const.ELECTRICITY_PRODUCED,
            positive=True,
            fit_intercept=False,
        )

        coefficient, intercept, r_squared = linear_regression_diagnostics(
            regression,
            reference,
            x_name=const.SOLAR_RADIATION,
            y_name=const.ELECTRICITY_PRODUCED,
        )

        yearly_results: list[PVYearResult] = []
        first_year = int(year_index[0])

        for year, yearly_frame in frame.groupby(year_index):
            if year == first_year and len(yearly_frame) < 12:
                continue

            prediction_series = predict_linear_regression(
                regression,
                yearly_frame,
                x_name=const.SOLAR_RADIATION,
            )

            actual = float(yearly_frame[const.ELECTRICITY_PRODUCED].sum())
            prediction = float(prediction_series.sum())
            error = actual - prediction
            if actual == 0:
                relative_error = 0.0 if prediction == 0 else None
            else:
                relative_error = error / actual

            yearly_results.append(
                PVYearResult(
                    year=int(year),
                    actual_production=actual,
                    predicted_production=prediction,
                    error=error,
                    relative_error=relative_error,
                    complete_year=len(yearly_frame) == 12,
                )
            )

        return PVLongTermAnalysisOutput(
            reference=input_data.reference,
            yearly_results=yearly_results,
            regression_diagnostics=PVRegressionDiagnostics(
                coefficient=coefficient,
                intercept=intercept,
                r_squared=r_squared,
            ),
        )
