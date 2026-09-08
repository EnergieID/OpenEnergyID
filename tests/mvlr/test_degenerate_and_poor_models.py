"""find_best_mvlr must return the best fit — and be honest about its quality.

Regression tests for AB#820. Two failure modes previously surfaced as one
opaque HTTP 400 from the DAE's /generate_regression endpoint:

- **cluster A**: a real fit that misses the caller's rsquared threshold. Today
  `find_best_mvlr` raises; after the fix it returns the fit with
  `is_valid=False`, so callers that deliberately relax the thresholds (e.g. v3's
  plant-performance page, which computes its own quality warning) can render it.
- **cluster B**: a *degenerate* fit whose core statistics are NaN — typically
  `df_resid == 0` after resampling. Today NaN slips through `validate()` (every
  NaN comparison is False) and is then rejected by pydantic. After the fix
  `validate()` rejects non-finite fits, and constructing the result raises the
  typed `DegenerateModelError` with observation counts, so callers can tell
  "not enough signal" from "not enough data".
"""

import math

import numpy as np
import pandas as pd
import pytest

from openenergyid.models import TimeDataFrame
from openenergyid.mvlr import (
    DegenerateModelError,
    MultiVariableRegressionInput,
    find_best_mvlr,
)
from openenergyid.mvlr.mvlr import MultiVariableLinearRegression

PRODUCTION = "energyProduction/solarPhotovoltaic"
SOLAR_REFERENCE = "solarPowerGeneration"


def _input(frame: pd.DataFrame, *, min_rsquared: float = 0.0) -> MultiVariableRegressionInput:
    """Build a MultiVariableRegressionInput; caller sets the validation threshold."""
    return MultiVariableRegressionInput.model_validate(
        {
            "timeZone": "Europe/Brussels",
            "independentVariables": [
                {"name": SOLAR_REFERENCE, "allowNegativeCoefficient": False},
            ],
            "dependentVariable": PRODUCTION,
            "frame": TimeDataFrame.from_pandas(frame).model_dump(),
            "granularities": ["P1D"],
            "allowNegativePredictions": False,
            "validationParameters": {"rsquared": min_rsquared, "f_pvalue": 1.0, "pvalues": 1.0},
        }
    )


def _clean_frame() -> pd.DataFrame:
    """A high-quality synthetic PV frame; adj R² should exceed 0.9."""
    idx = pd.date_range("2025-01-01", periods=90, freq="D", tz="Europe/Brussels")
    reference = 1.0 + (np.arange(len(idx)) % 35) / 8.0
    production = 4.35 * reference + 0.15
    return pd.DataFrame(
        {PRODUCTION: production, SOLAR_REFERENCE: reference},
        index=idx,
    )


def _moderate_frame() -> pd.DataFrame:
    """A real-but-modest positive relationship: adj R² in the low 0.x range."""
    idx = pd.date_range("2025-01-01", periods=60, freq="D", tz="Europe/Brussels")
    rng = np.random.default_rng(seed=7)
    reference = np.abs(rng.normal(loc=5.0, scale=2.0, size=len(idx)))
    production = 1.2 * reference + rng.normal(loc=2.0, scale=6.0, size=len(idx))
    return pd.DataFrame(
        {PRODUCTION: production, SOLAR_REFERENCE: reference},
        index=idx,
    )


def _degenerate_frame() -> pd.DataFrame:
    """Two rows with two parameters (intercept + one exog) → df_resid == 0."""
    idx = pd.date_range("2025-01-01", periods=2, freq="D", tz="Europe/Brussels")
    return pd.DataFrame(
        {PRODUCTION: [3.0, 7.0], SOLAR_REFERENCE: [1.0, 2.0]},
        index=idx,
    )


class TestGoodModel:
    """Regression: an ordinary passing fit is returned unchanged."""

    def test_returns_is_valid_true(self) -> None:
        """A model that meets the thresholds carries is_valid=True."""
        result = find_best_mvlr(_input(_clean_frame(), min_rsquared=0.75))
        assert result.is_valid is True
        assert result.validation_message is None
        assert result.r2_adj > 0.75


class TestClusterA:
    """A fit that misses the caller's rsquared threshold — the majority case."""

    def test_poor_but_real_model_is_returned_with_is_valid_false(self) -> None:
        """A moderate model against a strict threshold gets returned, not raised."""
        result = find_best_mvlr(_input(_moderate_frame(), min_rsquared=0.99))
        assert result.is_valid is False
        assert math.isfinite(result.r2_adj)
        assert 0 < result.r2_adj < 0.99

    def test_validation_message_carries_the_real_number(self) -> None:
        """The message names the actual adjusted R², not the clamped 0.000 of 0.2.0."""
        result = find_best_mvlr(_input(_moderate_frame(), min_rsquared=0.99))
        assert result.validation_message is not None
        assert f"{result.r2_adj:.3f}" in result.validation_message
        assert "0.99" in result.validation_message


class TestClusterB:
    """A fit that is not merely poor but structurally impossible to summarise."""

    def test_degenerate_fit_raises_named_error(self) -> None:
        """df_resid == 0 raises DegenerateModelError, not a bare ValueError."""
        with pytest.raises(DegenerateModelError, match=r"df_resid"):
            find_best_mvlr(_input(_degenerate_frame()))

    def test_error_carries_observation_counts(self) -> None:
        """Callers must be able to read nobs / df_model / df_resid off the error."""
        with pytest.raises(DegenerateModelError) as excinfo:
            find_best_mvlr(_input(_degenerate_frame()))
        assert excinfo.value.nobs == 2
        assert excinfo.value.df_resid == 0
        assert excinfo.value.df_model == 1

    def test_degenerate_error_still_subclasses_ValueError(self) -> None:
        """Existing broad `except ValueError` (e.g. in EnergyID.DAE) keeps working."""
        with pytest.raises(ValueError):
            find_best_mvlr(_input(_degenerate_frame()))


class TestValidateRejectsNonFiniteFits:
    """MultiVariableLinearRegression.validate() no longer silently accepts NaN."""

    def test_nan_fit_is_rejected(self) -> None:
        """The three <> guards would pass a NaN fit; the new non-finite guard blocks it."""
        frame = _degenerate_frame()
        mvlr = MultiVariableLinearRegression(
            data=frame,
            y=PRODUCTION,
            granularity="P1D",
        )
        mvlr.do_analysis()
        # Sanity check that we've reproduced the NaN regime.
        assert math.isnan(float(mvlr.fit.f_pvalue))
        # And validate() correctly refuses it, regardless of thresholds.
        assert mvlr.validate(min_rsquared=0.0, max_f_pvalue=1.0, max_pvalues=1.0) is False
