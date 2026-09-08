# MVLR: distinguishing degenerate fits from poor fits

## Problem Statement

`find_best_mvlr` currently raises `ValueError("No valid model found. Best R²: X (need ≥Y)")`
whenever it cannot return a model. That single exit path swallows four different situations
and reports them all as the same string to the caller.

Observed in production (`func-energyid-dae-py-prod` logs, 4 days):

| Cause | Log message | Occurrences | Real category |
| --- | --- | ---: | --- |
| `df_resid == 0` after resampling | `2 validation errors for MultiVariableRegressionResult` | 12 | degenerate — no model exists |
| adjusted R² finite, below caller's threshold | `Best R²: 0.670 (need ≥0.75)` | 1 | poor-but-real model |
| adjusted R² negative or NaN, clamped to 0.000 | `Best R²: 0.000 (need ≥0.0)` | 16 | mostly degenerate |
| statistical fit was found but its stats are NaN | (same message) | included above | degenerate |

Two downstream effects from EnergyID's v3 monolith:

1. **The `plant-performance` callers pass `minimumRSquared: 0`** so a weak model can be
   rendered with an explanatory warning by the SPA. That relaxed path currently cannot
   succeed for any of the failure modes above.
2. **The nightly `BenchmarkingEngine` batch treats a null model as "invalid" and *deletes*
   the cached `plant-performance-model` `RecordConfig`**, so records that could support a
   weak model repeatedly lose their cache entry every night.

See [AB#820](https://dev.azure.com/energyid/EnergyID/_workitems/edit/820) (opened against
the DAE + this library), which links [AB#812](https://dev.azure.com/energyid/EnergyID/_workitems/edit/812)
(user-visible symptom in v3) and [AB#667](https://dev.azure.com/energyid/EnergyID/_workitems/edit/667)
(v3's DAE DTOs are non-nullable `double`, which constrains what this library may emit).

## Solution

Split the two categories at the source, in this library, and make `find_best_mvlr` return
the best available fit rather than deciding for the caller.

- **Poor-but-real model** — adjusted R² is finite (possibly negative) but below the
  caller's threshold: return the fit with `is_valid=False` and a `validation_message`
  stating the real numbers. Callers that pass a strict threshold to reject weak models get
  the same information they used to (via `is_valid`) plus the ability to render the model
  themselves if they choose to.
- **Degenerate fit** — one or more of `r2`, `r2_adj`, `f_stat`, `prob_f_stat` is not
  finite (typically because `df_resid == 0` after resampling coarsened the frame past the
  number of parameters): raise a typed `DegenerateModelError(ValueError)` carrying `nobs`,
  `df_model` and `df_resid` so the caller can explain what went wrong.

`DegenerateModelError` subclasses `ValueError` so existing `except ValueError` catches
(notably in `EnergyID.DAE/src/fastapi_app.py`) keep working unmodified. New callers can
narrow the catch to distinguish causes.

## API Changes

### `MultiVariableRegressionResult`

Two new fields, both additive (defaults preserve existing serialisation):

```python
is_valid: bool = Field(default=True, alias="isValid")
validation_message: str | None = Field(default=None, alias="validationMessage")
```

The lower bound on `r2_adj` is dropped:

```python
r2_adj: float = Field(le=1, alias="rSquaredAdjusted")   # was ge=0, le=1
```

A negative adjusted R² is the well-defined value for a model that fits worse than the mean;
constraining it to `[0, 1]` made the "relaxed threshold" path unable to return anything.
`r2` remains `ge=0, le=1` (it is bounded by construction).

**AB#667 constraint:** `f_stat` and `prob_f_stat` remain required `float` and non-nullable —
`EnergyID.Core/Models/PythonAnalytics/RegressionModelResult.cs` declares them as
non-nullable `double` in v3, and emitting `null` there would trigger exactly the
`JsonSerializationException` documented on AB#667. A negative or unusual `double`
deserializes fine; `null` does not.

### `MultiVariableRegressionResult.from_mvlr`

Now takes two keyword-only arguments and raises `DegenerateModelError` if the core
statistics are not finite:

```python
@classmethod
def from_mvlr(
    cls,
    mvlr: MultiVariableLinearRegression,
    *,
    is_valid: bool = True,
    validation_message: str | None = None,
) -> "MultiVariableRegressionResult":
    ...
```

### `MultiVariableLinearRegression.validate`

Explicitly rejects a non-finite fit before its three `<`/`>` guards. Previously every
NaN comparison was `False`, so a degenerate fit silently satisfied all three checks and
returned `True`.

### `find_best_mvlr`

Same signature. The behaviour change:

- If at least one granularity's fit passes `validate()`, return the first passing fit
  (unchanged).
- Otherwise, if at least one granularity produced a representable fit, return the one with
  the highest adjusted R² and `is_valid=False` — instead of raising.
- Only if *every* granularity's fit is degenerate does it raise (now
  `DegenerateModelError` rather than a bare `ValueError`).

## Implementation Details

- `find_best_mvlr` now seeds its tracker with `-math.inf`, so negative adjusted R² values
  are ranked correctly rather than being masked by `max(0, ...)`.
- `_finite_or_none` on `IndependentVariableResult` is unchanged (a pre-existing pattern for
  the variable-level stats).
- No new dependencies.

## Testing Strategy

`tests/mvlr/test_degenerate_and_poor_models.py` covers:

- **TestGoodModel** — regression: a good fit still returns `is_valid=True`.
- **TestClusterA** — a moderate fit against a strict threshold returns with
  `is_valid=False`; `validation_message` carries the real adjusted R² and the threshold.
- **TestClusterB** — a 2-row frame with 2 parameters produces `df_resid == 0`; the raise
  is `DegenerateModelError`, carries `nobs`/`df_model`/`df_resid`, and still satisfies a
  broad `except ValueError`.
- **TestValidateRejectsNonFiniteFits** — the `validate()` non-finite guard.

## Files

- `openenergyid/mvlr/models.py` — new `DegenerateModelError`, relaxed `r2_adj` bound,
  two new fields, `from_mvlr` guard
- `openenergyid/mvlr/mvlr.py` — non-finite check in `validate()`
- `openenergyid/mvlr/main.py` — rewrite of `find_best_mvlr` (35 → 79 lines)
- `openenergyid/mvlr/__init__.py` — re-export `DegenerateModelError`
- `tests/mvlr/__init__.py`, `tests/mvlr/test_degenerate_and_poor_models.py` — new
- `docs/specs/mvlr-degenerate-models.md` — this file
- `README.md` — one paragraph under the MVLR section

## Out of Scope

- Changes to v3 (`EnergyID.Core/Services/Impl/DataAnalyticsEngine.cs` discards the DAE
  error body; `RegressionModelResult.cs` DTOs are non-nullable) — recorded on AB#820
  for the v3 owners.
- The DAE's `except ValueError` in `src/fastapi_app.py` — will land in a separate DAE
  hotfix once this library is released, since it depends on `DegenerateModelError`.
- `resample_input_data` fabricating `0.0` for gap periods rather than NaN
  (`mvlr/helpers.py:28`) — a broader behaviour change touching every analysis, needs its
  own spec.

## Backward Compatibility

- **Wire format**: `isValid` and `validationMessage` are additive; existing clients
  ignore them and continue to work. `rSquaredAdjusted` can now be negative, which is a
  new value range for consumers to expect.
- **Python API**: `find_best_mvlr(data)` keeps its signature and return type.
- **Behavioural**: `find_best_mvlr` no longer raises for a poor-but-real model. Callers
  that today catch that raise and treat it as "no model" should check `result.is_valid`
  instead.
- **Exception type**: the degenerate-fit path now raises `DegenerateModelError`, still a
  `ValueError` subclass.
