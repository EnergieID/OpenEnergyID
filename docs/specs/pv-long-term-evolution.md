# Long-term PV evolution

## Problem Statement

A PV system's production varies from year to year mainly because the weather varies. To see
whether a system itself is performing worse — degradation, soiling, a failing string, an
inverter that trips — the weather has to be taken out first. `openenergyid.pv_evolution`
answers one question per calendar year:

> Given the irradiance this year, how much should this system have produced, judged by how it
> behaved in its own reference period, and how far was it off?

The system's own history is the reference, so the analysis needs no metadata: no peak power,
tilt, azimuth or shading description.

## Method

### 1. Daily input

One row per local calendar day, with:

| Column | Meaning | Required |
| --- | --- | --- |
| `electricity_produced` | PV production | yes |
| `solar_radiation` | irradiance on a tilted plane (typically 35° facing south) | yes |
| `solar_radiation_horizontal` | global horizontal irradiance, same unit | no |

A day without irradiance cannot be predicted, so it is dropped from both the actual and the
predicted totals. When `solar_radiation_horizontal` is used, a day needs both series; the days
that have the tilted series but not the horizontal one are reported in a
`horizontal_irradiance_incomplete` warning, since a horizontal series that starts late removes
whole years. A day without production is left out of the reference fit, but counts as zero in the yearly totals
(an outage is real lost production).

### 2. Reference period and buckets

The reference period is the first 12 months of data, or an explicit `baselineStart` /
`baselineEnd`. Its usable days are summed into about twelve month-like buckets (25–35 days
each; true calendar months when the period is exactly one complete year). Daily data is too
noisy to fit on directly. Monthly sums average out clouds and timing, and still keep the
seasonal shape that the model has to learn.

### 3. The model

Production per bucket is fitted as a weighted sum of irradiance, with no intercept and with
non-negative weights:

```
P = a · G_tilted                      (tilted series only)
P = a · G_tilted + b · G_horizontal   (both series)
a, b ≥ 0
```

This is non-negative least squares (`LinearRegression(positive=True, fit_intercept=False)`).
No intercept, because no light means no production. Non-negative, because a system cannot
produce less when there is more light.

### 4. Yearly result

Each calendar year's monthly irradiance sums go through the fitted model to give the predicted
production. The output per year is:

- `error = actual − predicted`
- `relativeError = error / predicted`, so −1 means no production at all
- `completeYear`, false for the current (incomplete) year or a year with dropped days

The first, partial year is skipped. A complete calendar year that is the reference period
itself is skipped as well, since it would trivially match.

`regressionDiagnostics` reports the weights (`coefficient` = `a`, `coefficientHorizontal` =
`b`, or null without the horizontal series), the intercept (always 0) and the R² of the fit on
the reference buckets.

## Why two irradiance series

### The bias of one tilted series

With only `solar_radiation`, the weight `a` is a single ratio of production to irradiance for
the whole year. That assumes the system follows the seasonal shape of a 35° south-facing
plane. That plane gains a lot in winter from its tilt toward the low sun, so its seasonal
curve is flatter than what many real systems see. Flatter roofs, east/west layouts, and
systems shaded in winter all produce relatively less in winter than the tilted series implies.

For a complete year this cancels: `a` was fitted on a complete year, so the winter
over-prediction and the summer under-prediction sum to roughly zero. For an incomplete year
it does not cancel. After January and February, the prediction is mostly winter months and
too high, and the year-to-date result reads far too low. By late summer it has swung the other
way.

### The mix

Horizontal irradiance has the opposite shape: strong in summer, weak in winter. A fitted mix
`a · G_tilted + b · G_horizontal` can produce any seasonal curve between the two. The ratio
`a : b` is chosen per system from its own reference period. In effect this is an orientation
fit, without asking anyone for the tilt or azimuth.

The weights are not a physical orientation and should not be read as one. They are the
combination of the two daily series that best reproduces the system's seasonal profile.

### Why the weights must be non-negative

The two series are strongly correlated (both follow the sun). With only about twelve buckets,
an unconstrained fit can choose a large positive weight on one and a large negative weight on
the other. These cancel on the reference period and extrapolate badly to any other year. With
the non-negativity constraint, the worst case is that one weight is zero. That is the
single-series model, so the mix can only match or improve on it within the reference period.

### Evidence

Evaluated on 215 systems with daily data from 2019 to 2025, comparing each
year-to-date result with the full-year result for the same year (median gap):

| Year-to-date after | Tilted series only | Tilted + horizontal |
| --- | ---: | ---: |
| February | −22% | −7% |
| April | −6% | −1% |
| August | +3% | +0.5% |

Complete-year results barely change. A full fit on hourly east, south and west plane-of-array
irradiance gave the same improvement, so the two daily series are enough.

## What this does not fix

- **Long-term drift.** A slow decline over the years is what the analysis is meant to show,
  whether it comes from the panels or from a trend in the irradiance source.
- **Changes to the installation.** Extra panels, a replaced inverter or a meter that changes
  scope break the comparison with the reference period. The analysis assumes one unchanged
  system.
- **Short incomplete years.** Two months of data remain noisier than a full year. The mix
  removes most of the systematic bias (see the table above), not the noise.

## Backward compatibility

`solar_radiation_horizontal` is optional. Without it, or when it holds no values, the model is
the single-series fit and the output is identical to before, with
`coefficientHorizontal: null`.
