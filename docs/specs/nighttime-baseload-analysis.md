# Nighttime-Only Baseload Analysis

## Problem Statement

When a user has PV (solar panels) but no production meter, the electricity import data is misleading during daylight hours. The import shows less than actual consumption because unmeasured PV production covers part of the load. This makes baseload calculation inaccurate.

**Current workaround (C# frontend):**
- Filter to only nighttime hours (23:00-05:00) before sending to backend
- Hardcoded hours don't account for seasonal sunrise/sunset variation

**Why this fails:**

| Season | Brussels Sunset | Brussels Sunrise | Hardcoded 23:00-05:00 |
|--------|-----------------|------------------|----------------------|
| Summer | ~21:45 | ~05:45 | Includes 1h+ of daylight at both ends |
| Winter | ~16:30 | ~09:00 | Misses 6+ hours of valid nighttime data |

## Solution

Add a `nighttime_only` mode to `BaseloadAnalyzer` that uses astronomical calculations (via pvlib) to determine actual nighttime for each day in the dataset.

## API Changes

### BaseloadAnalyzer Constructor

```python
class BaseloadAnalyzer:
    def __init__(
        self,
        timezone: str,
        quantile: float = 0.05,
        nighttime_only: bool = False,
        location: tuple[float, float] | None = None,
        solar_elevation_threshold: float = 0.0,
    ):
        """
        Parameters
        ----------
        timezone : str
            Timezone for analysis (e.g., "Europe/Brussels").

        quantile : float, default=0.05
            Percentile for baseload detection (5% = lowest ~72 min/day).

        nighttime_only : bool, default=False
            If True, filter to only nighttime readings before analysis.
            Use this when PV production exists but is not metered.

        location : tuple[float, float] | None, default=None
            Latitude and longitude for solar position calculation.
            Only used when nighttime_only=True.
            If None, defaults to Brussels (50.85, 4.35).

        solar_elevation_threshold : float, default=0.0
            Sun elevation angle (degrees) below which is considered night.
            - 0.0: Geometric sunset (sun at horizon)
            - -6.0: Civil twilight (recommended for PV, sky visibly dark)
            - -12.0: Nautical twilight (conservative)
        """
```

### Behavior

1. **Default behavior unchanged**: `nighttime_only=False` produces identical results to current implementation.

2. **Nighttime filtering**: When `nighttime_only=True`, readings are filtered to only include timestamps where solar elevation < threshold before any analysis.

3. **Location defaulting**: If `nighttime_only=True` and `location=None`, Brussels coordinates (50.85, 4.35) are used as default.

## Implementation Details

### Solar Position Calculation

Use `pvlib.solarposition.get_solarposition()` which is already a project dependency:

```python
import pvlib
import pandas as pd

def _calculate_night_mask(
    timestamps: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    elevation_threshold: float,
) -> pd.Series:
    """Return boolean mask where True = nighttime."""
    solar_pos = pvlib.solarposition.get_solarposition(
        timestamps, latitude, longitude
    )
    return solar_pos["elevation"] < elevation_threshold
```

### Performance Optimization

Solar position calculation is deterministic for a given location and timestamp. For a year of 15-minute data (~35,000 rows), the calculation takes <1 second. No caching needed for typical use cases.

### Filtering Integration

The nighttime filter is applied early in the `analyze()` method, before daily baseload calculation:

```python
def analyze(self, power_lf: pl.LazyFrame, reporting_granularity: str = "1h"):
    power_lf = self._ensure_power_frame(power_lf)

    # Apply nighttime filter if enabled
    if self.nighttime_only:
        power_lf = self._apply_nighttime_filter(power_lf)

    # Continue with existing analysis...
```

## Sparse Data Considerations

### The Challenge

Night duration varies significantly by season. Brussels data (50.85°N, 4.35°E):

| Date | Geometric (0°) | Civil (-6°) | Nautical (-12°) |
|------|----------------|-------------|-----------------|
| Jan 15 (winter) | 16.0h / 64 readings | 14.5h / 58 readings | 13.0h |
| Mar 20 (equinox) | 12.0h / 48 readings | 10.8h / 43 readings | 9.5h |
| Jun 21 (summer) | 7.8h / 31 readings | 5.8h / 23 readings | 3.8h |
| Sep 22 (equinox) | 12.0h / 48 readings | 10.8h / 43 readings | 9.5h |
| Dec 21 (winter) | 16.2h / 65 readings | 14.8h / 59 readings | 13.2h |

With `quantile=0.05`:
- Summer night (civil): 23 readings → 1 reading defines baseload
- Winter night (civil): 58 readings → 2-3 readings define baseload

### Why This Still Works

1. **Median aggregation is robust**: The daily baseload is the 5th percentile of that night's readings. Even with 23 readings, this captures the lowest sustained consumption.

2. **Monthly/global medians smooth noise**: Individual night variations are smoothed by taking medians across many nights.

3. **Alternative approaches investigated**: Mathematical research into better baseload detection methods without submetering found that simple quantile + median approach is effective. More sophisticated methods require additional data (e.g., separate PV meter, appliance signatures).

4. **Practical validation**: The C# frontend has been using hardcoded nighttime filtering successfully. This enhancement makes it more accurate, not fundamentally different.

### Minimum Data Requirements

The existing `_empty_result()` handling applies: if filtering results in zero valid readings, an empty result is returned. Callers should check for this.

## Metric Validity

When `nighttime_only=True`, all metrics are computed from nighttime data only:

| Metric | Interpretation |
|--------|----------------|
| `global_median_baseload` | True baseload (no PV contamination) |
| `monthly_median_baseload_in_watt` | True monthly baseload |
| `average_daily_baseload_in_watt` | True baseload per reporting period |
| `total_consumption_in_kilowatthour` | Nighttime consumption only |
| `baseload_ratio` | Nighttime baseload ratio (still valid) |
| `consumption_not_due_to_baseload_in_kilowatthour` | Nighttime non-baseload only |

The caller knows they're using nighttime-only mode and should interpret consumption totals accordingly.

## Examples

### Basic Usage

```python
# Standard analysis (full day)
analyzer = BaseloadAnalyzer(timezone="Europe/Brussels")
result = analyzer.analyze(power_data, "1mo")

# Nighttime-only for homes with unmeasured PV
analyzer = BaseloadAnalyzer(
    timezone="Europe/Brussels",
    nighttime_only=True,
)
result = analyzer.analyze(power_data, "1mo")
```

### With Custom Location

```python
# Amsterdam location with civil twilight threshold
analyzer = BaseloadAnalyzer(
    timezone="Europe/Amsterdam",
    nighttime_only=True,
    location=(52.37, 4.90),
    solar_elevation_threshold=-6.0,
)
```

### Conservative Threshold

```python
# Very conservative - only analyze when sky is fully dark
analyzer = BaseloadAnalyzer(
    timezone="Europe/Brussels",
    nighttime_only=True,
    solar_elevation_threshold=-12.0,  # Nautical twilight
)
```

## Testing Strategy

### Unit Tests

1. **Filter accuracy**: Verify that known timestamps are correctly classified as day/night for Brussels summer/winter solstices.

2. **Threshold behavior**: Test that different elevation thresholds produce expected hour counts.

3. **Default location**: Verify Brussels is used when location=None.

4. **Backward compatibility**: Verify `nighttime_only=False` produces identical results to current implementation.

### Integration Tests

1. **Real data comparison**: Run analysis with and without nighttime filtering on data known to have PV impact. Nighttime-only should show higher baseload.

2. **Empty result handling**: Test with data that has no nighttime readings (edge case).

## Files to Modify

| File | Changes |
|------|---------|
| `openenergyid/baseload/analysis.py` | Add parameters, filtering method |
| `openenergyid/baseload/__init__.py` | No changes needed |
| `tests/baseload/test_analysis.py` | Add nighttime-only tests |

## Dependencies

- `pvlib>=0.13.0` - Already in pyproject.toml, no new dependencies required.

## Backward Compatibility

100% backward compatible:
- All new parameters have defaults
- `nighttime_only=False` (default) produces identical behavior
- No changes to return types or result structure

## Recommended Defaults

For typical Belgian/Dutch residential use with unmeasured PV:

```python
analyzer = BaseloadAnalyzer(
    timezone="Europe/Brussels",
    quantile=0.05,
    nighttime_only=True,
    # location defaults to Brussels (50.85, 4.35)
    # solar_elevation_threshold defaults to 0.0 (geometric sunset)
)
```

**Threshold recommendations:**
- `0.0` (default): Safe choice, captures all hours when sun is below horizon
- `-6.0`: Civil twilight - more conservative, ensures no residual PV production. Use if data shows anomalies around sunrise/sunset.

The geometric threshold (0.0) is recommended as default because:
1. PV panels produce negligible power when sun is at/below horizon
2. It maximizes available nighttime data, especially in summer
3. Simple to understand and explain

## C# Frontend Migration

Once implemented, the C# frontend can be simplified:

**Before (hardcoded hours):**
```csharp
// Filter to 23:00-05:00 before sending to DAE
var nighttimeData = data.Where(x => x.Timestamp.Hour >= 23 || x.Timestamp.Hour < 5);
```

**After (let Python handle it):**
```csharp
// Send all data, let DAE filter astronomically
var allData = data;  // No filtering needed
// API call with nighttime_only=true parameter
```

This moves the logic to a single source of truth and eliminates the seasonal accuracy issues.
