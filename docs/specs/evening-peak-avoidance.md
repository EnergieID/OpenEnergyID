# Evening Peak Avoidance ("Avondpiek mijden")

## Problem Statement

Energie van Utrecht runs the campaign *Piek Mijden Huishoudens Utrecht* in the winter of
2026–2027. Around 600 households in the Utrecht congestion area are asked to shift electricity
use out of the 16:00–21:00 window during November through February, easing congestion on the
provincial grid. Stedin and the Province of Utrecht co-fund it; Stedin treats it explicitly as a
pilot.

A nudge campaign only works if participants can see how they are doing. The platform therefore
needs an analysis that answers two different questions:

| Question | Quantity |
|---|---|
| How hard does this connection load the grid at its worst moment? | the highest quarter-hour power in the window, in kW |
| How much of the day sits in the evening — what could actually be shifted? | the share of net daily offtake falling in the window, in percent |

Neither existing analysis answers them. The capacity analysis finds the single highest
quarter-hour per month across the whole day, with no notion of a window; the baseload analysis
finds the continuous minimum, which is the opposite end of the distribution.

**Why both quantities, and not just the peak.** The peak is the grid's story and the share is
the participant's. They diverge routinely: an oven at 19:00 on an otherwise quiet day makes a
tall peak with a middling share, while a long low evening on a day when nothing else ran makes a
high share with no peak worth mentioning. Spreading consumption across the day is good; pulling
it all into one moment is not. Showing only the peak would reward a household for simply using
less, rather than for shifting.

## Solution

A new `openenergyid.evening_peak` package. Given the two gross meter registers of one
connection, it returns per-day peaks and shares, weekly medians as a reference line, and the
highest peaks with the curve of their own day.

The unit of analysis is a **single connection**. Aggregating across participants — for the
monthly interim reports and the final settlement — happens outside this library, on exported
data.

### Contract

Decided with the front-end team, and fixed:

- **Two gross series**, `grossOfftake` and `grossInjection`, each in **kWh per quarter-hour**.
  This is what `energyBalanceByCarrier` at `PT15M` already returns, and what the smart meter's
  P4 port delivers: two non-negative registers rather than one signed net value.
- **`grossInjection` is optional.** A connection without production omits it.
- **Timestamps label the start of the interval.** The window is half-open,
  `[window_start, window_end)`, so the default covers the twenty quarter-hours 16:00 … 20:45.
- Power in kW is `kWh × 4`.

### Injection is clipped per quarter-hour, before summation

```python
net = max(gross_offtake - gross_injection, 0)   # per quarter-hour
```

This is the whole reason the share behaves. Clipping after summation would let a sunny midday
offset an evening peak and produce a share above 100% — or a negative denominator. Clipping per
quarter-hour keeps the share in 0–100% and comparable between households with and without solar
panels, which is what makes a league table across participants meaningful at all.

## API

```python
class EveningPeakAnalyzer:
    def __init__(
        self,
        timezone: str,
        window_start: dt.time = dt.time(16, 0),
        window_end: dt.time = dt.time(21, 0),
        peak_share_threshold: float = 0.37,
        min_day_coverage: float = 0.9,
    ):
        """
        timezone
            IANA timezone of the connection. Day boundaries and the window are
            evaluated against this local wall clock.

        window_start, window_end
            The evening window, half-open as [window_start, window_end).

        peak_share_threshold
            Share below which a day counts as a good day, as a fraction. 0.37 is the
            peak share of an average Dutch household. One fixed value for all
            participants, configured centrally; adjustable in consultation, but not
            settable per workspace.

        min_day_coverage
            Minimum fraction of a day's quarter-hours required before its share is
            reported.
        """

    def prepare_net_offtake(self, gross_offtake_lf, gross_injection_lf=None) -> pl.LazyFrame
    def analyze(self, net_lf) -> EveningPeakAnalysisResult
    def peak_moments(self, net_lf, num_peaks=10) -> list[PeakMoment]
```

`EveningPeakInput` and `EveningPeakOutput` are the HTTP bodies of the Data Analytics Engine
endpoint. Their field descriptions are the front end's documentation, via the generated OpenAPI
schema, so they are written for that audience rather than as internal notes.

The threshold is echoed back in the response as `thresholdInPercent`, so a reader of the numbers
knows what "below threshold" meant without having to know the configuration.

## Implementation Details

Polars and pandera internally, following `openenergyid/baseload`; pydantic `TimeSeries` models at
the edge, following `openenergyid/capacity`, because that is what the DAE consumes directly.

### Three things that are easy to get wrong

**1. Minute-of-day arithmetic overflows.** Polars' `dt.hour()` and `dt.minute()` return `Int8`,
so the natural `hour * 60 + minute` silently wraps: 16:00 becomes −34, the window matches no
quarter-hours at all, and the daily totals stay perfectly plausible. The window predicate casts
to `Int32` before multiplying. This produced correct-looking output with a zero-length window on
the first run, and is exactly the class of bug that reaches production.

**2. A local day is not always 96 quarter-hours.** Under `Europe/Amsterdam` the October
fall-back day has 100 and the March spring-forward day has 92. Day coverage is therefore measured
against a computed expectation:

```python
expected_quarters = (day.dt.offset_by("1d") - day).dt.total_minutes() // 15
```

Both `dt.truncate("1d")` and `offset_by("1d")` are DST-correct on a timezone-aware column, but
only if the conversion to local time happens first. A constant 96 would have marked every
October fall-back day incomplete and dropped it from the counts.

European DST transitions happen at 02:00/03:00 local time and so never fall inside an evening
window, which is why `expected_window_quarters` can be derived from the wall-clock window length
alone. A window that did straddle a transition would need that computed per day.

**3. Pandera skips value checks on a LazyFrame.** Validating a `LazyFrame` checks columns and
dtypes only; `ge`/`le` checks run on a collected `DataFrame`. The per-day and per-week results
are therefore validated eagerly — they are one row per day, so collecting costs nothing — and
that is where `share <= 100` actually fires. It is the check that would catch a regression in the
clipping.

Timestamp columns are deliberately **not declared** in the schemas: pandera's Polars `DateTime`
type cannot express "a datetime in any time zone", so declaring it either rejects the frames the
analysis produces or, with coercion on, silently strips the time zone the whole analysis depends
on. Correct local-time behaviour is asserted by tests instead.

### Incomplete days

Peak and share fail independently, because they need different things:

| Condition | `evening_peak_in_kilowatt` | `evening_peak_share_in_percent` |
|---|---|---|
| Window fully measured, day coverage met | value | value |
| Window fully measured, day too sparse | value | null |
| Any window quarter-hour missing | null | null |
| No offtake at all that day | 0.0 | null |

Without the day-coverage rule, the partial first and last day of any export produce a share
against an incomplete denominator, which reads as a very good or very bad day when it is neither.

`measured_days` counts days with a reported share, and is the denominator of the "49 of 120 days"
figure in the design. Late entrants simply have fewer measured days — there is no closing date
for the campaign, so this has to degrade gracefully rather than be an error.

### A gapless daily index

The daily result covers every calendar day between the first and last measurement. A day with no
measurements at all is present with null metrics and `observed_quarters == 0`, rather than
missing from the index. A chart drawn straight from the series then breaks where the data breaks;
were the rows simply absent, a client would draw a straight line across a three-day outage and
show it as normal consumption.

### Weekly medians

Monday-aligned via `group_by_dynamic(every="1w", start_by="monday")`, computed over complete days
only, and keyed by the Monday of each week — the same shape as baseload's
`monthly_median_baseloads`. A partial first week still keys to its Monday. Render as a step line.

### Peak moments

One peak per day by construction, so — unlike `CapacityAnalysis.find_peaks_with_surroundings`,
which takes the largest quarter-hours across the whole series and must then discard neighbours of
an already-chosen peak — no de-duplication is needed. Each moment carries the **full local day**
at quarter-hour resolution, not just the surrounding hour, because the sparkline's job is to show
whether the peak was an isolated spike or the top of a long evening.

## Testing Strategy

`tests/evening_peak/`, 101 tests. Data is synthesised in code from a shared builder in
`conftest.py` that walks real time and converts back, so DST-affected days come out with the
right number of quarter-hours.

Beyond the arithmetic, the cases worth having:

- a midday spike larger than the whole evening does not become the evening peak, but does lower
  the share
- clipping is per quarter-hour: 5 kWh of midday injection does not offset 1 kWh of evening
  offtake
- the 25-hour October day yields a *lower* share than its neighbours, because the extra hour of
  baseload lands in the denominator
- the same instants split into one day under `Europe/Amsterdam` and two under `Europe/Lisbon`
- a day of zeroes yields a null share rather than a division error
- the schema rejects a share above 100%, and validation leaves the time zone intact

## Files

| Path | |
|---|---|
| `openenergyid/evening_peak/analysis.py` | `EveningPeakAnalyzer`, `EveningPeakAnalysisResult`, `PeakMoment`, `summarize` |
| `openenergyid/evening_peak/models.py` | `EveningPeakInput`, `EveningPeakOutput` and friends |
| `openenergyid/evening_peak/schemas.py` | pandera result guards |
| `tests/evening_peak/` | `conftest.py`, `test_analysis.py`, `test_models.py` |
| `data/evening_peak/make_sample.py` | seeded generator for the demo fixture |
| `data/evening_peak/evening_peak_sample.json` | one campaign winter, synthetic |
| `demo_evening_peak.ipynb` | demo, registered in `tests/nb.py` |

## Demo data

Synthetic, and deliberately so: the demo fixture is committed to a public repository, and a real
household's quarter-hourly consumption is personal data. It is generated from appliance events
rather than from an average profile — a standing base load plus discrete events, with one evening
archetype per day (cooking 3–5 kW, an ordinary evening in at 1–2 kW, nobody home at ~150 W,
occasionally something heavy overlapping the window).

That jaggedness is the point rather than decoration. A smoothed profile would hide the day-to-day
variation the participant is meant to act on, and would let a front end be built against a narrow
band of values that never occurs in practice. The generator is tuned so the mean peak share lands
near 37%, with the spread coming from the archetypes: median 36.5%, 10th percentile 26%, 90th
percentile 50%, peaks from 0.19 to 5.64 kW, and 60 of 117 measured days below the threshold.

Validation against real data was done separately, against a personal EnergyID record over two
winters, and is not committed.

## Out of scope

- **The Trends heatmap** (card 3 of the design). The caller already holds the quarter-hourly data
  it sent; returning a day × quarter-hour matrix would send the same ~11k values back. The front
  end builds it, as it already does for the capacity analysis.
- **Aggregation across participants**, the monthly export and the incentive calculation. This
  library answers for one connection.
- **Deciding who is rewarded.** The analysis reports the share and counts the days under the
  threshold; what is rewarded, how much and in what form is EvU's call.
- **Restricting the analysis to Energie van Utrecht workspaces.** That is a platform-level gate,
  not something this library or the endpoint can enforce — the endpoint analyses whatever series
  it is given.

## Backward Compatibility

Entirely additive: a new package, a new set of models, no change to any existing module. Released
as `0.1.42`.
