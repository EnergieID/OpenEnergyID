"""Generate the synthetic demo dataset for the evening peak avoidance analysis.

Run from the repository root::

    uv run python data/evening_peak/make_sample.py

Writes ``evening_peak_sample.json``, an ``EveningPeakInput`` payload covering one
campaign winter (1 November 2026 - 28 February 2027) of quarter-hourly gross offtake
and injection for a single Dutch household.

Why synthetic, and why jagged
-----------------------------
A real household is not a smooth curve with noise on top. It is a standing base load
plus a handful of discrete appliances switching on and off, so consecutive days differ
a great deal: an oven at 19:00 draws 3-5 kW, an evening where nobody is home draws
barely more than the fridge. The demo data is therefore assembled from appliance
events rather than from an average profile, because the whole point of the analysis is
the day-to-day variation a participant can act on. Tuning the spread down to something
tidy would hide exactly that.

The generator is seeded, so the committed fixture is reproducible.
"""

import datetime as dt
import json
import random
from pathlib import Path
from zoneinfo import ZoneInfo

TIMEZONE = "Europe/Amsterdam"
ZONE = ZoneInfo(TIMEZONE)
FIRST_DAY = dt.date(2026, 11, 1)
LAST_DAY = dt.date(2027, 2, 28)
QUARTERS_PER_HOUR = 4
SEED = 20261101

# Scales the winter heating load, which runs mostly outside the evening window and
# so lowers the peak share without touching the evening peaks. Together with the
# standing load and the daytime chores it is tuned to put the mean peak share on the
# 37% of an average Dutch household; the day-to-day spread comes from the evening
# archetypes below, not from here.
HEATING_INTENSITY = 0.10

# Days dropped entirely, standing in for a gap in the P4 feed.
GAP = (dt.date(2027, 1, 12), dt.date(2027, 1, 13), dt.date(2027, 1, 14))

OUTPUT = Path(__file__).with_name("evening_peak_sample.json")


def quarter_index(hour: int, minute: int = 0) -> int:
    """Index of a local wall-clock time within its day."""
    return hour * QUARTERS_PER_HOUR + minute // 15


def add_event(day_kilowatt: list[float], start: int, quarters: int, kilowatt: float) -> None:
    """Add an appliance drawing ``kilowatt`` for ``quarters`` quarter-hours."""
    for offset in range(quarters):
        position = start + offset
        if 0 <= position < len(day_kilowatt):
            day_kilowatt[position] += kilowatt


def build_day(
    day: dt.date, quarters: int, rng: random.Random, campaign_progress: float
) -> list[float]:
    """Return one day of net household demand, in kW per quarter-hour.

    ``campaign_progress`` runs 0 to 1 across the winter and mildly damps the evening,
    standing in for participants actually shifting load as the campaign goes on.
    """
    weekend = day.weekday() >= 5
    damping = 1.0 - 0.15 * campaign_progress

    # Standing load: fridge, router, standby. Drifts slowly, not per quarter-hour.
    base = rng.uniform(0.15, 0.25)
    day_kilowatt = [base + rng.uniform(-0.02, 0.02) for _ in range(quarters)]

    # Winter heating cycling through the cold hours, largely outside the window.
    for block_start in (
        quarter_index(0, 0),
        quarter_index(4, 0),
        quarter_index(10, 0),
        quarter_index(13, 0),
    ):
        if rng.random() < 0.75:
            add_event(
                day_kilowatt,
                block_start + rng.randint(0, 6),
                rng.randint(4, 10),
                rng.uniform(0.4, 1.1) * HEATING_INTENSITY,
            )

    # Morning: kettle, shower, breakfast. Later and lazier at the weekend.
    morning = quarter_index(9, 0) if weekend else quarter_index(6, 45)
    morning += rng.randint(-2, 2)
    add_event(day_kilowatt, morning, rng.randint(1, 2), rng.uniform(1.8, 2.4))
    add_event(day_kilowatt, morning + 2, rng.randint(2, 4), rng.uniform(0.3, 0.7))

    # Daytime chores, outside the evening window. These enlarge the day's total
    # without touching the evening, so they push the peak share down - one of the
    # honest ways a day lands well below the threshold.
    if rng.random() < 0.60:
        add_event(day_kilowatt, quarter_index(rng.randint(9, 14)), 2, rng.uniform(1.5, 1.9))
    if rng.random() < 0.45:
        add_event(day_kilowatt, quarter_index(rng.randint(11, 15)), 3, rng.uniform(2.2, 2.8))
    if weekend and rng.random() < 0.7:
        add_event(day_kilowatt, quarter_index(rng.randint(10, 15)), 5, rng.uniform(0.9, 1.6))
    # A load deliberately run overnight on the cheap tariff.
    if rng.random() < 0.35:
        add_event(day_kilowatt, quarter_index(rng.randint(1, 4)), 3, rng.uniform(1.6, 2.4))

    # The evening, 16:00-21:00. One archetype per day.
    roll = rng.random()
    evening_start = quarter_index(16, 0)
    if roll < 0.08:
        # Nobody home: barely more than the standing load all evening.
        add_event(day_kilowatt, evening_start, 20, rng.uniform(0.0, 0.05))
    elif roll < 0.45:
        # Cooking on the hob or in the oven, plus the usual lights and screens.
        add_event(day_kilowatt, evening_start, 20, rng.uniform(0.25, 0.5))
        cook = quarter_index(rng.randint(17, 19), rng.choice([0, 15, 30, 45]))
        add_event(day_kilowatt, cook, rng.randint(2, 4), rng.uniform(2.6, 4.4) * damping)
        if rng.random() < 0.4:
            add_event(day_kilowatt, cook - 1, 1, rng.uniform(1.8, 2.2) * damping)
    elif roll < 0.90:
        # An ordinary evening in.
        add_event(day_kilowatt, evening_start, 20, rng.uniform(0.3, 0.6))
        add_event(
            day_kilowatt,
            quarter_index(rng.randint(17, 20), rng.choice([0, 15, 30, 45])),
            rng.randint(1, 3),
            rng.uniform(0.8, 1.7) * damping,
        )
    else:
        # Something heavy overlapping the window: tumble dryer, or a car on charge.
        add_event(day_kilowatt, evening_start, 20, rng.uniform(0.3, 0.6))
        add_event(
            day_kilowatt,
            quarter_index(rng.randint(17, 19)),
            rng.randint(3, 6),
            rng.uniform(3.4, 5.2) * damping,
        )

    # Late evening wind-down, after the window closes.
    add_event(day_kilowatt, quarter_index(21, 0), rng.randint(2, 6), rng.uniform(0.2, 0.5))

    return day_kilowatt


def build_injection(day: dt.date, quarters: int, rng: random.Random) -> list[float]:
    """Return one day of PV injection, in kW per quarter-hour.

    A modest array on a Dutch roof: nothing in November and December worth speaking of,
    a little around midday from late January onwards.
    """
    if day.month not in (1, 2) or (day.month == 1 and day.day < 20):
        return [0.0] * quarters

    seasonal = 0.6 if day.month == 1 else 1.0
    weather = rng.choice([0.0, 0.15, 0.4, 0.8, 1.0])
    peak = 2.6 * seasonal * weather
    if peak == 0:
        return [0.0] * quarters

    sunrise, sunset = quarter_index(9, 0), quarter_index(17, 0)
    midday = (sunrise + sunset) / 2
    half_span = (sunset - sunrise) / 2

    injection = [0.0] * quarters
    for position in range(sunrise, min(sunset, quarters)):
        shape = max(0.0, 1.0 - ((position - midday) / half_span) ** 2)
        injection[position] = round(peak * shape * rng.uniform(0.85, 1.0), 4)
    return injection


def main() -> None:
    """Generate the fixture and report what it contains."""
    rng = random.Random(SEED)
    total_days = (LAST_DAY - FIRST_DAY).days + 1

    index: list[str] = []
    offtake: list[float] = []
    injection: list[float] = []

    for offset in range(total_days):
        day = FIRST_DAY + dt.timedelta(days=offset)
        if day in GAP:
            continue

        midnight = dt.datetime.combine(day, dt.time(0, 0), tzinfo=ZONE)
        next_midnight = dt.datetime.combine(day + dt.timedelta(days=1), dt.time(0, 0), tzinfo=ZONE)
        quarters = round((next_midnight - midnight).total_seconds() / 900)

        progress = offset / (total_days - 1)
        demand = build_day(day, quarters, rng, progress)
        production = build_injection(day, quarters, rng)

        for position in range(quarters):
            timestamp = midnight + dt.timedelta(minutes=15 * position)
            # Self-consumption first: only the surplus reaches the meter, and only the
            # shortfall is taken from the grid. Gross registers are never negative.
            net = demand[position] - production[position]
            index.append(timestamp.isoformat())
            offtake.append(round(max(net, 0.0) / QUARTERS_PER_HOUR, 4))
            injection.append(round(max(-net, 0.0) / QUARTERS_PER_HOUR, 4))

    payload = {
        "timeZone": TIMEZONE,
        "grossOfftake": {"index": index, "data": offtake},
        "grossInjection": {"index": index, "data": injection},
        "windowStart": "16:00",
        "windowEnd": "21:00",
        "peakShareThreshold": 0.37,
        "numPeakMoments": 10,
        "reference": "demo-household-utrecht",
    }
    # Compact: the payload is ~11k quarter-hours in two series, and it is regenerated
    # wholesale by this script rather than edited by hand.
    OUTPUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")

    print(f"wrote {OUTPUT.relative_to(Path.cwd())}")
    print(f"  quarter-hours : {len(index)}")
    print(f"  days          : {total_days - len(GAP)} of {total_days} ({len(GAP)}-day gap)")
    print(f"  offtake total : {sum(offtake):.1f} kWh")
    print(f"  injection total: {sum(injection):.1f} kWh")


if __name__ == "__main__":
    main()
