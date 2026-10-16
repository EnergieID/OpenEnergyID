"""Caveats about the ex-ante data a simulation is based on."""

import pandas as pd

from .. import const
from ..abstractsim import DataCoverage, SimulationWarning, SimulationWarningCode


def _days_with_data(series: pd.Series | None) -> int:
    if series is None:
        return 0
    index = pd.DatetimeIndex(series.dropna().index)
    return int(index.normalize().nunique())


def check_ex_ante(data: pd.DataFrame) -> tuple[DataCoverage, list[SimulationWarning]]:
    """Describe how much data the simulation had, and flag what makes its result less reliable.

    Without injection data the simulation cannot tell consumption from production, and a
    simulated battery has no surplus to charge from. Intervals with a missing value are
    not simulated, so the result covers only the intervals with data.
    """
    delivered = data.get(const.ELECTRICITY_DELIVERED)
    exported = data.get(const.ELECTRICITY_EXPORTED)
    has_injection = exported is not None and not exported.isna().all()

    incomplete = pd.Series(False, index=data.index)
    if delivered is not None:
        incomplete |= delivered.isna()
    if has_injection:
        incomplete |= exported.isna()
    intervals_incomplete = int(incomplete.sum())

    coverage = DataCoverage(
        start=data.index.min() if len(data.index) else None,
        end=data.index.max() if len(data.index) else None,
        days_with_delivered=_days_with_data(delivered),
        days_with_exported=_days_with_data(exported),
        intervals=len(data.index),
        intervals_incomplete=intervals_incomplete,
    )

    warnings: list[SimulationWarning] = []
    if not has_injection:
        warnings.append(
            SimulationWarning(
                code=SimulationWarningCode.NO_INJECTION_DATA,
                message=(
                    f"No injection data in '{const.ELECTRICITY_EXPORTED}', so nothing is "
                    "assumed to be injected. With existing solar panels this overstates "
                    "consumption and self-consumption, and leaves a simulated battery no "
                    "surplus to charge from."
                ),
                context={"column": const.ELECTRICITY_EXPORTED},
            )
        )
    if intervals_incomplete:
        warnings.append(
            SimulationWarning(
                code=SimulationWarningCode.ENERGY_DATA_INCOMPLETE,
                message=(
                    f"{intervals_incomplete} of {len(data.index)} intervals miss offtake or "
                    "injection data. The simulation result covers only the intervals with data."
                ),
                context={"intervals": len(data.index), "incomplete": intervals_incomplete},
            )
        )

    return coverage, warnings
