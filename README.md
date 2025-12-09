# OpenEnergyID

Open Source Python library for energy data analytics and simulations

[*more info for developers*](DEVELOPERS.md)

## Baseload analysis
- Use `BaseloadAnalyzer(timezone="Europe/Brussels")`, prepare data with `prepare_power_series(energy_lf)` and then call `analyze(power_lf, "1h")`.
- Accepts either energy (`timestamp`/`total` in kWh per 15 min) or precomputed power (`timestamp`/`power` watts); gapped or zero-valued intervals are kept and handled safely.
- Outputs energy splits (baseload vs total) and baseload ratios per chosen reporting granularity, keeping computations lazy via Polars `LazyFrame`.
