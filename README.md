# OpenEnergyID

Open Source Python library for energy data analytics and simulations.

OpenEnergyID is a powerful Python library that provides a wide range of tools for energy data analysis and simulation. Whether you are a data scientist, researcher, or developer working in the energy sector, OpenEnergyID can help you gain valuable insights from your data and build sophisticated models.

[*more info for developers*](DEVELOPERS.md)

## Getting Started

To get started with OpenEnergyID, you can install it using `pip`:

```bash
pip install openenergyid
```

## Analyses

OpenEnergyID provides a variety of analysis modules to help you work with your energy data.

### Baseload Analysis

The baseload analysis module helps you determine the baseload consumption of a building or a portfolio of buildings.

- Use `BaseloadAnalyzer(timezone="Europe/Brussels")`, prepare data with `prepare_power_series(energy_lf)` and then call `analyze(power_lf, "1h")`.
- Accepts either energy (`timestamp`/`total` in kWh per 15 min) or precomputed power (`timestamp`/`power` watts); gapped or zero-valued intervals are kept and handled safely.
- For homes with unmeasured PV, use `nighttime_only=True` to filter to nighttime readings only (uses pvlib for solar position).
- Outputs energy splits (baseload vs total) and baseload ratios per chosen reporting granularity, keeping computations lazy via Polars `LazyFrame`.

### Capacity Analysis

The capacity analysis module helps you identify peaks in your power data.

```python
from openenergyid.capacity import CapacityAnalysis

analyzer = CapacityAnalysis(data=power_series, threshold=2.5)
peaks = analyzer.find_peaks()
```

### Dynamic Tariff Analysis

The dynamic tariff analysis module helps you analyze the impact of dynamic tariffs on your energy costs.

```python
from openenergyid.dyntar import calculate_dyntar_columns

df_with_dyntar = calculate_dyntar_columns(df)
```

### Energy Sharing

The energy sharing module helps you simulate energy sharing scenarios.

```python
from openenergyid.energysharing import calculate

result = calculate(df, method=CalculationMethod.OPTIMAL)
```

### Multivariate Linear Regression (MVLR)

The MVLR module helps you build multivariate linear regression models to predict energy consumption.

```python
from openenergyid.mvlr import find_best_mvlr

model = find_best_mvlr(data)
```

### PV Simulation

The PV simulation module helps you simulate the output of a photovoltaic system.

```python
from openenergyid.pvsim import get_simulator, apply_simulation

simulator = get_simulator(input)
simulation_results = simulator.simulate()
df_with_pv = apply_simulation(df, simulation_results)
```

### Simulation Evaluation

The simulation evaluation module helps you evaluate the results of your energy simulations.

```python
from openenergyid.simeval import evaluate

evaluation = evaluate(df)
```
