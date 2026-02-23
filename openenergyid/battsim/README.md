# Battery Simulation (`battsim`)

This module simulates behind-the-meter battery behavior and integrates it into the generic OpenEnergyID simulation
pipeline.

## What `battsim` does

`battsim` models how a battery changes:

- grid offtake (`electricity_delivered`)
- grid injection (`electricity_exported`)
- battery state and usage metrics (`state_of_energy`, `electricity_charged`, `electricity_discharged`,
  `battery_cycles`)

The current implementation is `selfconsumptionbatterysimulation`:

- charge battery from local surplus (`electricity_exported`)
- discharge battery to reduce grid offtake (`electricity_delivered`)
- respect battery constraints (`capacity`, `power`, `loss_factor`)

## Where it fits in the full simulation flow

`run_simulation(...)` executes four outputs:

1. `ex_ante`: evaluation of original input data
2. `simulation_result`: evaluation of raw battery simulator output
3. `ex_post`: evaluation after applying battery impact onto original data
4. `comparison`: `ex_post - ex_ante`

Important:

- `simulation_result` answers: "What did the battery model produce?"
- `ex_post` answers: "What would the metered profile look like after applying the simulation?"

## Public API

- Input model:
  - `SelfConsumptionBatterySimulationInput`
- Simulator:
  - `SelfConsumptionBatterySimulator`
- Entry helpers:
  - `get_simulator(...)`
  - `apply_simulation(...)`

## Input requirements

Expected time-indexed columns:

- `electricity_delivered`
- `electricity_exported`
- optional: `electricity_produced` (passed through if present). Not needed for the battery simulation on its own, but needed to calculate selfconsumption in the ex-ante and ex-post.
- optional: `price_electricity_delivered` & `price_electricity_exported`. If included, cost calculations are included in the evaluation.

Parameters:

- `capacity` (kWh)
- `power` (kW)
- `initial_charge` (kWh, default `0.0`)
- `loss_factor` (fraction, default `0.05`)
- `result_resolution` (default `15min`)

## Output columns (simulator results)

- `electricity_delivered`: residual grid offtake after battery discharge
- `electricity_exported`: residual grid export after battery charging
- `state_of_energy`: battery state of energy (kWh)
- `electricity_charged`: charged energy (negative sign convention in final result)
- `electricity_discharged`: discharged energy
- `battery_cycles`: equivalent full cycles

## Sign conventions

- `electricity_charged` is negative in final simulator output
- `electricity_discharged` is positive

## Current behavior and caveats

- The simulator is deterministic and rule-based (not market-optimized).
- It applies power/energy constraints per time step.
- Losses are represented with a simple factor (`loss_factor`).

## Example (high-level)

```python
from openenergyid.sim import FullSimulationInput, run_simulation

# input_ = FullSimulationInput(...)
result = await run_simulation(input_, session=None)

print(result.simulation_result["total"])
print(result.ex_post["total"])
```
