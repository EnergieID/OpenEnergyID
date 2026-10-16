"""Self-consumption battery simulation module."""

import math
from typing import Literal, Self

import pandas as pd
from pydantic import Field, model_validator

from ... import const
from ..abstract import BatterySimulationInputAbstract, BatterySimulator


class SelfConsumptionBatterySimulationInput(BatterySimulationInputAbstract):
    """
    Input parameters for the SelfConsumptionBatterySimulator.
    """

    type: Literal["selfconsumptionbatterysimulation"] = Field(
        "selfconsumptionbatterysimulation", frozen=True
    )  # tag
    capacity: float = Field(
        ...,
        description="Battery capacity in kWh",
        examples=[10.0, 20.0, 50.0],
    )
    power: float | None = Field(
        None,
        description=(
            "Battery power in kW, for both charging and discharging. "
            "Required unless `charge_power` and `discharge_power` are both given."
        ),
        examples=[5.0, 10.0, 20.0],
    )
    charge_power: float | None = Field(
        None,
        description="Maximum charge power in kW. Defaults to `power`.",
        examples=[5.0, 10.0, 20.0],
    )
    discharge_power: float | None = Field(
        None,
        description="Maximum discharge power in kW. Defaults to `power`.",
        examples=[5.0, 10.0, 20.0],
    )
    initial_charge: float = Field(
        0.0,
        description="Initial state of charge in kWh",
        examples=[0.0, 5.0, 10.0],
    )
    loss_factor: float = Field(
        0.05,
        description=(
            "Battery loss factor, applied once on charging and once on discharging: "
            "charging stores (1 - l) of the energy, discharging takes (1 + l) from the "
            "battery per kWh delivered. The round-trip efficiency is (1 - l) / (1 + l)."
        ),
        examples=[0.01, 0.05, 0.1],
    )

    @model_validator(mode="after")
    def _resolve_powers(self) -> Self:
        if self.charge_power is None:
            self.charge_power = self.power
        if self.discharge_power is None:
            self.discharge_power = self.power
        if self.charge_power is None or self.discharge_power is None:
            raise ValueError("Supply `power`, or both `charge_power` and `discharge_power`.")
        return self


class SelfConsumptionBatterySimulator(BatterySimulator):
    """
    Simulator for battery self-consumption optimization.
    """

    def __init__(
        self,
        capacity: float,
        power: float | None = None,
        initial_charge: float = 0.0,
        loss_factor: float = 0.05,
        charge_power: float | None = None,
        discharge_power: float | None = None,
        **kwargs,
    ):
        """
        Initialize the simulator with battery parameters.

        Args:
            capacity: Battery capacity in kWh.
            power: Battery power in kW, for both directions.
            initial_charge: Initial state of charge in kWh.
            loss_factor: Battery loss factor, applied once per direction.
            charge_power: Maximum charge power in kW. Defaults to `power`.
            discharge_power: Maximum discharge power in kW. Defaults to `power`.
        """
        super().__init__(**kwargs)

        charge_power = power if charge_power is None else charge_power
        discharge_power = power if discharge_power is None else discharge_power
        if charge_power is None or discharge_power is None:
            raise ValueError("Supply `power`, or both `charge_power` and `discharge_power`.")

        self.capacity = capacity
        self.charge_power = charge_power
        self.discharge_power = discharge_power
        self.initial_charge = initial_charge
        self.loss_factor = loss_factor

    def simulate(self, **kwargs) -> pd.DataFrame:
        results = self.batt_sim(
            energy_offtake=self.data[const.ELECTRICITY_DELIVERED],
            energy_injection=self.data[const.ELECTRICITY_EXPORTED],
            capacity=self.capacity,
            charge_power=self.charge_power,
            discharge_power=self.discharge_power,
            initial_charge=self.initial_charge,
            loss=self.loss_factor,
        )

        if self.capacity != 0.0:
            results[const.BATTERY_CYCLES] = (
                results[const.ELECTRICITY_CHARGED] + results[const.ELECTRICITY_DISCHARGED]
            ) / (self.capacity * 2)
        else:
            results[const.BATTERY_CYCLES] = 0.0

        results[const.ELECTRICITY_CHARGED] = results[const.ELECTRICITY_CHARGED] * -1

        if const.ELECTRICITY_PRODUCED in self.data.columns:
            results[const.ELECTRICITY_PRODUCED] = self.data[const.ELECTRICITY_PRODUCED]

        return results

    @staticmethod
    def batt_sim(
        energy_offtake: pd.Series,
        energy_injection: pd.Series,
        capacity: float,
        charge_power: float,
        discharge_power: float,
        initial_charge: float = 0.0,
        loss: float = 0.0,
    ) -> pd.DataFrame:
        """
        Naive implementation of a simulated battery.

        Input:
            energy_offtake: Energy from the grid in kWh
            energy_injection: Energy to the grid in kWh
            capacity: Maximum charge in kWh.
            charge_power: Maximum charge power in kW
            discharge_power: Maximum discharge power in kW
            initial_charge: starting state of charge in kWh, default 0
        Output:
            pd.DataFrame, 2 columns
            state_of_energy: state of energy in kWh
            energy: energy charged/discharged in kWh. Discharge is positive, charge is negative

        Naive approach:
            Injection is put into the battery, Offtake is taken from the battery, as long as
            charge is available and it does not exceed the power.
            An interval where offtake or injection is missing (NaN) leaves the battery idle.
            Within one interval the battery first discharges, then charges in the time that
            is left: if discharging took a fraction f of the interval at discharge_power,
            charging gets (1 - f) of it at charge_power.
        Possible improvements:
            Extra "losses" due to momentary peaks that are not captured in the 15min averages
        """
        # Init empty arrays to fill
        soes = []
        charge = []
        discharge = []
        new_offtake = []
        new_injection = []
        # charge_discharge = []
        # new_grids = []
        soe = initial_charge

        if capacity == 0.0:
            res = pd.DataFrame(
                index=energy_offtake.index,
                data={
                    const.STATE_OF_ENERGY: 0,
                    const.ELECTRICITY_CHARGED: 0,
                    const.ELECTRICITY_DISCHARGED: 0,
                    const.ELECTRICITY_DELIVERED: energy_offtake,
                    const.ELECTRICITY_EXPORTED: energy_injection,
                },
            )
        else:
            for offtake, injection in zip(energy_offtake.values, energy_injection.values):
                # Without both offtake and injection the net balance is unknown, so the
                # battery stays idle and the interval passes through unchanged
                if math.isnan(offtake) or math.isnan(injection):
                    discharge.append(0.0)
                    charge.append(0.0)
                    new_offtake.append(offtake)
                    new_injection.append(injection)
                    soes.append(soe)
                    continue

                # `offtake`kWh is requested from the battery
                requested = offtake
                # However the battery needs to contain more than that, to account for loss
                energy_to_withdraw = requested + requested * loss
                # But if the soc is lower than that, or the maximum energy we can draw (power/4), we need to limit
                # So what actually is withdrawn:
                withdrawn = min(soe, discharge_power / 4, energy_to_withdraw)
                # This is delivered with exception of the loss
                delivered = withdrawn / (1 + loss)

                soe = soe - withdrawn
                discharge.append(delivered)
                new_offtake.append(offtake - delivered)

                # Share of the interval that discharging took
                discharge_share = withdrawn * 4 / discharge_power if discharge_power else 0.0
                net_power = charge_power * (1 - discharge_share)

                # Externally, `injection`kWh is available to charge with
                available = injection
                # However, there are some losses
                energy_to_insert = available - available * loss
                # We will try to insert it. But limited by the available space in the battery and power
                inserted = min(capacity - soe, net_power / 4, energy_to_insert)
                # But what we see from the outside is with inclusion of the loss
                if inserted != 0:
                    delivered = inserted / (1 - loss)
                else:
                    delivered = 0

                soe = soe + inserted
                charge.append(delivered)
                new_injection.append(injection - delivered)

                soes.append(soe)
            res = pd.DataFrame(
                index=energy_offtake.index,
                data={
                    const.STATE_OF_ENERGY: soes,
                    const.ELECTRICITY_CHARGED: charge,
                    const.ELECTRICITY_DISCHARGED: discharge,
                    const.ELECTRICITY_DELIVERED: new_offtake,
                    const.ELECTRICITY_EXPORTED: new_injection,
                },
            )
            res[const.STATE_OF_ENERGY] = res[const.STATE_OF_ENERGY].shift(1).fillna(initial_charge)
        return res
