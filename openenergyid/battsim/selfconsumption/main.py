"""Self-consumption battery simulation module."""

from typing import Literal

import pandas as pd
from pydantic import Field

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
    power: float = Field(
        ...,
        description="Battery power in kW",
        examples=[5.0, 10.0, 20.0],
    )
    initial_charge: float = Field(
        0.0,
        description="Initial state of charge in kWh",
        examples=[0.0, 5.0, 10.0],
    )
    loss_factor: float = Field(
        0.05,
        description="Battery loss factor per cycle",
        examples=[0.01, 0.05, 0.1],
    )


class SelfConsumptionBatterySimulator(BatterySimulator):
    """
    Simulator for battery self-consumption optimization.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        capacity: float,
        power: float,
        initial_charge: float = 0.0,
        loss_factor: float = 0.05,
        **kwargs,
    ):
        """
        Initialize the simulator with battery parameters.

        Args:
            capacity: Battery capacity in kWh.
            power: Battery power in kW.
            initial_charge: Initial state of charge in kWh.
            loss_factor: Battery loss factor per cycle.
        """
        super().__init__(**kwargs)

        self.data = data
        self.capacity = capacity
        self.power = power
        self.initial_charge = initial_charge
        self.loss_factor = loss_factor

    def simulate(self, **kwargs) -> pd.DataFrame:
        results = self.batt_sim(
            energy_offtake=self.data[const.ELECTRICITY_DELIVERED],
            energy_injection=self.data[const.ELECTRICITY_EXPORTED],
            capacity=self.capacity,
            power=self.power,
            initial_charge=self.initial_charge,
            loss=self.loss_factor,
        )

        results[const.BATTERY_CYCLES] = (
            results[const.ELECTRICITY_CHARGED] + results[const.ELECTRICITY_DISCHARGED]
        ) / (self.capacity * 2)

        results[const.ELECTRICITY_CHARGED] = results[const.ELECTRICITY_CHARGED] * -1

        return results

    @staticmethod
    def batt_sim(
        energy_offtake: pd.Series,
        energy_injection: pd.Series,
        capacity: float,
        power: float,
        initial_charge: float = 0.0,
        loss: float = 0.0,
    ) -> pd.DataFrame:
        """
        Naive implementation of a simulated battery.

        Input:
            energy_offtake: Energy from the grid in kWh
            energy_injection: Energy to the grid in kWh
            capacity: Maximum charge in kWh.
            power: Maximum power in kW
            initial_charge: starting state of charge in kWh, default 0
        Output:
            pd.DataFrame, 2 columns
            state_of_energy: state of energy in kWh
            energy: energy charged/discharged in kWh. Discharge is positive, charge is negative

        Naive approach:
            Injection is put into the battery, Offtake is taken from the battery, as long as
            charge is available and it does not exceed the power.
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
                # `offtake`kWh is requested from the battery
                requested = offtake
                # However the battery needs to contain more than that, to account for loss
                energy_to_withdraw = requested + requested * loss
                # But if the soc is lower than that, or the maximum energy we can draw (power/4), we need to limit
                # So what actually is withdrawn:
                withdrawn = min(soe, power / 4, energy_to_withdraw)
                # This is delivered with exception of the loss
                delivered = withdrawn / (1 + loss)

                soe = soe - withdrawn
                discharge.append(delivered)
                new_offtake.append(offtake - delivered)

                net_power = power - withdrawn * 4

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
