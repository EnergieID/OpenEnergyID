"""
PVLib-based models for photovoltaic system simulation.

Defines models for PV modules, arrays, inverters, systems, locations, and model chains,
with methods to instantiate corresponding pvlib objects and retrieve weather data.
"""

import datetime as dt

import pandas as pd
import pvlib
from pydantic import BaseModel

from .weather import get_weather


class PVWattsModule(BaseModel):
    """
    Model for PVWatts module parameters.
    """

    pdc0: float = 420
    gamma_pdc: float = -0.003


class PVLibArray(BaseModel):
    """
    Model for a PV array, including mounting, module, and temperature parameters.
    """

    mount: pvlib.pvsystem.FixedMount | pvlib.pvsystem.SingleAxisTrackerMount
    module_parameters: PVWattsModule | dict = PVWattsModule()
    modules_per_string: int
    strings: int = 1
    temperature_model_parameters: dict = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_glass"
    ]

    class Config:
        """Pydantic model configuration."""

        extra = "allow"

    def create_array(self) -> pvlib.pvsystem.Array:
        """
        Instantiate a pvlib.pvsystem.Array from this model.
        """
        params = self.model_dump(exclude={"mount", "module_parameters"})

        mp = (
            self.module_parameters
            if isinstance(self.module_parameters, dict)
            else self.module_parameters.model_dump()
        )

        return pvlib.pvsystem.Array(mount=self.mount, module_parameters=mp, **params)


class PVWattsInverter(BaseModel):
    """
    Model for PVWatts inverter parameters.
    """

    pdc0: float


class PVLibPVSystem(BaseModel):
    """
    Model for a PV system, consisting of arrays and inverter parameters.
    """

    arrays: list[PVLibArray] | PVLibArray
    inverter_parameters: PVWattsInverter | dict

    class Config:
        """Pydantic model configuration."""

        extra = "allow"

    def create_system(self) -> pvlib.pvsystem.PVSystem:
        """
        Instantiate a pvlib.pvsystem.PVSystem from this model.
        """
        params = self.model_dump(exclude={"arrays", "inverter_parameters"})

        arrays = (
            [array.create_array() for array in self.arrays]
            if isinstance(self.arrays, list)
            else [self.arrays.create_array()]
        )

        ip = (
            self.inverter_parameters
            if isinstance(self.inverter_parameters, dict)
            else self.inverter_parameters.model_dump()
        )

        return pvlib.pvsystem.PVSystem(arrays=arrays, inverter_parameters=ip, **params)


class PVLibLocation(BaseModel):
    """
    Model for a geographic location.
    """

    latitude: float
    longitude: float
    tz: str
    altitude: float | None = None
    name: str | None = None

    def create_location(self) -> pvlib.location.Location:
        """
        Instantiate a pvlib.location.Location from this model.
        """
        params = self.model_dump()

        return pvlib.location.Location(**params)


class PVLibModelChain(BaseModel):
    """
    Model for a PVLib ModelChain, including system, location, and simulation period.
    """

    system: PVLibPVSystem
    location: PVLibLocation
    start: dt.date
    end: dt.date

    class Config:
        """Pydantic model configuration."""

        extra = "allow"

    def create_modelchain(self) -> pvlib.modelchain.ModelChain:
        """
        Instantiate a pvlib.modelchain.ModelChain from this model.
        """
        params = self.model_dump(exclude={"system", "location", "start", "end"})

        return pvlib.modelchain.ModelChain(
            system=self.system.create_system(),
            location=self.location.create_location(),
            **params,
        )

    def get_weather(self) -> pd.DataFrame:
        """
        Retrieve weather data for the model's location and date range.
        """
        location = self.location
        return get_weather(
            latitude=location.latitude,
            longitude=location.longitude,
            start=self.start,
            end=self.end,
            tz=location.tz,
        )


class PVLibPVWattsModelChain(PVLibModelChain):
    """
    Specialized PVLibModelChain for PVWatts, with default AOI and DC models.
    """

    aoi_model: str = "physical"
    dc_model: str = "pvwatts"
