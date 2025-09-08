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


class PVLibFixedMount(BaseModel):
    """
    Model for a fixed mount system.
    """

    surface_tilt: float = 0
    surface_azimuth: float = 180
    racking_model: str | None = "open_rack"
    module_height: float | None = None

    class Config:
        """Pydantic model configuration."""

        extra = "allow"

    def create_mount(self) -> pvlib.pvsystem.FixedMount:
        """
        Instantiate a pvlib.pvsystem.FixedMount from this model.
        """
        params = self.model_dump()
        return pvlib.pvsystem.FixedMount(**params)


class PVLibSingleAxisTrackerMount(BaseModel):
    """ """

    axis_tilt: float = 0
    axis_azimuth: float = 180
    max_angle: float | tuple = 90
    backtrack: bool = True
    gcr: float = 2.0 / 7.0
    cross_axis_tilt: float = 0
    racking_model: str | None = "open_rack"
    module_height: float | None = None

    class Config:
        """Pydantic model configuration."""

        extra = "allow"

    def create_mount(self) -> pvlib.pvsystem.SingleAxisTrackerMount:
        """
        Instantiate a pvlib.pvsystem.SingleAxisTrackerMount from this model.
        """
        params = self.model_dump()
        return pvlib.pvsystem.SingleAxisTrackerMount(**params)


class PVLibArray(BaseModel):
    """
    Model for a PV array, including mounting, module, and temperature parameters.
    """

    mount: PVLibFixedMount | PVLibSingleAxisTrackerMount = PVLibFixedMount()
    albedo: float | None = None
    surface_type: str | None = None
    module_type: str | None = "glass_glass"
    module_parameters: PVWattsModule | dict = PVWattsModule()
    temperature_model_parameters: dict | None = None
    modules_per_string: int = 1
    strings: int = 1
    array_losses_parameters: dict | None = None
    name: str | None = None

    class Config:
        """Pydantic model configuration."""

        extra = "allow"

    def create_array(self) -> pvlib.pvsystem.Array:
        """
        Instantiate a pvlib.pvsystem.Array from this model.
        """
        params = self.model_dump(exclude={"mount", "module_parameters"})

        mount = (
            self.mount
            if isinstance(self.mount, pvlib.pvsystem.SingleAxisTrackerMount)
            else self.mount.create_mount()
        )

        mp = (
            self.module_parameters
            if isinstance(self.module_parameters, dict)
            else self.module_parameters.model_dump()
        )

        return pvlib.pvsystem.Array(mount=mount, module_parameters=mp, **params)


class PVWattsInverter(BaseModel):
    """
    Model for PVWatts inverter parameters.
    """

    pdc0: float


class PVLibPVSystem(BaseModel):
    """
    Model for a PV system, consisting of arrays and inverter parameters.
    """

    arrays: list[PVLibArray]
    inverter_parameters: PVWattsInverter | dict
    name: str | None = None

    class Config:
        """Pydantic model configuration."""

        extra = "allow"

    def create_system(self) -> pvlib.pvsystem.PVSystem:
        """
        Instantiate a pvlib.pvsystem.PVSystem from this model.
        """
        params = self.model_dump(exclude={"arrays", "inverter_parameters"})

        arrays = [array.create_array() for array in self.arrays]

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
    tz: str = "UTC"
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

    def get_weather(self, start: dt.date, end: dt.date) -> pd.DataFrame:
        """
        Retrieve weather data for the model's location and date range.
        """
        location = self.location
        return get_weather(
            latitude=location.latitude,
            longitude=location.longitude,
            start=start,
            end=end,
            tz=location.tz,
        )


class PVLibPVWattsModelChain(PVLibModelChain):
    """
    Specialized PVLibModelChain for PVWatts, with default AOI and DC models.
    """

    aoi_model: str = "physical"
    dc_model: str = "pvwatts"
