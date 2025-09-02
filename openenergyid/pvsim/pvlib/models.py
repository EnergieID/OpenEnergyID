import typing

import pandas as pd
import pvlib
from pydantic import BaseModel


def get_weather(
    latitude: float, longitude: float, year: int, utc_offset: int, tz: str
) -> pd.DataFrame:
    weather = pvlib.iotools.get_pvgis_tmy(
        latitude=latitude, longitude=longitude, coerce_year=year, roll_utc_offset=utc_offset
    )[0]
    weather = typing.cast(pd.DataFrame, weather)
    weather = weather.tz_convert(tz)
    weather.index.name = None

    # Add the first row again as the last row, but make the time that of the last row + 1h
    first_row = weather.iloc[0]
    last_index = typing.cast(pd.Timestamp, weather.last_valid_index())
    first_row.name = last_index + pd.Timedelta(hours=1)
    weather = pd.concat([weather, first_row.to_frame().T])
    weather = weather.resample("15min").interpolate()
    # drop last row
    weather = weather.iloc[:-1]
    return weather


class PVWattsModule(BaseModel):
    pdc0: float = 420
    gamma_pdc: float = -0.003


class PVLibArray(BaseModel):
    mount: pvlib.pvsystem.FixedMount | pvlib.pvsystem.SingleAxisTrackerMount
    module_parameters: PVWattsModule | dict = PVWattsModule()
    modules_per_string: int
    strings: int = 1
    temperature_model_parameters: dict = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_glass"
    ]

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def create_array(self) -> pvlib.pvsystem.Array:
        params = self.model_dump(exclude={"mount", "module_parameters"})

        mp = (
            self.module_parameters
            if isinstance(self.module_parameters, dict)
            else self.module_parameters.model_dump()
        )

        return pvlib.pvsystem.Array(mount=self.mount, module_parameters=mp, **params)


class PVWattsInverter(BaseModel):
    pdc0: float


class PVLibPVSystem(BaseModel):
    arrays: list[PVLibArray] | PVLibArray
    inverter_parameters: PVWattsInverter | dict

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def create_system(self) -> pvlib.pvsystem.PVSystem:
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
    latitude: float
    longitude: float
    tz: str
    altitude: float | None = None
    name: str | None = None

    def create_location(self) -> pvlib.location.Location:
        params = self.model_dump()

        return pvlib.location.Location(**params)


class PVLibModelChain(BaseModel):
    system: PVLibPVSystem
    location: PVLibLocation
    year: int
    utc_offset: int
    aoi_model: str = "physical"
    dc_model: str = "pvwatts"

    class Config:
        extra = "allow"
        arbitrary_types_allowed = True

    def create_modelchain(self) -> pvlib.modelchain.ModelChain:
        params = self.model_dump(exclude={"system", "location", "year", "utc_offset"})

        return pvlib.modelchain.ModelChain(
            system=self.system.create_system(), location=self.location.create_location(), **params
        )

    def get_weather(self) -> pd.DataFrame:
        location = self.location
        return get_weather(
            latitude=location.latitude,
            longitude=location.longitude,
            year=self.year,
            utc_offset=self.utc_offset,
            tz=location.tz,
        )
