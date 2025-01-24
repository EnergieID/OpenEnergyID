import pandera.polars as pa
from pandera.engines.polars_engine import DateTime


class PowerReadingSchema(pa.DataFrameModel):
    """Validates input energy readings"""

    timestamp: DateTime = pa.Field()
    total: float = pa.Field(ge=0)

    class Config:
        coerce = True


class PowerSeriesSchema(pa.DataFrameModel):
    """Validates converted power series"""

    timestamp: DateTime = pa.Field()
    power: float = pa.Field(ge=0)


class BaseloadResultSchema(pa.DataFrameModel):
    """Validates analysis results"""

    timestamp: DateTime = pa.Field()
    consumption_due_to_baseload_in_kilowatthour: float = pa.Field(ge=0)
    total_consumption_in_kilowatthour: float = pa.Field(ge=0)
    average_daily_baseload_in_watt: float = pa.Field(ge=0)
    average_power_in_watt: float = pa.Field(ge=0)
    consumption_not_due_to_baseload_in_kilowatthour: float
    baseload_ratio: float = pa.Field(ge=0, le=2)
