import pandera as pa

from pandera.typing import Series


class InputModel(pa.DataFrameModel):
    # Example of a simple schema
    column1: int = pa.Field(le=10)
    column2: float = pa.Field(lt=-1.2)
    column3: str = pa.Field(str_startswith="value_")

    @pa.check("column3")
    @classmethod
    def column_3_check(cls, series: Series[str]) -> Series[bool]:
        """Check that column3 values have two elements after being split with '_'"""
        return series.str.split("_", expand=True).shape[1] == 2


class OutputModel(pa.DataFrameModel):
    column1: int = pa.Field(le=10)
    column2: float = pa.Field(lt=-1.2)
    column3: str = pa.Field(str_startswith="value_")

    @pa.check("column3")
    @classmethod
    def column_3_check(cls, series: Series[str]) -> Series[bool]:
        """Check that column3 values have two elements after being split with '_'"""
        return series.str.split("_", expand=True).shape[1] == 2

    # Dict output required for serialization by FastAPI & Pydantic
    class Config:
        to_format = "dict"
        to_format_kwargs = {"orient": "split"}
