"""Example of a Pandera schema for input and output data validation."""

import pandera as pa

from pandera.typing import Series, Index


class InputModel(pa.DataFrameModel):
    """Pandera schema for input data validation."""

    index: Index[int] = pa.Field(ge=0)
    column1: Series[int] = pa.Field(le=10)
    column2: Series[float] = pa.Field(lt=-1.2)
    column3: Series[str] = pa.Field(str_startswith="value_")

    @pa.check("column3")
    @classmethod
    def column_3_check(cls, series: Series[str]) -> bool:
        """Check that column3 values have two elements after being split with '_'"""
        return series.str.split("_", expand=True).shape[1] == 2


class OutputModel(pa.DataFrameModel):
    """Pandera schema for output data validation."""

    index: Index[int] = pa.Field(ge=0)
    column1: Series[int] = pa.Field(le=10)
    column2: Series[float] = pa.Field(lt=-1.2)
    column3: Series[str] = pa.Field(str_startswith="value_")

    @pa.check("column3")
    @classmethod
    def column_3_check(cls, series: Series[str]) -> bool:
        """Check that column3 values have two elements after being split with '_'"""
        return series.str.split("_", expand=True).shape[1] == 2

    class Config:
        """Pandera schema configuration."""

        # Define the output data format
        to_format = "dict"
        to_format_kwargs = {"orient": "list"}
