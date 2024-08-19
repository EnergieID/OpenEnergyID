"""Example of a Pandera schema for input and output data validation."""

import pandera.polars as pap
import polars as pl
# from pandera.typing import Series


class InputModel(pap.DataFrameModel):
    """Pandera schema for input data validation."""

    # index: Optional[Index[int]]
    column1: int = pap.Field(le=10)
    column2: pl.Float64 = pap.Field(lt=-1.2)
    column3: str = pap.Field(str_startswith="value_")

    @pap.check("column3")
    @classmethod
    def column_3_check(cls, series: pl.Series) -> bool:
        """Check that column3 values have two elements after being split with '_'"""
        return len(series.str.split("_")) == 2


class OutputModel(pap.DataFrameModel):
    """Pandera schema for output data validation."""

    # index: pl.int = pap.Field(ge=0)
    column1: int = pap.Field(le=10)
    column2: float = pap.Field(lt=-1.2)
    column3: str = pap.Field(str_startswith="value_")

    @pap.check("column3")
    @classmethod
    def column_3_check(cls, series: pl.Series) -> bool:
        """Check that column3 values have two elements after being split with '_'"""
        return len(series.str.split("_")) == 2

    class Config:
        """Pandera schema configuration."""

        # Define the output data format
        to_format = "dict"
        to_format_kwargs = {"orient": "list"}
