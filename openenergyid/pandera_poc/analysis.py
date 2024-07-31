"""Proof of concept of a data analysis module using pandera."""

from pandera.typing import DataFrame
from .models import InputModel, OutputModel


def analyse(df: DataFrame[InputModel]) -> DataFrame[OutputModel]:
    """Perform analysis on the input data and return the output data."""
    # Validate input data
    InputModel.validate(df)

    # Perform analysis
    pass

    # Validate output data
    OutputModel.validate(df)

    return df
