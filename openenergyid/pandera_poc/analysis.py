from .models import InputModel, OutputModel
import pandas as pd


def analyse(df: InputModel | pd.DataFrame) -> OutputModel:
    # Validate input data
    df = InputModel.validate(df)

    # Perform analysis
    pass

    # Validate output data
    df = OutputModel.validate(df)

    return df
