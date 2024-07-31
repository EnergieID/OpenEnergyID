from .models import InputModel, OutputModel


def analyse(df: InputModel) -> OutputModel:
    # Validate input data
    InputModel.validate(df)

    # Perform analysis
    pass

    # Validate output data
    OutputModel.validate(df)

    return df
