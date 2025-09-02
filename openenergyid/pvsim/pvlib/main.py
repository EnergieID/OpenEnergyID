import pandas as pd
import pvlib

from openenergyid.pvsim.abstract import PVSimulator

from .models import PVLibModelChain


class PVLibSimulator(PVSimulator):
    def __init__(self, modelchain: pvlib.modelchain.ModelChain, weather: pd.DataFrame):
        super().__init__()

        self.modelchain = modelchain
        self.weather = weather

    def simulate(self, **kwargs) -> pd.Series:
        # Run model
        self.modelchain.run_model(self.weather)

        results = self.modelchain.results
        ac = results.ac

        if ac is None:
            raise ValueError("AC power is None")

        # Convert W to kWh
        energy = ac * 0.25 / 1000

        return energy

    @classmethod
    def from_pydantic(cls, modelchain: PVLibModelChain) -> "PVLibSimulator":
        mc = modelchain.create_modelchain()
        weather = modelchain.get_weather()

        return cls(modelchain=mc, weather=weather)
