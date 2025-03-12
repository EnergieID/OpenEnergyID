from abc import ABC, abstractmethod

import pandas as pd


class PVSimulator(ABC):
    @abstractmethod
    def simulate(self, **kwargs) -> pd.DataFrame:
        pass
