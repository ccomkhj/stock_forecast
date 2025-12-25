from abc import ABC, abstractmethod
import pandas as pd
from datetime import date

class BaseProvider(ABC):
    @abstractmethod
    def get_ohlcv(self, symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        pass
