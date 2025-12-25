from abc import ABC, abstractmethod
import pandas as pd

class BaseModel(ABC):
    @abstractmethod
    def train(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def predict(self, df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        """
        Returns DataFrame with columns:
        - date (future date)
        - predicted_return
        - predicted_price
        """
        pass
