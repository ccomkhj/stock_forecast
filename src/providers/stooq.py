import pandas_datareader.data as web
import pandas as pd
from datetime import date
from .base import BaseProvider

class StooqProvider(BaseProvider):
    def get_ohlcv(self, symbol: str, start: date, end: date, interval: str = "1d") -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("Stooq only supports '1d' interval via pandas_datareader")
        
        try:
            # pandas_datareader expects datetime or string, date objects usually work too
            df = web.DataReader(symbol, 'stooq', start=start, end=end)
            
            if df is None or df.empty:
                return pd.DataFrame()
            
            # Stooq returns index as Date, descending.
            df = df.sort_index(ascending=True).reset_index()
            
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            
            # Add metadata
            df['symbol'] = symbol
            df['provider'] = 'stooq'
            
            # Ensure columns exist and order
            required_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'provider']
            
            # Sometimes volume is missing or named differently? Stooq usually has 'Volume'.
            # If any required column is missing, fill with 0 or NaN?
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0.0
            
            # Convert date to datetime if not already (reset_index might make it Timestamp)
            df['date'] = pd.to_datetime(df['date'])
            
            return df[required_cols]
            
        except Exception as e:
            print(f"Error fetching from Stooq for {symbol}: {e}")
            return pd.DataFrame()
