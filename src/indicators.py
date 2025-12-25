import pandas as pd
import numpy as np

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Appends indicator columns to the dataframe.
    Returns a new dataframe with added columns.
    """
    df = df.copy()
    if df.empty:
        return df

    # Ensure sorted by date
    df = df.sort_values('date')
    close = df['close']

    # SMA
    df['sma_20'] = close.rolling(window=20).mean()
    df['sma_60'] = close.rolling(window=60).mean()
    df['sma_130'] = close.rolling(window=130).mean()

    # EMA
    df['ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['ema_60'] = close.ewm(span=60, adjust=False).mean()
    df['ema_130'] = close.ewm(span=130, adjust=False).mean()

    # Returns (simple percentage change)
    df['returns'] = df['close'].pct_change()

    # Rolling Volatility (20-day std of returns)
    df['volatility_20'] = df['returns'].rolling(window=20).std()

    # RSI (14-day) - using Wilder's Smoothing (alpha=1/14)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # alpha = 1/N for Wilder's
    alpha = 1.0 / 14.0
    ema_up = up.ewm(alpha=alpha, adjust=False).mean()
    ema_down = down.ewm(alpha=alpha, adjust=False).mean()
    
    rs = ema_up / ema_down
    df['rsi_14'] = 100 - (100 / (1 + rs))

    return df
