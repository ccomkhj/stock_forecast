import pandas as pd
from src.indicators import calculate_indicators

def prepare_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """
    Prepares features for the model.
    Adds indicators if missing.
    Generates lagged features and other stats.
    Removes rows with NaNs.
    """
    if 'sma_20' not in df.columns:
        df = calculate_indicators(df)
        
    df = df.copy()
    
    # Lagged features
    # r(t) is today's return (known at close of day t).
    # We want to predict future returns r(t+1)...r(t+H) using data available at t.
    
    for i in range(lags):
        df[f'lag_return_{i}'] = df['returns'].shift(i)
        df[f'lag_close_{i}'] = df['close'].shift(i)
        df[f'lag_rsi_{i}'] = df['rsi_14'].shift(i)
        df[f'lag_vol_{i}'] = df['volatility_20'].shift(i)

    # MA Distances
    df['dist_sma_20'] = df['close'] - df['sma_20']
    df['dist_sma_60'] = df['close'] - df['sma_60']
    df['dist_sma_130'] = df['close'] - df['sma_130']
    
    # Rolling mean of returns (e.g., 5-day)
    df['rolling_mean_return_5'] = df['returns'].rolling(window=5).mean()

    # Drop NaNs resulting from indicators and lags
    df = df.dropna()
    
    return df
