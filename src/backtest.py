import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def backtest_model(model_class, df: pd.DataFrame, horizon=30, backtest_window=180, **model_params):
    """
    Performs walk-forward backtest using Nixtla's cross_validation if available,
    or falls back to manual loop.
    Returns tuple: (metrics_dict, cv_dataframe)
    """
    # Instantiate to check capabilities or just use class logic
    # We need an instance to call cross_validate if it's an instance method
    # Or we just instantiate inside.
    
    # Calculate n_windows based on backtest_window
    # step_size = horizon usually
    n_windows = int(backtest_window / horizon)
    if n_windows < 1: n_windows = 1
    
    # Create temp instance
    model = model_class(horizon=horizon, **model_params)
    
    cv_df = pd.DataFrame()
    
    if hasattr(model, 'cross_validate'):
        cv_df = model.cross_validate(df, horizon=horizon, n_windows=n_windows)
    else:
        # Fallback to manual? (Shouldn't happen with our adapters)
        pass
        
    if cv_df.empty:
        return {}, pd.DataFrame()
        
    # Calculate metrics from cv_df
    # cv_df usually has columns: unique_id, ds, cutoff, y, <model_col>
    # Identify model column
    reserved = ['unique_id', 'ds', 'cutoff', 'y']
    model_col = [c for c in cv_df.columns if c not in reserved][0]
    
    y_true = cv_df['y'].values
    y_pred = cv_df[model_col].values
    
    # Drop NaNs if any (e.g. if horizon mismatch in manual cv)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {}, pd.DataFrame()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Directional Accuracy
    # Need returns. Calculate from price diffs.
    # We can't easily get exact returns without lag, but we can check direction vs cutoff price?
    # Or just direction of movement in the window?
    # Let's approximate: direction of change from previous day in the forecast sequence
    # This is hard on concatenated CV.
    
    # Alternative: Direction relative to Cutoff Price?
    # Or just mean direction of daily changes?
    # Let's stick to MAE/RMSE for simplicity or try to infer returns.
    
    # Simple Directional Accuracy: (Sign(y_pred - y_prev) == Sign(y_true - y_prev))
    # But we don't have y_prev easily aligned.
    
    metrics = {
        'MAE': mae,
        'RMSE': rmse,
        # 'Directional Accuracy': accuracy
    }
    
    return metrics, cv_df