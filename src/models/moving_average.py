import pandas as pd
import numpy as np
from .base import BaseModel
from .config import get_model_config

class MovingAverageModel(BaseModel):
    def __init__(self, horizon=30, **kwargs):
        conf = get_model_config("Moving Average")
        params = conf.get('params', {}).copy()
        
        self.window = kwargs.get('window', params.get('window', 20))
        self.last_mean_return = 0.0
        self.last_price = 0.0
        self.last_date = None
        self.volatility = 0.0

    def train(self, df: pd.DataFrame):
        if df.empty:
            return
        
        df = df.sort_values('date')
        
        if 'returns' not in df.columns:
            returns = df['close'].pct_change()
        else:
            returns = df['returns']
            
        rolling_mean = returns.rolling(window=self.window).mean()
        if rolling_mean.dropna().empty:
             self.last_mean_return = 0.0
        else:
             self.last_mean_return = rolling_mean.iloc[-1]
        
        rolling_std = returns.rolling(window=self.window).std()
        if rolling_std.dropna().empty:
            self.volatility = 0.01 
        else:
            self.volatility = rolling_std.iloc[-1]
             
        self.last_price = df['close'].iloc[-1]
        self.last_date = df['date'].iloc[-1]

    def predict(self, df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        if self.last_date is None:
            self.train(df)
            
        future_dates = pd.bdate_range(start=self.last_date + pd.Timedelta(days=1), periods=horizon).tolist()

        predictions = []
        current_price = self.last_price
        z_score = 1.645
        
        for i, date in enumerate(future_dates):
            pred_ret = self.last_mean_return
            current_price = current_price * (1 + pred_ret)
            
            t = i + 1
            sigma_t = self.volatility * np.sqrt(t)
            
            lower = current_price * (1 - z_score * sigma_t)
            upper = current_price * (1 + z_score * sigma_t)
            
            predictions.append({
                'date': date,
                'predicted_return': pred_ret,
                'predicted_price': current_price,
                'predicted_lower': lower,
                'predicted_upper': upper
            })
            
        return pd.DataFrame(predictions)
    
    # Manual CV since MA isn't in Nixtla
    def cross_validate(self, df: pd.DataFrame, horizon: int, n_windows: int, step_size: int = None) -> pd.DataFrame:
        if step_size is None:
            step_size = horizon
            
        df = df.sort_values('date').reset_index(drop=True)
        results = []
        
        # Determine splits
        # We need at least window + horizon points
        min_required = self.window + horizon
        if len(df) < min_required:
            return pd.DataFrame()
            
        # Walk backward from end to create windows
        end_idx = len(df)
        
        for i in range(n_windows):
            cutoff_idx = end_idx - horizon - (i * step_size)
            if cutoff_idx < self.window:
                break
                
            train = df.iloc[:cutoff_idx].copy()
            valid = df.iloc[cutoff_idx : cutoff_idx + horizon].copy()
            
            if len(valid) == 0:
                continue
                
            # Train on train set (create new instance to avoid state corruption)
            model = MovingAverageModel(horizon=horizon, window=self.window)
            model.train(train)
            preds = model.predict(train, horizon=horizon) # horizon usually matches valid length
            
            # Align preds with valid dates (if dates match)
            # MA predict generates dates based on last training date
            # Check alignment
            valid = valid.reset_index(drop=True)
            preds = preds.reset_index(drop=True)
            
            # Combine
            # Result format: ds, cutoff, y, <model_name>
            # preds has predicted_price
            
            cutoff_date = train['date'].iloc[-1]
            
            combined = pd.DataFrame({
                'unique_id': 'A',
                'ds': preds['date'],
                'cutoff': cutoff_date,
                'y': valid['close'], # actuals
                'MovingAverageModel': preds['predicted_price'] # prediction
            })
            
            results.append(combined)
            
        if not results:
            return pd.DataFrame()
            
        return pd.concat(results).reset_index(drop=True)
