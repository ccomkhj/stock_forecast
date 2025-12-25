import pandas as pd
import numpy as np
from .base import BaseModel
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals
from statsforecast import StatsForecast

from statsforecast.utils import ConformalIntervals

class MLForecastAdapter(BaseModel):
    def __init__(self, model, horizon=30, lags=None, date_features=None, diff_order=1, **kwargs):
        self.horizon = horizon
        self.lags = lags if lags else [1, 2, 5]
        self.date_features = date_features if date_features else ['month', 'dayofweek']
        self.diff_order = diff_order
        self.model_impl = model
        self.mlf = None
        self.last_price = 0.0
        self.last_date = None
        
    def _prepare_data(self, df: pd.DataFrame):
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').dropna()
        
        # Robust Frequency: Reindex to 'B'
        min_date = data['date'].min()
        max_date = data['date'].max()
        full_idx = pd.bdate_range(start=min_date, end=max_date)
        
        data_indexed = data.set_index('date')
        data_filled = data_indexed.reindex(full_idx)
        data_filled['close'] = data_filled['close'].ffill()
        
        # Drop initial NaNs if any (before first valid data)
        data_filled = data_filled.dropna(subset=['close']).reset_index().rename(columns={'index': 'date'})
        
        nixtla_df = pd.DataFrame({
            'unique_id': 'A',
            'ds': data_filled['date'],
            'y': data_filled['close']
        })
        return nixtla_df, data_filled['close'].iloc[-1], data_filled['date'].iloc[-1]

    def train(self, df: pd.DataFrame):
        nixtla_df, last_price, last_date = self._prepare_data(df)
        self.last_price = last_price
        self.last_date = last_date
        
        target_transforms = [Differences([self.diff_order])] if self.diff_order > 0 else []

        self.mlf = MLForecast(
            models=[self.model_impl],
            freq='B',
            lags=self.lags,
            date_features=self.date_features,
            target_transforms=target_transforms
        )
        
        # Robust Interval Logic
        n_samples = len(nixtla_df)
        n_windows = 2
        if n_samples < (self.horizon * n_windows + 20):
            n_windows = 1
            
        prediction_intervals = None
        if n_samples >= (self.horizon + 10):
            prediction_intervals = PredictionIntervals(n_windows=n_windows, h=self.horizon)
            
        self.mlf.fit(nixtla_df, prediction_intervals=prediction_intervals)

    def predict(self, df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        if self.mlf is None:
            self.train(df)
            
        has_intervals = hasattr(self.mlf, 'prediction_intervals') and self.mlf.prediction_intervals is not None
        
        if has_intervals:
            preds = self.mlf.predict(horizon, level=[90])
        else:
            preds = self.mlf.predict(horizon)
        
        cols = preds.columns
        # Assumes single model in MLForecast
        # mlforecast column name is usually the class name, e.g. RandomForestRegressor
        # We find the column that is NOT ds, unique_id, or interval cols
        model_col = [c for c in cols if c not in ['ds', 'unique_id'] and '-lo-' not in c and '-hi-' not in c][0]
        
        pred_prices = preds[model_col].values
        future_dates = preds['ds'].values
        
        predictions = []
        prev_price = self.last_price
        
        has_lo = any('-lo-90' in c for c in cols)
        has_hi = any('-hi-90' in c for c in cols)
        
        lo_prices = preds[[c for c in cols if '-lo-90' in c][0]].values if has_lo else None
        hi_prices = preds[[c for c in cols if '-hi-90' in c][0]].values if has_hi else None
        
        for i, date in enumerate(future_dates):
            price = pred_prices[i]
            ret = (price - prev_price) / prev_price if prev_price != 0 else 0
            
            rec = {
                'date': date,
                'predicted_return': ret,
                'predicted_price': price
            }
            if has_lo and has_hi:
                rec['predicted_lower'] = lo_prices[i]
                rec['predicted_upper'] = hi_prices[i]
            
            predictions.append(rec)
            prev_price = price
            
        return pd.DataFrame(predictions)
    
    def cross_validate(self, df: pd.DataFrame, horizon: int, n_windows: int, step_size: int = None) -> pd.DataFrame:
        # Nixtla CV
        nixtla_df, _, _ = self._prepare_data(df)
        
        # Re-init model to ensure clean state or use existing configuration
        target_transforms = [Differences([self.diff_order])] if self.diff_order > 0 else []
        
        mlf_cv = MLForecast(
            models=[self.model_impl],
            freq='B',
            lags=self.lags,
            date_features=self.date_features,
            target_transforms=target_transforms
        )
        
        # Check size
        required = horizon * n_windows + 10
        if len(nixtla_df) < required:
            # Fallback or reduce windows
            n_windows = max(1, int((len(nixtla_df) - 10) / horizon))
            if n_windows < 1:
                return pd.DataFrame()
        
        cv_results = mlf_cv.cross_validation(
            nixtla_df,
            n_windows=n_windows,
            h=horizon,
            step_size=step_size if step_size else horizon
        )
        
        # Normalize result format: ds, cutoff, y, predicted
        # mlforecast cv returns: unique_id, ds, cutoff, y, <model_name>
        return cv_results


class StatsForecastAdapter(BaseModel):
    def __init__(self, model, horizon=30, **kwargs):
        self.horizon = horizon
        self.model_impl = model
        self.sf = None
        self.last_price = 0.0
        
    def _prepare_data(self, df: pd.DataFrame):
        data = df.copy()
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').dropna()
        
        min_date = data['date'].min()
        max_date = data['date'].max()
        full_idx = pd.bdate_range(start=min_date, end=max_date)
        
        data_indexed = data.set_index('date')
        data_filled = data_indexed.reindex(full_idx)
        data_filled['close'] = data_filled['close'].ffill()
        data_filled = data_filled.dropna(subset=['close']).reset_index().rename(columns={'index': 'date'})
        
        nixtla_df = pd.DataFrame({
            'unique_id': 'A',
            'ds': data_filled['date'],
            'y': data_filled['close']
        })
        return nixtla_df, data_filled['close'].iloc[-1]

    def train(self, df: pd.DataFrame):
        nixtla_df, last_price = self._prepare_data(df)
        self.last_price = last_price
        
        self.sf = StatsForecast(
            models=[self.model_impl],
            freq='B',
            n_jobs=1
        )
        
        n_samples = len(nixtla_df)
        n_windows = 2
        
        if n_samples < (self.horizon * n_windows + 20):
            n_windows = 1
            
        if n_samples < (self.horizon + 10):
            self.sf.fit(nixtla_df)
        else:
            self.sf.fit(nixtla_df, prediction_intervals=ConformalIntervals(h=self.horizon, n_windows=n_windows))

    def predict(self, df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        if self.sf is None:
            self.train(df)
        
        try:
            preds = self.sf.predict(h=horizon, level=[90])
        except Exception:
            preds = self.sf.predict(h=horizon)
        
        cols = preds.columns
        model_col = [c for c in cols if c not in ['ds', 'unique_id'] and '-lo-' not in c and '-hi-' not in c][0]
        
        pred_prices = preds[model_col].values
        future_dates = preds['ds'].values
        
        predictions = []
        prev_price = self.last_price
        
        has_lo = any('-lo-90' in c for c in cols)
        has_hi = any('-hi-90' in c for c in cols)
        
        lo_prices = preds[[c for c in cols if '-lo-90' in c][0]].values if has_lo else None
        hi_prices = preds[[c for c in cols if '-hi-90' in c][0]].values if has_hi else None
        
        for i, date in enumerate(future_dates):
            price = pred_prices[i]
            ret = (price - prev_price) / prev_price if prev_price != 0 else 0
            
            rec = {
                'date': date,
                'predicted_return': ret,
                'predicted_price': price
            }
            if has_lo and has_hi:
                rec['predicted_lower'] = lo_prices[i]
                rec['predicted_upper'] = hi_prices[i]
                
            predictions.append(rec)
            prev_price = price
            
        return pd.DataFrame(predictions)
        
    def cross_validate(self, df: pd.DataFrame, horizon: int, n_windows: int, step_size: int = None) -> pd.DataFrame:
        nixtla_df, _ = self._prepare_data(df)
        
        sf_cv = StatsForecast(
            models=[self.model_impl],
            freq='B',
            n_jobs=1
        )
        
        required = horizon * n_windows + 10
        if len(nixtla_df) < required:
            n_windows = max(1, int((len(nixtla_df) - 10) / horizon))
            if n_windows < 1:
                return pd.DataFrame()
        
        cv_results = sf_cv.cross_validation(
            df=nixtla_df,
            h=horizon,
            step_size=step_size if step_size else horizon,
            n_windows=n_windows
        )
        
        return cv_results