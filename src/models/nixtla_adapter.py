import pandas as pd
import numpy as np
import optuna
from .base import BaseModel
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from mlforecast.utils import PredictionIntervals
from statsforecast import StatsForecast
from statsforecast.utils import ConformalIntervals
from sklearn.metrics import mean_absolute_error

class MLForecastAdapter(BaseModel):
    def __init__(self, model, horizon=30, lags=None, lag_transforms=None, date_features=None, diff_order=1, **kwargs):
        self.horizon = horizon
        self.lags = lags if lags else [1, 2, 5]
        self.lag_transforms = lag_transforms if lag_transforms else {}
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
        
        # Add technical indicators if not present
        if 'rsi_14' not in data.columns:
            from src.indicators import calculate_indicators
            data = calculate_indicators(data)
            
        min_date = data['date'].min()
        max_date = data['date'].max()
        full_idx = pd.bdate_range(start=min_date, end=max_date)
        
        data_indexed = data.set_index('date')
        data_filled = data_indexed.reindex(full_idx)
        # Forward fill all columns including indicators
        data_filled = data_filled.ffill()
        
        data_filled = data_filled.dropna(subset=['close']).reset_index().rename(columns={'index': 'date'})
        
        # We can pass exogenous variables to MLForecast
        # For now let's just use y, but we could add RSI/Vol to nixtla_df
        # Actually MLForecast.fit(df) uses all columns except unique_id, ds, y as exogenous.
        
        nixtla_df = pd.DataFrame({
            'unique_id': 'A',
            'ds': data_filled['date'],
            'y': data_filled['close']
        })
        
        # Add exogenous features
        exo_cols = ['rsi_14', 'volatility_20', 'sma_20', 'sma_60']
        for col in exo_cols:
            if col in data_filled.columns:
                nixtla_df[col] = data_filled[col]
        
        return nixtla_df, data_filled['close'].iloc[-1], data_filled['date'].iloc[-1]

    def train(self, df: pd.DataFrame, tune=False):
        nixtla_df, last_price, last_date = self._prepare_data(df)
        self.last_price = last_price
        self.last_date = last_date
        
        if tune:
            self.tune(nixtla_df)
            
        target_transforms = [Differences([self.diff_order])] if self.diff_order > 0 else []

        self.mlf = MLForecast(
            models=[self.model_impl],
            freq='B',
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            date_features=self.date_features,
            target_transforms=target_transforms
        )
        
        n_samples = len(nixtla_df)
        n_windows = 2
        if n_samples < (self.horizon * n_windows + 20):
            n_windows = 1
            
        prediction_intervals = None
        if n_samples >= (self.horizon + 10):
            prediction_intervals = PredictionIntervals(n_windows=n_windows, h=self.horizon)
            
        self.mlf.fit(nixtla_df, prediction_intervals=prediction_intervals)

    def tune(self, nixtla_df: pd.DataFrame, n_trials=20):
        # Specific tuning for RF and LGBM
        model_name = self.model_impl.__class__.__name__
        
        def objective(trial):
            if "RandomForest" in model_name:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10)
                }
                new_model = self.model_impl.__class__(**params, random_state=42)
            elif "LGBM" in model_name:
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0)
                }
                new_model = self.model_impl.__class__(**params, random_state=42, verbosity=-1)
            else:
                return 0.0

            # Use a quick cross-validation to evaluate
            target_transforms = [Differences([self.diff_order])] if self.diff_order > 0 else []
            mlf_tune = MLForecast(
                models=[new_model],
                freq='B',
                lags=self.lags,
                lag_transforms=self.lag_transforms,
                date_features=self.date_features,
                target_transforms=target_transforms
            )
            
            # Use single window CV for speed during tuning
            cv = mlf_tune.cross_validation(
                nixtla_df,
                n_windows=1,
                h=self.horizon,
                step_size=self.horizon
            )
            
            if cv.empty:
                return float('inf')
            
            # Calculate MAE
            actual = cv['y'].values
            pred = cv[new_model.__class__.__name__].values
            return mean_absolute_error(actual, pred)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        # Update model_impl with best params
        best_params = study.best_params
        if "LGBM" in model_name:
            best_params['verbosity'] = -1
        self.model_impl = self.model_impl.__class__(**best_params, random_state=42)

    def predict(self, df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        if self.mlf is None:
            self.train(df)
        
        has_intervals = hasattr(self.mlf, 'prediction_intervals') and self.mlf.prediction_intervals is not None
        if has_intervals:
            preds = self.mlf.predict(horizon, level=[90])
        else:
            preds = self.mlf.predict(horizon)
        
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
        nixtla_df, _, _ = self._prepare_data(df)
        target_transforms = [Differences([self.diff_order])] if self.diff_order > 0 else []
        
        mlf_cv = MLForecast(
            models=[self.model_impl],
            freq='B',
            lags=self.lags,
            lag_transforms=self.lag_transforms,
            date_features=self.date_features,
            target_transforms=target_transforms
        )
        
        required = horizon * n_windows + 10
        if len(nixtla_df) < required:
            n_windows = max(1, int((len(nixtla_df) - 10) / horizon))
            if n_windows < 1:
                return pd.DataFrame()
        
        cv_results = mlf_cv.cross_validation(
            nixtla_df,
            n_windows=n_windows,
            h=horizon,
            step_size=step_size if step_size else horizon
        )
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

    def train(self, df: pd.DataFrame, tune=False):
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
