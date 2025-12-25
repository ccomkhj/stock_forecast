from typing import List, Dict, Any
from mlforecast.lag_transforms import RollingMean, RollingStd, ExponentiallyWeightedMean

class ModelConfig:
    # Feature Engineering
    lags: List[int] = [1, 2, 5, 10, 20]
    lag_transforms: Dict[int, List[Any]] = {
        1: [RollingMean(window_size=5), RollingStd(window_size=5), RollingMean(window_size=20)],
        5: [RollingMean(window_size=5)]
    }
    date_features: List[str] = ['month', 'dayofweek']
    diff_order: int = 1
    
    # Base Hyperparameters
    rf_params: Dict[str, Any] = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42,
        'n_jobs': -1
    }
    
    lgbm_params: Dict[str, Any] = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': -1,
        'random_state': 42,
        'verbosity': -1,
        'n_jobs': -1
    }
    
    automfles_params: Dict[str, Any] = {
        'season_length': 5,
    }
    
    ma_params: Dict[str, Any] = {
        'window': 20
    }

def get_model_config(model_name: str) -> Dict[str, Any]:
    config = ModelConfig()
    if model_name == "Random Forest":
        return {
            'lags': config.lags,
            'lag_transforms': config.lag_transforms,
            'date_features': config.date_features,
            'diff_order': config.diff_order,
            'params': config.rf_params
        }
    elif model_name == "LightGBM":
        return {
            'lags': config.lags,
            'lag_transforms': config.lag_transforms,
            'date_features': config.date_features,
            'diff_order': config.diff_order,
            'params': config.lgbm_params
        }
    elif model_name == "AutoMFLES":
        return {
            'params': config.automfles_params
        }
    elif model_name == "Moving Average":
        return {
            'params': config.ma_params
        }
    return {}