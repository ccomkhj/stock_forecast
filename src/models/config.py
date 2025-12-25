from typing import List, Dict, Any

class ModelConfig:
    # Feature Engineering
    lags: List[int] = [1, 2, 3, 4, 5, 10, 20]
    date_features: List[str] = ['month', 'dayofweek', 'day', 'week']
    # Transformations (represented as strings or simple configs for now)
    diff_order: int = 1
    
    # Model Hyperparameters
    rf_params: Dict[str, Any] = {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    }
    
    lgbm_params: Dict[str, Any] = {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': -1,
        'random_state': 42,
        'verbosity': -1
    }
    
    automfles_params: Dict[str, Any] = {
        'season_length': 5, # Weekly seasonality for daily data
        # test_size is usually dynamic based on horizon
    }
    
    ma_params: Dict[str, Any] = {
        'window': 20
    }

def get_model_config(model_name: str) -> Dict[str, Any]:
    config = ModelConfig()
    if model_name == "Random Forest":
        return {
            'lags': config.lags,
            'date_features': config.date_features,
            'diff_order': config.diff_order,
            'params': config.rf_params
        }
    elif model_name == "LightGBM":
        return {
            'lags': config.lags, # Maybe more lags for LGBM?
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
