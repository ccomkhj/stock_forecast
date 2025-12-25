from lightgbm import LGBMRegressor
from .nixtla_adapter import MLForecastAdapter
from .config import get_model_config

class LightGBMModel(MLForecastAdapter):
    def __init__(self, horizon=30, **kwargs):
        conf = get_model_config("LightGBM")
        
        lgbm_params = conf.get('params', {}).copy()
        # Filter kwargs that match lgbm params or update
        # Simplified: just update. LGBM kwargs usually accepted.
        lgbm_params.update({k:v for k,v in kwargs.items() if k not in ['lags', 'date_features', 'diff_order']})
        
        lags = kwargs.get('lags', conf.get('lags'))
        date_features = kwargs.get('date_features', conf.get('date_features'))
        diff_order = kwargs.get('diff_order', conf.get('diff_order', 1))
        
        model = LGBMRegressor(**lgbm_params)
        super().__init__(model, horizon=horizon, lags=lags, date_features=date_features, diff_order=diff_order, **kwargs)