from sklearn.ensemble import RandomForestRegressor
from .nixtla_adapter import MLForecastAdapter
from .config import get_model_config

class RandomForestModel(MLForecastAdapter):
    def __init__(self, horizon=30, **kwargs):
        # Load config
        conf = get_model_config("Random Forest")
        
        # Merge kwargs with config params (kwargs override)
        rf_params = conf.get('params', {}).copy()
        rf_params.update({k:v for k,v in kwargs.items() if k in rf_params})
        
        lags = kwargs.get('lags', conf.get('lags'))
        date_features = kwargs.get('date_features', conf.get('date_features'))
        diff_order = kwargs.get('diff_order', conf.get('diff_order', 1))
        
        model = RandomForestRegressor(**rf_params)
        
        super().__init__(model, horizon=horizon, lags=lags, date_features=date_features, diff_order=diff_order, **kwargs)