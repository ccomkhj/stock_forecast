from statsforecast.models import AutoMFLES
from .nixtla_adapter import StatsForecastAdapter
from .config import get_model_config

class AutoMFLESModel(StatsForecastAdapter):
    def __init__(self, horizon=30, **kwargs):
        conf = get_model_config("AutoMFLES")
        params = conf.get('params', {}).copy()
        
        # Override/Set specific params
        season_length = kwargs.pop('season_length', params.get('season_length', 5))
        test_size = kwargs.pop('test_size', horizon)
        
        model = AutoMFLES(season_length=season_length, test_size=test_size) 
        super().__init__(model, horizon=horizon, **kwargs)