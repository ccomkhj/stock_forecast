import pytest
import pandas as pd
import numpy as np
from src.indicators import calculate_indicators
from src.features import prepare_features
from src.models.moving_average import MovingAverageModel
from src.models.random_forest import RandomForestModel
from src.models.lgbm import LightGBMModel
from src.models.automfles import AutoMFLESModel

def test_indicators():
    dates = pd.date_range(start='2023-01-01', periods=100)
    df = pd.DataFrame({
        'date': dates,
        'close': np.random.rand(100) * 100
    })
    res = calculate_indicators(df)
    assert 'sma_20' in res.columns
    assert 'rsi_14' in res.columns
    assert 'volatility_20' in res.columns

def test_features():
    dates = pd.date_range(start='2023-01-01', periods=200)
    df = pd.DataFrame({
        'date': dates,
        'close': np.linspace(100, 200, 200),
        'open': np.linspace(100, 200, 200),
        'high': np.linspace(100, 200, 200),
        'low': np.linspace(100, 200, 200),
        'volume': [1000] * 200,
        'symbol': 'TEST',
        'provider': 'test'
    })
    res = prepare_features(df)
    assert 'lag_return_0' in res.columns
    assert 'dist_sma_20' in res.columns
    assert not res.empty

def test_ma_model():
    dates = pd.date_range(start='2023-01-01', periods=50)
    df = pd.DataFrame({
        'date': dates,
        'close': np.linspace(100, 110, 50)
    })
    model = MovingAverageModel(window=10)
    model.train(df)
    preds = model.predict(df, horizon=5)
    assert len(preds) == 5
    assert 'predicted_price' in preds.columns

def test_rf_model():
    # Test Nixtla/MLForecast Wrapper
    dates = pd.date_range(start='2023-01-01', periods=60)
    df = pd.DataFrame({
        'date': dates,
        'close': np.linspace(100, 110, 60),
        'returns': np.random.normal(0, 0.01, 60)
    })
    model = RandomForestModel(horizon=5)
    model.train(df)
    preds = model.predict(df, horizon=5)
    assert len(preds) == 5
    assert 'predicted_return' in preds.columns

def test_lgbm_model():
    dates = pd.date_range(start='2023-01-01', periods=60)
    df = pd.DataFrame({
        'date': dates,
        'close': np.linspace(100, 110, 60),
        'returns': np.random.normal(0, 0.01, 60)
    })
    model = LightGBMModel(horizon=5)
    model.train(df)
    preds = model.predict(df, horizon=5)
    assert len(preds) == 5
    assert 'predicted_return' in preds.columns

def test_automfles_model():
    dates = pd.date_range(start='2023-01-01', periods=60)
    df = pd.DataFrame({
        'date': dates,
        'close': np.linspace(100, 110, 60) + np.sin(np.linspace(0, 10, 60))
    })
    model = AutoMFLESModel(horizon=5)
    model.train(df)
    preds = model.predict(df, horizon=5)
    assert len(preds) == 5
    assert 'predicted_price' in preds.columns