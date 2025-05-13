from sktime.forecasting.arima import ARIMA, AutoARIMA
from joblib import Memory
import pandas as pd

location = "~/.cache"
memory = Memory(location, verbose=1)


@memory.cache
def train_arima(data_path: str, **params) -> ARIMA:
    print("Training ARIMA model...")
    data = pd.read_pickle(data_path)
    arima = ARIMA(**params)
    return arima.fit(data)


@memory.cache
def train_auto_arima(data_path: str, **params) -> ARIMA:
    print("Training ARIMA model...")
    data = pd.read_pickle(data_path)
    arima = AutoARIMA(**params)
    return arima.fit(data)
