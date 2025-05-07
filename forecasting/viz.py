import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


def plot_predictions(train_df, test_df, short_forecast, long_forecast=None, fitted=None, name=None, days_before=1):
    time_to_viz = pd.Timestamp(test_df.index.min()) - pd.Timedelta(days=days_before)
    real = pd.concat([train_df, test_df]).loc[time_to_viz:, 'total_consumption']
    fitted = fitted.loc[time_to_viz:] if fitted is not None else None
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.plot(real, label='real values', marker='o', markersize=3)
    if fitted is not None:
        ax.plot(fitted, label='fitted values')

    ax.plot(short_forecast, label='forecasted values (short)')
    if long_forecast is not None:
        ax.plot(long_forecast, label='forecasted values (long)')
    ax.set_title(name)
    ax.legend()
    return round(mean_absolute_error(test_df['total_consumption'], short_forecast), 2)