# Time series forecasting

This repository consists of teaching material on time series forecasting for the MSc in Data Science of Barcelona School of Economics. The content of the course covers:
- Time-series data. Why does it need specific methods for modelling?
- Understanding trends, seasonalities, cycles.
- Methods for time-series forecasting: ARIMA, SARIMA, boosting.
- Constraint-based optimization: how to use time series forecasts in automated decision-making.

The theoretical content can be found in `slides.html`. The applied content can be found in `class-prepared-energy-consumption.ipynb`.

The evaluation will be done based on:
- 3 exercises on time series concepts that can be found in `exercises`.
- A final project explained in `final_project`.

## Installation

```
pip install uv==0.10.2
uv sync
```

## Final project

The final project consists of a task to forecast demand at a food delivery company.

### Data

You can find the train and mock test data under data. You can check your predictions are correctly formatted with check_output_format.py

