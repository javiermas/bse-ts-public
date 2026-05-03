import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import adfuller
    from pathlib import Path

    HORIZON = 12
    y = pd.read_csv(
        Path(__file__).parent / "data" / "02_bayes_product.csv",
        index_col="date", parse_dates=True,
    )["value"].values
    n_train = len(y) - HORIZON
    y_train, y_test = y[:n_train], y[n_train:]
    return HORIZON, adfuller, n_train, np, plt, sm, y, y_test, y_train


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 02: The Brief

    You have just joined the quantitative research team at **Bayes Capital**.
    Your manager drops a dataset on your desk and says:

    > *"We're getting a new structured product through approval.
    > The underlying index has twelve years of monthly history.
    > I need a twelve-month forecast with a 95% prediction interval at each step —
    > the client is sophisticated and wants to understand their exposure.
    > Use what you know."*

    The data is in `y`. The last 12 months are held out for evaluation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Explore the data

    Plot the series. Then use `adfuller` to test for a unit root on the training portion.
    What do you conclude about the series?
    """)
    return


@app.cell
def _(adfuller, n_train, plt, y, y_train):
    stat, p, *_ = adfuller(y_train)
    print(f"ADF stat={stat:.3f}, p={p:.4f}")

    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(y, color="#4C72B0", linewidth=0.9)
    ax1.axvline(n_train, color="grey", linestyle=":", linewidth=1, label="Train / test split")
    ax1.set_title("Bayes Capital product index")
    ax1.legend()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Build a model and forecast

    - Split: all but the last 12 months as training, last 12 as test.
    - Fit `sm.OLS` using `y_{t-1}` and a linear time trend as features.
      Use `sm.add_constant` and `sm.OLS` — this gives you prediction intervals
      via `get_prediction().summary_frame(alpha=0.05)`.
    - Report the in-sample R² and the estimated β coefficient.
    - Forecast 12 months ahead **iteratively**: at each step, feed the previous
      prediction back as the input feature. Record the 95% prediction interval
      at each step.
    - Plot the forecast and prediction intervals against the held-out 12 months.
    """)
    return


@app.cell
def _(HORIZON, n_train, np, plt, sm, y, y_test, y_train):
    t_train = np.arange(1, len(y_train))
    X_train = sm.add_constant(np.column_stack([y_train[:-1], t_train]))
    ols = sm.OLS(y_train[1:], X_train).fit()
    print(f"R²={ols.rsquared:.4f},  β={ols.params[1]:.4f},  γ={ols.params[2]:.4f}")

    preds, lo, hi = [], [], []
    last = float(y_train[-1])
    for h in range(HORIZON):
        t_step = n_train + h
        x = sm.add_constant([[last, t_step]], has_constant="add")
        f = ols.get_prediction(x).summary_frame(alpha=0.05)
        preds.append(f["mean"].iloc[0])
        lo.append(f["obs_ci_lower"].iloc[0])
        hi.append(f["obs_ci_upper"].iloc[0])
        last = preds[-1]
    preds, lo, hi = np.array(preds), np.array(lo), np.array(hi)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(range(n_train - 60, n_train), y[n_train - 60:n_train],
             color="#4C72B0", label="Train (last 60)")
    ax2.plot(range(n_train, n_train + HORIZON), y_test,
             color="#4C72B0", linestyle="--", label="Actual")
    ax2.plot(range(n_train, n_train + HORIZON), preds,
             color="#C44E52", lw=2, label="OLS forecast")
    ax2.fill_between(range(n_train, n_train + HORIZON), lo, hi,
                     color="#C44E52", alpha=0.15, label="95% PI")
    ax2.axvline(n_train, color="grey", linestyle=":", lw=1)
    ax2.legend(fontsize=9)
    ax2.set_title("OLS(lag-1 + trend) forecast")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Reflection

    1. Your estimated β is close to 1. Is that what you expected given the series you
       plotted? What does β ≈ 1 imply for a multi-step iterative forecast?

    2. Look at where your 12-step forecast ends up relative to where the training
       series was heading. Does the forecast follow the series' momentum, or does
       something else happen?

    3. Examine the width of your 95% prediction intervals at step 1 and step 12.
       What pattern do you see? Is that consistent with the uncertainty you would
       expect from a series with a unit root?

    4. Your model uses last month's value and a linear time trend. Adding the trend
       helps with the level, but if the way shocks propagate has structure beyond
       these features — for instance, if yesterday's shock partially offsets today's —
       can OLS recover that? What would you need to change?
    """)
    return


if __name__ == "__main__":
    app.run()
