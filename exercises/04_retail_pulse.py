import marimo

__generated_with = "0.23.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    sales_data = pd.read_csv(
        Path(__file__).parent / "data" / "04_retail_pulse.csv",
        parse_dates=["date"],
    )
    sales_data = sales_data.set_index("date")
    sales_data.index.freq = "MS"
    sales = sales_data["sales"]
    return mo, np, pd, plt, sales, sales_data


@app.cell(hide_code=True)
def _(mo, np, pd, plt, sales, sales_data):
    _start = pd.Timestamp(sales.index[0]).strftime("%b %Y")
    _end = pd.Timestamp(sales.index[-1]).strftime("%b %Y")
    mo.md(
        f"**Data loaded:** `sales` — {len(sales_data)} monthly observations"
        f" ({_start}–{_end}),"
        f" mean {np.mean(sales):,.0f} units/month."
        f" (`plt`, `pd`, `np` are available in your cells.)"
    )
    _ = plt.rcParams
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 04: Retail Pulse — forecasting OTC allergy tablet demand

    After a couple years at the IEA, you're ready for a new challenge.
    You're considering buying an appartment and with the current prices you'd need
    a second job to afford it. So when a recruiter from the pharmaceutical
    industry reaches out, you jump at the opportunity to apply your data
    skills in a new context.

    You joined the procurement analytics team at **Medistor**, a mid-size
    generics distributor that supplies independent pharmacies across the country.
    Your first assignment is the top-selling OTC product in the allergy
    category: a private-label antihistamine that has been quietly gaining
    shelf space over the past decade.

    The category manager, **Elena Voss**, drops a CSV on your desk:

    > "Sixteen years of monthly sales. We need a 12-month forward forecast —
    > point estimate and a 95% interval — to feed into next year's procurement
    > plan. The spring peak is what keeps me up at night: if we under-call it
    > we lose shelf space, if we over-call it we write off expired stock.
    > Make sure the model earns its keep on that month."

    The data:

    - `sales` — monthly units sold, January 2008 through December 2023
      (192 months). Integer counts. Clean series; no missing values.

    You will:

    1. Explore the series — trend, seasonal pattern, and what they suggest
       about the right modelling approach.
    2. Fit `SARIMAX`, hold out the last 12 months, and evaluate on the spring
       peak.
    3. Refit on all data and deliver the 12-month forecast with intervals.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Explore the series

    Plot the raw series and any transformations that help you characterise it.
    Decompose or compute rolling statistics if that helps.

    What seasonal pattern is present? Is there a trend, and what shape does it
    take? Which calendar month is the spring peak? What does the seasonal
    structure suggest about the right modelling approach — and does that
    suggestion change depending on what you think is driving the seasonality?
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fit SARIMAX and evaluate on a held-out period

    Hold out the last 12 months. Fit a `SARIMAX` model from `statsmodels` on
    the remaining data, then produce 12-step-ahead forecasts on the held-out
    window.

    Defend the specification you chose. What is the MAE on the held-out window?
    Does the model capture the spring peak adequately?
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Deliver the forecast

    Refit the model on all 192 months, then forecast 12 months ahead from the
    end of the series. Extract a 95% prediction interval.

    Present the full 12-month forecast table (point estimate and interval for
    each month). Call out the spring peak month specifically: what is the
    point forecast and how wide is the interval? Is the model precise enough
    to be useful for procurement planning?
    """)
    return


@app.cell
def _():
    # Your code here
    return


if __name__ == "__main__":
    app.run()
