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

    _df = pd.read_csv(
        Path(__file__).parent / "data" / "03_world_co2.csv",
        index_col="year",
    )
    co2 = _df["co2"].dropna()
    population = _df["population"]
    co2_per_capita = _df["co2_per_capita"].dropna()
    return co2, co2_per_capita, mo, population


@app.cell(hide_code=True)
def _(co2, co2_per_capita, population):
    print(f"co2:            {int(co2.index.min())}–{int(co2.index.max())}  ({len(co2)} obs)")
    print(f"population:     {int(population.index.min())}–{int(population.index.max())}  ({len(population)} obs)")
    print(f"co2_per_capita: {int(co2_per_capita.index.min())}–{int(co2_per_capita.index.max())}  ({len(co2_per_capita)} obs)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 03: A new chapter — forecasting world CO2

    The structured-products book at **Bayes Capital** has been good to you:
    the structured-product brief from your first weeks, the OLS forecast that
    looked reasonable until you diagnosed the MA dynamics it was missing — the
    head of research has stopped second-guessing your model choices. But you
    keep catching yourself wondering whether forecasting bond yields and equity
    indices is the most useful thing you could be doing with what you've
    learned.

    Then a recruiter from the **International Energy Agency** in Paris reaches
    out. Someone there saw your analysis and remembered the analyst who
    insisted on inspecting the series before letting the software pick orders.
    They have a position open on the modelling team, and they're skipping
    straight to a take-home test:

    > *"Annual world CO2 emissions, 1960 through 2025. We need a forecast to
    > 2050 and a defensible argument for why we should believe it. Our outlook
    > report goes to the climate negotiation in November and the policy team
    > needs to know whether the curve bends — and if so, when. Pick whatever
    > model class you like. We care more about how you reason about it than
    > which family of estimators wins."*

    The data on your desk:

    - `co2` — annual World CO2 emissions 1960–2025 (OWID / Global Carbon
      Project; 2025 is provisional).
    - `population` — World population 1960–2050, historical through 2024 and
      UN WPP 2024 medium-variant projections to 2050.
    - `co2_per_capita` — tonnes CO2 per person, computed, 1960–2025.

    Unlike anything you ever shipped at Bayes, **there is no held-out truth
    here.** You are forecasting the actual future.

    You will:

    1. Explore the series.
    2. Fit a couple of ARIMA specifications and compare their forecasts to 2050.
    3. Pick a model and answer the IEA's question.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Explore the data

    Plot the series and any transformations of it that help you decide what
    to model.
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Fit ARIMA models and forecast to 2050

    Fit a couple of ARIMA specifications, plot the point forecasts on the same
    axis, and pick the one you'd defend.
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Recommendation

    1. Do emissions peak — and if so, when?
    2. What could make your forecast wrong?
    """)
    return


if __name__ == "__main__":
    app.run()
