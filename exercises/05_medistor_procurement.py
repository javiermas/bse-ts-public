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
    from ortools.linear_solver import pywraplp

    forecast = pd.read_csv(
        Path(__file__).parent / "data" / "05_medistor_procurement.csv",
        parse_dates=["month"],
    )
    forecast = forecast.set_index("month")
    return forecast, mo, np, pd, plt, pywraplp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 05: Optimal procurement under demand uncertainty

    You have been in the procurement analytics team at **Medistor** for three months.
    The demand forecast you delivered last week — twelve monthly point estimates with
    95% prediction intervals for the private-label antihistamine — is now sitting on
    Elena Voss's desk. She walks over:

    > "Good work on the forecast. Now comes the hard part. I need a procurement
    > schedule for 2024: how many units do we order each month, and when? We have
    > 65,000 units available at the start of January — the last December shipment
    > just arrived. The supplier works on a one-month lead time, so whatever we
    > order this month arrives at the start of next month. They also have a
    > pallet minimum: every order must be a whole number of pallets, and each
    > pallet holds 10,000 units. The warehouse holds at most 120,000. Give me
    > a plan that minimises cost, and tell me what changes if we plan against
    > the upper end of the forecast interval instead of the point estimate."

    The data on your desk:

    - `forecast` — twelve-row table with columns `forecast`, `lower_95`,
      `upper_95` for January–December 2024, from your SARIMAX model in
      Exercise 4.

    Cost and constraint parameters (define as constants in the cell below):

    | Parameter | Value |
    |---|---|
    | Unit purchase cost | €4.20 / unit |
    | Unit selling price | €8.50 / unit |
    | Holding cost | €0.15 / unit / month |
    | Stockout cost | €3.00 / unit short (lost margin + emergency reorder) |
    | Warehouse capacity | 120,000 units |
    | Starting inventory | 65,000 units (January 1 on-hand) |
    | Lead time | 1 month (order placed in month $t$ arrives at start of month $t+1$) |
    | Pallet size (minimum order unit) | 10,000 units; all orders must be a whole number of pallets |

    You will:

    1. Formulate the problem — decision variables, objective, constraints.
    2. Solve the LP relaxation (ignore the pallet constraint) on the point forecast.
    3. Solve the MILP (integer pallet multiples) and compare.
    4. Solve the MILP on the upper-95% demand scenario and quantify the cost of robustness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Parameters

    Define the cost and constraint constants here. Downstream cells consume them.
    """)
    return


@app.cell
def _():
    UNIT_COST = 4.20
    SELL_PRICE = 8.50
    HOLDING_COST = 0.15
    STOCKOUT_COST = 3.00
    CAPACITY = 120_000
    I_START = 65_000
    PALLET_SIZE = 10_000
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Problem formulation

    Before writing any code, write down in plain English (or maths):

    1. **Decision variables** — what are you choosing, and for which time periods?
    2. **Objective function** — what are you minimising, and why?
    3. **Constraints** — list every hard limit. Which constraint introduces integrality?
    4. **Lead time** — how does the one-month lead time appear in the inventory
       balance equation?

    *Hint on the objective*: purchasing a unit costs €4.20; selling it earns
    €8.50; a stockout costs €3.00. What is the effective penalty for a unit
    short — i.e., the combined cost of lost margin and the emergency premium?
    """)
    return


@app.cell
def _():
    # Your written answer here (use a comment block or a mo.md cell)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. LP relaxation — point forecast

    Ignore the pallet constraint for now. Treat order quantities as continuous
    non-negative variables and solve with `pywraplp` using the `GLOP` solver.

    The inventory balance with one-month lead time is:

    $$I_t = I_{t-1} + o_{t-1} - d_t, \quad t = 1, \ldots, 12$$

    where $o_{t-1}$ is the order placed in month $t-1$ (arriving at the start
    of month $t$), $d_t$ is demand in month $t$, and $I_0 = 65{,}000$.

    **Note on the objective**: you are minimising total cost, which includes
    unit purchase cost, holding cost, and stockout penalties. Because units
    that are stocked out generate lost revenue in addition to the emergency
    reorder premium, the effective per-unit stockout cost is
    $p_{\text{sell}} + p_{\text{stockout}}$, not just $p_{\text{stockout}}$
    alone.

    Solve and print the monthly plan: order quantity, end-of-month inventory,
    and holding cost for each month. Report the total cost.
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. MILP — pallet constraint on the point forecast

    Now enforce that every order must be a whole number of pallets (10,000 units
    each). Replace the continuous order variable with an integer variable
    $k_t \in \mathbb{Z}_{\geq 0}$ representing the number of pallets, so that
    the physical order quantity is $o_t = 10{,}000 \cdot k_t$.

    Use the `SCIP` solver (it handles integer variables; `GLOP` does not).

    Compare the MILP plan to the LP plan:

    - How do the monthly order quantities differ?
    - What is the cost premium from imposing the pallet constraint?
    - Does inventory accumulate or deplete differently over the year?

    Plot both plans (LP and MILP) on the same axes: order quantities and
    end-of-month inventory side by side.
    """)
    return


@app.cell
def _():
    # Your code here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Robust procurement — planning against the upper bound

    Re-solve the MILP using the **upper 95% prediction interval** as the demand
    scenario. This is the "plan for the worst reasonable case" strategy.

    Questions to answer:

    1. How does the procurement schedule change? Which months are most affected?
    2. What is the total cost of the robust plan versus the point-forecast plan?
    3. If actual demand turns out to be the point forecast, how much excess
       inventory does the robust plan leave at year-end, and what does that
       cost in holding charges?
    4. **Is the extra cost of robustness justified?** Calculate the break-even
       probability: how likely does Elena need to believe that demand will hit
       the upper bound before the robust plan is worth the premium?

    *There is no single right answer to question 4. Your job is to lay out the
    trade-off clearly enough that Elena can make the call.*
    """)
    return


@app.cell
def _():
    # Your code here
    return


if __name__ == "__main__":
    app.run()
