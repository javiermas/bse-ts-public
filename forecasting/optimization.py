from ortools.linear_solver import pywraplp
import pandas as pd


def battery_optimization_milp(
    cost_series,
    demand_series,
    battery_capacity=15,
    charge_limit=3,
    discharge_limit=3,
):
    """
    Optimize battery charging/discharging to minimize energy cost.

    Args:
        cost_series (pd.Series): Energy cost (cents per kWh) at each time step.
        demand_series (pd.Series): Household energy demand (kWh) at each time step.
        battery_capacity (float): Max battery storage capacity (kWh).
        charge_limit (float): Max charging power (kW).
        discharge_limit (float): Max discharging power (kW).
        efficiency (float): Battery round-trip efficiency.

    Returns:
        pd.DataFrame: Optimized battery operation (charge, discharge, storage).
    """
    solver = pywraplp.Solver.CreateSolver("SCIP")
    if not solver:
        raise RuntimeError("Solver not available.")

    times = cost_series.index
    T = len(times)  # Number of time steps

    # Decision variables
    charge = [solver.NumVar(0, charge_limit, f"charge_{t}") for t in range(T)]
    discharge = [solver.NumVar(0, discharge_limit, f"discharge_{t}") for t in range(T)]
    storage = [solver.NumVar(0, battery_capacity, f"storage_{t}") for t in range(T)]
    is_charging = [solver.BoolVar(f"is_charging_{t}") for t in range(T)]

    # Constraints
    for t in range(T):
        if t == 0:
            # Initial storage (assume battery starts empty)
            solver.Add(storage[t] == (0 + charge[t] - discharge[t]))
        else:
            # Storage balance
            solver.Add(storage[t] == (storage[t - 1] + charge[t] - discharge[t]))

        # Storage must stay within limits
        solver.Add(storage[t] >= 0)
        solver.Add(storage[t] <= battery_capacity)

        # Prevent simultaneous charging & discharging using binary variable
        solver.Add(charge[t] <= charge_limit * is_charging[t])
        solver.Add(discharge[t] <= discharge_limit * (1 - is_charging[t]))

        # Discharge cannot exceed demand
        solver.Add(discharge[t] <= demand_series.iloc[t])

    # Objective: Minimize total energy cost
    effective_demand = [demand_series.iloc[t] + charge[t] - discharge[t] for t in range(T)]
    total_cost = solver.Sum(effective_demand[t] * cost_series.iloc[t] for t in range(T))
    solver.Minimize(total_cost)

    # Solve
    status = solver.Solve()
    if status != pywraplp.Solver.OPTIMAL:
        raise RuntimeError("No optimal solution found.")

    # Extract solution
    result = pd.DataFrame(
        {
            "storage": [storage[t].solution_value() for t in range(T)],
            "is_charging": [is_charging[t].solution_value() for t in range(T)],
            "demand": demand_series.values,
            "charge": [charge[t].solution_value() for t in range(T)],
            "discharge": [round(discharge[t].solution_value(), 4) for t in range(T)],
            "effective_demand": [
                round(e.solution_value(), 4) for e in effective_demand
            ],
            "cost": cost_series.values,
            "effective_cost": [
                round((effective_demand[t] * cost_series.iloc[t]).solution_value(), 4)
                for t in range(T)
            ],
        },
        index=cost_series.index,
    )

    return result
