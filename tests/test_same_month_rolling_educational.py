"""Educational tests: same-calendar-month rolling features use only prior same months.

Students can read the assertions as executable specifications of "no leakage" and of
exactly what ``sm_roll_mean_w`` means.

The helpers use irregular timestamps on purpose: only ``index`` order and calendar
month matter; a full ``freq="MS"`` grid is **not** required for this logic.

"""

import numpy as np
import pandas as pd
import pytest

from forecasting.modelling import same_month_roll_mean_column, same_month_rolling_means


def _series_at(dates: list[str], values: list[float]) -> pd.Series:
    """DatetimeIndex without asserting ``freq`` (sparse / same-calendar subsamples need no offset)."""
    return pd.Series(values, index=pd.to_datetime(dates))


def test_sm_roll_mean_w2_matches_hand_calculation():
    """At 2024-01: prior Januarys are 2022→10 and 2023→20; mean of last 2 is 15."""
    z = _series_at(["2022-01-01", "2023-01-01", "2024-01-01"], [10.0, 20.0, 30.0])
    at = pd.to_datetime(["2024-01-01"])
    out = same_month_rolling_means(z, [2], at)
    col = same_month_roll_mean_column(2)
    assert col in out.columns
    assert out.loc[at[0], col] == pytest.approx(15.0)


def test_sm_roll_fewer_than_w_prior_months_averages_all():
    """Only one prior January → mean of that single value (min_periods-style behaviour)."""
    z = _series_at(["2022-01-01", "2023-01-01"], [10.0, 20.0])
    at = pd.to_datetime(["2023-01-01"])
    out = same_month_rolling_means(z, [3], at)
    col = same_month_roll_mean_column(3)
    assert out.loc[at[0], col] == pytest.approx(10.0)


def test_sm_roll_first_occurrence_of_month_is_nan():
    z = _series_at(["2022-01-01"], [10.0])
    at = pd.to_datetime(["2022-01-01"])
    out = same_month_rolling_means(z, [1], at)
    assert np.isnan(out.loc[at[0], same_month_roll_mean_column(1)])


def test_sm_roll_does_not_use_current_timestamp():
    """Current row value must not enter the rolling mean (strict ``index < ts``)."""
    z = _series_at(["2022-01-01", "2023-01-01", "2024-01-01"], [10.0, 20.0, 999.0])
    at = pd.to_datetime(["2024-01-01"])
    out = same_month_rolling_means(z, [1], at)
    assert out.loc[at[0], same_month_roll_mean_column(1)] == pytest.approx(20.0)


def test_sm_roll_isolated_by_calendar_month():
    """February rolling must not mix January observations."""
    z = pd.concat(
        [
            _series_at(["2022-01-01"], [100.0]),
            _series_at(["2022-02-01"], [1.0]),
            _series_at(["2023-02-01"], [3.0]),
        ]
    ).sort_index()
    at = pd.to_datetime(["2023-02-01"])
    out = same_month_rolling_means(z, [2], at)
    assert out.loc[at[0], same_month_roll_mean_column(2)] == pytest.approx(1.0)


def test_future_spike_same_month_never_seen_by_earlier_at():
    """A later timestamp in the calendar cannot contaminate features at an earlier ``at``."""
    z = _series_at(["2023-01-01", "2024-01-01", "2025-01-01"], [10.0, 20.0, 1.0e6])
    at = pd.to_datetime(["2024-01-01"])
    out = same_month_rolling_means(z, [2], at)
    assert out.loc[at[0], same_month_roll_mean_column(2)] == pytest.approx(10.0)


@pytest.mark.parametrize("w", [0, -1, -3])
def test_sm_roll_rejects_non_positive_window(w: int):
    z = _series_at(["2022-01-01"], [1.0])
    at = pd.to_datetime(["2023-01-01"])
    with pytest.raises(ValueError, match="positive"):
        same_month_rolling_means(z, [w], at)
