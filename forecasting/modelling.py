"""Forecast modelling — SARIMA (pmdarima) and direct monthly horizons (XGBoost)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from pmdarima import ARIMA as PmdARIMA
from xgboost import XGBRegressor


class MonthlyHorizonModel(ABC):
    """Tabular forecaster: fit on history, predict a future DatetimeIndex."""

    @abstractmethod
    def fit(self, levels: pd.Series): ...

    @abstractmethod
    def predict(self, future_index: pd.DatetimeIndex) -> pd.Series: ...


DEFAULT_MONTHLY_LAG_MONTHS: tuple[int, ...] = (12, 13, 24)


def monthly_feature_columns(lag_months: Iterable[int]) -> list[str]:
    return [lag_column_name(k) for k in lag_months] + ["month", "t_idx"]


def same_month_roll_mean_column(window: int) -> str:
    return f"sm_roll_mean_{int(window)}"


def xgb_feature_column_names(
    lag_months: Iterable[int],
    rolling_same_month_windows: Sequence[int],
) -> list[str]:
    names = monthly_feature_columns(lag_months)
    for w in rolling_same_month_windows:
        names.append(same_month_roll_mean_column(w))
    return names


def same_month_rolling_means(
    z: pd.Series,
    windows: Sequence[int],
    at: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Mean of ``z`` over the last ``w`` *prior* timestamps with the same calendar month."""
    z = z.astype(float)
    out = pd.DataFrame(index=at)
    for w in windows:
        if int(w) < 1:
            raise ValueError("rolling_same_month_windows must be positive integers")
        w = int(w)
        col = same_month_roll_mean_column(w)
        vals: list[float] = []
        for ts in at:
            m = ts.month
            prev = z[(z.index < ts) & (z.index.month == m) & z.notna()]
            chunk = prev.iloc[-w:]
            vals.append(float(chunk.mean()) if len(chunk) else np.nan)
        out[col] = vals
    return out


def lag_column_name(lag_months_back: int) -> str:
    return f"lag_{lag_months_back}"


def _to_scaled_series(levels: pd.Series, use_log: bool) -> pd.Series:
    y = levels.astype(float)
    if use_log:
        return pd.Series(np.log(y.values), index=y.index, name="scaled")
    return y.rename("scaled")


def supervised_monthly_frame(
    z: pd.Series,
    lag_months: Sequence[int],
    *,
    seasonal_diff_target: bool = False,
) -> pd.DataFrame:
    z = z.astype(float)
    frame = pd.DataFrame(index=z.index)
    for k in lag_months:
        frame[lag_column_name(k)] = z.shift(k)
    frame["month"] = frame.index.month.astype(float)
    frame["t_idx"] = np.arange(len(frame), dtype=float)
    frame["target"] = z.diff(12) if seasonal_diff_target else z
    return frame


def future_feature_matrix(
    history: pd.Series,
    future_index: pd.DatetimeIndex,
    lag_months: Sequence[int],
) -> pd.DataFrame:
    feature_cols = monthly_feature_columns(lag_months)
    hist = history.astype(float)
    filler = pd.Series(np.nan, index=future_index, dtype=float)
    y_all = pd.concat([hist, filler]).sort_index()
    frame_all = pd.DataFrame({"z": y_all})
    for k in lag_months:
        frame_all[lag_column_name(k)] = frame_all["z"].shift(k)
    frame_all["month"] = frame_all.index.month.astype(float)
    frame_all["t_idx"] = np.arange(len(frame_all), dtype=float)
    return frame_all.loc[future_index, feature_cols]


def xgb_horizon_feature_matrix(
    history_z: pd.Series,
    future_index: pd.DatetimeIndex,
    lag_months: Sequence[int],
    rolling_same_month_windows: Sequence[int],
    feature_cols: list[str],
) -> pd.DataFrame:
    hist = history_z.astype(float)
    filler = pd.Series(np.nan, index=future_index, dtype=float)
    z_all = pd.concat([hist, filler]).sort_index()
    frame_all = pd.DataFrame({"z": z_all})
    for k in lag_months:
        frame_all[lag_column_name(k)] = frame_all["z"].shift(k)
    frame_all["month"] = frame_all.index.month.astype(float)
    frame_all["t_idx"] = np.arange(len(frame_all), dtype=float)
    X = frame_all.loc[future_index, monthly_feature_columns(lag_months)]
    if rolling_same_month_windows:
        rolls = same_month_rolling_means(
            z_all, rolling_same_month_windows, future_index
        )
        X = pd.concat([X, rolls], axis=1)
    return X[feature_cols]


def backtest_ranges(
    index: pd.DatetimeIndex,
    *,
    initial_train_span: pd.DateOffset,
    test_span: pd.DateOffset,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    first_test_start = index[0] + initial_train_span
    candidate_starts = pd.date_range(first_test_start, index[-1], freq=test_span)
    ranges: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for test_start in candidate_starts:
        test_end = test_start + test_span - pd.offsets.MonthBegin(1)
        if test_end <= index[-1]:
            train_end = test_start - pd.offsets.MonthBegin(1)
            ranges.append((train_end, test_start, test_end))
    return ranges


def _flatten_models_arg(models: object) -> list[MonthlyHorizonModel]:
    """Normalize `models` passed to `run_horizon_backtest`.

    Accepts a single model, a list/tuple of models, or legacy ``(label, model)`` pairs.
    """
    if isinstance(models, MonthlyHorizonModel):
        return [models]
    normalized: list[MonthlyHorizonModel] = []
    try:
        iterable = iter(models)
    except TypeError as e:
        raise TypeError(
            "run_horizon_backtest(..., models): expected an iterable of MonthlyHorizonModel "
            f"instances (or one instance). Got {type(models).__name__!r}."
        ) from e
    for item in iterable:
        if isinstance(item, tuple) and len(item) == 2 and isinstance(
            item[1], MonthlyHorizonModel
        ):
            normalized.append(item[1])
        elif isinstance(item, MonthlyHorizonModel):
            normalized.append(item)
        else:
            raise TypeError(
                "Each entry in models must be a MonthlyHorizonModel or a (str, model) tuple; "
                f"got {type(item).__name__!r}. If you see 'cannot unpack ... Model', your "
                "notebook or an old copy of this module may still use "
                "'for name, model in models' while passing plain model instances — use "
                "'for model in models' or restart the marimo kernel after updating."
            )
    return normalized


def run_horizon_backtest(
    levels: pd.Series,
    models: Sequence[MonthlyHorizonModel],
    *,
    initial_train_span: pd.DateOffset = pd.DateOffset(years=10),
    test_span: pd.DateOffset = pd.DateOffset(years=1),
) -> pd.DataFrame:
    rows: list[dict] = []
    model_instances = _flatten_models_arg(models)
    for train_end, test_start, test_end in backtest_ranges(
        levels.index,
        initial_train_span=initial_train_span,
        test_span=test_span,
    ):
        train_levels = levels.loc[:train_end]
        test_ix = levels.loc[test_start:test_end].index
        actual = levels.loc[test_ix]
        for model in model_instances:
            preds = model.fit(train_levels).predict(test_ix)
            mae = float(np.mean(np.abs(preds.values - actual.values)))
            rows.append(
                {
                    "model": str(model),
                    "train_end": train_levels.index[-1],
                    "test_start": test_ix[0],
                    "test_end": test_ix[-1],
                    "mae_units": mae,
                }
            )
    return pd.DataFrame(rows)


def summarize_horizon_backtest_mae(
    backtest_scores: pd.DataFrame,
    *,
    ewm_span: int = 3,
) -> pd.DataFrame:
    """One row per model: mean, last window, EWMA (by test window), and std of MAE."""
    ordered = backtest_scores.sort_values(["model", "test_start"])
    rows: list[dict] = []
    for name, sub in ordered.groupby("model", sort=False):
        x = sub["mae_units"]
        n = len(x)
        span = max(1, min(ewm_span, n))
        rows.append(
            {
                "model": name,
                "mean": float(x.mean()),
                "last": float(x.iloc[-1]),
                "exp_weighted_mean": float(
                    x.ewm(span=span, adjust=False).mean().iloc[-1]
                ),
                "std": float(x.std(ddof=1)) if n > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


class SARIMAMonthlyModel(MonthlyHorizonModel):
    """Seasonal ARIMA (`pmdarima`). If `use_log`, fit is on log levels and forecasts are exponentiated."""

    def __init__(
        self,
        order: tuple[int, int, int] = (0, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (0, 1, 1, 12),
        *,
        use_log: bool = False,
        **arima_kw,
    ):
        self._order = order
        self._seasonal_order = seasonal_order
        self._use_log = use_log
        self._arima_kw = arima_kw

    def __repr__(self) -> str:
        return (
            f"SARIMAMonthlyModel(order={self._order!r}, seasonal_order={self._seasonal_order!r}, "
            f"use_log={self._use_log!r})"
        )

    def __str__(self) -> str:
        p, d, q = self._order
        P, D, Q, m = self._seasonal_order
        m_suffix = "\u2081\u2082" if m == 12 else f" [{m}]"
        log_part = " · log" if self._use_log else ""
        return f"SARIMA ({p},{d},{q})({P},{D},{Q}){m_suffix}{log_part}"

    def fit(self, levels: pd.Series):
        y = levels.astype(float)
        vals = np.log(y.values) if self._use_log else y.values
        self._fitted = PmdARIMA(
            order=self._order,
            seasonal_order=self._seasonal_order,
            **self._arima_kw,
        ).fit(vals)
        return self

    def predict(self, future_index: pd.DatetimeIndex) -> pd.Series:
        raw = np.asarray(self._fitted.predict(n_periods=len(future_index)), dtype=float)
        out = np.exp(raw) if self._use_log else raw
        return pd.Series(out, index=future_index, name="sarima")


class XGBDirectHorizonModel(MonthlyHorizonModel):
    """Direct horizon booster. Optional log scale and optional seasonal (12‑month) differencing for the target."""

    def __init__(
        self,
        lag_months: Sequence[int] | None = None,
        *,
        use_log: bool = False,
        use_diff: bool = False,
        rolling_same_month_windows: Sequence[int] = (),
        n_estimators: int = 300,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        objective: str = "reg:squarederror",
        random_state: int = 0,
    ):
        self._lag_months = tuple(lag_months or DEFAULT_MONTHLY_LAG_MONTHS)
        self._rolling_windows = tuple(
            dict.fromkeys(int(w) for w in rolling_same_month_windows)
        )
        if any(w < 1 for w in self._rolling_windows):
            raise ValueError("rolling_same_month_windows must be positive integers")
        self._feature_cols = xgb_feature_column_names(
            self._lag_months, self._rolling_windows
        )
        self._use_log = use_log
        self._use_diff = use_diff
        self._xgb_kw = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            random_state=random_state,
            n_jobs=-1,
        )

    def __repr__(self) -> str:
        return (
            f"XGBDirectHorizonModel(lag_months={self._lag_months!r}, "
            f"use_log={self._use_log!r}, use_diff={self._use_diff!r}, "
            f"rolling_same_month_windows={self._rolling_windows!r})"
        )

    def __str__(self) -> str:
        lag_s = ",".join(str(x) for x in self._lag_months)
        parts = [f"lags {lag_s}"]
        if self._use_log:
            parts.append("log")
        if self._use_diff:
            parts.append("Δ₁₂ target")
        if self._rolling_windows:
            rw = ",".join(str(w) for w in self._rolling_windows)
            parts.append(f"same-month roll [{rw}]")
        return "XGBoost direct (" + ", ".join(parts) + ")"

    def fit(self, levels: pd.Series):
        self._history_scaled = _to_scaled_series(levels, self._use_log)
        fit_df = supervised_monthly_frame(
            self._history_scaled,
            self._lag_months,
            seasonal_diff_target=self._use_diff,
        )
        if self._rolling_windows:
            rolls = same_month_rolling_means(
                self._history_scaled,
                self._rolling_windows,
                fit_df.index,
            )
            fit_df = pd.concat([fit_df, rolls], axis=1)
        fit_df = fit_df.dropna()
        self._model = XGBRegressor(**self._xgb_kw)
        self._model.fit(
            fit_df[self._feature_cols].to_numpy(),
            fit_df["target"].to_numpy(),
        )
        return self

    def predict(self, future_index: pd.DatetimeIndex) -> pd.Series:
        X = xgb_horizon_feature_matrix(
            self._history_scaled,
            future_index,
            self._lag_months,
            self._rolling_windows,
            self._feature_cols,
        )
        raw = self._model.predict(X.to_numpy())
        if self._use_diff:
            filler = pd.Series(np.nan, index=future_index, dtype=float)
            combo = pd.concat([self._history_scaled, filler]).sort_index()
            z_lag12 = combo.shift(12).loc[future_index].to_numpy(dtype=float)
            raw = raw + z_lag12
        out = np.exp(raw) if self._use_log else raw
        return pd.Series(out, index=future_index, name="xgb_direct")


class SeasonalNaive12Model(MonthlyHorizonModel):
    """Same month last year, optionally shifted by last realized YoY change (in log or level space)."""

    def __init__(self, *, use_log: bool = False, use_diff: bool = False):
        self._use_log = use_log
        self._use_diff = use_diff

    def __repr__(self) -> str:
        return (
            f"SeasonalNaive12Model(use_log={self._use_log!r}, use_diff={self._use_diff!r})"
        )

    def __str__(self) -> str:
        parts = []
        if self._use_log:
            parts.append("log")
        if self._use_diff:
            parts.append("+YoY wedge")
        extra = ", ".join(parts) if parts else "level only"
        return f"Seasonal naive (t-12)[{extra}]"

    def fit(self, levels: pd.Series):
        self._levels = levels.astype(float).copy()
        return self

    def predict(self, future_index: pd.DatetimeIndex) -> pd.Series:
        z = _to_scaled_series(self._levels, self._use_log)
        lag_ix = future_index - pd.DateOffset(years=1)
        base_z = z.reindex(lag_ix).astype(float)
        if self._use_diff:
            last_ts = z.index[-1]
            wedge = float(
                z.loc[last_ts]
                - z.loc[last_ts - pd.DateOffset(years=1)]
            )
            raw = base_z.to_numpy(dtype=float) + wedge
        else:
            raw = base_z.to_numpy(dtype=float)
        out = np.exp(raw) if self._use_log else raw
        return pd.Series(out, index=future_index, name="seasonal_naive_12")
