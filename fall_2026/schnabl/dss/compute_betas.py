"""Return bank-level beta estimates with 5% winsorization."""

from __future__ import annotations

import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)
from settings import REUSABLE_CODE_DIR

sys.path.append(str(REUSABLE_CODE_DIR))

import numpy as np
import pandas as pd

from get_data import PAPER_STOCKS, get_beta_panel


def compute_beta(outcome, rate_changes, *, controls=None, zero_tolerance=0) -> float:
    """Return the sum of OLS slopes on current and lagged rate changes.

    Each row is a quarter; rate_changes has one column per current/lagged
    change. Include an intercept and omit rows with nonfinite inputs.
    Optional controls enter the regression but are excluded from the beta sum.
    Return NaN for insufficient observations, deficient rank, or zero slopes,
    preserving the replication sample rule. ``zero_tolerance`` optionally
    excludes near-zero policy slopes, as in the AI audit.
    """
    y = np.asarray(outcome, dtype=float)
    x = np.asarray(rate_changes, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x):
        raise ValueError(
            "Supply a one-dimensional outcome and matching rate-change matrix"
        )
    if x.shape[1] == 0:
        raise ValueError("Supply at least one rate-change column")
    n_target = x.shape[1]
    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 2 or len(controls) != len(y):
            raise ValueError("Controls must be a matrix matching the outcome length")
        x = np.column_stack([x, controls])
    x = np.column_stack([np.ones(len(y)), x])
    valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
    if valid.sum() < x.shape[1]:
        return float("nan")
    coefficients, _, rank, _ = np.linalg.lstsq(x[valid], y[valid], rcond=None)
    if rank != x.shape[1] or np.any(np.abs(coefficients[1 : 1 + n_target]) <= zero_tolerance):
        return float("nan")
    return float(coefficients[1 : 1 + n_target].sum())


def fit_cross_bank_regression(data, outcome, regressors):
    """OLS with an intercept, HC1 robust standard errors, and t inference."""
    import statsmodels.api as sm

    sample = data[[outcome, *regressors]].replace([np.inf, -np.inf], np.nan).dropna()
    x = sm.add_constant(sample[regressors], has_constant="add")
    if len(sample) <= x.shape[1] or np.linalg.matrix_rank(x) < x.shape[1]:
        raise ValueError(
            "Insufficient observations or deficient rank in cross-bank regression"
        )
    return sm.OLS(sample[outcome], x).fit(cov_type="HC1", use_t=True)


def winsorize(values: pd.Series, tail: float) -> pd.Series:
    """Cap both tails independently; missing observations remain missing."""
    if not 0 <= tail < 0.5:
        raise ValueError("winsorization tail must be in [0, 0.5)")
    if tail == 0:
        return values.copy()
    lower, upper = values.quantile([tail, 1 - tail])
    return values.clip(lower, upper)


def compute_betas(
    panel=None,
    rates=None,
    *,
    include_loan_categories=True,
    beta_tail=0.05,
    start="1994Q1",
    end="2013Q4",
):
    """Return raw and winsorized betas for the common bank sample."""
    d = get_beta_panel(panel=panel, rates=rates, start=start, end=end)
    outcomes = ["spread_beta", *PAPER_STOCKS]
    if not include_loan_categories:
        outcomes = outcomes[:-2]
    return estimate_bank_betas(d, outcomes=outcomes, beta_tail=beta_tail, min_obs=6)


def estimate_bank_betas(
    d,
    *,
    outcomes=("spread_beta", "deposit_flow_beta"),
    beta_tail=0.05,
    min_obs=12,
    max_lag=4,
    regressor_columns=None,
    control_columns=(),
    winsorizer=winsorize,
    zero_tolerance=0,
):
    """Estimate bank betas from a prepared panel; winsorize only final estimates.

    ``winsorizer`` selects the final-beta capping rule; defaults preserve the
    baseline's interpolated quantiles. The audit uses Stata order statistics.
    """
    if not isinstance(max_lag, int) or max_lag < 0:
        raise ValueError("max_lag must be a nonnegative integer")
    columns = (
        [f"x{lag}" for lag in range(max_lag + 1)]
        if regressor_columns is None
        else list(regressor_columns)
    )
    arrays = d[columns].to_numpy(float)
    controls = d[list(control_columns)].to_numpy(float)
    responses = d[[name + "_y" for name in outcomes]].to_numpy(float)
    records = []
    for cert, indices in d.groupby("cert", sort=True).indices.items():
        x = arrays[indices]
        group = d.iloc[indices]
        row = {"cert": cert}
        for column in ["assets", "real_assets"]:
            if column in group:
                row["mean_" + column] = group[column].mean()
        for i, name in enumerate(outcomes):
            y = responses[indices, i]
            z = controls[indices]
            valid = (
                np.isfinite(y) & np.isfinite(x).all(axis=1) & np.isfinite(z).all(axis=1)
            )
            row[name + "_nobs"] = int(valid.sum())
            row[name + "_raw"] = (
                compute_beta(y, x, controls=z, zero_tolerance=zero_tolerance)
                if valid.sum() >= min_obs else np.nan
            )
        records.append(row)
    result = pd.DataFrame(records)
    for name in outcomes:
        raw = result[name + "_raw"]
        result[name] = winsorizer(raw, beta_tail)
    # The original program merges all seven beta files with keep if _merge==3.
    result = result.dropna(subset=[name + "_raw" for name in outcomes]).reset_index(
        drop=True
    )
    result.attrs["summary"] = {
        "first_quarter": str(d.quarter.min()),
        "last_quarter": str(d.quarter.max()),
        "paper_mean_spread_beta": 0.54,
        "beta_winsor_tail": beta_tail,
    }
    return result


if __name__ == "__main__":
    betas = compute_betas()


def compute_cycle_betas(panel, rates):
    """Deposit-rate change from 2021Q4 to 2023Q4 / target-rate change."""
    d = panel.sort_values(["cert", "quarter"]).copy()
    g = d.groupby("cert")
    adjacent = d.quarter.astype("int64").sub(g.quarter.shift().astype("int64")).eq(1)
    deposits = g.domdep.shift().where(adjacent)
    d["rate"] = (400 * d.intexpdomdep / deposits).where(
        deposits.gt(0) & d.intexpdomdep.ge(0) & d.assets.gt(0))
    wide = d.pivot(index="cert", columns="quarter", values="rate")
    change = rates.loc["2023Q4", "fed_funds_target_pct"] - rates.loc["2021Q4", "fed_funds_target_pct"]
    return ((wide[pd.Period("2023Q4")] - wide[pd.Period("2021Q4")]) / change).rename("cycle_beta").dropna().reset_index()
