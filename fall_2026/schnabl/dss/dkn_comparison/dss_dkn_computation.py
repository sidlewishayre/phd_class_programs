"""Prepare the sequential DKN-to-DSS comparison without estimating regressions."""

import sys
from pathlib import Path

import pandas as pd

CWD = Path(__file__).resolve().parents[1]
sys.path.append(str(CWD))

from get_data import quarterly_expense
from dkn_comparison.dkn_computation import build_dataset
from compute_betas import estimate_bank_betas


def compute_comparison_betas(panel, end):
    """Adapt total-expense inputs to the shared bank-beta estimator."""
    end_tq = pd.Period(end, freq="Q").ordinal + 40
    data = panel.loc[panel.tq.le(end_tq)].sort_values(["rssdid", "tq"]).copy()
    # The shared estimator calls its grouping key cert; these remain RSSD IDs.
    data = data.rename(columns={"rssdid": "cert"})
    data["quarter"] = pd.PeriodIndex.from_ordinals(data.tq.to_numpy() - 40, freq="Q")
    groups = data.groupby("cert", sort=False)
    adjacent = data.tq.sub(groups.tq.shift()).eq(1)
    data["expense_beta_y"] = groups.expense_rate.diff().where(adjacent)
    data["x0"] = groups.fedfunds.diff().where(adjacent)
    groups = data.groupby("cert", sort=False)
    for lag in range(1, 4):
        contiguous = data.tq.sub(groups.tq.shift(lag)).eq(lag)
        data[f"x{lag}"] = groups.x0.shift(lag).where(contiguous)
    return estimate_bank_betas(
        data, outcomes=("expense_beta",), max_lag=3,
        min_obs=60, beta_tail=0.05,
    ).rename(columns={"cert": "rssdid"})


def prepare_comparison(reports, rates):
    aggregate = build_dataset(reports, rates)
    banks = reports.rename(columns={"id_rssd": "rssdid", "date": "quarter"})
    banks = banks.sort_values(["rssdid", "quarter"]).copy()
    banks["qid"] = banks.quarter.dt.to_period("Q").astype("int64")
    banks["expense"] = quarterly_expense(banks, "ytdint_exp", "ytd")
    previous = banks.groupby("rssdid").assets.shift()
    adjacent = banks.qid.sub(banks.groupby("rssdid").qid.shift()).eq(1)
    banks["average_assets"] = (banks.assets + previous.where(adjacent)) / 2
    banks = banks.loc[
        banks.ent_type_cd.eq(1) & banks.assets.gt(0) & banks.deposits.gt(0)
        & banks.domestic_dep.ge(banks.deposits / 2)
    ].copy()
    cutoff = banks.groupby("qid").assets.transform("quantile", q=0.01)
    banks = banks.loc[banks.assets.ge(cutoff)].copy()
    banks["tq"] = banks.qid + 40
    banks = banks.merge(aggregate[["tq", "fedfunds"]], on="tq", validate="many_to_one")
    banks["expense_rate"] = 4 * banks.expense / banks.average_assets.where(banks.average_assets.gt(0))
    # Preserve the same total-expense cohort and denominator in aggregate and bank steps.
    valid = banks.dropna(subset=["expense_rate"])
    sums = valid.groupby("tq")[["expense", "average_assets"]].sum()
    aggregate["expense_rate"] = aggregate.tq.map(4 * sums.expense / sums.average_assets)
    aggregate["d_expense_rate"] = aggregate.expense_rate.diff()
    aggregate["deposit_share"] = aggregate.deposits / aggregate.assets
    panel = banks[["rssdid", "tq", "expense_rate", "fedfunds"]]
    if panel.duplicated(["rssdid", "tq"]).any():
        raise ValueError("Duplicate bank-quarter observations")
    return aggregate, panel
