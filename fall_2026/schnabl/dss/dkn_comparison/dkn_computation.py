"""Build the quarterly aggregate deposit-spread data for DKN Table 3."""

import sys
from pathlib import Path

import pandas as pd

CWD = Path(__file__).resolve().parents[1]
sys.path.append(str(CWD))

from get_data import load_dkn_call_reports, load_dkn_rates, quarterly_expense
from settings import DKN_REGRESSION_FILE

START, END = "1984Q1", "2021Q2"
VARIABLES = ["spread", "fedfunds", "swap1", "swap5"]


def aggregate_deposits(reports):
    """Compute quarterly flows before applying the commercial-bank screens."""
    banks = reports.rename(columns={"id_rssd": "rssdid", "date": "quarter"})
    banks = banks.sort_values(["rssdid", "quarter"])
    quarters = banks.quarter.dt.to_period("Q")
    banks["qid"] = quarters.astype("int64")
    banks["expense"] = quarterly_expense(banks, "ytdint_exp_dep", "ytd")
    # The NY Fed files provide total assets and do not identify card interest income.
    banks = banks.loc[
        banks.ent_type_cd.eq(1)
        & banks.assets.gt(0)
        & banks.deposits.gt(0)
        & banks.domestic_dep.ge(banks.deposits / 2)
    ]
    grouped = banks.groupby(quarters.loc[banks.index])
    cutoff = grouped.assets.transform("quantile", q=0.01)
    banks = banks.loc[banks.assets.ge(cutoff)].dropna(subset=["expense"])
    grouped = banks.groupby(quarters.loc[banks.index])
    return grouped[["assets", "deposits", "expense"]].sum().assign(
        banks=grouped.size()
    )


def build_dataset(reports, monthly_rates):
    monthly_rates = monthly_rates.sort_index()
    rates = monthly_rates.groupby(monthly_rates.index.to_period("Q")).mean() / 100
    rates = rates.reindex(pd.period_range(rates.index.min(), END, freq="Q"))
    rates["fedfunds"] = rates.FEDFUNDS
    for years in (1, 5):
        past_yield = rates[f"GS{years}"].shift(1).rolling(4 * years).mean()
        rates[f"swap{years}"] = past_yield - rates.fedfunds

    quarters = pd.period_range(START, END, freq="Q")
    data = aggregate_deposits(reports).join(rates[["fedfunds", "swap1", "swap5"]])
    data = data.reindex(quarters)
    data["spread"] = (data.fedfunds * data.deposits - 4 * data.expense) / data.assets
    incomplete = data.index[data[VARIABLES].isna().any(axis=1)]
    if len(incomplete):
        raise ValueError(f"Missing Table 3 inputs in {', '.join(map(str, incomplete))}")

    # Difference inside the estimation sample so Panel B starts in 1984Q2.
    data = data.join(data[VARIABLES].diff().add_prefix("d_"))
    data.insert(0, "quarter", quarters.astype(str))
    data.insert(1, "tq", quarters.astype("int64") + 40)
    return data.reset_index(drop=True)


def main():
    data = build_dataset(load_dkn_call_reports(START, END), load_dkn_rates())
    DKN_REGRESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_stata(DKN_REGRESSION_FILE, write_index=False, version=118)
    print(f"Saved {len(data)} quarters to {DKN_REGRESSION_FILE}")


if __name__ == "__main__":
    main()
