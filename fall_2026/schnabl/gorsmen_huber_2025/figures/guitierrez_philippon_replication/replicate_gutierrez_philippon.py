#!/usr/bin/env python3
"""Reconstruct and extend Gutiérrez–Philippon's Net Investment vs. Q chart.

The specification follows Figure 4 of *Investment-less Growth*:
  NI_t/K_{t-1} = alpha + beta Q_{t-1} + error_t,
estimated over 1990–2001 and extrapolated afterward.

The script uses the latest-vintage FRED/Financial Accounts data, so the
historical values can differ from the vintage used in the published paper.
"""

from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
RAW.mkdir(exist_ok=True)

SERIES = {
    "equity": "NCBCEL",
    "liabilities": "TLBSNNCB",
    "financial_assets": "TFAABSNNCB",
    "inventories": "BOGZ1LM105020015Q",
    "corporate_fixed_assets": "BOGZ1FL105015085Q",
    "noncorporate_fixed_assets": "BOGZ1FL115015005Q",
    "gross_fixed_investment": "BOGZ1FA145019005Q",
    "consumption_fixed_capital": "BOGZ1FA146300005Q",
}


def load_series(name: str, fred_id: str) -> pd.Series:
    path = RAW / f"{fred_id}.csv"
    if not path.exists() or path.stat().st_size < 100:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}"
        urlretrieve(url, path)
    frame = pd.read_csv(path, na_values=".")
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    return frame.set_index("observation_date")[fred_id].rename(name)


quarterly = pd.concat(
    [load_series(name, fred_id) for name, fred_id in SERIES.items()], axis=1
)

stock_names = [
    "equity",
    "liabilities",
    "financial_assets",
    "inventories",
    "corporate_fixed_assets",
    "noncorporate_fixed_assets",
]
flow_names = ["gross_fixed_investment", "consumption_fixed_capital"]

# Financial-account stocks are end-of-period; flows are reported at SAAR, so
# the annual flow is the average of the four quarterly rates.
annual = pd.concat(
    [
        quarterly[stock_names].resample("YE").last(),
        quarterly[flow_names].resample("YE").mean(),
    ],
    axis=1,
)
annual.index = annual.index.year
annual.index.name = "year"

annual["tobins_q"] = (
    annual["equity"]
    + annual["liabilities"]
    - annual["financial_assets"]
    - annual["inventories"]
) / annual["corporate_fixed_assets"]

annual["nonfinancial_business_capital"] = (
    annual["corporate_fixed_assets"] + annual["noncorporate_fixed_assets"]
)
annual["net_investment"] = (
    annual["gross_fixed_investment"] - annual["consumption_fixed_capital"]
)
annual["net_investment_rate"] = annual["net_investment"] / annual[
    "nonfinancial_business_capital"
].shift(1)
annual["lagged_q"] = annual["tobins_q"].shift(1)

train = annual.loc[1990:2001, ["net_investment_rate", "lagged_q"]].dropna()
X = np.column_stack([np.ones(len(train)), train["lagged_q"].to_numpy()])
y = train["net_investment_rate"].to_numpy()
alpha, beta = np.linalg.lstsq(X, y, rcond=None)[0]

annual["fitted_net_investment_rate"] = alpha + beta * annual["lagged_q"]
annual["residual"] = (
    annual["net_investment_rate"] - annual["fitted_net_investment_rate"]
)
annual["cumulative_gap"] = annual.loc[1990:, "residual"].cumsum()

plot = annual.loc[1990:2025].copy()
plot.to_csv(ROOT / "gutierrez_philippon_replication_data.csv", float_format="%.8f")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharex=True)

axes[0].plot(
    plot.index,
    plot["net_investment_rate"],
    color="#2f6b25",
    linewidth=1.8,
    label="Net investment",
)
axes[0].plot(
    plot.index,
    plot["fitted_net_investment_rate"],
    color="#e66b2e",
    linewidth=1.8,
    linestyle="--",
    label="Fitted values using lagged Q",
)
axes[0].axvline(2015.5, color="#888888", linewidth=0.8, linestyle=":")
axes[0].set_ylabel("NI/K")
axes[0].set_xlabel("Year")
axes[0].set_title("Net investment (actual and predicted with Q)", pad=8)
axes[0].legend(frameon=False, ncol=2, loc="upper left")
axes[0].grid(axis="y", color="#dddddd", linewidth=0.7)

axes[1].plot(
    plot.index,
    plot["cumulative_gap"],
    color="#2f6b25",
    linewidth=1.8,
    label="Cumulative gap",
)
axes[1].plot(
    plot.index,
    plot["residual"],
    color="#e66b2e",
    linewidth=1.8,
    linestyle="--",
    label="Residual",
)
axes[1].axhline(0, color="#9b1c31", linewidth=0.8, linestyle=":")
axes[1].axvline(2015.5, color="#888888", linewidth=0.8, linestyle=":")
axes[1].set_ylabel("Regression residuals")
axes[1].set_xlabel("Year")
axes[1].set_title("Prediction residuals (by period and cumulative)", pad=8)
axes[1].legend(frameon=False, ncol=2, loc="lower left")
axes[1].grid(axis="y", color="#dddddd", linewidth=0.7)

fig.tight_layout()
fig.savefig(ROOT / "gutierrez_philippon_replication_extended.png", dpi=220)

print(f"alpha={alpha:.8f}; beta={beta:.8f}")
print(
    plot.loc[
        [2001, 2015, 2019, 2020, 2021, 2022, 2023, 2024, 2025],
        [
            "net_investment_rate",
            "fitted_net_investment_rate",
            "residual",
            "cumulative_gap",
        ],
    ]
)
