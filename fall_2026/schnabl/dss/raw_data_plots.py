"""Create the six requested quarterly-change figures from the cleaned panel."""

import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)
from settings import REUSABLE_CODE_DIR

sys.path.append(str(REUSABLE_CODE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

from get_data import get_data

import settings


def main(panel: pd.DataFrame | None = None):
    out = settings.RESULTS_DIR
    out.mkdir(exist_ok=True)
    variables = [
        ("d_deposit_spread_pct", "Deposit spread", "Percentage points"),
        ("d_total_deposits", "Total deposits", "Trillions of dollars"),
        ("d_log_assets", "Log assets", "Log points"),
        ("d_securities", "Securities", "Trillions of dollars"),
        ("d_total_loans", "Total loans", "Trillions of dollars"),
        ("d_fed_funds_target_pct", "Federal funds target rate", "Percentage points"),
    ]
    change_columns = [column for column, _, _ in variables]
    if panel is None:
        panel = get_data(output_columns=["quarter", *change_columns])
    else:
        panel = panel.loc[:, ["quarter", *change_columns]]
    grouped = panel.groupby("quarter", sort=True)
    means = grouped[change_columns].mean()
    counts = grouped[change_columns].count()
    # Source monetary amounts are in thousands of dollars.
    monetary_columns = ["d_total_deposits", "d_securities", "d_total_loans"]
    means[monetary_columns] = means[monetary_columns] / 1_000_000_000
    means.to_csv(out / "quarterly_mean_changes.csv")
    counts.to_csv(out / "quarterly_change_counts.csv")
    dates = pd.PeriodIndex(means.index, freq="Q").to_timestamp(how="end")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for ax, (column, title, unit) in zip(axes.flat, variables):
        values = means[column]
        ax.plot(dates, values, color="#176087", linewidth=1.35)
        ax.axhline(0, color="0.5", linewidth=0.6)
        ax.set(title=title, ylabel=unit, xlabel="Year")
        ax.tick_params(axis="x", labelbottom=True)
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(axis="y", alpha=0.18)
        individual, single_ax = plt.subplots(figsize=(8, 3.4))
        single_ax.plot(dates, values, color="#176087", linewidth=1.35)
        single_ax.axhline(0, color="0.5", linewidth=0.6)
        single_ax.set(title=title + ": quarterly change", ylabel=unit, xlabel="Year")
        single_ax.xaxis.set_major_locator(mdates.YearLocator(2))
        single_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        single_ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        single_ax.grid(axis="y", alpha=0.18)
        individual.savefig(out / f"{column}.pdf")
        individual.savefig(out / f"{column}.png", dpi=180)
        plt.close(individual)
    fig.suptitle(
        f"Quarterly changes: commercial banks, {pd.Period(panel.quarter.min(), freq='Q').year}–"
        f"{pd.Period(panel.quarter.max(), freq='Q').year} sample",
        y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out / "quarterly_changes.pdf")
    fig.savefig(out / "quarterly_changes.png", dpi=180)
    plt.close(fig)
    print(
        "Saved six individual charts, combined chart, quarterly means and counts in results/"
    )


if __name__ == "__main__":
    main()
