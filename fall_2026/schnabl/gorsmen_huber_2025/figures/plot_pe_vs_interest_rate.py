#!/usr/bin/env python3
"""Plot the P/E ratio, long-term rates, and investment over time.

The script downloads Robert Shiller's official monthly workbook from his Yale
website, caches it locally, and writes ``pe_vs_interest_rate.png`` alongside
this file. Reading the legacy XLS workbook requires the ``xlrd`` package.
"""

from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "ie_data.xlsx"
OUTPUT_PATH = ROOT / "pe_vs_interest_rate.png"
DATA_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
INVESTMENT_DATA_PATH = (
    ROOT
    / "guitierrez_philippon_replication"
    / "gutierrez_philippon_replication_data.csv"
)


def download_data() -> None:
    """Download the workbook using headers accepted by the Yale web server."""
    request = Request(DATA_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=60) as response, DATA_PATH.open("wb") as output:
        output.write(response.read())


def load_data() -> pd.DataFrame:
    """Download (if needed), clean, and return Shiller's monthly data."""
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size < 1_000:
        download_data()

    try:
        raw = pd.read_excel(DATA_PATH, sheet_name="Data", header=None, engine="xlrd")
    except ImportError as exc:
        raise SystemExit(
            "Reading Shiller's .xls file requires xlrd. Install it with: "
            "python3 -m pip install xlrd"
        ) from exc

    # Locate the header instead of relying on a fixed number of introductory rows.
    header_rows = raw.index[raw.iloc[:, 0].astype(str).str.strip().eq("Date")]
    if header_rows.empty:
        raise ValueError("Could not locate the data header in Shiller's workbook")
    header_row = int(header_rows[0])
    data = raw.iloc[header_row + 1 :].copy()
    data.columns = [str(value).strip() for value in raw.iloc[header_row]]

    def find_column(*names: str) -> str:
        normalized = {str(column).lower().replace(" ", ""): column for column in data}
        for name in names:
            match = normalized.get(name.lower().replace(" ", ""))
            if match is not None:
                return match
        raise ValueError(f"Could not find any of these columns: {names}")

    date_column = find_column("Date")
    pe_column = find_column("CAPE", "PE10")
    rate_column = find_column("Rate GS10", "Long Interest Rate")

    shiller_date = pd.to_numeric(data[date_column], errors="coerce")
    years = np.floor(shiller_date)
    months = np.rint((shiller_date - years) * 100)
    data["date"] = pd.to_datetime(
        {"year": years, "month": months, "day": 1}, errors="coerce"
    )
    data["pe_ratio"] = pd.to_numeric(data[pe_column], errors="coerce")
    data["interest_rate"] = pd.to_numeric(data[rate_column], errors="coerce")

    # Zeroes in the source denote unavailable observations, not economic data.
    data = data.loc[
        data["date"].ge("1990-01-01")
        & data["pe_ratio"].gt(0)
        & data["interest_rate"].gt(0),
        ["date", "pe_ratio", "interest_rate"],
    ].copy()
    return data


def main() -> None:
    shiller = load_data().set_index("date")
    shiller_annual = shiller[["pe_ratio", "interest_rate"]].resample("YE").mean()

    investment = pd.read_csv(INVESTMENT_DATA_PATH).set_index("year")
    investment.index = pd.to_datetime(investment.index.astype(str) + "-12-31")

    data = shiller_annual.join(
        investment[["net_investment_rate"]], how="inner"
    ).loc["1990-01-01":"2025-12-31"].dropna()
    baseline_mean = data.loc["1990-01-01":"1999-12-31"].mean()
    normalized = data.div(baseline_mean)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    pe_line = ax.plot(
        normalized.index,
        normalized["pe_ratio"],
        color="#2f6b25",
        linewidth=1.6,
        label="P/E ratio",
    )[0]
    rate_line = ax.plot(
        normalized.index,
        normalized["interest_rate"],
        color="#e66b2e",
        linewidth=1.4,
        label="Long-term interest rate",
    )[0]
    investment_line = ax.plot(
        normalized.index,
        normalized["net_investment_rate"],
        color="#2f5f98",
        linewidth=1.6,
        label="Net investment / capital (NI/K)",
    )[0]

    ax.axhline(1, color="#c62828", linewidth=1.1, linestyle=":")
    ax.set_title("P/E Ratio, Long-Term Interest Rates, and Investment")
    ax.set_xlabel("Year")
    ax.set_xlim(pd.Timestamp("1990-01-01"), pd.Timestamp("2025-12-31"))
    ax.set_xticks(pd.date_range("1990-01-01", "2025-01-01", freq="5YS"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Ratio to 1990s average")
    ax.grid(color="#dddddd", linewidth=0.7, alpha=0.8)
    ax.legend(
        handles=[pe_line, rate_line, investment_line],
        frameon=False,
        ncol=3,
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
