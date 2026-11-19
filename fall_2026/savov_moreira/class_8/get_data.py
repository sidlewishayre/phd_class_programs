"""Build the quarterly and annual consumption/return data sets for class 7.

The script downloads BEA series through FRED and monthly returns from the
Ken French Data Library.  Downloaded source files are cached in ``data/raw``;
pass ``--refresh`` to replace the cache.

Outputs
-------
data/quarterly_data.csv : 1947Q2--2025Q4, excluding 2020Q1--2021Q4
data/annual_data.csv    : 1948--2025, excluding 2020--2021

All 33 variables after ``date`` are continuously compounded (log) growth
rates or real returns expressed as decimals, not percentages.
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

FRED_SERIES = {
    "real_nondurables": "A796RX0Q048SBEA",
    "real_services": "A797RX0Q048SBEA",
    "current_nondurables": "A796RC0Q052SBEA",
    "current_services": "A797RC0Q052SBEA",
}

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
FRENCH_FILES = {
    "factors": "F-F_Research_Data_Factors_CSV.zip",
    "book_to_market": "Portfolios_Formed_on_BE-ME_CSV.zip",
    "size": "Portfolios_Formed_on_ME_CSV.zip",
    "momentum": "10_Portfolios_Prior_12_2_CSV.zip",
}


def download(url: str, destination: Path, refresh: bool) -> Path:
    """Download *url* unless a nonempty cached file is available."""
    if destination.exists() and destination.stat().st_size > 0 and not refresh:
        return destination

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=(20, 180),
        )
        response.raise_for_status()
        contents = response.content
    except Exception as error:
        raise RuntimeError(f"Could not download {url}") from error

    destination.write_bytes(contents)
    return destination


def load_fred(refresh: bool) -> pd.DataFrame:
    """Load the four quarterly NIPA series and align them by quarter."""
    frames = []
    for name, series_id in FRED_SERIES.items():
        path = download(
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
            RAW_DIR / f"{series_id}.csv",
            refresh,
        )
        frame = pd.read_csv(path)
        frame.columns = ["date", name]
        frame["date"] = pd.to_datetime(frame["date"])
        frame[name] = pd.to_numeric(frame[name], errors="coerce")
        frames.append(frame.set_index("date"))

    nipa = pd.concat(frames, axis=1, join="inner").sort_index()
    if nipa.isna().any().any():
        bad_dates = nipa.index[nipa.isna().any(axis=1)].strftime("%Y-%m-%d").tolist()
        raise ValueError(f"Missing NIPA observations on: {bad_dates}")
    nipa.index = nipa.index.to_period("Q")
    return nipa


def zip_text(path: Path) -> str:
    """Read the only CSV contained in a Ken French ZIP archive."""
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in {path}, found {len(csv_names)}")
        return archive.read(csv_names[0]).decode("utf-8", errors="replace")


def monthly_block(text: str, heading_pattern: str | None = None) -> pd.DataFrame:
    """Extract the first YYYYMM monthly block after an optional heading."""
    lines = text.splitlines()
    start_at = 0
    if heading_pattern is not None:
        heading = re.compile(heading_pattern, flags=re.IGNORECASE)
        matches = [i for i, line in enumerate(lines) if heading.search(line)]
        if not matches:
            raise ValueError(f"Could not find table heading {heading_pattern!r}")
        start_at = matches[0] + 1

    first_data = next(
        (i for i in range(start_at, len(lines)) if re.match(r"^\s*\d{6}\s*,", lines[i])),
        None,
    )
    if first_data is None or first_data == 0:
        raise ValueError("Could not find a monthly return table")

    header = first_data - 1
    end = first_data
    while end < len(lines) and re.match(r"^\s*\d{6}\s*,", lines[end]):
        end += 1

    frame = pd.read_csv(io.StringIO("\n".join(lines[header:end])), skipinitialspace=True)
    frame = frame.rename(columns={frame.columns[0]: "yyyymm"})
    frame["yyyymm"] = frame["yyyymm"].astype(str).str.strip()
    frame.index = pd.PeriodIndex(frame.pop("yyyymm"), freq="M")
    return frame.apply(pd.to_numeric, errors="coerce")


def load_french(refresh: bool) -> pd.DataFrame:
    """Load market, T-bill, and 30 value-weighted decile returns."""
    texts = {}
    for name, filename in FRENCH_FILES.items():
        path = download(f"{FRENCH_BASE}/{filename}", RAW_DIR / filename, refresh)
        texts[name] = zip_text(path)

    factors = monthly_block(texts["factors"])
    market = factors["Mkt-RF"] + factors["RF"]
    returns = pd.DataFrame({"market": market, "rf": factors["RF"]})

    specifications = {
        "bm": (texts["book_to_market"], r"Value Weight Returns\s*--\s*Monthly"),
        "size": (texts["size"], r"Average Value Weight Returns\s*--\s*Monthly"),
        "momentum": (texts["momentum"], r"Value Weight Returns\s*--\s*Monthly"),
    }
    for prefix, (text, heading) in specifications.items():
        table = monthly_block(text, heading)
        # The 10 deciles are the last ten columns in the B/M and size files;
        # the momentum file contains exactly ten columns.
        deciles = table.iloc[:, -10:].copy()
        deciles.columns = [f"{prefix}_{number}" for number in range(1, 11)]
        returns = returns.join(deciles, how="inner")

    returns = returns.replace([-99.99, -999.0], np.nan) / 100.0
    requested = returns.loc["1947-01":"2025-12"]
    if requested.isna().any().any():
        locations = np.argwhere(requested.isna().to_numpy())[:10]
        details = [(str(requested.index[i]), requested.columns[j]) for i, j in locations]
        raise ValueError(f"Missing French returns in the requested sample: {details}")
    return returns


def aggregate_log_returns(monthly: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Compound simple monthly returns and return their log gross returns."""
    if (monthly <= -1).any().any():
        raise ValueError("A monthly simple return is no greater than -100 percent")
    groups = monthly.index.asfreq(frequency)
    expected_count = 3 if frequency == "Q" else 12
    counts = monthly.groupby(groups).size()
    complete = counts[counts == expected_count].index
    return np.log1p(monthly).groupby(groups).sum().loc[complete]


def consumption_and_inflation(nipa: pd.DataFrame) -> pd.DataFrame:
    """Construct consumption and its implicit price deflator."""
    result = pd.DataFrame(index=nipa.index)
    result["consumption"] = nipa["real_nondurables"] + nipa["real_services"]
    nominal = nipa["current_nondurables"] + nipa["current_services"]
    result["deflator"] = nominal / result["consumption"]
    result["consumption_growth"] = np.log(result["consumption"]).diff()
    result["inflation"] = np.log(result["deflator"]).diff()
    return result


def assemble(
    nipa: pd.DataFrame,
    monthly: pd.DataFrame,
    exclude_covid: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the requested 34-column quarterly and annual data sets."""
    macro = consumption_and_inflation(nipa)

    quarterly_nominal = aggregate_log_returns(monthly, "Q")
    quarterly = macro[["consumption_growth"]].join(
        quarterly_nominal.sub(macro["inflation"], axis=0), how="inner"
    )
    quarterly = quarterly.loc["1947Q2":"2025Q4"]
    if exclude_covid:
        quarterly = quarterly.loc[
            ~quarterly.index.to_timestamp().year.isin([2020, 2021])
        ]

    q4 = macro.loc[macro.index.quarter == 4, ["consumption", "deflator"]].copy()
    q4.index = q4.index.year
    annual_macro = pd.DataFrame(index=q4.index)
    annual_macro["consumption_growth"] = np.log(q4["consumption"]).diff()
    annual_macro["inflation"] = np.log(q4["deflator"]).diff()

    annual_nominal = aggregate_log_returns(monthly, "Y")
    annual_nominal.index = annual_nominal.index.year
    annual = annual_macro[["consumption_growth"]].join(
        annual_nominal.sub(annual_macro["inflation"], axis=0), how="inner"
    )
    annual = annual.loc[1948:2025]
    if exclude_covid:
        annual = annual.loc[~annual.index.isin([2020, 2021])]

    expected_columns = ["consumption_growth", "market", "rf"] + [
        f"{group}_{number}"
        for group in ("bm", "size", "momentum")
        for number in range(1, 11)
    ]
    quarterly = quarterly[expected_columns]
    annual = annual[expected_columns]
    validate(quarterly, annual, exclude_covid)
    return quarterly, annual


def validate(
    quarterly: pd.DataFrame,
    annual: pd.DataFrame,
    exclude_covid: bool = True,
) -> None:
    """Fail loudly if source changes make the output incomplete or malformed."""
    expected_rows = (307, 76) if exclude_covid else (315, 78)
    for name, frame, rows in (
        ("quarterly", quarterly, expected_rows[0]),
        ("annual", annual, expected_rows[1]),
    ):
        if frame.shape != (rows, 33):
            raise ValueError(f"Unexpected {name} shape {frame.shape}; expected {(rows, 33)}")
        if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"The {name} data contain missing or non-finite values")


def write_outputs(quarterly: pd.DataFrame, annual: pd.DataFrame) -> None:
    quarterly_output = quarterly.copy()
    quarterly_output.index = quarterly_output.index.astype(str)
    quarterly_output.index.name = "date"

    annual_output = annual.copy()
    annual_output.index.name = "date"

    quarterly_output.to_csv(DATA_DIR / "quarterly_data.csv", float_format="%.10f")
    annual_output.to_csv(DATA_DIR / "annual_data.csv", float_format="%.10f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="redownload all source files instead of using data/raw",
    )
    arguments = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)
    nipa = load_fred(arguments.refresh)
    monthly = load_french(arguments.refresh)
    quarterly, annual = assemble(nipa, monthly)
    write_outputs(quarterly, annual)

    print(f"Wrote {DATA_DIR / 'quarterly_data.csv'}: {quarterly.shape[0]} rows, 34 columns")
    print(f"Wrote {DATA_DIR / 'annual_data.csv'}: {annual.shape[0]} rows, 34 columns")


if __name__ == "__main__":
    main()
