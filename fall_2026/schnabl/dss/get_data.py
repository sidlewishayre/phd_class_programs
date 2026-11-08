"""Build the available commercial-bank panel; see DATA_NOTES.md for source caveats."""

from __future__ import annotations

import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)
sys.path.append(os.path.abspath(os.path.join(CWD, "../../../../reusable_code")))

import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from database_management.fred_pull import get_fred_series

ROOT = Path(__file__).resolve().parent
COLUMNS = [
    "rssdid",
    "cert",
    "yq",
    "assets",
    "domdep",
    "foreigndep",
    "intexpalldep",
    "securities",
    "loans",
]


def download_call_reports(
    start: str = "1994Q1",
    end: str = "2000Q4",
    data_dir: Path = ROOT / "data",
) -> dict[pd.Period, Path]:
    """Cache official Chicago Fed quarterly Call Report ZIPs, fetching only missing files.

    Defaults to the years absent from the supplied Stata file. The archive
    supports 1976–2010; later filings require the FFIEC source. Returns a mapping
    from reporting quarter to local archive path. No observations are loaded.
    Existing files are validated and reused, never refreshed or overwritten.
    Downloads stream to temporary files; only complete, valid ZIPs enter the cache.
    Source: chicagofed.org/banking/financial-institution-reports/commercial-bank-data
    """
    first, last = pd.Period(start, freq="Q"), pd.Period(end, freq="Q")
    if first > last or first < pd.Period("1976Q1") or last > pd.Period("2010Q4"):
        raise ValueError("Require 1976Q1 <= start <= end <= 2010Q4")
    cache = Path(data_dir) / "call_reports_online"
    cache.mkdir(parents=True, exist_ok=True)
    base_url = (
        "https://www.chicagofed.org/-/media/others/banking/"
        "financial-institution-reports/commercial-bank-data"
    )
    paths = {}
    for quarter in pd.period_range(first, last, freq="Q"):
        stem = f"call{quarter.year % 100:02d}{quarter.quarter * 3:02d}"
        path = cache / f"{stem}.zip"
        if not path.exists():
            url = f"{base_url}/{stem}-zip.zip"
            with tempfile.TemporaryDirectory(prefix=".download_", dir=cache) as directory:
                temporary = Path(directory) / path.name
                with requests.get(url, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as destination:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                destination.write(block)
                with zipfile.ZipFile(temporary) as archive:
                    if not any(name.lower().endswith(".xpt") for name in archive.namelist()):
                        raise ValueError(f"No SAS XPORT dataset in {url}")
                    if archive.testzip() is not None:
                        raise ValueError(f"Corrupt Call Report archive from {url}")
                temporary.replace(path)
        # Do not replace a bad local file silently or make an unsolicited request.
        with zipfile.ZipFile(path) as archive:
            if not any(name.lower().endswith(".xpt") for name in archive.namelist()):
                raise ValueError(f"No SAS XPORT dataset in cached file {path}")
        paths[quarter] = path
    return paths


def _select_xport_columns(reader, columns: list[str]):
    """Project XPORT records before pandas decodes thousands of unused fields.

    Preserve the full record stride and original byte offsets in a sparse NumPy
    dtype. Pandas still handles SAS missing values and IBM numeric conversion.
    This small adapter relies on the installed pandas XportReader layout.
    """
    positions = [reader.columns.index(column) for column in columns]
    fields = [reader.fields[index] for index in positions]
    dtype = np.dtype({
        "names": [f"s{index}" for index in range(len(positions))],
        "formats": [f"S{field['field_length']}" for field in fields],
        "offsets": [reader._dtype.fields[f"s{index}"][1] for index in positions],
        "itemsize": reader.record_length,
    })
    reader.columns = columns
    reader.fields = fields
    reader._dtype = dtype


def load_online_call_reports(
    columns: list[str],
    start: str = "1994Q1",
    end: str = "2000Q4",
    data_dir: Path = ROOT / "data",
    chunksize: int = 500,
) -> pd.DataFrame:
    """Read selected raw MDRM fields from cached/downloaded historical files.

    Unlike ``load_call_reports``, these are raw regulatory fields (e.g.
    RSSD9001, RCFD2170, RIAD4170), not the supplied Stata file's friendly aliases.
    Income fields remain YTD. This function does not append them to the cleaned
    panel or assume that their definitions match local aliases. ``source_quarter``
    identifies each archive's reporting quarter. SAS files are read in small
    chunks without extracting or loading a complete quarterly dataset in memory.
    """
    if not columns or chunksize <= 0:
        raise ValueError("Supply at least one raw column and a positive chunksize")
    columns = list(dict.fromkeys(column.upper() for column in columns))
    if "SOURCE_QUARTER" in columns:
        raise ValueError("source_quarter is a generated column, not a source field")
    paths = download_call_reports(start=start, end=end, data_dir=data_dir)
    pieces = []
    for quarter, path in paths.items():
        with zipfile.ZipFile(path) as archive:
            members = [name for name in archive.namelist() if name.lower().endswith(".xpt")]
            if len(members) != 1:
                raise ValueError(f"Expected one XPORT dataset in {path}; found {members}")
            with archive.open(members[0]) as source:
                with pd.read_sas(source, format="xport", encoding="latin1", chunksize=chunksize) as reader:
                    missing = set(columns) - set(reader.columns)
                    if missing:
                        raise ValueError(f"Columns missing in {quarter}: {sorted(missing)}")
                    _select_xport_columns(reader, columns)
                    for chunk in reader:
                        selected = chunk.loc[:, columns].copy()
                        selected["source_quarter"] = quarter
                        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


# Verified against 8,762 matching RSSD observations in the 2001Q1 archive.
# Monetary fields are quarter-end stocks in thousands; RIAD4170 is a YTD flow.
HISTORICAL_FIELDS = {
    "rssdid": "RSSD9001", "cert": "RSSD9050", "assets": "RCFD2170",
    "domdep": "RCON2200", "intexpalldep": "RIAD4170", "loans": "RCFD2122",
    "reloans": "RCFD1410", "ciloans": "RCON1766",
    "loans_unearnedinc": "RCFD1400",
    "name": "RSSD9010", "state": "RSSD9200",
}


def harmonize_historical_reports(raw: pd.DataFrame) -> pd.DataFrame:
    """Map Chicago Fed raw fields to the local aliases, retaining YTD timing.

    RCFD stocks have consolidated domestic-plus-foreign scope; C&I loans use
    RCON1766 domestic-office scope to match the supplied alias. Securities are
    held-to-maturity at amortized cost plus available-for-sale at fair value.
    Missing foreign deposits are inferred only from reported consolidated minus
    domestic deposits. RSSD9331=1 identifies domestic commercial banks.
    Raw monetary amounts/IDs are reported as integers; rounding removes XPORT
    conversion residue near zero (e.g. 5.4e-79) without inventing missing values.
    """
    raw = raw.copy()
    numeric = raw.select_dtypes(include="number").columns
    raw[numeric] = raw[numeric].round()
    mapped = {alias: raw[code] for alias, code in HISTORICAL_FIELDS.items() if code in raw}
    d = pd.DataFrame(mapped)
    d["yq"] = pd.PeriodIndex(raw.source_quarter, freq="Q").to_timestamp()
    d["securities"] = raw.RCFD1754 + raw.RCFD1773 if "RCFD1754" in raw else raw.RCFD0390
    if "RIAD4172" in raw:
        d["intexpdomdep"] = raw.RIAD4170 - raw.RIAD4172.fillna(0)
    inferred_foreign = (raw.RCFD2200 - raw.RCON2200).where(lambda x: x.ge(0))
    d["foreigndep"] = raw.RCFN2200.combine_first(inferred_foreign)
    d["commercial_bank"] = raw.RSSD9331.eq(1).where(raw.RSSD9331.notna())
    d["expense_timing"] = "ytd"
    d["data_source"] = "chicago_fed"
    for column in ["name", "state"]:
        if column in d:
            d[column] = d[column].str.strip()
    return d


def load_historical_call_reports(
    data_dir: Path, columns: list[str] | None = None,
    end: str = "2000Q4",
    start: str = "1994Q1",
) -> pd.DataFrame:
    """Read/harmonize the cached 1994–2000 observations in bounded chunks."""
    aliases = list(dict.fromkeys(COLUMNS + (columns or [])))
    fields = [HISTORICAL_FIELDS[alias] for alias in aliases if alias in HISTORICAL_FIELDS]
    fields += ["RSSD9331", "RCFD2200", "RCFN2200"]
    if "intexpdomdep" in aliases:
        fields += ["RIAD4172"]
    frames = []
    if pd.Period(start, freq="Q") < pd.Period("1994Q1"):
        earlier = load_online_call_reports(fields + ["RCFD0390"], start=start, end="1993Q4", data_dir=data_dir)
        frames.append(harmonize_historical_reports(earlier))
    raw = load_online_call_reports(fields + ["RCFD1754", "RCFD1773"], start="1994Q1", end=end, data_dir=data_dir)
    frames.append(harmonize_historical_reports(raw))
    d = pd.concat(frames, ignore_index=True)
    # Requested local-only aliases are unavailable historically, not zero.
    for column in aliases:
        if column not in d:
            d[column] = np.nan
    return d


def fred_cpi() -> pd.Series:
    """Load annual FRED CPI, normalized to 2010 = 1 for real assets."""
    monthly = get_fred_series("CPIAUCSL")
    annual = monthly.groupby(monthly.index.year).mean()
    return (annual / annual.loc[2010]).rename("cpi").rename_axis("year")


def fred_rates(data_dir: Path, refresh: bool = False) -> pd.DataFrame:
    cache = data_dir / "fred"
    series = []
    for code in ["DFEDTAR", "DFEDTARU", "DFEDTARL"]:
        path = cache / f"{code}.csv"
        if refresh or not path.exists():
            s = get_fred_series(code).loc["1993-01-01":"2013-12-31"]
        else:
            s = pd.read_csv(path, index_col="date", parse_dates=True)[code]
        series.append(s.rename(code))
    daily = pd.concat(series, axis=1).reindex(pd.date_range("1993-01-01", "2013-12-31"))
    # Explicit switch: do not carry the discontinued point target into the range era.
    daily["fed_funds_target_pct"] = daily.DFEDTAR.where(
        daily.index < pd.Timestamp("2008-12-16"), (daily.DFEDTARU + daily.DFEDTARL) / 2
    )
    if daily.fed_funds_target_pct.isna().any():
        raise ValueError(
            "Missing daily FRED targets; refusing to average partial quarters"
        )
    daily.index.name = "date"
    q = daily.fed_funds_target_pct.groupby(daily.index.to_period("Q")).mean().to_frame()
    q["d_fed_funds_target_pct"] = q.fed_funds_target_pct.diff()
    q["fed_funds_target_end_pct"] = daily.fed_funds_target_pct.groupby(daily.index.to_period("Q")).last()
    q.index.name = "quarter"
    return q


def fdic_records(data_dir: Path) -> pd.DataFrame:
    """Use cached historical charter classes and YTD expenses (CB is not a charter flag)."""
    path = data_dir / "fdic_historical.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required: chartertype is missing in the supplied 2001–2013 data."
        )
    d = pd.read_csv(
        path,
        usecols=["CERT", "REPDTE", "BKCLASS", "EDEP"],
        dtype={"REPDTE": str, "BKCLASS": str},
    )
    d["quarter"] = pd.to_datetime(d.REPDTE, format="%Y%m%d").dt.to_period("Q")
    if d.duplicated(["CERT", "quarter"]).any():
        raise ValueError("Duplicate FDIC certificate-quarter keys")
    return d


def load_call_reports(
    data_dir: Path,
    chunksize: int = 25000,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Read selected columns, retaining only the analysis window plus its lag."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    pieces = []
    with pd.read_stata(
        data_dir / "call_reports.dta",
        columns=list(dict.fromkeys(COLUMNS + (columns or []))),
        convert_categoricals=False,
        chunksize=chunksize,
    ) as reader:
        for chunk in reader:
            keep = chunk.yq.between("1993-10-01", "2013-12-31")
            if keep.any():
                pieces.append(chunk.loc[keep].copy())
    if not pieces:
        raise ValueError("No Call Report observations in the requested window")
    return pd.concat(pieces, ignore_index=True)


def detect_expense_input(d: pd.DataFrame) -> tuple[str, dict]:
    """Identify transformed aliases by comparing with cached regulatory YTD flows."""
    groups = d.groupby(["rssdid", d.quarter.dt.year], sort=False)
    cumulative = groups.intexpalldep.cumsum(skipna=False)
    complete = groups.cumcount().add(1).eq(d.quarter.dt.quarter)
    comparable = complete & d.quarter.dt.quarter.gt(1) & d.EDEP.gt(0)
    comparable &= d.intexpalldep.notna() & cumulative.notna()
    scores = {
        "quarterly": (
            float(
                np.isclose(
                    cumulative[comparable], d.loc[comparable, "EDEP"], atol=2, rtol=0
                ).mean()
            )
            if comparable.any()
            else 0.0
        ),
        "ytd": (
            float(
                np.isclose(
                    d.loc[comparable, "intexpalldep"],
                    d.loc[comparable, "EDEP"],
                    atol=2,
                    rtol=0,
                ).mean()
            )
            if comparable.any()
            else 0.0
        ),
    }
    winner = max(scores, key=scores.get)
    if scores[winner] < 0.95 or abs(scores["quarterly"] - scores["ytd"]) < 0.5:
        raise ValueError(
            f"Cannot identify expense timing reliably: {scores}; specify expense_input"
        )
    return winner, {
        "expense_detection_rows": int(comparable.sum()),
        "expense_detection_match_fractions": scores,
    }


def quarterly_expense(d: pd.DataFrame, column: str, kind: str) -> pd.Series:
    """Missing quarters never stand in for zero, and Q1 resets the YTD counter."""
    x = d[column]
    if kind not in {"quarterly", "ytd"}:
        raise ValueError(f"Unknown expense timing: {kind}")
    if kind == "quarterly":
        return x.where(x >= 0)
    group = d.groupby("rssdid", sort=False)
    previous = group[column].shift()
    adjacent = d.qid.sub(group.qid.shift()).eq(1)
    flow = (x - previous).where(adjacent & d.quarter.dt.quarter.ne(1))
    flow = flow.where(d.quarter.dt.quarter.ne(1), x)
    return flow.where(flow >= 0)


def clean_panel(
    d: pd.DataFrame,
    rates: pd.DataFrame,
    fdic: pd.DataFrame,
    expense_input: str = "auto",
    include_presample: bool = False,
) -> tuple[pd.DataFrame, dict]:
    audit = {"source_rows": len(d), "expense_input": expense_input}
    d["quarter"] = pd.to_datetime(d.yq).dt.to_period("Q")
    audit["source_first_quarter"] = str(d.quarter.min())
    audit["source_last_quarter"] = str(d.quarter.max())
    d = d.loc[d.quarter.between(pd.Period("1993Q2"), pd.Period("2013Q4"))].copy()
    if (
        d[["rssdid", "quarter"]].isna().any().any()
        or d.duplicated(["rssdid", "quarter"]).any()
    ):
        raise ValueError("Missing or duplicate bank-quarter keys")
    d = d.sort_values(["rssdid", "quarter"])
    d["qid"] = d.quarter.astype("int64")
    d = d.merge(
        fdic[["CERT", "quarter", "BKCLASS", "EDEP"]],
        left_on=["cert", "quarter"],
        right_on=["CERT", "quarter"],
        how="left",
        validate="many_to_one",
        indicator="fdic_match",
    )
    # Use the Chicago Fed's historical entity type before 2001 and the cached
    # FDIC charter class thereafter. CB describes community-bank status only.
    fdic_commercial = d.BKCLASS.isin(["N", "NM", "SM"]).where(d.BKCLASS.notna())
    if "commercial_bank" in d:
        d["commercial_bank"] = d.commercial_bank.combine_first(fdic_commercial)
    else:
        d["commercial_bank"] = fdic_commercial
    audit["unclassified_rows"] = int(d.commercial_bank.isna().sum())
    audit["excluded_noncommercial_rows"] = int(d.commercial_bank.eq(False).sum())
    historical = d.get("expense_timing", pd.Series(index=d.index, dtype="object")).eq("ytd")
    local = d.loc[~historical]
    if expense_input == "auto":
        if local.empty:
            expense_input = "ytd"
        else:
            expense_input, detection = detect_expense_input(local)
            audit.update(detection)
    audit["expense_input"] = expense_input
    audit["historical_ytd_rows"] = int(historical.sum())
    audit["expense_input_by_source"] = {"local_stata": expense_input, "chicago_fed": "ytd"}
    # Compute both source flows before filtering or computing cross-source lags.
    local_flow = quarterly_expense(d, "intexpalldep", expense_input)
    ytd_flow = quarterly_expense(d, "intexpalldep", "ytd")
    d["deposit_interest_expense_q"] = local_flow.where(~historical, ytd_flow)
    cumulative = d.groupby(["rssdid", d.quarter.dt.year]).intexpalldep.cumsum(skipna=False)
    d["deposit_expense_ytd_check"] = (
        cumulative if expense_input == "quarterly" else d.intexpalldep
    ).where(~historical, d.intexpalldep)
    complete_year = (
        d.groupby(["rssdid", d.quarter.dt.year])
        .cumcount()
        .add(1)
        .eq(d.quarter.dt.quarter)
    )
    comparable = complete_year & d.EDEP.notna() & d.deposit_expense_ytd_check.notna()
    audit["fdic_expense_comparison_rows"] = int(comparable.sum())
    audit["fdic_expense_ytd_match_fraction"] = float(
        np.isclose(
            d.loc[comparable, "deposit_expense_ytd_check"],
            d.loc[comparable, "EDEP"],
            atol=2,
            rtol=0,
        ).mean()
    )
    # Missing foreign deposits are not assumed to equal zero.
    d["total_deposits"] = d.domdep + d.foreigndep
    d["deposit_expense_rate_pct"] = (
        400 * d.deposit_interest_expense_q / d.total_deposits
    ).where(d.total_deposits.gt(0))
    d["log_assets"] = np.log(d.assets.where(d.assets.gt(0)))
    d["total_loans"] = d.loans.where(d.loans.ge(0))
    d["securities"] = d.securities.where(d.securities.ge(0))
    d = d.merge(
        rates, left_on="quarter", right_index=True, how="left", validate="many_to_one"
    )
    if d.fed_funds_target_pct.isna().any():
        raise ValueError("Unmatched FRED quarter")
    d["deposit_spread_pct"] = d.fed_funds_target_pct - d.deposit_expense_rate_pct
    d = d.loc[d.commercial_bank.eq(True) & d.assets.gt(0) & d.total_deposits.gt(0)].copy()
    d = d.sort_values(["rssdid", "quarter"])
    g = d.groupby("rssdid", sort=False)
    adjacent = d.qid.sub(g.qid.shift()).eq(1)
    for col in [
        "deposit_spread_pct",
        "total_deposits",
        "log_assets",
        "securities",
        "total_loans",
    ]:
        d["d_" + col] = g[col].diff().where(adjacent)
    for col in ["total_deposits", "securities", "total_loans"]:
        prior = g[col].shift()
        d["growth_" + col] = (d[col] / prior - 1).where(adjacent & prior.gt(0))
        d["d_log_" + col] = (
            np.log(d[col].where(d[col].gt(0))) - np.log(prior.where(prior.gt(0)))
        ).where(adjacent)
    lower = "1993Q2" if include_presample else "1994Q1"
    d = d.loc[d.quarter.between(pd.Period(lower), pd.Period("2013Q4"))].copy()
    observed_quarters = set(d.quarter)
    audit.update(
        output_rows=len(d),
        banks=int(d.rssdid.nunique()),
        first_quarter=str(d.quarter.min()),
        last_quarter=str(d.quarter.max()),
        missing_requested_quarters=[
            str(q)
            for q in pd.period_range("1994Q1", "2013Q4", freq="Q")
            if q not in observed_quarters
        ],
    )
    audit["missing_values"] = {
        c: int(d[c].isna().sum()) for c in d if c.startswith(("d_", "deposit_"))
    }
    d["quarter"] = d.quarter.astype(str)
    d["quarter_date"] = pd.PeriodIndex(d.quarter, freq="Q").to_timestamp()
    d["rssdid"] = d.rssdid.astype("int64")
    return d.drop(columns=["yq", "CERT", "fdic_match"]), audit


def get_data(
    columns: list[str] | None = None,
    *,
    data_dir: Path = ROOT / "data",
    chunksize: int = 25000,
    expense_input: str = "auto",
    refresh_fred: bool = False,
    output_columns: list[str] | None = None,
    include_presample: bool = False,
) -> pd.DataFrame:
    """Return the cleaned commercial-bank panel without writing output files.

    ``columns`` adds source fields to the required cleaning inputs. These fields
    survive in the returned panel. ``output_columns`` optionally restricts the
    returned fields, including derived measures. Reads are chunked and restricted to the time
    window before concatenation. Cleaning diagnostics are in ``df.attrs['audit']``.
    Missing Chicago Fed archives may be fetched and cached. Historical FDIC
    charter classes for the local source must already exist.
    """
    data_dir = Path(data_dir)
    rates = fred_rates(data_dir, refresh_fred)
    fdic = fdic_records(data_dir)
    d = load_call_reports(data_dir, chunksize, columns)
    d["data_source"] = "local_stata"
    d["expense_timing"] = "local"
    first_local = pd.Period(d.yq.min(), freq="Q")
    if first_local > pd.Period("1994Q1"):
        end = min(first_local - 1, pd.Period("2000Q4"))
        historical = load_historical_call_reports(data_dir, columns, end=str(end),
                                                    start="1993Q2" if include_presample else "1994Q1")
        # Fill only quarters preceding the first supplied local observation.
        d = pd.concat([historical, d], ignore_index=True)
    panel, audit = clean_panel(d, rates, fdic, expense_input, include_presample=include_presample)
    if output_columns is not None:
        panel = panel.loc[:, output_columns].copy()
    panel.attrs["audit"] = audit
    return panel


AUTHOR_SOURCE = "https://dataverse.harvard.edu/api/access/datafile/"


def author_replication_file(filename: str, file_id: int, data_dir: Path = ROOT / "data") -> Path:
    """Cache an explicitly requested original input from doi:10.7910/DVN/KHNXYJ."""
    path = Path(data_dir) / "replication" / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".download_") as directory:
            temporary = Path(directory) / filename
            with requests.get(f"{AUTHOR_SOURCE}{file_id}?format=original", stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as destination:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            destination.write(chunk)
            temporary.replace(path)
    return path


def load_author_call_reports() -> pd.DataFrame:
    """Explicit benchmark source; read only 13 original inputs in chunks."""
    path = author_replication_file("sample_call_final.dta", 3021712)
    columns = ["cert", "totdep", "assets", "loans", "reloans", "ciloansother", "securities",
               "depexp", "date", "ff_tar_quar_avg", "ff_tar", "quar", "assets_2010"]
    with pd.read_stata(path, columns=columns, chunksize=25000, convert_categoricals=False) as reader:
        return pd.concat(list(reader), ignore_index=True)
