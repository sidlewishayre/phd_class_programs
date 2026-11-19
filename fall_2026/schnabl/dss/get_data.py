"""Build the available commercial-bank panel; see DATA_NOTES.md for source caveats."""

from __future__ import annotations

import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)
from settings import REUSABLE_CODE_DIR

sys.path.append(str(REUSABLE_CODE_DIR))

import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from database_management.fred_pull import get_fred_series

import settings

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
    cache = settings.CALL_REPORTS_CACHE_DIR
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
            with tempfile.TemporaryDirectory(
                prefix=".download_", dir=cache
            ) as directory:
                temporary = Path(directory) / path.name
                with requests.get(url, stream=True, timeout=(15, 120)) as response:
                    response.raise_for_status()
                    with temporary.open("wb") as destination:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                destination.write(block)
                with zipfile.ZipFile(temporary) as archive:
                    if not any(
                        name.lower().endswith(".xpt") for name in archive.namelist()
                    ):
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
    dtype = np.dtype(
        {
            "names": [f"s{index}" for index in range(len(positions))],
            "formats": [f"S{field['field_length']}" for field in fields],
            "offsets": [reader._dtype.fields[f"s{index}"][1] for index in positions],
            "itemsize": reader.record_length,
        }
    )
    reader.columns = columns
    reader.fields = fields
    reader._dtype = dtype


def load_online_call_reports(
    columns: list[str],
    start: str = "1994Q1",
    end: str = "2000Q4",
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
    paths = download_call_reports(start=start, end=end)
    pieces = []
    for quarter, path in paths.items():
        with zipfile.ZipFile(path) as archive:
            members = [
                name for name in archive.namelist() if name.lower().endswith(".xpt")
            ]
            if len(members) != 1:
                raise ValueError(
                    f"Expected one XPORT dataset in {path}; found {members}"
                )
            with archive.open(members[0]) as source:
                with pd.read_sas(
                    source, format="xport", encoding="latin1", chunksize=chunksize
                ) as reader:
                    missing = set(columns) - set(reader.columns)
                    if missing:
                        raise ValueError(
                            f"Columns missing in {quarter}: {sorted(missing)}"
                        )
                    _select_xport_columns(reader, columns)
                    for chunk in reader:
                        selected = chunk.loc[:, columns].copy()
                        selected["source_quarter"] = quarter
                        pieces.append(selected)
    return pd.concat(pieces, ignore_index=True)


# Verified against 8,762 matching RSSD observations in the 2001Q1 archive.
# Monetary fields are quarter-end stocks in thousands; RIAD4170 is a YTD flow.
HISTORICAL_FIELDS = {
    "rssdid": "RSSD9001",
    "cert": "RSSD9050",
    "assets": "RCFD2170",
    "equity": "RCFD3210",
    "liabilities": "RCFD2948",
    "domdep": "RCON2200",
    "intexpalldep": "RIAD4170",
    "loans": "RCFD2122",
    "reloans": "RCFD1410",
    "ciloans": "RCON1766",
    "loans_unearnedinc": "RCFD1400",
    "name": "RSSD9010",
    "state": "RSSD9200",
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
    mapped = {
        alias: raw[code] for alias, code in HISTORICAL_FIELDS.items() if code in raw
    }
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
    columns: list[str] | None = None,
    end: str = "2000Q4",
    start: str = "1994Q1",
) -> pd.DataFrame:
    """Read/harmonize the cached 1994–2000 observations in bounded chunks."""
    aliases = list(dict.fromkeys(COLUMNS + (columns or [])))
    fields = [
        HISTORICAL_FIELDS[alias] for alias in aliases if alias in HISTORICAL_FIELDS
    ]
    fields += ["RSSD9331", "RCFD2200", "RCFN2200"]
    if "intexpdomdep" in aliases:
        fields += ["RIAD4172"]
    frames = []
    if pd.Period(start, freq="Q") < pd.Period("1994Q1"):
        earlier = load_online_call_reports(
            fields + ["RCFD0390"], start=start, end="1993Q4"
        )
        frames.append(harmonize_historical_reports(earlier))
    raw = load_online_call_reports(
        fields + ["RCFD1754", "RCFD1773"], start="1994Q1", end=end
    )
    frames.append(harmonize_historical_reports(raw))
    d = pd.concat(frames, ignore_index=True)
    # Requested local-only aliases are unavailable historically, not zero.
    for column in aliases:
        if column not in d:
            d[column] = np.nan
    return d


def fred_cpi() -> pd.Series:
    """Load annual FRED CPI, normalized to 2010 = 1 for real assets."""
    path = settings.FRED_CACHE_DIR / "CPIAUCSL.csv"
    if path.exists():
        monthly = pd.read_csv(path, index_col="observation_date", parse_dates=True)["CPIAUCSL"]
    else:
        monthly = get_fred_series("CPIAUCSL")
    annual = monthly.groupby(monthly.index.year).mean()
    return (annual / annual.loc[2010]).rename("cpi").rename_axis("year")


def fred_rates(refresh: bool = False) -> pd.DataFrame:
    cache = settings.FRED_CACHE_DIR
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
    q["fed_funds_target_end_pct"] = daily.fed_funds_target_pct.groupby(
        daily.index.to_period("Q")
    ).last()
    q.index.name = "quarter"
    return q


def fdic_records() -> pd.DataFrame:
    """Use cached historical charter classes and YTD expenses (CB is not a charter flag)."""
    path = settings.FDIC_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is required for the 2001–2013 FDIC charter-class filter."
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


def _read_ny_fed_reports(kind, columns, start, end, chunksize):
    """Extract the publisher's Stata file and retain the requested quarters."""
    release = f"call-reports-{kind}-Jan2026"
    cache = settings.NY_FED_CALL_REPORTS_DIR / f"{release}.zip"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache.parent) as directory:
        if not cache.exists():
            download = Path(directory) / cache.name
            url = f"{settings.NY_FED_CALL_REPORTS_URL}/{release}"
            with requests.get(url, stream=True, timeout=120) as response:
                response.raise_for_status()
                with download.open("wb") as target:
                    shutil.copyfileobj(response.raw, target)
            with zipfile.ZipFile(download):
                pass
            download.replace(cache)
        with zipfile.ZipFile(cache) as archive:
            member, = [name for name in archive.namelist() if name.endswith(".dta")]
            stata_file = Path(directory) / "reports.dta"
            with archive.open(member) as source, stata_file.open("wb") as target:
                shutil.copyfileobj(source, target)
        with pd.read_stata(stata_file, columns=columns, convert_categoricals=False,
                           chunksize=chunksize) as reader:
            return pd.concat(
                [chunk.loc[chunk.date.between(start, end)] for chunk in reader],
                ignore_index=True,
            )


def load_call_reports(
    chunksize: int = 25000,
    columns: list[str] | None = None,
    *,
    start: str = "1994Q1",
    end: str = "2013Q4",
) -> pd.DataFrame:
    """Load New York Fed Call Reports with local aliases and YTD expenses."""
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    start = max(pd.Period(start, freq="Q"), pd.Period("1994Q1")).start_time
    end = pd.Period(end, freq="Q").end_time
    aliases = {
        "id_rssd": "rssdid", "id_fdic_cert": "cert", "date": "yq",
        "chtr_type_cd": "chartertype", "nm_lgl": "name", "state_abbr_nm": "state",
        "domestic_dep": "domdep", "foreign_dep": "foreigndep",
        "deposits": "totdep", "ln_tot": "loans", "ln_tot_gross": "loans_unearnedinc",
        "ln_re": "reloans", "ln_ci": "ciloans", "liab_tot_unadj": "liabilities",
        "ytdint_exp_dep": "intexpalldep",
    }
    balance = _read_ny_fed_reports(
        "balance-sheets",
        [field for field in aliases if field != "ytdint_exp_dep"]
        + ["ent_type_cd", "assets", "securities", "equity", "insured_deposits"],
        start, end, chunksize,
    )
    income = _read_ny_fed_reports(
        "income-statements", ["id_rssd", "date", "ytdint_exp_dep", "ytdint_exp_fgn"],
        start, end, chunksize,
    )
    d = balance.merge(income, on=["id_rssd", "date"], how="left",
                      validate="one_to_one").rename(columns=aliases)
    if d.empty:
        raise ValueError("No Call Report observations in the requested window")
    d["intexpdomdep"] = d.intexpalldep - d.ytdint_exp_fgn.fillna(0)
    inferred_foreign = (d.totdep - d.domdep).where(lambda x: x.ge(0))
    d["foreigndep"] = d.foreigndep.combine_first(inferred_foreign)
    d["commercial_bank"] = d.ent_type_cd.eq(1).where(
        d.ent_type_cd.notna() & d.yq.lt("2001-01-01")
    )
    d["data_source"] = "ny_fed"
    d["expense_timing"] = "ytd"
    # Preserve the robustness input as an uninsured domestic deposit proxy.
    d["uninsureddep2"] = (d.domdep - d.insured_deposits).where(
        d.insured_deposits.gt(0) & d.insured_deposits.le(d.domdep)
    )
    for column in columns or []:
        if column not in d:
            d[column] = np.nan
    return d[list(dict.fromkeys(
        COLUMNS + (columns or []) + ["commercial_bank", "data_source", "expense_timing"]
    ))]


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
    # Use the historical entity type before 2001 and the cached FDIC
    # charter class thereafter. CB describes community-bank status only.
    fdic_commercial = d.BKCLASS.isin(["N", "NM", "SM"]).where(d.BKCLASS.notna())
    if "commercial_bank" in d:
        d["commercial_bank"] = d.commercial_bank.combine_first(fdic_commercial)
    else:
        d["commercial_bank"] = fdic_commercial
    audit["unclassified_rows"] = int(d.commercial_bank.isna().sum())
    audit["excluded_noncommercial_rows"] = int(d.commercial_bank.eq(False).sum())
    historical = d.get("expense_timing", pd.Series(index=d.index, dtype="object")).eq(
        "ytd"
    )
    local = d.loc[~historical]
    if expense_input == "auto":
        if local.empty:
            expense_input = "ytd"
        else:
            expense_input, detection = detect_expense_input(local)
            audit.update(detection)
    audit["expense_input"] = expense_input
    audit["historical_ytd_rows"] = int(historical.sum())
    if "data_source" in d:
        audit["expense_input_by_source"] = {
            source: "ytd" if historical.loc[group.index].all() else expense_input
            for source, group in d.groupby("data_source")
        }
    # Compute both source flows before filtering or computing cross-source lags.
    local_flow = quarterly_expense(d, "intexpalldep", expense_input)
    ytd_flow = quarterly_expense(d, "intexpalldep", "ytd")
    d["deposit_interest_expense_q"] = local_flow.where(~historical, ytd_flow)
    cumulative = d.groupby(["rssdid", d.quarter.dt.year]).intexpalldep.cumsum(
        skipna=False
    )
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
    d = d.loc[
        d.commercial_bank.eq(True) & d.assets.gt(0) & d.total_deposits.gt(0)
    ].copy()
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
    New York Fed archives are downloaded and cached on first use. The source
    starts at 1994Q1, including when include_presample is requested.
    """
    rates = fred_rates(refresh=refresh_fred)
    fdic = fdic_records()
    d = load_call_reports(chunksize=chunksize, columns=columns)
    panel, audit = clean_panel(
        d, rates, fdic, expense_input, include_presample=include_presample
    )
    if output_columns is not None:
        panel = panel.loc[:, output_columns].copy()
    panel.attrs["audit"] = audit
    return panel


PAPER_STOCKS = {
    "deposit_flow_beta": "domestic_deposits",
    "asset_flow_beta": "assets",
    "securities_flow_beta": "securities",
    "loan_flow_beta": "gross_loans",
    "re_loan_flow_beta": "reloans",
    "ci_loan_flow_beta": "ciloans",
}


def fred_mmf_assets(refresh=False) -> pd.Series:
    """Quarter-end total MMF financial assets, millions of USD, NSA (FRED)."""
    if refresh or not settings.MMF_AUM_CACHE_FILE.exists():
        assets = get_fred_series(settings.MMF_FRED_SERIES).rename("mmf_assets")
        assets.index.name = "date"
        settings.MMF_AUM_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        assets.to_csv(settings.MMF_AUM_CACHE_FILE)
    else:
        assets = pd.read_csv(settings.MMF_AUM_CACHE_FILE, index_col="date",
                             parse_dates=True).mmf_assets
    assets.index = assets.index.to_period("Q")
    if assets.index.has_duplicates:
        raise ValueError("Duplicate quarters in FRED MMF assets")
    return assets.sort_index()


def fred_aggregate_deposits(refresh=False) -> pd.Series:
    """Last weekly observation per quarter, commercial-bank deposits in millions.

    FRED DPSACBW027NBOG is NSA and reported in billions of dollars.
    """
    path = settings.MMF_DEPOSIT_CACHE_FILE
    if refresh or not path.exists():
        deposits = get_fred_series(settings.MMF_DEPOSIT_SERIES).rename("deposits")
        deposits.index.name = "date"
        path.parent.mkdir(parents=True, exist_ok=True)
        deposits.to_csv(path)
    else:
        deposits = pd.read_csv(path, index_col="date", parse_dates=True).deposits
    deposits = deposits.sort_index()
    return deposits.groupby(deposits.index.to_period("Q")).last() * 1000


def compute_mmf_share_inputs(assets, deposits):
    """Both inputs are quarterly stocks in millions; changes are percentage points."""
    quarters = pd.period_range("1993Q4", "2013Q4", freq="Q")
    data = pd.DataFrame({"mmf_assets": assets, "aggregate_deposits": deposits}).reindex(quarters)
    valid = data.mmf_assets.gt(0) & data.aggregate_deposits.gt(0)
    data["mmf_share"] = (data.mmf_assets / (data.mmf_assets + data.aggregate_deposits)).where(valid)
    data["d_mmf_share"] = 100 * data.mmf_share.diff()
    if data[["mmf_assets", "aggregate_deposits", "mmf_share"]].isna().any().any():
        raise ValueError("Missing aggregate MMF/deposit stocks in the estimation window")
    return data


def get_mmf_panel(panel=None, mmf_assets=None, deposits=None, *, ffr_lags=4) -> pd.DataFrame:
    """Bank spread levels and MMF share changes, with current/lagged FFR changes.

    Spread uses the same annualized domestic deposit rate as spread_beta.
    MMF share is MMF assets / (MMF assets + aggregate commercial-bank deposits).
    Its quarterly first difference is measured in percentage points.
    Inputs are untrimmed; the beta estimator winsorizes final bank coefficients.
    """
    d = get_beta_panel(panel=panel, max_lag=ffr_lags)
    assets = fred_mmf_assets() if mmf_assets is None else mmf_assets.copy()
    deposits = fred_aggregate_deposits() if deposits is None else deposits.copy()
    share_inputs = compute_mmf_share_inputs(assets, deposits)
    settings.MMF_DIR.mkdir(parents=True, exist_ok=True)
    share_inputs.to_csv(settings.MMF_SHARE_INPUT_FILE, index_label="quarter")
    d["mmfflow"] = d.quarter.map(share_inputs.d_mmf_share)
    d["mmf_assets"] = d.quarter.map(share_inputs.mmf_assets)
    d["mmf_share"] = d.quarter.map(share_inputs.mmf_share)
    d["mmf_beta_y"] = d.deposit_spread
    return d[["cert", "quarter", "deposit_spread", "mmfflow", "mmf_assets", "mmf_share",
              "mmf_beta_y", *[f"x{i}" for i in range(ffr_lags + 1)]]].copy()


def load_mmf_control_betas() -> pd.DataFrame:
    """Load the matched, winsorized bank data for the MMF-control tables."""
    return pd.read_csv(settings.MMF_CONTROL_DIR / "bank_betas.csv")


def load_mmf_betas() -> pd.DataFrame:
    """Read the per-bank estimates produced by compute_betas.py."""
    return pd.read_csv(settings.MMF_BETAS_FILE)


def load_robustness_results() -> pd.DataFrame:
    """Load saved cross-bank coefficients for inspection or regenerating plots."""
    return pd.read_csv(settings.ROBUSTNESS_COEFFICIENTS_FILE)


def load_robustness_regression_inputs() -> pd.DataFrame:
    """Load saved bank-level regression inputs for extending robustness runs."""
    return pd.read_stata(settings.ROBUSTNESS_DATA_FILE)


def load_robustness_quartile_coverage() -> pd.DataFrame:
    """Load the saved characteristic-quartile sample coverage."""
    return pd.read_csv(settings.ROBUSTNESS_DIR / "quartile_coverage.csv")


def get_robustness_panel(max_lag=5) -> pd.DataFrame:
    """Load beta inputs and quarterly characteristics for robustness exercises.

    Leverage is assets/equity; funding share is total deposits/liabilities.
    Uninsured share uses domestic deposits less reported insured deposits
    as a proxy. The share is missing where insured deposits are unavailable.
    """
    panel = get_data(
        columns=["reloans", "ciloans", "intexpdomdep", "loans_unearnedinc",
                 "equity", "liabilities", "uninsureddep2"],
        include_presample=True,
    )
    d = get_beta_panel(panel=panel, max_lag=max_lag)
    positive_assets = d.assets.where(d.assets.gt(0))
    positive_equity = d.equity.where(d.equity.gt(0))
    d["leverage"] = positive_assets / positive_equity
    d["deposit_funding_share"] = d.total_deposits / d.liabilities.where(d.liabilities.gt(0))
    d["uninsured_share"] = (
        d.uninsureddep2.where(d.uninsureddep2.ge(0)) / d.domdep.where(d.domdep.gt(0))
    ).where(lambda s: s.le(1))
    d["size"] = d.real_assets.where(d.real_assets.gt(0))
    return d


def get_beta_panel(panel=None, rates=None, *, max_lag=4, start="1994Q1", end="2013Q4") -> pd.DataFrame:
    """Load and prepare untrimmed quarterly inputs for bank beta estimation."""
    if panel is None:
        panel = get_data(
            columns=["reloans", "ciloans", "intexpdomdep", "loans_unearnedinc"],
            include_presample=True,
        )
    d = panel.copy()
    d["quarter"] = pd.PeriodIndex(d.quarter, freq="Q")
    d = d.sort_values(["cert", "quarter"]).reset_index(drop=True)
    if d.duplicated(["cert", "quarter"]).any() or d.cert.isna().any():
        raise ValueError(
            "Paper procedure requires unique, nonmissing certificate-quarter keys"
        )
    d["domestic_deposits"] = d.domdep
    # The local gross-loan alias is sparsely reported; retain net-of-unearned
    # loans as an explicitly documented proxy where gross loans are absent.
    d["gross_loans"] = d.loans_unearnedinc.combine_first(d.loans)
    g = d.groupby("cert", sort=False)
    adjacent = d.quarter.astype("int64").sub(g.quarter.shift().astype("int64")).eq(1)
    flow = (d.intexpdomdep - g.intexpdomdep.shift()).where(adjacent)
    flow = flow.where(d.quarter.dt.quarter.ne(1), d.intexpdomdep)
    d["expense_q"] = d.intexpdomdep.where(d.expense_timing.ne("ytd"), flow)
    cpi = fred_cpi()
    d["real_assets"] = d.assets / d.quarter.dt.year.map(cpi)
    if rates is None:
        rates = fred_rates()
    d["ff_tar"] = d.quarter.map(rates.fed_funds_target_end_pct)
    d["ff_tar_quar_avg"] = d.quarter.map(rates.fed_funds_target_pct)
    for beta, column in PAPER_STOCKS.items():
        logged = np.log(d[column].where(d[column].gt(0)))
        growth = 100 * logged.groupby(d.cert).diff().where(adjacent)
        d[beta + "_y"] = growth
    denominator = g.domestic_deposits.shift().where(adjacent)
    rate = 400 * d.expense_q / denominator.where(denominator.gt(0))
    spread = d.ff_tar_quar_avg - rate
    d["deposit_spread"] = spread
    d["spread_beta_y"] = spread.groupby(d.cert).diff().where(adjacent)
    d["x0"] = d.ff_tar.groupby(d.cert).diff().where(adjacent)
    if start == "1994Q1":
        # The target is common to all banks. Seed the first policy change
        # from FRED so 1995Q1 has all four lags without 1993 bank reports.
        first = pd.Period(start)
        targets = rates.fed_funds_target_end_pct.reindex([first - 1, first])
        d.loc[d.quarter.eq(first), "x0"] = targets.diff().iloc[-1]
        d = d.loc[d.quarter.between(pd.Period(start), pd.Period(end))].copy()
    g = d.groupby("cert", sort=False)
    qid = d.quarter.astype("int64")
    for lag in range(1, max_lag + 1):
        d[f"x{lag}"] = g.x0.shift(lag).where(
            qid.sub(g.quarter.shift(lag).astype("int64")).eq(lag)
        )
    return d.loc[d.quarter.between(pd.Period(start), pd.Period(end))].copy()



def load_ai_audit_inputs():
    """Read the audit's prepared panel and daily policy targets for comparisons."""
    panel = pd.read_pickle(settings.AI_AUDIT_PANEL_FILE)
    daily = pd.read_csv(settings.AI_AUDIT_RATES_FILE, parse_dates=["observation_date"])
    daily["target"] = daily.DFEDTAR.fillna((daily.DFEDTARU + daily.DFEDTARL) / 2)
    daily["quarter"] = daily.observation_date.dt.to_period("Q")
    rates = daily.groupby("quarter").target.agg(
        fed_funds_target_end_pct="last", fed_funds_target_pct="mean"
    )
    return panel, rates


def get_out_of_sample_panel():
    """Recent quarterly reports with the existing endpoint commercial-bank filter."""
    columns = ["cert", "yq", "assets", "domdep", "intexpdomdep", "loans",
               "loans_unearnedinc", "securities", "reloans", "ciloans"]
    d = load_call_reports(columns=columns, start="2021Q1", end="2023Q4")
    d["quarter"] = d.yq.dt.to_period("Q")
    d["qid"] = d.quarter.astype("int64")
    d = d.sort_values(["rssdid", "quarter"])
    d["intexpdomdep"] = quarterly_expense(d, "intexpdomdep", "ytd")
    fdic = pd.read_csv(settings.OUT_OF_SAMPLE_FDIC_FILE)
    eligible = fdic.assign(commercial=fdic.BKCLASS.isin(["N", "NM", "SM"]))
    eligible = eligible.groupby("CERT").commercial.agg(["all", "count"])
    d = d[d.cert.isin(eligible.index[eligible["all"] & eligible["count"].eq(2)])].copy()
    d["expense_timing"] = "quarterly"
    targets = pd.read_csv(settings.OUT_OF_SAMPLE_RATES_FILE, index_col="date", parse_dates=True)
    midpoint = (targets.DFEDTARU + targets.DFEDTARL) / 2
    rates = midpoint.groupby(midpoint.index.to_period("Q")).agg(["mean", "last"])
    rates.columns = ["fed_funds_target_pct", "fed_funds_target_end_pct"]
    return d, rates


def load_dkn_call_reports(start, end, chunksize=25000):
    """Read only the New York Fed fields needed for aggregate deposit spreads."""
    first = pd.Period(start, freq="Q").start_time
    last = pd.Period(end, freq="Q").end_time
    keys = ["id_rssd", "date"]
    balance = _read_ny_fed_reports(
        "balance-sheets",
        keys + ["ent_type_cd", "assets", "deposits", "domestic_dep"],
        first, last, chunksize,
    )
    income = _read_ny_fed_reports(
        "income-statements", keys + ["ytdint_exp_dep"], first, last, chunksize,
    )
    return balance.merge(income, on=keys, how="left", validate="one_to_one")


def load_dkn_rates():
    """Read the cached monthly effective fed funds rate and Treasury yields."""
    rates = pd.read_csv(settings.DKN_RATES_FILE, parse_dates=["observation_date"])
    return rates.set_index("observation_date")[["FEDFUNDS", "GS1", "GS5"]]


def load_dss_dkn_reports():
    """DKN inputs plus publisher-labeled total YTD interest expense."""
    reports = load_dkn_call_reports("1984Q1", "2021Q2")
    income = _read_ny_fed_reports(
        "income-statements", ["id_rssd", "date", "ytdint_exp"],
        pd.Timestamp("1984-01-01"), pd.Timestamp("2021-06-30"), 25000,
    )
    return reports.merge(income, on=["id_rssd", "date"], validate="one_to_one")
