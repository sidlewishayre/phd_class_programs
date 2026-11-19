# Call Report preparation

```python
from get_data import get_data
banks = get_data(columns=["reloans", "ciloans"])
print(banks.attrs["audit"])
```

`get_data()` loads the New York Fed balance-sheet and income-statement files,
joins them on RSSD ID and quarter, and retains 1994Q1–2013Q4. The out-of-sample
loader uses the same source for 2021Q1–2023Q4. The archives are cached in
`data/nyfed`. Deposit interest expense is year-to-date and converted to
quarterly flows during preparation. Commercial-bank classification continues
to use historical entity type before 2001 and FDIC charter class thereafter.
The 1994Q1 policy change uses the difference between the cached 1993Q4 and
1994Q1 quarter-end FRED targets. This supplies the fourth policy lag for
1995Q1 without using pre-1994 Call Reports.

The sections below document the prior Chicago Fed and supplied Stata inputs.

## Commercial-bank classification correction

For 1994–2000, RSSD9331=1 identifies commercial banks using the official MDRM
entity-type definition. For the local source, historical FDIC BKCLASS values
N, NM and SM identify national, state nonmember and state member commercial
banks. Savings institutions and foreign branches are excluded; unknown classes
remain excluded. The previous use of CB=1 was incorrect: CB indicates
**community-bank status**, not commercial-bank status, and excluded major
commercial banks. The filter is now corrected throughout the local period.
No additional FDIC downloads are needed.

FDIC source: https://www.fdic.gov/system/files/2024-06/currentcommunitybankingreferencedata-notestousers.pdf
Bank-class definitions: https://api.fdic.gov/banks/docs/

## Historical variable crosswalk

Historical fields are used over 1994–2000. Monetary stocks are reported in
thousands of dollars. RCFD denotes consolidated domestic-plus-foreign scope;
RCON denotes domestic-office scope.

| Local alias | Historical field/construction | Scope/timing |
| --- | --- | --- |
| assets | RCFD2170 | Consolidated quarter-end assets |
| domdep | RCON2200 | Domestic quarter-end deposits |
| foreigndep | RCFN2200; if missing, RCFD2200 minus RCON2200 | Foreign deposits, inferred only from two reported stocks |
| intexpalldep | RIAD4170 | YTD deposit interest expense |
| securities | RCFD1754 + RCFD1773 | HTM at amortized cost plus AFS at fair value |
| loans | RCFD2122 | Consolidated loans/leases net of unearned income |
| reloans | RCFD1410 | Loans secured by real estate |
| ciloans | RCON1766 | Domestic-office C&I loans |

These choices use the local MDRM dictionary and the Chicago Fed's consistency
notes, then an overlapping 2001Q1 source file was checked against the local
aliases. Among 8,762 matching RSSD observations, certificates matched exactly;
assets matched in 99.92%, domestic deposits 99.76%, deposit expense 99.99%,
loans 99.97%, securities 99.98%, inferred foreign deposits 99.99%, real-estate
loans 100%, and domestic C&I loans 99.99%, using a half-unit tolerance. Source
vintages differ, so discrepancies remain; these checks do not establish an
exact original WRDS construction-code crosswalk. In particular RCFD1766 was
rejected for C&I because it differed for banks with foreign-office lending.

XPORT conversion can represent a reported zero by a tiny number near 5.4e-79.
Historical integer monetary fields and identifiers are rounded to their reported
integer precision; NaNs remain missing. Unknown requested historical aliases
remain missing rather than being assigned an unverified regulatory mapping.

Dictionary/provenance: `data/call_report_dictionary_sources.md` and
`data/MDRM_CSV.csv`.
Consistency notes: https://www.chicagofed.org/-/media/others/banking/financial-institution-reports/notes-on-forming-consistent-time-series-pdf.pdf

## Expenses, rates, and changes

Historical income fields retain YTD timing until cleaning: Q1 is the reported
value, and Q2–Q4 subtract the immediately preceding quarter of the same bank
and year. Missing prior quarters do not imply zero. The local expense alias
is already quarterly; auto detection verifies this against cached FDIC EDEP.
It requires at least 95% agreement and a 50 percentage point lead over the
alternative timing interpretation. Explicit `expense_input` overrides apply
to local records; historical records always retain their known YTD timing.
Flows are normalized before combining bank differences across 2000–2001.
Negative quarterly flows are treated as missing.

Total deposits equal domestic plus foreign deposits; missing components remain
missing. Annualized deposit expense in percentage points is `400 * quarterly
expense / total_deposits`. Deposit spread is the quarterly mean target minus
this rate. The existing rate convention uses DFEDTAR before 2008-12-16 and
DFEDTARU thereafter (the target-range upper bound). This differs from the
midpoint convention described in the paper. Existing cached FRED series are
reused; absent series use the reusable `fred_pull.py` helper.

`d_*` fields are consecutive-quarter changes in spreads, deposits, log assets,
securities and loans. Rate changes are percentage points. The Fed funds change
uses the full quarterly rate series. Gaps in bank observations produce missing
bank differences. Nonpositive assets/deposits are excluded; negative securities
and loans become missing. No blanket complete-case filter, winsorization of
input observations, merger adjustment or fabrication of lag quarters is applied.

## Betas, regressions, and writeup

`compute_betas()` returns a DataFrame without writing files. As in equation (21)
of `dss.pdf`, bank-level OLS includes an intercept and the current plus four
lagged Fed funds changes. Each beta sums the five slopes. Flow outcomes are
100 times log changes, with the rate regressor in percentage points. At least
seven observations and full regressor rank are required per outcome. Raw betas
are retained; unsuffixed betas cap both tails at the 5th/95th percentiles.

`beta_regs.py` estimates six bank-level flow-beta regressions on spread beta,
with robust SEs, using reg_runner/Stata. Panel A uses all identified banks.
Panel B uses banks at or above the 95th percentile of sample mean assets among
banks with identified spread betas. The PDF specifies the top 5% by assets
but not a reference quarter, so mean assets is an implementation choice.
Winsorization uses the full sample before selecting large banks.

Figure VI uses 100 rank bins of spread beta, stable bank-ID tie breaking,
available outcome means in each bin, and fitted bank-level OLS lines. Figure
VII applies the same plotting procedure to large banks. Temporary bank-beta
files are deleted after Stata finishes; tables, logs and figures are retained
under `results/beta_regs/`.

Both `compute_betas.py` and `beta_regs.py` refresh `results/beta_comparison.tex`
from current estimates and sample dates. The writeup includes this generated
fragment. `raw_data_plots.py` scales monetary mean changes from thousands to
trillions and derives its displayed sample years from the current data.
Recompile `pset1_writeup.tex` after regenerating the analysis outputs.

## Cached historical downloads

`download_call_reports()` caches the 1994Q1–2000Q4 Chicago Fed ZIP archives in
`data/call_reports_online/`; only absent files are fetched. Downloads stream to
temporary files, are validated and atomically moved into the cache. The helper
supports 1976–2010. The 2001Q1 overlap archive is also cached for validation.
`load_online_call_reports(columns=[...])` remains available for raw MDRM fields;
`load_historical_call_reports()` performs the verified projection used by cleaning.

Source: https://www.chicagofed.org/banking/financial-institution-reports/commercial-bank-data-complete-1976-2000

## Out-of-sample analysis

`out_of_sample.py` uses `compute_betas()` for 2022Q1–2023Q4 and reuses
`beta_regs.regression_specs()` for Table IX. Recent lag construction retains
2021 observations to construct four lags before restricting the sample. The
historical default sample is unchanged. Cycle deposit-rate betas are computed
in `compute_betas.py` for the mean comparison and persistence regression.
The historical mean deposit-rate beta is one minus mean spread beta.
The recent commercial-bank filter requires classification at both endpoints.
Regressions use reg_runner/Stata. `out_of_sample.tex` is generated by reg-runner and imports its generated
mean, persistence, and Table IX tables from `tables/`, with no narrative text.

Cached recent inputs: `data/fdic_out_of_sample.csv` (FDIC financials endpoint,
20211231 and 20231231, CERT/REPDTE/BKCLASS/EDEPDOM) and
`data/fred/out_of_sample_targets.csv` (FRED DFEDTARU/DFEDTARL, 2021–2023).
