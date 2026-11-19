# Table 9, Panel A: deposit-flow beta on deposit-spread beta

With the New York Fed data, the coefficient is **−3.854** (HC1 SE 0.244,
11,051 banks). It moved **−0.0714** from the original estimate.

The original replication coefficient was **−3.782**, versus **−3.610** in Drechsler,
Savov, and Schnabl (QJE, 2017). The absolute relative difference is **4.77%**,
meeting the requested 20% tolerance on the first completed specification.
No alternative specifications were searched to move the coefficient toward the target.

| Statistic | Paper | Replication |
|---|---:|---:|
| Spread-beta coefficient | −3.610 | −3.782 |
| Heteroskedasticity-robust SE | 0.203 | 0.245 |
| Banks | 11,091 | 11,511 |
| R² | 0.132 | 0.092 |

This is a close coefficient replication, **not an exact replication of the sample or R²**.
The raw spread-beta mean is 0.573; its 10th and 90th percentiles are 0.362 and
0.851. The paper reports approximately 0.54, 0.31, and 0.89, respectively.

## Run

From this directory:

```bash
bash download_data.sh
/home/sidlh/Documents/python_envs/data_env/bin/python prepare_data.py
/home/sidlh/Documents/python_envs/data_env/bin/python replicate_table9.py
```

The preparation script downloads and caches the New York Fed Call Report
archives through the shared loader. It joins balance-sheet and income-statement
records on RSSD ID and quarter, retains 1994Q1–2013Q4, and converts year-to-date
deposit interest expense to quarterly flows before writing `analysis_panel.pkl`.

The method and checks below describe the original replication. The estimation
script uses the same procedure with the newly prepared input panel.

## Method

The paper's equation (21), pp. 1864–1865, is implemented with an intercept,
the current quarterly change in the Fed funds target, and four quarterly lags.
The posted `depositspread_020117.do` clarifies the following preprocessing:

1. Identify banks by FDIC certificate. Use consecutive quarterly observations;
   do not difference across reporting gaps or across banks.
2. Use total deposits and total deposit interest expense. Historical deposits
   are `RCFD2200`, falling back to `RCON2200`; supplied deposits are `domdep`
   plus `foreigndep`. Historical expense is `RIAD4170`; supplied expense is
   `intexpalldep`.
3. Convert historical year-to-date expense into quarterly expense by subtracting
   the previous quarter within each calendar year. The supplied expense variable
   is already quarterly and is not differenced again. Annualized deposit rate
   in percentage points is `400 × quarterly expense / lagged total deposits`.
4. Winsorize deposit-rate levels and `100 × Δ log(balance)` for each balance-sheet
   outcome at 1% in each tail. Use the original Stata `winsor` convention:
   replace `floor(0.01 × N)` values at each end with the adjacent order statistic.
5. Deposit spread is the calendar-day quarterly average Fed funds target minus
   the deposit rate. Its quarterly change is the spread first-stage outcome.
   Deposit growth is the deposit-flow first-stage outcome. Policy regressors
   use changes in the **quarter-end** target. After December 15, 2008, use the
   midpoint of the target range.
6. Estimate each bank's OLS separately for each outcome and sum the five policy
   coefficients. Require at least seven usable observations and full regressor
   rank; exclude effectively zero policy slopes (absolute value below 1e−12).
   This implements the authors' exclusion of missing or zero/omitted slopes.
   There is no imposed 20-quarter minimum (it is commented out in their code).
7. Winsorize each beta distribution at 1% **before** intersecting the samples.
   Retain banks with valid betas for spread, deposits, assets, securities,
   loans, real-estate loans, and C&I loans, as in the authors' Table 9 code.
8. Regress deposit-flow beta on spread beta with an intercept, equal bank
   weights, and HC1 robust standard errors (Stata `regress, robust`).

The 10% beta winsorization discussed alongside Figure VI is for the figures;
the posted code uses **1%** for Table 9. No merger-growth exclusion from the
separate Table 8 code is imposed on Table 9.

Historical balance-sheet codes used only to reproduce the common sample:
assets `2170`; loans `2122`; real-estate loans `1410`; C&I loans `1766`;
securities `1754 + 1773`. Use the `RCFD` prefix, falling back to `RCON`.
The supplied columns are `assets`, `loans`, `reloans`, `ciloans`, and `securities`.

## Checks and remaining differences

- An independently downloaded 2001 Q1 raw file matched supplied expense for
  99.99% of 8,762 matched banks and domestic deposits for 99.76%.
- For the first five banks in an independent 2001 Q2 raw extract, raw YTD
  expense exactly equaled supplied Q1 plus Q2 expense; see
  `expense_conversion_check.csv`.
- The final panel has unique certificate-quarter keys. Missing dates are
  respected in quarterly differences and policy lags. Logs require positive
  balances; infinite rates are excluded before winsorization.
- Historical records are restricted to commercial-bank charter code 200.
  The supplied charter field is missing for all observations in 2001–2013,
  so records with unknown charter are retained. This leaves an imperfectly
  harmonized commercial-bank population and may retain some savings banks
  in the later period. The difference is not resolved by selecting a sample
  to match the target coefficient.
- The authors' code winsorizes rates and growth before restricting to
  1994–2013, using a longer prepared input history. Here preprocessing uses
  the reconstructed 1994–2013 panel. No 1993 bank observations are supplied,
  so initial bank differences are unavailable. This is one reason not to
  expect identical beta distributions or sample size.
- Contemporary raw-file revisions, the supplied dataset's prior processing,
  and the bank-classification differences above can also affect the result.
  The authors' raw-to-prepared-data construction is not included in their
  posted estimation program, so the deposit and auxiliary balance-sheet
  mappings are reconstructed from Call Report definitions.

## Outputs and sources

- `results.json`: full-precision estimates and comparison with the paper.
- `bank_betas.csv`: raw and winsorized betas and first-stage observation counts
  for every bank in the final common sample.
- `preparation.log`, `estimation.log`: run output and peak-memory measurements.
- `dss.txt`: text extracted from the supplied paper.
- `authors_depositspread.do`, `authors_banks.do`, `authors_readme.pdf`,
  `authors_data_dictionary.xml`: downloaded documentation, retained for audit.

Public sources:

- [Chicago Fed historical raw Call Reports, 1976–2000](https://www.chicagofed.org/banking/financial-institution-reports/commercial-bank-data-complete-1976-2000):
  all 28 quarters from 1994 Q1 to 2000 Q4 were downloaded and used.
- [Authors' replication package](https://doi.org/10.7910/DVN/KHNXYJ), especially
  [depositspread_020117.do](https://dataverse.harvard.edu/api/access/datafile/3021715).
- [Fed funds target](https://fred.stlouisfed.org/series/DFEDTAR),
  [upper limit](https://fred.stlouisfed.org/series/DFEDTARU), and
  [lower limit](https://fred.stlouisfed.org/series/DFEDTARL).
- [Stata winsor implementation](http://fmwww.bc.edu/repec/bocode/w/winsor.ado).
- [Federal Reserve charter definitions](https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/hmda/hmda-definitions-1990-2017.pdf).

Downloaded November 9, 2026. The FRED download contains dates outside the
estimation window; only dates matched to the 1994–2013 panel affect estimates.
