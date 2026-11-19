# Call Report dictionary sources

Downloaded on 2026-11-07 from the Board of Governors of the Federal Reserve System:

- Documentation: https://www.federalreserve.gov/apps/mdrm/download_mdrm.htm
- Download: https://www.federalreserve.gov/apps/mdrm/pdf/MDRM.zip
- Online lookup: https://www.federalreserve.gov/apps/mdrm/
- Period-specific forms and instructions: https://www.ffiec.gov/resources/reporting-forms

`MDRM_CSV.csv` and `README File for MDRM.txt` were extracted without modification from the original `MDRM.zip` download. The dictionary covers multiple regulatory collections, including Call Reports, and historical records. A download date is not the effective date of every definition.

SHA-256:

```text
cdfc5aba6a73c0d39ec32978ccea031fbe6da8d6b8ea6745fe2a238d3035787f  MDRM.zip
2c464d415c6b043366f9d94f61579df6ab3e1396349f2ba33f2124945b23eeef  MDRM_CSV.csv
```

`call_reports_variable_labels.csv` is local metadata, not a downloaded regulatory crosswalk. It was extracted from the variable-name and variable-label sections of the Stata release 118 header in `call_reports.zip/call_reports.dta`, without reading observations or modifying the archive. It has 414 rows. Many friendly or derived variable names have blank embedded labels. The source file's construction code and a verified alias-to-MDRM mapping are not present in this project; definitions for those aliases remain unresolved until supported by documentation.

The dataset skill and lookup helper are in `~/.codex/skills/datasets/call-report-variables/`.
