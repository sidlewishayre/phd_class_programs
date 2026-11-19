"""Recreate the two deposit columns of DKN Table 3 with Stata via reg_runner."""

import sys
from pathlib import Path

CWD = Path(__file__).resolve().parents[1]
sys.path.append(str(CWD))

from settings import DKN_REGRESSION_FILE, DKN_RESULTS_DIR, REUSABLE_CODE_DIR

sys.path.append(str(REUSABLE_CODE_DIR))
from reg_runner import run_tables

PANELS = {
    "levels": "Panel A. Regression in levels",
    "changes": "Panel B. Regression in changes",
}
LABELS = {"fedfunds": r"r^{*}_{t}", "swap1": r"\ell_{1t}", "swap5": r"\ell_{5t}"}

# Stata's newey fits OLS coefficients but does not store their R-squared.
STATA_SETUP = """
tsset tq, quarterly
capture program drop dkn_newey
program define dkn_newey, eclass
    quietly regress `0'
    local r2 = e(r2)
    newey `0', lag(8)
    ereturn scalar r2 = `r2'
end
"""


def get_regression(panel, column):
    prefix = "d_" if panel == "changes" else ""
    regressors = ["fedfunds"] if column == 1 else LABELS
    return {
        "id": f"{panel}_{column}",
        "data": str(DKN_REGRESSION_FILE),
        "table": panel,
        "dep_var": "Deposits",
        "stata_code": STATA_SETUP,
        "reg_code": {
            "reg_fn": "dkn_newey",
            "dep_var": prefix + "spread",
            "indep_vars": [prefix + name for name in regressors],
        },
    }


def main():
    regressions = [
        get_regression(panel, column) for panel in PANELS for column in (1, 2)
    ]
    tables = [
        {
            "id": panel,
            "title": title,
            "clean_variables": True,
            "var_map": {
                **{
                    ("d_" if panel == "changes" else "")
                    + name: "$"
                    + (r"\Delta " if panel == "changes" else "")
                    + label
                    + "$"
                    for name, label in LABELS.items()
                },
                "cons": "Intercept",
            },
        }
        for panel, title in PANELS.items()
    ]
    run_tables(regressions, str(DKN_RESULTS_DIR), tables=tables, reload=True)


if __name__ == "__main__":
    main()
