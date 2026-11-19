"""Seven-column bridge from DKN Table 3 to DSS Table I's published expense beta.

Run with the project's data_env Python. Outputs go to results/dss_dkn_compare.
Columns 1--2 retain the requested raw spread slopes; the comparable expense
sensitivity is also stored as a separate statistic, using mean D/A minus slope.
Columns 3--6 estimate total expense / average assets directly. Column 7 is a
published benchmark, not an estimated regression. Remaining differences from
DSS include bank screens, quarter-average asset approximation, and data vintage.
"""

import sys
from pathlib import Path

CWD = Path(__file__).resolve().parents[1]
sys.path.append(str(CWD))

from get_data import load_dss_dkn_reports, load_dkn_rates
from dkn_comparison.dss_dkn_computation import prepare_comparison, compute_comparison_betas
from settings import (
    REUSABLE_CODE_DIR,
    DSS_DKN_COMPARE_DIR,
    DSS_DKN_PANEL_FILE,
    DSS_DKN_AGGREGATE_FILE,
)

sys.path.append(str(REUSABLE_CODE_DIR))
from reg_runner import run_tables

STATA_SETUP = r"""
capture program drop bridge
program define bridge, eclass
    syntax varlist(min=2 max=2), step(integer)
    tempname b V slope expense N
    if `step' <= 3 {
        tsset tq
        if `step' == 1 quietly newey spread fedfunds, lag(8)
        if `step' == 2 quietly newey d_spread d_fedfunds, lag(8)
        if `step' == 3 quietly newey d_expense_rate d_fedfunds, lag(8)
        if `step' == 1 local x fedfunds
        else local x d_fedfunds
        scalar `slope' = _b[`x']
        matrix `V' = (_se[`x']^2)
        scalar `N' = e(N)
        scalar `expense' = `slope'
        if `step' <= 2 {
            quietly summarize deposit_share if e(sample), meanonly
            scalar `expense' = r(mean) - `slope'
        }
    }
    if `step' == 4 {
        tsset tq
        quietly newey d_expense_rate d_fedfunds L(1/3).d_fedfunds, lag(8)
        scalar `N' = e(N)
        quietly lincom d_fedfunds + L.d_fedfunds + L2.d_fedfunds + L3.d_fedfunds
        scalar `slope' = r(estimate)
        scalar `expense' = `slope'
        matrix `V' = (r(se)^2)
    }
    if inlist(`step', 5, 6) {
        // Python's shared estimator supplies the winsorized mean and bank count.
        scalar `slope' = python_beta_`step'[1]
        scalar `expense' = `slope'
        scalar `N' = python_n_`step'[1]
        matrix `V' = (0)
    }
    if `step' == 7 {
        scalar `slope' = .345
        scalar `expense' = .345
        scalar `N' = 8086
        matrix `V' = (0)
    }
    matrix `b' = (`slope')
    matrix colnames `b' = beta
    matrix colnames `V' = beta
    matrix rownames `V' = beta
    // Discard newey's r(table); reg-runner must read the posted beta below.
    quietly summarize expense_rate, meanonly
    return clear
    ereturn post `b' `V'
    ereturn scalar N = `N'
    ereturn scalar expense_beta = `expense'
    ereturn local depvar expense_rate
    ereturn local cmd bridge
end
"""


def specifications():
    labels = [
        "DKN levels",
        "DKN changes",
        "Total expense",
        "FFR lags",
        "Bank-level betas",
        "DSS sample",
        "Published DSS",
    ]
    return [
        {
            "id": f"step_{step}",
            "table": "comparison",
            "data": str(DSS_DKN_AGGREGATE_FILE),
            "stata_code": STATA_SETUP,
            "header": labels[step - 1],
            "reg_code": {
                "reg_fn": "bridge",
                "dep_var": "expense_rate",
                "indep_vars": ["fedfunds"],
                "options": {"step": str(step)},
            },
            "save_stats": ["N", "expense_beta"],
        }
        for step in range(1, 8)
    ]


def table_specifications():
    return [
        {
            "id": "comparison",
            "title": "Sequential comparison of DKN and DSS",
            "clean_variables": True,
            "var_map": {
                "beta": "Reported beta",
                "d_fedfunds": "Reported beta",
                "dfedfunds": "Reported beta",
                "cons": "Constant",
                "expense_rate": "Interest sensitivity",
            },
            "save_stats": ["N", "expense_beta"],
            "save_stats_name_dict": {
                "N": "Observations / banks",
                "expense_beta": "Comparable expense beta",
            },
        }
    ]


def remove_dependent_label():
    """Omit the generic outcome label and constant with its statistic row."""
    for path in DSS_DKN_COMPARE_DIR.rglob("*.tex"):
        lines = path.read_text().splitlines(keepends=True)
        kept = []
        skip_statistic = False
        for line in lines:
            if skip_statistic:
                skip_statistic = False
                continue
            if line.lstrip().startswith("Constant &"):
                skip_statistic = True
                continue
            if "Dependent variable:" not in line and not line.lstrip().startswith(
                "Comparable expense beta &"
            ):
                kept.append(line)
        path.write_text("".join(kept))


def main():
    aggregate, panel = prepare_comparison(load_dss_dkn_reports(), load_dkn_rates())
    DSS_DKN_COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    for step, end in ((5, "2021Q2"), (6, "2017Q4")):
        betas = compute_comparison_betas(panel, end)
        if betas.empty:
            raise ValueError(f"No valid bank betas through {end}")
        betas.to_csv(DSS_DKN_COMPARE_DIR / f"bank_betas_{end}.csv", index=False)
        aggregate[f"python_beta_{step}"] = betas.expense_beta.mean()
        aggregate[f"python_n_{step}"] = len(betas)
    aggregate.to_stata(DSS_DKN_AGGREGATE_FILE, write_index=False, version=118)
    panel.to_stata(DSS_DKN_PANEL_FILE, write_index=False, version=118)
    run_tables(
        specifications(),
        str(DSS_DKN_COMPARE_DIR),
        tables=table_specifications(),
        reload=True,
        overwrite=True,
        pdf_kwargs={"raw_latex_labels": True, "separate_tables": True},
    )
    remove_dependent_label()


if __name__ == "__main__":
    main()
