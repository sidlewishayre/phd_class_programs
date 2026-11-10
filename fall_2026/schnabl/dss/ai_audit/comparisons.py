"""Compare four cumulative methodologies for deposit beta on spread beta.

Run with the project's data_env Python. Outputs go to
results/ai_audit_comparisons (Stata tables/logs, CSV coefficients, bank betas,
and a methodology manifest). All columns retain a common sample across the
seven first-stage outcomes, but run only one cross-bank regression per method.

Columns 2 and 3 use the audit's 1% winsorization of both quarterly inputs and
final betas. Column 4 additionally uses the audit's prepared source sample,
policy lags, and minimum observations.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

CWD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(CWD)

import settings

sys.path.append(str(settings.REUSABLE_CODE_DIR))
from reg_runner import run_tables

from compute_betas import estimate_bank_betas
from get_data import (
    PAPER_STOCKS,
    fred_rates,
    get_beta_panel,
    get_data,
    load_ai_audit_inputs,
)
from ai_audit.replicate_table9 import winsor

OUTCOMES = ["spread_beta", *PAPER_STOCKS]
METHODS = [
    dict(
        id="baseline",
        label="Replication",
        input_tail=0,
        beta_tail=0.05,
        deposits="Domestic",
        sample="Baseline",
        min_obs=6,
    ),
    dict(
        id="quarterly_winsor",
        label="Winsorization",
        input_tail=0.01,
        beta_tail=0.01,
        deposits="Domestic",
        sample="Baseline",
        min_obs=6,
    ),
    dict(
        id="total_deposits",
        label="Total Deposits",
        input_tail=0.01,
        beta_tail=0.01,
        deposits="Total",
        sample="Baseline",
        min_obs=6,
    ),
    dict(
        id="full_audit",
        label="AI Replication",
        input_tail=0.01,
        beta_tail=0.01,
        deposits="Total",
        sample="Audit",
        min_obs=7,
    ),
]


def change_quarterly_inputs(source, prepared, rates, *, total_deposits=False):
    """Insert audit-style trimming between baseline preparation and beta fitting.

    Cap deposit rates before differencing spreads, and cap each quarterly stock
    growth series. Use the retained 1994--2013 panel for the trimming cutoffs.
    Preserve valid presample levels when calculating the first spread change.
    """
    d = source.copy()
    d["quarter"] = pd.PeriodIndex(d.quarter, freq="Q")
    d = d.sort_values(["cert", "quarter"]).reset_index(drop=True)
    g = d.groupby("cert", sort=False)
    adjacent = d.quarter.astype("int64").sub(g.quarter.shift().astype("int64")).eq(1)
    retained = d.quarter.between(pd.Period("1994Q1"), pd.Period("2013Q4"))
    deposits = d.domdep + d.foreigndep.fillna(0) if total_deposits else d.domdep
    expense = d.intexpalldep if total_deposits else d.intexpdomdep
    quarterly = (expense - expense.groupby(d.cert).shift()).where(adjacent)
    quarterly = quarterly.where(d.quarter.dt.quarter.ne(1), expense)
    expense = expense.where(d.expense_timing.ne("ytd"), quarterly)
    prior = deposits.groupby(d.cert).shift().where(adjacent)
    rate = (400 * expense / prior.where(prior.gt(0))).replace([np.inf, -np.inf], np.nan)
    # Get exact Stata order-statistic cutoffs, then apply to presample rates too.
    trimmed = winsor(rate.loc[retained])
    rate = rate.clip(trimmed.min(), trimmed.max())
    spread = d.quarter.map(rates.fed_funds_target_pct) - rate
    d["spread_beta_y"] = spread.groupby(d.cert).diff().where(adjacent)
    if total_deposits:
        logged = np.log(deposits.where(deposits.gt(0)))
        d["deposit_flow_beta_y"] = 100 * logged.groupby(d.cert).diff().where(adjacent)
    updates = d.loc[retained, ["cert", "quarter", "spread_beta_y"]].copy()
    replaced = ["spread_beta_y"]
    if total_deposits:
        updates["deposit_flow_beta_y"] = d.loc[retained, "deposit_flow_beta_y"]
        replaced.append("deposit_flow_beta_y")
    result = prepared.drop(columns=replaced).merge(
        updates, on=["cert", "quarter"], validate="one_to_one"
    )
    for name in PAPER_STOCKS:
        result[name + "_y"] = winsor(result[name + "_y"])
    return result


def prepare_audit_inputs(panel, rates):
    """Map the audit preparation into the baseline bank-beta estimator's schema."""
    d = panel.sort_values(["cert", "q"]).copy()
    d["quarter"] = d.dateq.dt.to_period("Q")
    d["ff_end"] = d.quarter.map(rates.fed_funds_target_end_pct)
    d["ff_avg"] = d.quarter.map(rates.fed_funds_target_pct)
    d["x0"] = d.quarter.map(rates.fed_funds_target_end_pct.diff())
    g = d.groupby("cert", sort=False)
    adjacent = d.q.sub(g.q.shift()).eq(1)
    rate = 400 * d.intexpalldep / g.totdep.shift().where(adjacent)
    spread = d.ff_avg - winsor(rate.replace([np.inf, -np.inf], np.nan))
    d["spread_beta_y"] = spread.groupby(d.cert).diff().where(adjacent)
    for name, stock in zip(
        PAPER_STOCKS, ["totdep", "assets", "securities", "loans", "reloans", "ciloans"]
    ):
        logged = np.log(d[stock].where(d[stock].gt(0)))
        d[name + "_y"] = winsor(100 * logged.groupby(d.cert).diff().where(adjacent))
    for lag in range(1, 5):
        d[f"x{lag}"] = g.x0.shift(lag).where(d.q.sub(g.q.shift(lag)).eq(lag))
    return d


def build_comparisons(source, rates, audit_panel, audit_rates):
    """Run the baseline stages, inserting changes at the input and beta stages."""
    baseline = get_beta_panel(panel=source, rates=rates)
    inputs = [
        baseline,
        change_quarterly_inputs(source, baseline, rates),
        change_quarterly_inputs(source, baseline, rates, total_deposits=True),
        prepare_audit_inputs(audit_panel, audit_rates),
    ]
    frames = []
    for specnum, (method, panel) in enumerate(zip(METHODS, inputs)):
        print(f"Estimating {method['label']}...", flush=True)
        options = dict(winsorizer=winsor) if method["input_tail"] else {}
        if method["sample"] == "Audit":
            options["zero_tolerance"] = 1e-12
        betas = estimate_bank_betas(
            panel,
            outcomes=OUTCOMES,
            beta_tail=method["beta_tail"],
            min_obs=method["min_obs"],
            **options,
        )
        betas["specnum"] = specnum
        frames.append(betas)
    return pd.concat(frames, ignore_index=True)


def run_regressions(betas):
    """Run the four cross-bank regressions and save the comparison table."""
    output = settings.AI_COMPARISONS_DIR
    output.mkdir(parents=True, exist_ok=True)
    betas.to_csv(output / "bank_betas.csv", index=False)
    pd.DataFrame(METHODS).to_csv(output / "methodologies.csv", index=False)
    data_file = output / "regression_inputs.dta"
    betas[["cert", "specnum", "deposit_flow_beta", "spread_beta"]].to_stata(
        data_file, write_index=False
    )
    regs = [
        dict(
            id=m["id"],
            data=str(data_file),
            table="comparisons",
            subset=m["label"],
            reg_code=dict(
                reg_fn="regress",
                dep_var="deposit_flow_beta",
                indep_vars=["spread_beta"],
                subset=f"specnum == {i}",
                options={"vce": "robust"},
            ),
            save_stats=["N", "r2"],
        )
        for i, m in enumerate(METHODS)
    ]
    tables = [
        dict(
            id="comparisons",
            title="Methodology comparison",
            clean_variables=False,
            index_vars=["spreadbeta"],
            display_se=True,
            latex_kwargs={"add_pvalue_lims": False},
            save_stats=["N", "r2"],
            var_map={
                "spread_beta": r"Spread-$\beta$",
                "spreadbeta": r"Spread-$\beta$",
                "deposit_flow_beta": r"Deposit-$\beta$",
                "cons": "Constant",
            },
        )
    ]
    results = run_tables(
        regs=regs,
        tables=tables,
        save_folder=str(output),
        overwrite=True,
        reload=True,
        open_pdf=False,
        pdf_kwargs={"separate_tables": True, "raw_latex_labels": True},
    )
    records = []
    for reg, method in zip(results.tables[0].regressions, METHODS):
        slope = reg.results["spreadbeta"]
        stats = {stat.id: stat.results for stat in reg.save_stats}
        records.append(
            dict(
                method,
                coefficient=slope.loc["b"],
                se=slope.loc["se"],
                ci_lower=slope.loc["ll"],
                ci_upper=slope.loc["ul"],
                n_banks=stats["N"],
                r_squared=stats["r2"],
            )
        )
    estimates = pd.DataFrame(records)
    estimates.to_csv(output / "coefficients.csv", index=False)
    print(estimates[["label", "coefficient", "se"]].to_string(index=False))
    return results


def main():
    source = get_data(
        columns=["reloans", "ciloans", "intexpdomdep", "loans_unearnedinc"],
        include_presample=True,
    )
    audit_panel, audit_rates = load_ai_audit_inputs()
    betas = build_comparisons(source, fred_rates(), audit_panel, audit_rates)
    return run_regressions(betas)


if __name__ == "__main__":
    main()
