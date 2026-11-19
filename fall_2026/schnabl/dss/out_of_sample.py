"""Out-of-sample beta comparisons and Table IX, using the existing estimators."""

import os
import re
import sys

import pandas as pd

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)

import settings

sys.path.append(str(settings.REUSABLE_CODE_DIR))
from reg_runner import run_tables
from beta_regs import classify_large_banks, regression_specs
from compute_betas import compute_betas, compute_cycle_betas
from get_data import get_out_of_sample_panel


def main():
    output = settings.OUT_OF_SAMPLE_DIR
    output.mkdir(parents=True, exist_ok=True)
    panel, rates = get_out_of_sample_panel()
    historical = compute_betas(start="1994Q1", end="2013Q4")
    recent = classify_large_banks(
        compute_betas(panel=panel, rates=rates, start="2022Q1", end="2023Q4")
    )
    cycle = compute_cycle_betas(panel, rates)
    matched = cycle.merge(
        historical[["cert", "spread_beta"]], on="cert", validate="one_to_one"
    )
    # Compare spread sensitivities in both periods.
    matched["cycle_spread_beta"] = 1 - matched.cycle_beta
    path = output / "persistence_inputs.dta"
    matched.to_stata(path, write_index=False)
    data_path = output / "table_ix_inputs.dta"
    recent.to_stata(data_path, write_index=False)
    recent.to_csv(output / "regression_betas.csv", index=False)
    regs, tables = regression_specs(data_path)
    for table in tables:
        table.update(clean_variables=False, index_vars=["spreadbeta"])
        table["var_map"].update({"deposit_flow_beta": r"Deposits-$\beta$",
                                 "depositflowbeta": r"Deposits-$\beta$"})
    regs.append(
        dict(
            id="persistence",
            table="persistence",
            data=str(path),
            reg_code=dict(
                reg_fn="regress",
                dep_var="cycle_spread_beta",
                indep_vars=["spread_beta"],
                options={"vce": "robust"},
            ),
            save_stats=["N", "r2"],
        )
    )
    tables.append(
        dict(
            id="persistence",
            clean_variables=True,
            display_se=True,
            var_map={
                "cycle_spread_beta": r"Cycle Spread-$\beta$",
                "cyclespreadbeta": r"Cycle Spread-$\beta$",
                "spread_beta": r"DSS Spread-$\beta$",
                "spreadbeta": r"DSS Spread-$\beta$",
                "cons": "Constant",
            },
            save_stats=["N", "r2"],
        )
    )
    means = pd.concat(
        [
            pd.DataFrame({"rate_beta": 1 - historical.spread_beta, "period": 0}),
            pd.DataFrame({"rate_beta": cycle.cycle_beta, "period": 1}),
        ],
        ignore_index=True,
    )
    mean_path = output / "mean_inputs.dta"
    means.to_stata(mean_path, write_index=False)
    regs += [
        dict(
            id=f"mean_{period}",
            table="mean_betas",
            data=str(mean_path),
            subset=label,
            reg_code=dict(
                reg_fn="regress",
                dep_var="rate_beta",
                indep_vars=[""],
                subset=f"period == {period}",
                options={"vce": "robust"},
            ),
            save_stats=["N"],
        )
        for period, label in [(0, "1994--2013"), (1, "2021Q4--2023Q4")]
    ]
    tables.insert(
        0,
        dict(
            id="mean_betas",
            title="Mean spreads",
            dep_vars=[None, None],
            clean_variables=True,
            display_se=True,
            var_map={"cons": r"Mean Spread-$\beta$", "rate_beta": r"Deposit-rate $\beta$"},
            save_stats=["N"],
        ),
    )
    result = run_tables(
        regs=regs,
        tables=tables,
        save_folder=str(output),
        overwrite=True,
        reload=True,
        open_pdf=False,
        pdf_kwargs={"raw_latex_labels": True, "separate_tables": True},
    )
    # reg-runner defaults to four significant figures for standard errors.
    result.table_list[0] = result.table_list[0].map(
        lambda value: f"({float(value[1:-1]):.3g})"
        if isinstance(value, str) and re.fullmatch(r"\([-+0-9.eE]+\)", value)
        else value
    )
    result.save_folder = str(settings.ROOT)
    result.latex_filepath = str(settings.OUT_OF_SAMPLE_TABLE_FILE)
    result.pdf_filepath = str(settings.ROOT / "out_of_sample.pdf")
    result.make_pdf(pdf_kwargs={"raw_latex_labels": True, "separate_tables": True})


if __name__ == "__main__":
    main()
