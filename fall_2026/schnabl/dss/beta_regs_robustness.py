"""Sample, bank-characteristic, winsorization, and lag robustness regressions.

Run without arguments. All cross-bank regressions use reg_runner/Stata OLS
with an intercept and HC1 robust standard errors. Betas sum coefficients on
current and four lagged target-rate changes, requiring 12 complete quarters
except in the exercise that varies this minimum in four-quarter steps.
The lag exercise reestimates betas with current through k lagged rate changes,
k=1,...,5; each specification uses its own available bank observations.
Only final bank betas are winsorized. Confidence intervals condition on the
estimated bank betas; they do not incorporate first-stage estimation error.
"""

import os
import sys

import matplotlib
import numpy as np
import pandas as pd

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)
from settings import REUSABLE_CODE_DIR

sys.path.append(str(REUSABLE_CODE_DIR))
from reg_runner import run_tables
from compute_betas import estimate_bank_betas, winsorize
from get_data import get_robustness_panel
import settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Extend this list with (prepared-panel column, display label) pairs.
STRATIFY_VARIABLES = [
    ("leverage", "Leverage"),
    ("uninsured_share", "Uninsured Deposit Share"),
    ("deposit_funding_share", "Deposit Funding Share"),
    ("size", "Assets"),
]
WINSOR_PERCENTAGES = [0, 1, 3, 5, 10, 15]
MIN_BETA_OBSERVATIONS = 12
LAG_COUNTS = [1, 2, 3, 4, 5]


def build_scenarios(panel, stratify_variables=None):
    """Reestimate window betas; hold raw full-sample betas fixed across tails."""
    variables = STRATIFY_VARIABLES if stratify_variables is None else stratify_variables
    scenarios, metadata, coverage = [], [], []

    def add(betas, **info):
        frame = betas[["cert", "deposit_flow_beta", "spread_beta"]].copy()
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 3 or frame.spread_beta.nunique() < 2:
            if info["family"] == "rolling":
                metadata.append(
                    dict(
                        specnum=len(metadata),
                        n_banks=len(frame),
                        status="Not estimable: insufficient banks or rate variation",
                        **info,
                    )
                )
                return
            raise ValueError(f"Insufficient banks or spread variation for {info}")
        frame = frame.rename(
            columns={"deposit_flow_beta": "depositbeta", "spread_beta": "spreadbeta"}
        )
        frame["specnum"] = len(metadata)
        scenarios.append(frame)
        metadata.append(
            dict(specnum=len(metadata), n_banks=len(frame), status="Estimated", **info)
        )

    first_year, last_year = panel.quarter.min().year, panel.quarter.max().year
    for start in range(first_year, last_year - 3):
        end = start + 4
        window = panel.loc[panel.quarter.dt.year.between(start, end)]
        # Current/lagged changes are prepared before slicing, preserving valid
        # pre-window monetary-policy lags without using future outcomes.
        betas = estimate_bank_betas(window, min_obs=MIN_BETA_OBSERVATIONS)
        add(
            betas,
            id=f"window_{start}_{end}",
            family="rolling",
            table="rolling",
            start_year=start,
            end_year=end,
            label=f"{start}–{end}",
        )

    raw = estimate_bank_betas(panel, beta_tail=0, min_obs=MIN_BETA_OBSERVATIONS)
    baseline = raw.copy()
    for name in ["deposit_flow_beta", "spread_beta"]:
        baseline[name] = winsorize(raw[name + "_raw"], 0.05)
    for variable, label in variables:
        means = (
            panel.groupby("cert", sort=True)[variable]
            .mean()
            .replace([np.inf, -np.inf], np.nan)
        )
        group = baseline.merge(
            means.rename("characteristic"), on="cert", validate="one_to_one"
        ).dropna(subset=["characteristic"])
        # Value cutoffs keep ties together; do not split equal values by bank ID.
        try:
            group["quartile"] = pd.qcut(
                group.characteristic, 4, labels=[1, 2, 3, 4]
            ).astype(int)
        except ValueError as error:
            raise ValueError(
                f"Cannot form four distinct value quartiles for {variable}"
            ) from error
        for q in range(1, 5):
            subset = group.loc[group.quartile.eq(q)]
            add(
                subset,
                id=f"quartile_{variable}_{q}",
                family="quartile",
                table=f"quartile_{variable}",
                variable=variable,
                label=label,
                quartile=q,
            )
            observed = panel.loc[panel.cert.isin(subset.cert) & panel[variable].notna()]
            coverage.append(
                dict(
                    variable=variable,
                    label=label,
                    quartile=q,
                    n_banks=len(subset),
                    excluded_missing=len(baseline) - len(group),
                    minimum=subset.characteristic.min(),
                    maximum=subset.characteristic.max(),
                    first_quarter=str(observed.quarter.min()),
                    last_quarter=str(observed.quarter.max()),
                )
            )

    for pct in WINSOR_PERCENTAGES:
        betas = raw.copy()
        for name in ["deposit_flow_beta", "spread_beta"]:
            betas[name] = winsorize(raw[name + "_raw"], pct / 100)
        add(
            betas,
            id=f"winsor_{pct}",
            family="winsor",
            table="winsor",
            winsor_pct=pct,
            label=f"{pct}%",
        )
    minimum_data, minimum_metadata = build_minimum_quarter_scenarios(panel)
    offset = len(metadata)
    minimum_data["specnum"] += offset
    minimum_metadata["specnum"] += offset
    scenarios.append(minimum_data)
    metadata.extend(minimum_metadata.to_dict("records"))
    lag_data, lag_metadata = build_lag_scenarios(panel)
    offset = len(metadata)
    lag_data["specnum"] += offset
    lag_metadata["specnum"] += offset
    scenarios.append(lag_data)
    metadata.extend(lag_metadata.to_dict("records"))
    return (
        pd.concat(scenarios, ignore_index=True),
        pd.DataFrame(metadata),
        pd.DataFrame(coverage),
    )


def build_minimum_quarter_scenarios(panel):
    """Vary complete-quarter requirements for both full-sample bank betas."""
    names = ["deposit_flow_beta", "spread_beta"]
    columns = [f"x{i}" for i in range(5)] + [name + "_y" for name in names]
    usable = np.isfinite(panel[columns].to_numpy(float)).all(axis=1)
    full_quarters = panel.loc[usable, "quarter"].nunique()
    thresholds = sorted(set(range(4, full_quarters + 1, 4)) | {full_quarters})
    raw = estimate_bank_betas(panel, beta_tail=0, min_obs=4)
    scenarios, metadata = [], []
    for specnum, minimum in enumerate(thresholds):
        selected = raw.loc[
            raw.deposit_flow_beta_nobs.ge(minimum) & raw.spread_beta_nobs.ge(minimum)
        ].copy()
        for name in names:
            selected[name] = winsorize(selected[name + "_raw"], 0.05)
        frame = selected[["cert", *names]].rename(
            columns={"deposit_flow_beta": "depositbeta", "spread_beta": "spreadbeta"}
        )
        frame["specnum"] = specnum
        estimable = len(frame) >= 3 and frame.spreadbeta.nunique() >= 2
        if estimable:
            scenarios.append(frame)
        metadata.append(
            dict(
                id=f"minimum_quarters_{minimum}",
                family="minimum_quarters",
                table="minimum_quarters",
                label=f"At least {minimum} quarters",
                specnum=specnum,
                min_quarters=minimum,
                n_banks=len(frame),
                full_usable_quarters=full_quarters,
                calendar_quarters=panel.quarter.nunique(),
                status=(
                    "Estimated"
                    if estimable
                    else "Not estimable: insufficient banks or rate variation"
                ),
            )
        )
    if not scenarios:
        raise ValueError("No minimum-quarter specification is estimable")
    return pd.concat(scenarios, ignore_index=True), pd.DataFrame(metadata)


def build_lag_scenarios(panel):
    """Reestimate both bank betas summing current through k lagged slopes."""
    scenarios, metadata = [], []
    for specnum, k in enumerate(LAG_COUNTS):
        betas = estimate_bank_betas(
            panel, max_lag=k, min_obs=MIN_BETA_OBSERVATIONS, beta_tail=0
        )
        # Use the valid deposit/spread pair sample for both winsorization
        # cutoffs, matching the existing full-sample winsorization exercise.
        for name in ["deposit_flow_beta", "spread_beta"]:
            betas[name] = winsorize(betas[name + "_raw"], 0.05)
        frame = betas[["cert", "deposit_flow_beta", "spread_beta"]].rename(
            columns={"deposit_flow_beta": "depositbeta", "spread_beta": "spreadbeta"}
        )
        if len(frame) < 3 or frame.spreadbeta.nunique() < 2:
            raise ValueError(f"Insufficient banks for {k} lags")
        frame["specnum"] = specnum
        scenarios.append(frame)
        metadata.append(
            dict(
                id=f"lags_{k}",
                family="lags",
                table="lags",
                label=f"{k} lags",
                lag_count=k,
                min_quarters=MIN_BETA_OBSERVATIONS,
                specnum=specnum,
                n_banks=len(frame),
                status="Estimated",
            )
        )
    return pd.concat(scenarios, ignore_index=True), pd.DataFrame(metadata)


def run_regressions(data, metadata, output):
    """Run all specifications through reg_runner and extract Stata's 95% CIs."""
    output.mkdir(parents=True, exist_ok=True)
    data_file = output / settings.ROBUSTNESS_DATA_FILE.name
    data.to_stata(data_file, write_index=False)
    available = (
        metadata.loc[metadata.status.eq("Estimated")]
        if "status" in metadata
        else metadata
    )
    regs = [
        dict(
            id=row.id,
            data=str(data_file),
            table=row.table,
            reg_code=dict(
                reg_fn="regress",
                dep_var="depositbeta",
                indep_vars=["spreadbeta"],
                subset=f"specnum == {row.specnum}",
                options={"vce": "robust"},
            ),
            subset=(f"Q{int(row.quartile)}" if row.family == "quartile" else row.label),
            save_stats=["N", "r2"],
        )
        for row in available.itertuples()
    ]
    tables = [
        dict(
            id=table_id,
            title=table_id.replace("_", " ").title(),
            clean_variables=False,
            index_vars=["spreadbeta"],
            display_se=True,
            var_map={"spreadbeta": "Spread beta", "depositbeta": "Deposit beta"},
            save_stats=["N", "r2"],
        )
        for table_id in available.table.unique()
    ]
    assert metadata.id.is_unique and metadata.specnum.is_unique
    assert len({table["id"] for table in tables}) == len(tables)
    results = run_tables(
        regs=regs,
        tables=tables,
        save_folder=str(output),
        overwrite=True,
        reload=True,
        open_pdf=False,
        pdf_kwargs={"separate_tables": True},
    )
    records = []
    for table in results.tables:
        for reg in table.regressions:
            coefs = reg.results["spreadbeta"]
            records.append(
                dict(
                    id=reg.id,
                    coefficient=coefs.loc["b"],
                    se=coefs.loc["se"],
                    ci_lower=coefs.loc["ll"],
                    ci_upper=coefs.loc["ul"],
                )
            )
    estimates = metadata.merge(
        pd.DataFrame(records), on="id", how="left", validate="one_to_one"
    )
    estimates.to_csv(output / "coefficients.csv", index=False)
    return estimates


def regression_coefficient(
    deposit_betas: list[float], spread_betas: list[float]
) -> float:
    """Return the intercept-including Stata OLS slope for supplied beta pairs."""
    data = pd.DataFrame(dict(depositbeta=deposit_betas, spreadbeta=spread_betas))
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 3 or data.spreadbeta.nunique() < 2:
        raise ValueError("Need at least three complete pairs with varying spread betas")
    data["specnum"] = 0
    metadata = pd.DataFrame(
        [
            dict(
                id="single",
                table="single",
                family="single",
                label="All banks",
                specnum=0,
                n_banks=len(data),
            )
        ]
    )
    return float(
        run_regressions(
            data, metadata, settings.ROBUSTNESS_DIR / "single"
        ).coefficient.iloc[0]
    )


def plot_coefficients(estimates, output):
    for family, x_column, xlabel, filename in [
        ("rolling", "end_year", "Five-year window ending in", "rolling_coefficients"),
        (
            "winsor",
            "winsor_pct",
            "Winsorization in each tail (%)",
            "winsorization_coefficients",
        ),
        (
            "minimum_quarters",
            "min_quarters",
            "Minimum usable quarters per bank beta",
            "minimum_quarters_coefficients",
        ),
        (
            "lags",
            "lag_count",
            "Number of quarterly FFR-change lags",
            "lag_coefficients",
        ),
    ]:
        d = estimates.loc[estimates.family.eq(family)].sort_values(x_column)
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        ax.errorbar(
            d[x_column],
            d.coefficient,
            yerr=np.vstack([d.coefficient - d.ci_lower, d.ci_upper - d.coefficient]),
            fmt="o-",
            capsize=4,
            color="#176087",
            label="OLS slope and robust 95% CI",
        )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set(xlabel=xlabel, ylabel="Deposit beta on spread beta: coefficient")
        ax.set_xticks(d[x_column])
        if family in ["rolling", "minimum_quarters"]:
            ax.tick_params(axis="x", rotation=45)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
        for extension in ["pdf", "png"]:
            fig.savefig(output / f"{filename}.{extension}", dpi=180)
        plt.close(fig)


def write_winsorization_note(estimates, output):
    """Write the zero-winsorization estimate below the writeup figure."""
    zero = estimates.loc[
        estimates.family.eq("winsor") & estimates.winsor_pct.eq(0)
    ].iloc[0]
    note = (
        rf"At 0\% winsorization, the coefficient is ${zero.coefficient:.4f}$ "
        rf"with a 95\% confidence interval of "
        rf"$[{zero.ci_lower:.4f}, {zero.ci_upper:.4f}]$." + "\n"
    )
    (output / "winsorization_note.tex").write_text(note)


def write_quartile_table(estimates, output):
    d = estimates.loc[estimates.family.eq("quartile")].copy()
    labels = dict(STRATIFY_VARIABLES)
    d = d.loc[d.variable.isin(labels)].copy()
    d["label"] = d.variable.map(labels)
    d["estimate"] = d.apply(lambda r: f"{r.coefficient:.3f} ({r.se:.3f})", axis=1)
    table = d.pivot(index="label", columns="quartile", values="estimate")
    table = table.reindex([label for _, label in STRATIFY_VARIABLES])
    table.columns = [f"Q{int(q)}" for q in table.columns]
    table.index.name = None
    table.to_csv(output / "quartile_coefficients.csv")
    table.to_latex(
        output / "quartile_coefficients.tex",
        escape=True,
    )
    return table


def main():
    output = settings.ROBUSTNESS_DIR
    output.mkdir(parents=True, exist_ok=True)
    print("Loading and preparing robustness panel...", flush=True)
    panel = get_robustness_panel()
    print("Estimating bank betas for rolling windows and full sample...", flush=True)
    data, metadata, coverage = build_scenarios(panel)
    coverage.to_csv(output / "quartile_coverage.csv", index=False)
    print(
        f"Running {metadata.status.eq('Estimated').sum()} cross-bank regressions...",
        flush=True,
    )
    estimates = run_regressions(data, metadata, output)
    plot_coefficients(estimates, output)
    write_quartile_table(estimates, output)
    write_winsorization_note(estimates, output)
    estimates.loc[estimates.family.eq("minimum_quarters")].to_csv(
        output / "minimum_quarters_coefficients.csv", index=False
    )
    estimates.loc[estimates.family.eq("lags")].to_csv(
        output / "lag_coefficients.csv", index=False
    )
    return estimates


if __name__ == "__main__":
    main()
