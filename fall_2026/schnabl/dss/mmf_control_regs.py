"""Table IX with additive FFR and MMF coefficients from the same regression.

All flow/spread/MMF betas are winsorized at 5% in each tail, before sample
splits. Baseline outcome betas retain their original winsorization; MMF and
conditional FFR betas are winsorized over their full bank population before merging.
Large-bank membership retains the baseline 95th-percentile size cutoff.
"""

import os
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)
from settings import REUSABLE_CODE_DIR

sys.path.append(str(REUSABLE_CODE_DIR))
from reg_runner import run_tables
from compute_betas import compute_betas, winsorize
from beta_regs import FLOW_LABELS, classify_large_banks
from get_data import load_mmf_betas, load_mmf_control_betas
import settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def mmf_decile_regressions(betas=None):
    """Deposit-flow beta on spread beta separately by MMF-beta decile."""
    betas = load_mmf_control_betas() if betas is None else betas.copy()
    betas["mmf_decile"] = pd.qcut(betas.mmf_beta, 10, labels=range(1, 11)).astype(int)
    output = settings.MMF_DECILE_DIR
    output.mkdir(parents=True, exist_ok=True)
    path = output / "regression_inputs.dta"
    betas.to_stata(path, write_index=False)
    regs = [dict(id=f"mmf_decile_{q}", table="mmf_deciles", data=str(path),
        subset=f"D{q}", reg_code=dict(reg_fn="regress", dep_var="deposit_flow_beta",
            indep_vars=["spread_beta"], subset=f"mmf_decile == {q}",
            options={"vce": "robust"}), save_stats=["N", "r2"])
        for q in range(1, 11)]
    tables = [dict(id="mmf_deciles", title="MMF-beta deciles",
        clean_variables=False, index_vars=["spreadbeta"], display_se=True,
        var_map={"spreadbeta": r"Spread-$\beta$", **FLOW_LABELS}, save_stats=["N", "r2"])]
    result = run_tables(regs=regs, tables=tables, save_folder=str(output),
        overwrite=True, reload=True, open_pdf=False,
        pdf_kwargs={"raw_latex_labels": True, "separate_tables": True})
    records = []
    for q, reg in enumerate(result.tables[0].regressions, 1):
        group = betas.loc[betas.mmf_decile.eq(q)]
        coef = reg.results["spreadbeta"]
        records.append(dict(decile=q, coefficient=coef.loc["b"], se=coef.loc["se"],
            ci_lower=coef.loc["ll"], ci_upper=coef.loc["ul"], n_banks=len(group),
            mmf_beta_min=group.mmf_beta.min(), mmf_beta_max=group.mmf_beta.max()))
    estimates = pd.DataFrame(records)
    estimates.to_csv(output / "coefficients.csv", index=False)
    table_path = output / "tables" / "mmf_deciles.tex"
    table_path.write_text(table_path.read_text().replace(
        "Robust t-statistics in parentheses", "Robust standard errors in parentheses"))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    "mmf_deciles.tex"], cwd=output, check=True, stdout=subprocess.DEVNULL)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ax.errorbar(estimates.decile, estimates.coefficient,
        yerr=np.vstack([estimates.coefficient - estimates.ci_lower,
                        estimates.ci_upper - estimates.coefficient]),
        fmt="o-", capsize=4, color="#176087", label="OLS slope and robust 95% CI")
    ax.axhline(0, color="gray", linewidth=.8)
    ax.set(xlabel="MMF-beta decile (1 = lowest, 10 = highest)",
           ylabel="Deposit beta on spread beta: coefficient", xticks=range(1, 11))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    for extension in ["pdf", "png"]:
        fig.savefig(output / f"decile_coefficients.{extension}", dpi=180)
    plt.close(fig)
    print(estimates.to_string(index=False), flush=True)
    return estimates


def spread_on_mmf_regressions():
    """Cross-sectional OLS of spread beta on MMF beta with robust SEs."""
    panels = [("all_banks", None), ("large_banks", "large_bank == 1")]
    regs = [dict(
        id=f"spread_on_mmf_{sample}", table="spread_on_mmf",
        data=str(settings.MMF_CONTROL_REGRESSION_FILE),
        reg_code=dict(reg_fn="regress", dep_var="spread_beta",
            indep_vars=["mmf_beta"], options={"vce": "robust"},
            **({"subset": subset} if subset else {})),
        subset=sample.replace("_", " ").title(), save_stats=["N", "r2"])
        for sample, subset in panels]
    tables = [dict(id="spread_on_mmf", title="Spread beta on MMF beta",
        clean_variables=True, display_se=True,
        var_map={"mmfbeta": r"MMF-$\beta$", "spreadbeta": r"Spread-$\beta$", "spread_beta": r"Spread-$\beta$"},
        save_stats=["N", "r2"])]
    result = run_tables(regs=regs, tables=tables,
        save_folder=str(settings.MMF_SPREAD_REG_DIR), overwrite=True,
        reload=True, open_pdf=False,
        pdf_kwargs={"raw_latex_labels": True, "separate_tables": True})
    table_path = settings.MMF_SPREAD_REG_DIR / "tables" / "spread_on_mmf.tex"
    table_path.write_text(table_path.read_text().replace(
        "Robust t-statistics in parentheses", "Robust standard errors in parentheses"))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    "spread_on_mmf.tex"], cwd=settings.MMF_SPREAD_REG_DIR,
                   check=True, stdout=subprocess.DEVNULL)
    records = []
    for (sample, _), reg in zip(panels, result.tables[0].regressions):
        coef = reg.results["mmfbeta"]
        records.append(dict(sample=sample, coefficient=coef.loc["b"],
            se=coef.loc["se"], ci_lower=coef.loc["ll"], ci_upper=coef.loc["ul"]))
    estimates = pd.DataFrame(records)
    estimates.to_csv(settings.MMF_SPREAD_REG_DIR / "coefficients.csv", index=False)
    plot_spread_on_mmf(load_mmf_control_betas(), result.tables[0].regressions)
    print(estimates.to_string(index=False), flush=True)
    return estimates


def plot_spread_on_mmf(betas, regressions):
    """Plot bank betas and fitted lines from the Stata regressions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True,
                             sharey=True, layout="constrained")
    samples = [("All banks", betas),
               ("Largest 5% of banks", betas.loc[betas.large_bank.eq(1)])]
    for ax, (title, sample), reg in zip(axes, samples, regressions):
        sample = sample.dropna(subset=["mmf_beta", "spread_beta"])
        slope = reg.results["mmfbeta"].loc["b"]
        se = reg.results["mmfbeta"].loc["se"]
        intercept = reg.results["cons"].loc["b"]
        x = np.linspace(sample.mmf_beta.min(), sample.mmf_beta.max(), 100)
        ax.scatter(sample.mmf_beta, sample.spread_beta, s=9, alpha=.15,
                   color="#176087", edgecolors="none", rasterized=True)
        ax.plot(x, intercept + slope * x, color="#b44732", linewidth=2,
                label="OLS fit")
        ax.set(title=title, xlabel=r"MMF $\beta$")
        ax.text(.04, .96, f"Slope = {slope:.4f} (robust SE {se:.4f})\n"
                f"N = {len(sample):,}", transform=ax.transAxes, va="top",
                fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right")
    axes[0].set_ylabel(r"Spread $\beta$")
    for extension in ["pdf", "png"]:
        fig.savefig(settings.MMF_SPREAD_REG_DIR / f"spread_on_mmf_plot.{extension}",
                    dpi=180)
    plt.close(fig)


def main(betas=None):
    output = settings.MMF_CONTROL_DIR
    output.mkdir(parents=True, exist_ok=True)
    if betas is None:
        print("Computing baseline bank betas (5% winsorization)...", flush=True)
        baseline = classify_large_banks(compute_betas(beta_tail=.05))
        mmf = load_mmf_betas()[["cert", "mmf_beta_raw", "ffr_beta_raw"]].copy()
        for name in ("mmf_beta", "ffr_beta"):
            mmf[name] = winsorize(mmf[name + "_raw"], .05)
        betas = baseline.merge(mmf[["cert", "mmf_beta", "ffr_beta"]], on="cert", how="inner",
                               validate="one_to_one").dropna(subset=["mmf_beta", "ffr_beta"])
    betas.to_csv(output / "bank_betas.csv", index=False)
    betas.to_stata(settings.MMF_CONTROL_REGRESSION_FILE, write_index=False)
    panels = [
        ("table_ix_mmf_panel_a", "Panel A: All banks", None),
        ("table_ix_mmf_panel_b", "Panel B: Large banks", "large_bank == 1"),
    ]
    regs = [dict(
        id=f"{table_id}_{outcome}", table=table_id,
        data=str(settings.MMF_CONTROL_REGRESSION_FILE),
        reg_code=dict(reg_fn="regress", dep_var=outcome,
            indep_vars=["ffr_beta", "mmf_beta"], options={"vce": "robust"},
            **({"subset": subset} if subset else {})), save_stats=["N", "r2"])
        for table_id, _, subset in panels for outcome in FLOW_LABELS]
    terms = {
        "ffrbeta": r"FFR-$\beta$ (MMF control)",
        "mmfbeta": r"MMF-$\beta$",
    }
    tables = [dict(id=table_id, title=title, clean_variables=False,
        index_vars=list(terms),
        var_map={**FLOW_LABELS, **terms},
        display_se=True, reg_subsets=[None] * len(FLOW_LABELS),
        save_stats=["N", "r2"], save_stats_name_dict={"r2": r"$R^2$"})
        for table_id, title, _ in panels]
    result = run_tables(regs=regs, tables=tables, save_folder=str(output),
        overwrite=True, reload=True, open_pdf=False,
        pdf_kwargs={"raw_latex_labels": True, "separate_tables": True})
    records = []
    for table in result.tables:
        for reg, outcome in zip(table.regressions, FLOW_LABELS):
            for term in terms:
                coefs = reg.results[term]
                records.append(dict(panel=table.title, outcome=outcome, regressor=term,
                    coefficient=coefs.loc["b"], se=coefs.loc["se"],
                    ci_lower=coefs.loc["ll"], ci_upper=coefs.loc["ul"]))
    estimates = pd.DataFrame(records)
    estimates.to_csv(output / "coefficients.csv", index=False)
    for table_id, _, _ in panels:
        path = output / "tables" / f"{table_id}.tex"
        path.write_text(path.read_text().replace(
            "Robust t-statistics in parentheses", "Robust standard errors in parentheses"))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                    "mmf_control_regs.tex"], cwd=output, check=True,
                   stdout=subprocess.DEVNULL)
    (output / "sample_summary.txt").write_text(
        f"Matched MMF banks: {len(betas)}\n"
        f"Matched large banks: {int(betas.large_bank.sum())}\n"
        "All betas winsorized at 5% per tail before sample splits.\n"
        "MMF beta is the bank spread-level slope on the quarterly change in MMF assets / (MMF assets + aggregate commercial-bank deposits), measured in percentage points, controlling for current and four lagged FFR changes.\n"
        f"Sources: FRED {settings.MMF_FRED_SERIES} (MMF assets) and {settings.MMF_DEPOSIT_SERIES} (deposits).\n"
        "FFR beta sums the current and four lagged FFR coefficients in that same regression.\n"
        "Stata: regress outcome ffr_beta mmf_beta, vce(robust).\n"
        "Regressions include an intercept; tables omit the constant row.\n")
    print(estimates.loc[estimates.outcome.eq("deposit_flow_beta")].to_string(index=False), flush=True)
    spread_on_mmf_regressions()
    mmf_decile_regressions(betas)
    return estimates


if __name__ == "__main__":
    main()
