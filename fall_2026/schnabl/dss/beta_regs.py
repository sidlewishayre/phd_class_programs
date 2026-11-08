"""Table IX Panels A/B and Figures VI/VII using the available bank sample.

Run this file without arguments. Regressions use the paper replication procedure and 5% winsorized bank betas,
an intercept, and heteroskedasticity-robust standard errors. The four plots use
100 equally populated spread-beta bins and bank-level OLS fitted lines.
Beta data are passed to run_tables through a temporary file.
Tables, Stata logs, and figures are written under results/beta_regs.
"""

from __future__ import annotations

import os
import sys

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)
sys.path.append(os.path.abspath(os.path.join(CWD, "../../../../reusable_code")))

import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reg_runner import run_tables

from compute_betas import compute_betas

ROOT = Path(__file__).resolve().parent
FLOW_LABELS = {
    "deposit_flow_beta": r"Deposit-$\beta$",
    "asset_flow_beta": r"Assets-$\beta$",
    "securities_flow_beta": r"Securities-$\beta$",
    "loan_flow_beta": r"Loans-$\beta$",
    "re_loan_flow_beta": r"RE loans-$\beta$",
    "ci_loan_flow_beta": r"C\&I loans-$\beta$",
}


SAMPLES = {
    "table_ix_panel_a": ("Panel A: All banks", None),
    "table_ix_panel_b": ("Panel B: Large banks", "large_bank == 1"),
}


def classify_large_banks(betas: pd.DataFrame) -> pd.DataFrame:
    """Figure VII: banks at/above the 95th percentile of bank size by assets.

    The authors' code ranks the common sample by mean CPI-adjusted assets
    (2010 dollars). Nominal mean assets are only a legacy fallback.
    Keep the full-sample winsorized betas when selecting the large-bank subset.
    """
    betas = betas.copy()
    size_column = "mean_real_assets" if "mean_real_assets" in betas else "mean_assets"
    eligible = betas.spread_beta.notna() & betas[size_column].gt(0)
    cutoff = betas.loc[eligible, size_column].quantile(0.95)
    if pd.isna(cutoff):
        raise ValueError("No bank sizes available to identify large banks")
    betas["large_bank"] = (eligible & betas[size_column].ge(cutoff)).astype(int)
    betas.attrs["large_bank_asset_cutoff"] = float(cutoff)
    return betas


def regression_specs(data_path: Path):
    """Six robust OLS specifications for each of the two Table IX samples."""
    regs = [
        {
            "id": outcome if table_id == "table_ix_panel_a" else outcome + "_large",
            "reg_code": dict(
                reg_fn="regress",
                dep_var=outcome,
                indep_vars=["spread_beta"],
                options={"vce": "robust"},
                **({"subset": subset} if subset else {}),
            ),
            "data": str(data_path),
            "table": table_id,
            "save_stats": ["N", "r2"],
        }
        for table_id, (_, subset) in SAMPLES.items()
        for outcome in FLOW_LABELS
    ]
    tables = [
        {
            "id": table_id,
            "title": title,
            # Keep index_vars exact; automatic cleaning adds the constant back.
            "clean_variables": False,
            "index_vars": ["spreadbeta"],
            "display_se": True,
            "reg_subsets": [None] * len(FLOW_LABELS),
            "var_map": {
                **FLOW_LABELS,
                "spread_beta": r"Spread-$\beta$",
                "spreadbeta": r"Spread-$\beta$",
                "cons": "Constant",
            },
            "save_stats": ["N", "r2"],
            "save_stats_name_dict": {"r2": r"$R^2$"},
        }
        for table_id, (title, _) in SAMPLES.items()
    ]
    assert len({reg["id"] for reg in regs}) == len(regs)
    assert len({table["id"] for table in tables}) == len(tables)
    assert all(reg["table"] in {table["id"] for table in tables} for reg in regs)
    return regs, tables


def make_binscatters(betas: pd.DataFrame, output: Path, name: str = "figure_vi"):
    """Common 100 rank bins; outcome means use available banks within each bin."""
    titles = [
        "Panel A: Deposits",
        "Panel B: Total assets",
        "Panel C: Securities",
        "Panel D: Loans",
    ]
    d = betas.loc[betas.spread_beta.notna()].copy()
    bank_key = "cert" if "cert" in d else "rssdid"
    d = d.sort_values(["spread_beta", bank_key])
    if len(d) < 100:
        raise ValueError(
            "At least 100 identified spread betas are required for 100 bins"
        )
    # Stable bank-ID ordering breaks ties introduced by winsorization.
    d["bin"] = np.arange(len(d)) * 100 // len(d)
    outcomes = list(FLOW_LABELS)[:4]
    bins = d.groupby("bin")[["spread_beta", *outcomes]].mean()
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for ax, outcome, title in zip(axes.flat, outcomes, titles):
        sample = d[["spread_beta", outcome]].dropna()
        x = np.column_stack([np.ones(len(sample)), sample.spread_beta])
        coefficients = np.linalg.lstsq(x, sample[outcome], rcond=None)[0]
        grid = np.linspace(sample.spread_beta.min(), sample.spread_beta.max(), 100)
        ax.scatter(bins.spread_beta, bins[outcome], s=16, color="#176087", alpha=0.8)
        ax.plot(
            grid,
            coefficients[0] + coefficients[1] * grid,
            color="#b54332",
            linewidth=1.4,
        )
        ax.set(title=title, xlabel=r"Spread $\beta$", ylabel=FLOW_LABELS[outcome])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return bins


def write_beta_comparison(betas: pd.DataFrame):
    """Update the writeup's comparison table from the current bank estimates."""
    summary = betas.attrs["summary"]
    first_year = pd.Period(summary["first_quarter"], freq="Q").year
    last_year = pd.Period(summary["last_quarter"], freq="Q").year
    table = "\n".join(
        [
            r"\begin{tabular}{lc}",
            r"\toprule",
            r" & Mean deposit spread $\beta$ \\",
            r"\midrule",
            f"Paper (1994--2013) & {summary['paper_mean_spread_beta']:.3f}" + r" \\",
            f"Our sample ({first_year}--{last_year}) & {betas.spread_beta_raw.mean():.3f}"
            + r" \\",
            f"Our sample, {100 * summary.get('beta_winsor_tail', .05):g}\\% winsorized & {betas.spread_beta.mean():.3f}"
            + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )
    path = ROOT / "results" / "beta_comparison.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table)
    return path



def main(panel: pd.DataFrame | None = None, *, beta_tail=0.05):
    name = "beta_regs" if beta_tail == 0.05 else f"beta_regs_{100*beta_tail:g}pct"
    output = ROOT / "results" / name
    output.mkdir(parents=True, exist_ok=True)
    betas = compute_betas(
        panel=panel, include_loan_categories=True, beta_tail=beta_tail
    )
    betas = classify_large_banks(betas)
    if beta_tail == 0.05:
        write_beta_comparison(betas)
    bins = make_binscatters(betas, output)
    large_bins = make_binscatters(
        betas.loc[betas.large_bank.eq(1)], output, "figure_vii"
    )
    with tempfile.TemporaryDirectory(prefix="dss_betas_") as directory:
        data_path = Path(directory) / "bank_betas.dta"
        betas[["cert", "spread_beta", "large_bank", *FLOW_LABELS]].to_stata(
            data_path, write_index=False
        )
        regs, tables = regression_specs(data_path)
        results = run_tables(
            regs=regs,
            tables=tables,
            save_folder=str(output),
            overwrite=True,
            reload=True,
            open_pdf=False,
            pdf_kwargs={"raw_latex_labels": True, "separate_tables": True},
        )
    return results, {"all_banks": bins, "large_banks": large_bins}


if __name__ == "__main__":
    main()
