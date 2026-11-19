"""Bank spread sensitivities to MMF share changes, conditional on FFR changes.

For each bank: spread_it = intercept_i + beta_i * change_MMF_share_t
+ sum_(k=0,...,4) gamma_ik * change_FFR_(t-k) + error_it.
Report the MMF coefficient and the sum of current/lagged FFR coefficients
from this same regression, retaining raw and winsorized versions of both.
Spread is in annualized percentage points; MMF share is assets / (assets + aggregate commercial-bank deposits),
and its quarterly change is measured in percentage points. Use 1994Q1--2013Q4 and at least 12 complete
quarters. Final bank coefficients are winsorized at 5% per tail; retain raw betas.
"""

import os
import sys

import matplotlib

CWD = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CWD)
from settings import REUSABLE_CODE_DIR

sys.path.append(str(REUSABLE_CODE_DIR))
from compute_betas import estimate_bank_betas, winsorize
from get_data import get_mmf_panel
import settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MIN_QUARTERS = 12
FFR_LAGS = 4


def estimate_mmf_betas(panel):
    """Extract both coefficient groups from the same OLS specification."""
    betas = estimate_bank_betas(
        panel,
        outcomes=("mmf_beta",),
        regressor_columns=["mmfflow"],
        control_columns=[f"x{i}" for i in range(FFR_LAGS + 1)],
        min_obs=MIN_QUARTERS,
        beta_tail=0,
    )
    ffr_panel = panel.assign(ffr_beta_y=panel.mmf_beta_y)
    ffr = estimate_bank_betas(
        ffr_panel,
        outcomes=("ffr_beta",),
        regressor_columns=[f"x{i}" for i in range(FFR_LAGS + 1)],
        control_columns=["mmfflow"],
        min_obs=MIN_QUARTERS,
        beta_tail=0,
    )
    betas = betas.merge(ffr, on="cert", validate="one_to_one")
    if not betas.mmf_beta_nobs.eq(betas.ffr_beta_nobs).all():
        raise ValueError("MMF and FFR coefficients must use the same observations")
    for name in ("mmf_beta", "ffr_beta"):
        betas[name] = winsorize(betas[name + "_raw"], .05)
    if betas.empty:
        raise ValueError("No banks meet the minimum-quarter and full-rank requirements")
    settings.MMF_DIR.mkdir(parents=True, exist_ok=True)
    betas["mmf_series"] = settings.MMF_FRED_SERIES
    betas.to_csv(settings.MMF_BETAS_FILE, index=False)
    return betas


def plot_histogram(betas):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, name, title in zip(
        axes, ("mmf_beta", "ffr_beta"), ("MMF coefficient", "FFR coefficient sum")
    ):
        ax.hist(betas[name].dropna(), bins=60, color="#176087",
                edgecolor="white", linewidth=0.4)
        ax.set(title=title, xlabel=r"MMF-$\beta$" if name == "mmf_beta"
               else r"FFR-$\beta$", ylabel="Number of banks")
        ax.spines[["top", "right"]].set_visible(False)
    for extension in ["pdf", "png"]:
        fig.savefig(settings.MMF_DIR / f"beta_histogram.{extension}", dpi=180)
    plt.close(fig)


def main():
    print("Loading bank panel and FRED total MMF assets...", flush=True)
    panel = get_mmf_panel(ffr_lags=FFR_LAGS)
    print("Estimating each bank's spread sensitivity to MMF share changes...", flush=True)
    betas = estimate_mmf_betas(panel)
    plot_histogram(betas)
    print(
        f"Saved histogram and {len(betas):,} bank betas to {settings.MMF_DIR}",
        flush=True,
    )
    print(betas.mmf_beta.describe().to_string(), flush=True)
    return betas


if __name__ == "__main__":
    main()
