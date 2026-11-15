"""Plot expected payout and certainty equivalent for a positive-part normal gamble.

Parameters: underlying normal SD $10,000, gamma=2, means -$60,000--$60,000.
The payout is max(X, 0). Utility is u(wealth+max(X, 0)),
where u(c)=c**(1-gamma)/(1-gamma),
or log(c) for gamma=1, and baseline wealth is $1 million.
Payoffs are measured in dollars. Expected utility is not a dollar amount.
The certainty-equivalent payout is inverse_u(E[u(wealth+max(X,0))])-wealth.
The right panel shows average slopes over $30,000 windows directed away from zero.
Expected payouts use a closed form; certainty equivalents use deterministic quadrature.
All calculations use integration, not simulation.

Run: python attenuation.py. Edit the constants below to change parameters.
"""

from pathlib import Path
import os
import sys
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "attenuation-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

CWD = Path(__file__).resolve().parent
sys.path.append(str(CWD))

from settings import ATTENUATION_PLOT, ATTENUATION_NUMBERS

SD = 50_000.0
GAMMA = 2.0
BASELINE_WEALTH = 100_000.0
MEAN_MIN = -60_000.0
MEAN_MAX = 60_000.0
N_POINTS = 481
WINDOW = 30_000.0


def gamble_moments(means, sd, gamma, wealth):
    """Return E[max(X,0)] and E[u(wealth+max(X,0))]."""
    means = np.asarray(means, dtype=float)
    if sd <= 0 or not np.isfinite(sd):
        raise ValueError("Standard deviation must be finite and positive.")
    if gamma < 0 or not np.isfinite(gamma):
        raise ValueError("Risk aversion must be finite and nonnegative.")
    if not np.all(np.isfinite(means)):
        raise ValueError("Means must be finite.")
    if wealth <= 0 or not np.isfinite(wealth):
        raise ValueError("Baseline wealth must be finite and positive.")
    a = means / sd
    expected_payoff = means * norm.cdf(a) + sd * norm.pdf(a)
    expected_utility = np.empty_like(means)
    for idx in np.ndindex(means.shape):
        center = a[idx]
        zero_probability = norm.cdf(-center)
        # Integrate wealth-normalized utility to avoid tiny-integral tolerances.
        if gamma == 1:
            integrand = lambda y: np.log1p(sd * y / wealth) * norm.pdf(y - center)
            expected_utility[idx] = (
                np.log(wealth) + quad(integrand, 0, np.inf, epsabs=1e-11)[0]
            )
        else:
            power = 1 - gamma
            integrand = lambda y: (1 + sd * y / wealth) ** power * norm.pdf(y - center)
            expected_utility[idx] = (
                wealth**power
                / power
                * (zero_probability + quad(integrand, 0, np.inf, epsabs=1e-11)[0])
            )
    return expected_payoff, expected_utility


def certainty_equivalent_payout(means):
    """Evaluate certainty-equivalent payouts by deterministic integration."""
    means = np.asarray(means, dtype=float)
    # Integrate utility gains directly to preserve precision for rare payouts.
    certainty_equivalent = np.empty_like(means)
    for idx, mean in enumerate(means):
        center = mean / SD
        if GAMMA == 1:
            gain = quad(
                lambda y: np.log1p(SD * y / BASELINE_WEALTH) * norm.pdf(y - center),
                0,
                np.inf,
                epsabs=1e-25,
                epsrel=1e-10,
            )[0]
            certainty_equivalent[idx] = BASELINE_WEALTH * np.expm1(gain)
        else:
            power = 1 - GAMMA
            gain = quad(
                lambda y: np.expm1(power * np.log1p(SD * y / BASELINE_WEALTH))
                * norm.pdf(y - center),
                0,
                np.inf,
                epsabs=1e-25,
                epsrel=1e-10,
            )[0]
            certainty_equivalent[idx] = BASELINE_WEALTH * np.expm1(
                np.log1p(gain) / power
            )
    return certainty_equivalent


def main():
    means = np.linspace(MEAN_MIN, MEAN_MAX, N_POINTS)
    payoff, _ = gamble_moments(means, SD, GAMMA, BASELINE_WEALTH)
    certainty_equivalent = certainty_equivalent_payout(means)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(means / 1000, payoff / 1000, label="Expected payout")
    axes[0].plot(
        means / 1000,
        np.maximum(means, 0) / 1000,
        linestyle="-",
        color="black",
        label="No uncertainty",
    )
    axes[0].plot(
        means / 1000,
        certainty_equivalent / 1000,
        color="tab:orange",
        label="Certainty-equivalent payout",
    )
    axes[0].set(title="Level", ylabel="Payout")
    # Average derivative equals the secant slope exactly, without estimating
    # pointwise derivatives or averaging a discretized derivative curve.
    for side, mask in enumerate((means <= 0, means >= 0)):
        centers = means[mask]
        lower = centers - WINDOW if side == 0 else centers
        upper = centers if side == 0 else centers + WINDOW
        lower_payoff, _ = gamble_moments(lower, SD, GAMMA, BASELINE_WEALTH)
        upper_payoff, _ = gamble_moments(upper, SD, GAMMA, BASELINE_WEALTH)
        payoff_slope = (upper_payoff - lower_payoff) / WINDOW
        ce_slope = (
            certainty_equivalent_payout(upper) - certainty_equivalent_payout(lower)
        ) / WINDOW
        no_uncertainty_slope = (np.maximum(upper, 0) - np.maximum(lower, 0)) / WINDOW
        label = side == 0
        axes[1].plot(
            centers / 1000,
            payoff_slope,
            color="tab:blue",
            label="Expected payout" if label else None,
        )
        axes[1].plot(
            centers / 1000,
            ce_slope,
            color="tab:orange",
            label="Certainty-equivalent payout" if label else None,
        )
        axes[1].plot(
            centers / 1000,
            no_uncertainty_slope,
            "-",
            color="black",
            label="No uncertainty" if label else None,
        )
    # Evaluate exactly at zero, independently of the plotting grid.
    endpoints = np.array([-WINDOW, 0.0, WINDOW])
    endpoint_payoff, _ = gamble_moments(endpoints, SD, GAMMA, BASELINE_WEALTH)
    endpoint_ce = certainty_equivalent_payout(endpoints)
    differences = []
    for values in (
        endpoint_payoff,
        endpoint_ce,
        np.maximum(endpoints, 0),
    ):
        left, right = np.diff(values) / WINDOW
        differences.append(float(right - left))
    axes[1].set(
        title="Derivative",
        ylabel="Payout Slope",
    )
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        legend_order = [labels.index(label) for label in (
            "No uncertainty", "Expected payout", "Certainty-equivalent payout"
        )]
        ax.legend([handles[i] for i in legend_order],
                  [labels[i] for i in legend_order], loc="upper left")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Equity Distance ($000s)")
        ax.grid(alpha=0.25)
    fig.suptitle("Expected payout of bankruptcy")
    ATTENUATION_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ATTENUATION_PLOT, dpi=180)
    plt.close(fig)
    reciprocals = [1.0 / difference if difference != 0 else np.inf
                   for difference in differences[:2]]
    ATTENUATION_NUMBERS.parent.mkdir(parents=True, exist_ok=True)
    ATTENUATION_NUMBERS.write_text(
        "% Generated by attenuation.py; do not edit manually.\n"
        f"\\newcommand{{\\UncertaintyMultiplier}}{{{reciprocals[0]:.1f}}}\n"
        f"\\newcommand{{\\RiskAversionMultiplier}}{{{reciprocals[1]:.1f}}}\n"
        f"\\newcommand{{\\MeasurementErrorSD}}{{{SD:,.0f}}}\n"
        f"\\newcommand{{\\BaselineWealth}}{{{BASELINE_WEALTH:,.0f}}}\n"
        f"\\newcommand{{\\RiskAversionCoefficient}}{{{GAMMA:g}}}\n"
    )
    for label, difference in zip(
        ("Expected payout", "Certainty-equivalent payout"),
        differences,
    ):
        reciprocal = 1.0 / difference if difference != 0 else np.inf
        print(f"{label}: {reciprocal:.6f}")


if __name__ == "__main__":
    main()
