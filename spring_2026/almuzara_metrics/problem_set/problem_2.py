import os
import sys
import torch
import numpy as np
import pandas as pd

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir))

sys.path.append(CWD)

sys.path.append("/home/sidlh/Documents/reusable_code")
from metrics.gmm import GMM


def get_data():
    df = pd.read_csv(os.path.join(CWD, "ps1_data.csv"))
    df["i"] = range(len(df))
    df = pd.melt(df, id_vars="i")
    df["var"] = df["variable"].str.split("_").str[0]
    df["t"] = df["variable"].str.split("_").str[1]
    df = df.drop(columns=["variable"]).set_index(["i", "t", "var"])["value"]
    df = df.unstack("var")

    y = df["income"].reset_index().pivot(index="i", columns="t", values="income").values
    c = (
        df["consumption"]
        .reset_index()
        .pivot(index="i", columns="t", values="consumption")
        .values
    )
    x = (y, c)
    return df


def get_diff(x, shift=1):
    return (x - np.roll(x, shift=shift, axis=1))[:, shift:]


def get_cov(x, y, shift=1, max_shift=2):
    x_0 = x[:, max_shift:].reshape(-1)
    x_1 = np.roll(y, shift=shift, axis=1)[:, max_shift:].reshape(-1)
    return torch.tensor((x_0 - x_0.mean()) * (x_1 - x_1.mean()), dtype=torch.float32)


def get_gmm_moment_fn(x, income_only=False):
    y, c = x
    delta_y = get_diff(y)
    delta_c = get_diff(c)

    # --- Precompute all covariances ---
    yy_2 = get_cov(delta_y, delta_y, shift=2)
    yy_1 = get_cov(delta_y, delta_y, shift=1)
    var_y = get_cov(delta_y, delta_y, shift=0)

    cc_m1 = get_cov(delta_c, delta_c, shift=1)
    var_c = get_cov(delta_c, delta_c, shift=0)

    yc_2 = get_cov(delta_y, delta_c, shift=2)
    yc_1 = get_cov(delta_y, delta_c, shift=1)
    yc_0 = get_cov(delta_y, delta_c, shift=0)

    def moment_fn(
        theta,
        income_only=income_only,
        yy_2=yy_2,
        yy_1=yy_1,
        var_y=var_y,
        cc_m1=cc_m1,
        var_c=var_c,
        yc_2=yc_2,
        yc_1=yc_1,
        yc_0=yc_0,
    ):
        if income_only:
            theta = torch.cat([theta, torch.zeros(4)])
        sigma_eta, phi_epsilon, sigma_epsilon, alpha, beta, phi_c, sigma_c = theta
        results = [
            # 1. cov(Δy_t, Δy_{t+2}) + Ψ_ε σ_ε
            yy_2 + phi_epsilon * sigma_epsilon,
            # 2. cov(Δy_t, Δy_{t+1}) + (Ψ_ε−1) σ_ε
            yy_1 + (phi_epsilon - 1) * sigma_epsilon,
            # 3. var(Δy_t) − [η + (2Ψ_ε − 2Ψ_ε + 2) σ_ε]
            var_y
            - (sigma_eta + (2 * phi_epsilon - 2 * phi_epsilon + 2) * sigma_epsilon),
            # 4. cov(Δc_t, Δc_{t−1}) − Ψ_c σ_c
            cc_m1 - phi_c * sigma_c,
            # 5. var(Δc_t) − [α η + αβ σ_ε + (1+Ψ_c) σ_c]
            var_c
            - (
                alpha * sigma_eta + alpha * beta * sigma_epsilon + (1 + phi_c) * sigma_c
            ),
            # 6. cov(Δy_t, Δc_{t+2}) + αβ Ψ_ε σ_ε
            yc_2 + alpha * beta * phi_epsilon * sigma_epsilon,
            # 7. cov(Δy_t, Δc_{t+1}) − αβ (Ψ_ε−1) σ_ε
            yc_1 - alpha * beta * (phi_epsilon - 1) * sigma_epsilon,
            # 8. cov(Δy_t, Δc_t) − (α η + αβ σ_ε)
            yc_0 - (alpha * sigma_eta + alpha * beta * sigma_epsilon),
        ]

        if income_only:
            results = results[:3]
        return torch.stack(results)

    return moment_fn


def organize_results(theta, var):
    labels = [
        "sigma_eta",
        "Phi_epsilon",
        "sigma_epsilon",
        "alpha",
        "beta",
        "Phi_c",
        "sigma_c",
    ]
    return pd.DataFrame(
        np.array([var.detach().numpy() for var in [theta, var.diag()]]),
        columns=labels[: len(theta)],
        index=["Estimate", "Standard Errors"],
    ).transpose()


def problem_2():
    df = get_data()
    ################
    ### Part (c) ###
    ################

    g_income_only = get_gmm_moment_fn(x, income_only=True)
    gmm_income = GMM(g=g_income_only, theta0=np.array([1] * 3))
    theta_income = gmm_income.fit(max_steps=20000)
    var_income = gmm_income.asymptotic_variance()
    results_income_only = organize_results(theta_income, var_income)

    ################
    ### Part (d) ###
    ################

    g = get_gmm_moment_fn(x, income_only=False)
    gmm = GMM(g=g, theta0=np.concatenate([theta_income, np.array([1] * 4)]))
    theta = gmm.fit(max_steps=10000, lr=1e-1, max_error=1e-2)
    var = gmm.asymptotic_variance()
    results = organize_results(theta, var)


if __name__ == "__main__":
    problem_2()
