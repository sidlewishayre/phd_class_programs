import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm

sys.path.append("/home/sidlh/Documents/reusable_code")
from other_utils.pandas_utils import add_id

###############
### inputs ####
###############

ns = [20, 200, 2000]
Ks = np.array([2, 10, 100, 200])
N = 200
assert Ks.min() >= 2

ns_summable = [100, 1000, 5000]
Ks_summable = np.array([1, 2, 3, 10])
N_summable = 2000

#####################
### generate data ###
#####################


def generate_data(n, K, model="junk"):
    z = np.array([np.random.normal(size=n) for _ in range(K)])
    e = np.array([np.random.normal(size=n) for _ in range(3)])
    if model == "junk":
        x = z[0] + z[1] + e[0] + e[1]
        y = 1 + x + e[0] + e[2]
    elif model == "summable":
        x = (1 / 10 * (z.shape[0] ** (-1 / 2))) * np.sum(z, axis=0) + e[0]
        y = 1 + x + e[1]
    return y, x, z, e


##############
### part a ###
##############


def OLS_with_constant(X, y, se=True):
    X = np.column_stack((np.ones(len(X)), X))
    XtX_inv = np.linalg.inv(X.T @ X)  # shape (k, k)
    beta = XtX_inv @ (X.T @ y)
    # homoskeastic standard error
    # residuals
    u = y - X @ beta  # shape (n,)
    # degrees of freedom
    n, k = X.shape
    # sigma^2 hat (use n - k)
    sigma2_hat = (u.T @ u) / (n - k)  # scalar
    # inverse of X'X (use pinv if worried about singularity)
    # standard errors
    std_err = np.sqrt(np.diag(XtX_inv * sigma2_hat))  # shape (k,)
    if not se:
        return beta, [0, 0]
    return beta, std_err


def IV_with_constant(X, y, Z, se=True):
    X = np.column_stack((np.ones(len(X)), X))
    Z = np.column_stack((np.ones(len(Z)), Z))
    Pz = Z @ np.linalg.inv(Z.T @ Z) @ Z.T
    beta = np.linalg.inv(X.T @ Pz @ X) @ (X.T @ Pz @ y)
    # residuals
    u = y - X @ beta
    # homoskedastic error variance
    sigma2_hat = u.T @ u / (X.shape[0] - X.shape[1])
    Ainv = np.linalg.inv(X.T @ Pz @ X)
    if not se:
        return beta, [0, 0]
    std_err = np.sqrt(np.diag(Ainv * sigma2_hat))
    return beta, std_err


def med_iqr(x):
    x = x[~np.isnan(x)]
    return {
        "median": np.median(x),
        "iqr": np.percentile(x, 75) - np.percentile(x, 25),
        "mean": np.mean(x),
        "std": np.std(x),
    }


def compare_iv_ols(n, K, N, plot=False, model="junk"):
    ols = []
    iv = []
    se = model != "summable"
    for i in range(N):
        y, x, z, e = generate_data(n, K, model=model)
        ols.append(OLS_with_constant(x.reshape(-1, 1), y, se=se))
        iv.append(IV_with_constant(x.reshape(-1, 1), y, z.T, se=se))
    ols = np.array(ols)
    iv = np.array(iv)
    if plot:
        plt.hist(ols[:, 1], alpha=0.5, label="OLS")
        plt.hist(iv[:, 1], label="IV")
        plt.show()
    if model == "summable":
        for v in [ols, iv]:
            v[:, 0, 1] = np.sqrt(n) * (v[:, 0, 1] - 1)
    return {
        k: {
            **med_iqr(v[:, 0, 1]),
            **({"std_err": pd.Series(v[:, 1, 1]).dropna().mean()} if se else {}),
        }
        for k, v in {"ols": ols, "iv": iv}.items()
    }


def plot_comparison(stat_results, title):
    ax = plt.subplot()
    for method in stat_results.index.get_level_values(0).unique():
        for K in stat_results.columns:
            stat_results.loc[method][K].plot(
                label=method.upper() if K == stat_results.columns.min() else "",
                c={"ols": "tab:blue", "iv": "tab:orange"}[method],
                ax=ax,
                alpha=np.log(K / max(stat_results.columns) + 1),
            )
    plt.xlabel("Log Number of Instruments")
    plt.ylabel("Value")
    plt.title("Comparison of IV and OLS by {}".format(title.upper()))
    plt.legend()
    plt.show()


def compile_results(model, ns, Ks, N):
    results = pd.concat(
        [
            add_id(
                add_id(
                    pd.DataFrame.from_records(
                        compare_iv_ols(n=n, K=K, N=N, model=model)
                    ),
                    value=K,
                    col="K",
                ),
                value=n,
                col="n",
            )
            for n in tqdm(ns)
            for K in Ks
        ]
    )
    results.index.name = "statistic"
    results = results.reset_index().melt(
        id_vars=["statistic", "n", "K"], var_name="method", value_name="value"
    )
    results["n"] = np.log10(results["n"])
    results["K"] = np.log10(results["K"])
    results.loc[results["statistic"] == "iqr", "value"] *= 1 / (
        norm.ppf(0.75) - norm.ppf(0.25)
    )
    pivot_results = results.pivot(
        index=["statistic", "method", "K"], columns=["n"], values="value"
    )

    for stat in pivot_results.index.get_level_values(0).unique():
        plot_comparison(stat_results=pivot_results.loc[stat], title=stat)

    return pivot_results


##############
### part a ###
##############

results = compile_results("junk", ns=ns, Ks=Ks, N=N)

##############
### part b ###
##############

results.loc["std_err", "iv"].round(3)
results.loc["iqr", "iv"].round(3)

##############
### part c ###
##############

results = compile_results("summable", ns=ns_summable, Ks=Ks_summable, N=N_summable)
results.xs(key="iv", level=1)
