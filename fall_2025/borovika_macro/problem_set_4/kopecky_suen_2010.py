import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

sigma_z = 1

####################
### Question 3.1 ###
####################


def block_matrix(Omega, pos):

    zero_col = np.zeros((Omega.shape[0], 1))
    zero_row = np.zeros((1, Omega.shape[1]))
    zero_scalar = np.zeros((1, 1))
    if pos == 0:
        block = [[Omega, zero_col], [zero_row, zero_scalar]]
    elif pos == 1:  # example: swap columns
        block = [[zero_col, Omega], [zero_scalar, zero_row]]
    elif pos == 2:  # swap rows
        block = [[zero_row, zero_scalar], [Omega, zero_col]]
    elif pos == 3:  # swap both
        block = [[zero_scalar, zero_row], [zero_col, Omega]]
    return np.block(block)


def get_transition_matrix(p, q, N):
    if N == 2:
        return np.array([[p, 1 - p], [1 - q, q]])
    Omega = get_transition_matrix(p=p, q=q, N=N - 1)
    P = np.sum(
        [
            prob * block_matrix(Omega, pos)
            for pos, prob in enumerate([p, 1 - p, 1 - q, q])
        ],
        axis=0,
    )
    P = P / np.sum(P, axis=1, keepdims=True)
    return P


def get_markov_state(N, rho):
    phi = np.sqrt(N - 1) * sigma_z
    p = (1 + rho) / 2
    q = p
    P = get_transition_matrix(p=p, q=q, N=N)
    Y = np.linspace(-phi, phi, N)
    return Y, P


def markov_iterate(P, iter=1000):
    n = P.shape[0]
    x = np.repeat(1 / n, n)
    x_list = [x]
    for _ in range(iter):
        x = np.dot(np.transpose(P), x)
        x_list.append(x)
    return x_list


def plot_distribution(Y, emp, theory_fn, title=None):
    if title is None:
        title = ""
    title = " " + title
    theory_values = theory_fn(Y)
    theory_values = theory_values / theory_values.sum()
    plt.plot(Y, theory_values, label="Theoretical", color="tab:orange")
    plt.bar(Y, emp, label="Empirical")
    plt.ylabel("Density")
    plt.xlabel("Y")
    plt.title("Theoretical vs Empirical Comparison" + title)
    plt.legend()
    plt.show()


def compute_comparison(rho, N):

    Y, P = get_markov_state(N=N, rho=rho)
    assert (np.abs(P.sum(axis=1) - 1) < 1e-3).all()
    results = markov_iterate(P)
    theory_fn = lambda x: norm.pdf(x, loc=0, scale=sigma_z)
    plot_distribution(
        Y,
        results[-1],
        theory_fn=theory_fn,
        title=f"(N={N}, {r"$rho$"}={rho})",
    )

    # auto covariance
    theory_covariance = lambda j, x: (rho**j) * theory_fn(x) ** 2
    empirical_covariance = lambda j: np.mean(
        [results[i] * results[i + j] for i in range(100, len(results) - j - 1)], axis=0
    )
    J = 10
    for j in range(1, J):
        plt.plot(
            Y,
            theory_covariance(j, Y),
            label=f"Theoretical Covariance" if j == 1 else None,
            alpha=1 - j / J,
            c="tab:blue",
        )
        plt.plot(
            Y,
            empirical_covariance(j),
            label=f"Empirical Covariance" if j == 1 else None,
            alpha=1 - j / J,
            c="tab:orange",
            linestyle="dashed",
        )
        plt.title("Empirical vs Theoretical Covariance")
        plt.legend()
    plt.show()
    return P


def main():
    ####################
    ### Question 3.2 ###
    ####################

    Ns = [2, 5, 10, 20]
    rhos = [0.95, 0.98, 0.999]

    Ps = [[compute_comparison(rho=rho, N=N) for rho in rhos] for N in Ns]

    ####################
    ### Question 3.3 ###
    ####################

    Ps[0]


if __name__ == "__main__":
    main()
