import os
import sys
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from value_fn_iteration import value_fn_iteration

from inputs import (
    F,
    F_valid,
    f_prime,
    beta,
    delta,
    f,
    alpha,
    gamma,
    utility,
)

################
### settings ###
################

N = 1000


def plot_consumption(x, g, alpha=alpha, delta=delta):
    c = f(x, alpha=alpha, delta=delta) - g
    plt.plot(x, c)
    plt.ylabel("Consumption")
    plt.xlabel("Capital")
    plt.xlim(x.min(), x.max())
    plt.ylim(c.min(), c.max())
    plt.show()
    return c


def compute_steady_state(alpha=alpha, beta=beta, delta=delta, f_prime=f_prime):
    return float(
        scipy.optimize.root(
            lambda k: (f_prime(k, alpha=alpha, delta=delta)) * beta - 1, x0=1
        ).x[0]
    )


def run_iteration(
    delta=delta,
    beta=beta,
    alpha=alpha,
    gamma=gamma,
    min_ratio=0.1,
    max_ratio=2,
    x=None,
    compare=None,
    plot=True,
    utility=utility,
):
    k_steady = compute_steady_state()
    if compare is not None:
        if len(compare) > 3:
            V_alt = compare[0]
        else:
            V_alt = None
        g_alt = compare[-3]
        c_alt = compare[-2]
        x_alt = compare[-1]

    iterate_value_fn = lambda x: value_fn_iteration(
        F=lambda x, y: F(x, y, delta=delta, alpha=alpha, gamma=gamma, utility=utility),
        x=x,
        F_valid=lambda x, y: F_valid(x, y, delta=delta, alpha=alpha),
        beta=beta,
        plot=compare is None and plot,
        max_iter=10000,
        tol=1e-10,
    )
    if x is not None:
        x = np.linspace(min_ratio * k_steady, max_ratio * k_steady, N)
    V, g = iterate_value_fn(x)
    if not plot:
        return V, g, x
    if V_alt is not None:
        plt.plot(x, V)
        plt.plot(x_alt, V_alt, label="Baseline", c="r")
        plt.legend()
        plt.ylabel("Value Function")
        plt.xlabel("Decision Rule")
        plt.show()
    if compare is not None:
        plt.plot(x, g)
        plt.plot(x_alt, g_alt, label="Baseline", c="r")
        plt.legend()
        plt.ylabel("Next Capital")
        plt.xlabel("This Capital")
        plt.xlim(x.min(), x.max())
        plt.ylim(g.min(), g.max())
        plt.show()
    if compare is not None:
        plt.plot(x_alt, c_alt, label="Baseline", c="r")
        plt.legend()
    c = plot_consumption(x, g=g, alpha=alpha, delta=delta)
    return V, g, c, x


def main():
    ####################
    ### Question 1.6 ###
    ####################

    run_iteration(min_ratio=0.95, max_ratio=1.1)

    ####################
    ### Question 1.7 ###
    ####################

    _, g, c, x = run_iteration()

    ####################
    ### Question 1.8 ###
    ####################

    # increase delta
    run_iteration(delta=0.1, compare=[g, c, x])

    # increase beta
    run_iteration(beta=0.95, compare=[g, c, x])

    # higher capital share
    run_iteration(alpha=0.5, compare=[g, c, x])

    # lower intertemporal elasticity of substiution
    run_iteration(gamma=3, compare=[g, c, x])

    #####################
    ### Question 1.17 ###
    #####################

    b = alpha * (1 - beta * alpha)
    a = (1 / (1 - beta)) * (
        np.log(1 - beta * b / (1 + beta * b))
        + beta * b * np.log((1 + beta * b))
        + beta * alpha
    )
    x = np.arange(0, 20)
    V = -(a + b * np.log(x))
    g = beta * b * x / (1 + beta * b)
    c = f(x) - g

    run_iteration(utility=lambda c, gamma: np.log(c), compare=[V, g, c, x])


if __name__ == "__main__":
    main()
