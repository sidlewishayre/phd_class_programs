import math

#################
### paramters ###
#################

delta = 0.05
beta = 0.98
alpha = 1 / 3
gamma = 2
A = 1


# stochastic parameters
rho = 0.95
sigma = 0.02
Sigma = (sigma**2) / (1 - rho**2)


def marginal_utility(c, gamma=gamma):
    return c ** (-gamma)


def utility(c, gamma=gamma):
    return (c ** (1 - gamma) - 1) / (1 - gamma)


def mpk(k, pi, Sigma=Sigma, A=A, alpha=alpha):
    return alpha * A * sigma_mult(pi=pi, Sigma=Sigma) * (k ** (alpha - 1))


def sigma_mult(pi, Sigma=Sigma):
    if pi is None:
        Sigma = 0
        pi = 0
    return (math.e ** (-(1 / 2) * Sigma)) * (math.e**pi)


def G(k, pi, A=A, Sigma=Sigma, alpha=alpha):
    return A * sigma_mult(pi=pi, Sigma=Sigma) * k**alpha


def f(k, pi=None, Sigma=Sigma, delta=delta, alpha=alpha):
    return G(k, alpha=alpha, pi=pi, Sigma=Sigma) + (1 - delta) * k


def F(x, y, pi, delta=delta, alpha=alpha, gamma=gamma, utility=utility):
    return -utility(f(x, pi=pi, delta=delta, alpha=alpha) - y, gamma=gamma)


def f_prime(k, pi=None, delta=delta, alpha=alpha):
    return mpk(k, pi=pi, alpha=alpha) + (1 - delta)


def F_valid(x, y, pi=None, delta=delta, alpha=alpha):
    ub = f(x, pi=pi, delta=delta, alpha=alpha)
    return (y >= 0) & (y <= ub)
