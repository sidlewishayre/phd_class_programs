#################
### paramters ###
#################

delta = 0.05
beta = 0.98
alpha = 1 / 3
gamma = 2
A = 1


def marginal_utility(c, gamma=gamma):
    return c ** (-gamma)


def utility(c, gamma=gamma):
    return (c ** (1 - gamma) - 1) / (1 - gamma)


def mpk(k, A=A, alpha=alpha):
    return alpha * A * (k ** (alpha - 1))


def G(k, A=A, alpha=alpha):
    return A * k**alpha


def f(k, delta=delta, alpha=alpha):
    return G(k, alpha=alpha) + (1 - delta) * k


def F(x, y, delta=delta, alpha=alpha, gamma=gamma, utility=utility):
    return -utility(f(x, delta=delta, alpha=alpha) - y, gamma=gamma)


def f_prime(k, delta=delta, alpha=alpha):
    return mpk(k, alpha=alpha) + (1 - delta)


def F_valid(x, y, delta=delta, alpha=alpha):
    ub = f(x, delta=delta, alpha=alpha)
    return (y >= 0) & (y <= ub)
