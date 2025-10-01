import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from tqdm import tqdm

################
### settings ###
################
N = 100

##############
### inputs ###
##############

beta = 0.97
delta = 0.0


def g(x, A=1, alpha=0.8):
    return A * x**alpha


def pi(s):
    return s**0.5


mu = np.arange(N)
f = 1 / (np.arange(N) + 1)
f = f / f.sum()

########################
### Bellman operator ###
########################


def reject_value(V, mu, phi, delta=delta):
    return V[
        np.minimum(np.maximum(np.int64(np.round(g(mu * phi) - delta * mu)), 0), N - 1)
    ]


def offer_value(V, phi, mu, rv=None):
    if rv is None:
        rv = reject_value(V=V, mu=mu, phi=phi)
    return np.dot(np.maximum(V, rv), f)


def next_value(V, phi, s, mu):
    rv = reject_value(V=V, mu=mu, phi=phi)
    return (
        mu * (1 - phi - s)
        + beta * pi(s) * offer_value(V=V, phi=phi, mu=mu, rv=rv)
        + beta * (1 - pi(s)) * rv
    )


def bellman_operator(V, return_args=False, mu=mu):
    next_value_aux = lambda x, mu: next_value(V=V, phi=x[0], s=x[1], mu=mu)
    next_opt = lambda mu: scipy.optimize.minimize(
        lambda x: -1 * next_value_aux(x=x, mu=mu),
        x0=[1 / 3, 1 / 3],
        bounds=((0, 1), (0, 1)),
    )
    results = np.vectorize(lambda x: next_opt(x))(mu)
    assert all(res.success for res in results)
    V_new = np.array([-res.fun for res in results])
    phi = np.array([res.x[0] for res in results])
    s = np.array([res.x[1] for res in results])
    if return_args:
        return phi, s
    return V_new


def plot_value_function(V):
    phi_star, s_star = bellman_operator(V, return_args=True)

    plt.plot(mu, phi_star, label="$\\phi$")
    plt.plot(mu, s_star, label="$s$")
    plt.legend()
    plt.xlabel("$\\mu$")
    plt.ylabel("Policy")
    plt.title("Optimal Search and Investment Policies")
    plt.show()


V = np.arange(N)
count = 0
loop_num = 100
for _ in tqdm(range(loop_num)):
    V_new = bellman_operator(V=V)
    if np.max(np.abs(V_new - V)) < 1e-2:
        V_star = V_new
        break
    V = V_new
    if count % 10 == 0:
        phi_star, s_star = bellman_operator(V, return_args=True)
        plot_value_function(V)
    count += 1

    # if count > loop_num - 1:
    #     raise ValueError("No convergence")

plot_value_function(V_star)
