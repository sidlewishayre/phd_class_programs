import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

############################
### Exercise 1; Part (c) ###
############################

# panel shape
n_mc = 100
N = 200
T = 7

# model parameters
beta = 1
r = 2
sigma = 1
Pi = np.array([1, 1])
rho = 0.5
alpha = 0.4


# simulation
def dgp(rho, sigma, alpha):
    size = (n_mc, N, T)
    epsilon = np.random.normal(loc=0, scale=sigma, size=size)
    v = np.empty(shape=size)
    v[:, :, 0] = epsilon[:, :, 0]
    for t in range(1, T):
        v[:, :, t] = rho * v[:, :, t - 1] + epsilon[:, :, t]
    z = np.random.normal(loc=0, scale=1, size=(*size, 2))
    u = np.random.normal(loc=0, scale=1, size=size)
    assert len(Pi.shape) == 1
    pi_z = (Pi.reshape(1, 1, 1, Pi.shape[0]) * z).sum(axis=-1)
    x = pi_z + alpha * epsilon + np.sqrt(1 - alpha**2) * u
    eta = np.random.normal(loc=0, scale=1, size=(n_mc, T))
    eta = np.repeat(eta, N, axis=0).reshape(size)
    assert eta.std(axis=1).max() < 1e-10
    y = beta * x + eta + v
    Sigma = sigma**2 * (np.ones(shape=T) + (rho ** np.linspace(0, T - 1, T)))
    Sigma[0] = 1
    return y, x, z, pi_z, Sigma


def first_difference_transform(x):
    return (x - np.roll(x, shift=1, axis=2))[:, :, 1:]


def within_groups_transform(x):
    return x - x.mean(axis=1, keepdims=True)


def orthogonal_deviations_transform(x):
    return x  # TODO


def reshape_for_reg(x, buffer=0):
    return x.reshape(x.shape[0], -1, *(x.shape[-buffer:] if buffer > 0 else ()))


def iv_reg(x, y, z):
    x = reshape_for_reg(x)
    y = reshape_for_reg(y)
    z = reshape_for_reg(z, buffer=1)
    z_T = np.moveaxis(z, -1, -2)
    x_hat = (z @ np.linalg.inv(z_T @ z) @ z_T @ x[..., None]).squeeze(-1)
    diff = lambda x: x - x.mean(axis=-1, keepdims=True)
    denominator = np.mean(diff(y) * diff(x_hat)).sum(axis=-1)
    numerator = np.var(x, axis=-1)
    beta_hat = numerator / denominator
    return beta_hat


def get_betas(rho, sigma, alpha):
    y, x, z, pi_z, Sigma = dgp(rho=rho, sigma=sigma, alpha=alpha)

    delta_y = first_difference_transform(y)
    delta_x = first_difference_transform(x)
    delta_z = first_difference_transform(z)
    beta_hat_fd = iv_reg(delta_x, delta_y, delta_z)

    y_within = within_groups_transform(y)
    x_within = within_groups_transform(x)
    z_within = within_groups_transform(z)
    beta_hat_wg = iv_reg(x_within, y_within, z_within)

    x_orthog = orthogonal_deviations_transform(x)
    y_orthog = orthogonal_deviations_transform(y)
    pi_z_orthog = orthogonal_deviations_transform(pi_z)

    def optimal_iv(pi_z, Sigma, x, y):
        x = reshape_for_reg(x)
        y = reshape_for_reg(y)
        inv_sigma = Sigma**-1
        X = pi_z * inv_sigma
        X = reshape_for_reg(X)
        pi_z = reshape_for_reg(pi_z)
        assert len(x.shape) == 2
        assert len(pi_z.shape) == 2
        assert len(y.shape) == 2
        denominator = (X * x).sum(axis=1)
        numerator = (X * y).sum(axis=1)
        return numerator / denominator

    beta_iv = optimal_iv(pi_z_orthog, Sigma, x_orthog, y_orthog)

    return beta_hat_fd, beta_hat_wg, beta_iv


def plot_betas(param, values):
    kwargs = {"rho": rho, "sigma": sigma, "alpha": alpha}
    model_kwargs = [
        {k: v if k != param else i for k, v in kwargs.items()} for i in values
    ]
    betas = [get_betas(**kwargs) for kwargs in model_kwargs]
    betas = np.array(betas)

    betas_df = pd.DataFrame(
        betas.var(axis=-1) / np.sqrt(n_mc),
        columns=["First Difference", "Within Groups", "Optimal IV"],
        index=values,
    )

    betas_df.plot()
    plt.xlabel(f"$\\{param}$")
    plt.ylabel("Monte Carlo var($\\beta$)")
    plt.title(f"Monte Carlo Estimate of the Sample Variance of $\\beta$ by $\\{param}$")
    plt.show()


if __name__ == "__main__":
    plot_betas(param="rho", values=np.linspace(-0.2, 1, 7))
    plot_betas(param="alpha", values=np.linspace(0.1, 0.8, 8))
    plot_betas(param="sigma", values=np.linspace(0.2, 1.4, 7))
