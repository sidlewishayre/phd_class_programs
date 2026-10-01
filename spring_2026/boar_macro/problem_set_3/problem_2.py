import os
import statistics
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir))

sys.path.append(CWD)
from problem_1 import get_markov_chain
from problem_1 import transform

sys.path.append("/home/sidlh/Documents/deep_virgil")
from bellman import Bellman
from architecture.exog_state import ExogenousState
from architecture.custom_layer import CustomLayer

mc = get_markov_chain()
phi = 0
beta = 0.95
R = 1.02


def crra(c, gamma):
    # CRRA utility function
    return (
        c ** (1 - gamma) / (1 - gamma)
        if gamma != 1
        else (c.log() if type(c) == torch.Tensor else np.log(c))
    )


def consumption_reward(a, a_prime, income, gamma):
    c = income + R * a - a_prime
    return crra(c, gamma=gamma)


def income_fluctuation_policy_iteration(
    sigma=0.06, phi=0, gamma=2, max_assets=10, plot=True
):

    mc = get_markov_chain(sigma=sigma)

    asset_range = (phi, max_assets)
    # TODO solve capital accumulation problem using Bellman class and compare to known solution
    r = CustomLayer(
        fn=lambda x, u: consumption_reward(
            a=x[:, :1], a_prime=u, income=x[:, 1:], gamma=gamma
        ),
        auto_grad=True,
    )

    exog_state = ExogenousState(
        transition_matrix=mc.P,
        exog_states=transform(mc.states),
    )

    self = Bellman(
        exog_state,
        state_dim=1,
        control_dim=1,
        exog_state_dim=1,
        state_input_range=[asset_range],
        control_input_range=[asset_range],
        beta=beta,
        reward_model=r,
        lom=CustomLayer(
            fn=lambda x: x, auto_grad=True
        ),  # TODO allow lom to depend on theta
        debug=False,
        inequality=True,
    )

    init_policy_fn = lambda x: x[:, :1]
    policy_fn, success = self.policy_fn_iteration(u_init=init_policy_fn, max_iter=100)

    if plot:
        policy_data = pd.DataFrame(
            policy_fn(self.pinn.training_grid.detach().numpy()).reshape(
                -1, len(mc.states)
            )
        )
        policy_data.columns = [f"z={z:.2f}" for z in mc.states]
        policy_data.index = self.pinn.training_grid.detach().numpy()[:, 0][
            :: len(mc.states)
        ]
        policy_data.plot()
    return policy_fn


def simulate_from_policy_fn(policy_fn, T=10000, buffer=1000):
    mc = get_markov_chain()
    np.random.seed(0)
    shock_value = np.random.uniform(size=T)
    states = transform(
        mc.simulate_uniform(
            start=statistics.median_low(mc.states), T=T, uniform_shocks=shock_value
        )
    )
    assets = [2]
    for state in states:
        x = torch.tensor([[assets[-1], state]])
        a_prime = policy_fn(x).item()
        assets.append(a_prime)

    assets = np.array(assets[buffer:])
    states = np.array(states[buffer:])
    return states, assets


# policy_fn = income_fluctuation_policy_iteration()
import joblib

save_filepath = "/home/sidlh/Documents/class_programs/spring_2026/boar_macro/problem_set_3/policy_fn.joblib"
# joblib.dump(policy_fn, save_filepath)
policy_fn = joblib.load(save_filepath)
states, assets = simulate_from_policy_fn(policy_fn)

# # debugging
# y = 2.0641
# a = 1
# x = torch.tensor([[a, y]])
# policy_fn = init_policy_fn
# dr_du, expectation, r_prime, integrand_dx, d_x_prime = self.pinn(
#     x=x,
#     u=self.control_grid,
#     policy_fn=policy_fn,
#     debug=True,
# )


# def plot_pinn(var):
#     if type(var) == torch.Tensor:
#         var = var.detach().numpy()
#     plt.plot(self.control_grid.detach().numpy(), var)


# def plot_true(fn):
#     u_values = self.control_grid.detach().numpy()
#     y = fn(u_values)
#     plt.plot(u_values, y)
#     return y


# marginal_utility_fn = lambda y, a, u: (y + R * a - u) ** (-gamma)


# plot_pinn(-dr_du)
# dr_du_direct = plot_true(lambda u: marginal_utility_fn(y, a, u=u))
# plt.title("Marginal Utility")
# plt.show()


# x_prime_direct = self.pde.h(self.control_grid)
# u_prime_direct = policy_fn(x_prime_direct)
# markov_expectation = lambda x: (
#     x.reshape(-1, mc.P.shape[0]).detach().numpy()
#     * mc.P[np.isclose(transform(np.array(mc.states)), y)]
# ).sum(axis=1)
# u_next = consumption_reward(
#     income=x_prime_direct[:, 1:2], a=x_prime_direct[:, 0:1], a_prime=u_prime_direct
# )
# mu_next = R * marginal_utility_fn(
#     y=x_prime_direct[:, 1:2], a=x_prime_direct[:, 0:1], u=u_prime_direct
# )
# expectation_direct = markov_expectation(mu_next)

# plot_pinn(markov_expectation(r_prime))
# plot_pinn(markov_expectation(u_next))
# plt.title("Utility Next")
# plt.show()


# # import pandas as pd
# # pd.Series((integrand_dx - mu_next).detach().numpy()[:, 0]).plot()
# plot_pinn(markov_expectation(integrand_dx))
# plot_pinn(markov_expectation(mu_next))
# plt.title("Marginal Utility Next")
# plt.show()


# # TODO: why is this wrong?
# plot_pinn(expectation)
# plot_pinn(expectation_direct)
# plt.title("Expectation")
# plt.show()

# plot_pinn(-dr_du - self.beta * expectation)
# plot_pinn(dr_du_direct[:, 0] - self.beta * expectation_direct)
# plt.title("Euler Equation Error")
# plt.show()
