import sys
import statistics
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

sys.path.append("/home/sidlh/Documents/reusable_code")
from macro.discretized_processes import (
    get_rouwenhorst_markov_chain,
    get_tauchen_markov_chain,
)

##############
### Inputs ###
##############

rho = 0.9
sigma = 0.06
w_bar = 1
transform = lambda w: np.exp(w_bar + np.array(w))
variance = lambda sigma, rho: sigma**2 / (1 - rho**2)


################
### Part (a) ###
################


def get_markov_chain(N=5, rho=rho, sigma=sigma, process_type="rouwenhorst"):
    if process_type == "tauchen":
        get_markov = get_tauchen_markov_chain
    elif process_type == "rouwenhorst":
        get_markov = get_rouwenhorst_markov_chain
    mc = get_markov(N=N, rho=rho, sigma=sigma)
    return mc


def simulate_process(
    N, plot=True, process_type="tauchen", rho=rho, sigma=sigma, transform=transform
):
    mc = get_markov_chain(N=N, rho=rho, sigma=sigma, process_type=process_type)
    data = mc.simulate(start=statistics.median_low(mc.states), T=10000)
    data = transform(data)
    if plot:
        plt.hist(data, density=True)
        x = np.linspace(mc.states[0], mc.states[-1], 100)
        plt.plot(
            transform(x),
            norm.pdf(x, loc=0, scale=np.sqrt(variance(sigma=sigma, rho=rho))),
            label="Asymptotic Distribution",
        )
        plt.title(f"Histogram of {process_type.capitalize()} Simulation")
        plt.show()
    return data


def main():
    data_tauchen_5 = simulate_process(N=5)

    ################
    ### Part (b) ###
    ################

    data_tauchen_7 = simulate_process(N=7)

    ################
    ### Part (c) ###
    ################

    rouwenhorst = simulate_process(N=5, process_type="rouwenhorst")

    ################
    ### Part (d) ###
    ################

    rho_new = 0.98
    sigma_new = np.sqrt(variance(sigma=sigma, rho=rho) * (1 - rho_new**2))
    simulate_process(N=5, rho=rho_new, sigma=sigma_new)
    simulate_process(N=5, process_type="rouwenhorst", rho=rho_new, sigma=sigma_new)


if __name__ == "__main__":
    main()
