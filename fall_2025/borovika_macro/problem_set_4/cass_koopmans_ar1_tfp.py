import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    "/home/sidlh/Documents/class_programs/fall_2025/borovika_ljungqvist_macro/problem_set_4"
)
from kopecky_suen_2010 import get_markov_state

sys.path.append(
    "/home/sidlh/Documents/class_programs/fall_2025/borovika_ljungqvist_macro/problem_set_3"
)
from iteration import compute_steady_state, run_iteration
from inputs import rho, Sigma

sys.path.append("/home/sidlh/Documents/reusable_code")
from macro.value_fn_iteration import simulate_path


simulate_path

# inputs
N = 21
N_k = 100
phi = np.sqrt(N - 1) * np.sqrt(Sigma)
pi = np.linspace(-phi, phi, N)

####################
### Question 4.1 ###
####################

k_steady = compute_steady_state()

####################
### Question 4.2 ###
####################
Y, P = get_markov_state(N=N, rho=rho)

# deterministic
V_0, g_0, x_0 = run_iteration(plot=False, N=N_k)
####################
### Question 4.3 ###
####################

# stochastic
V, g, x = run_iteration(plot=False, P=P, pi=pi, N=N_k)

plt.plot(x_0, g_0, label="Baseline")
plt.plot(x, g[0], label="Stochastic")
plt.xlabel("Initial Capital")
plt.ylabel("Decision Rule")
plt.legend()
plt.show()

####################
### Question 4.4 ###
####################

for i in range(len(g)):
    plt.plot(x, g[i], label=f"$z_{{{i}}}$")
    plt.xlabel("Initial Capital")
    plt.ylabel("Decision Rule")
    plt.legend()
plt.show()


####################
### Question 4.5 ###
####################

paths = [simulate_path(g, x=x, x_0=None, P=P, iter=1000)[0] for i in range(1000)]
paths = np.array(paths)

long_run_sample = paths[:, 100:].flatten()
plt.hist(long_run_sample, bins=20, density=True)
plt.title("Density of Capital")
plt.xlabel("Capital")
plt.show()


####################
### Question 4.6 ###
####################

long_run_sample.mean()


####################
### Question 4.7 ###
####################

paths = np.array(
    [
        simulate_path(g, x=x, x_0=k_steady, pi_0=P.shape[0] - 1, P=P, iter=50)[0]
        for i in range(1000)
    ]
)

x_vals = np.arange(paths.shape[1])
plt.plot(x_vals, paths.mean(axis=0), c="r", linewidth=3)
for i in range(paths.shape[0]):
    plt.plot(x_vals, paths[i, :], linewidth=0.01)
plt.title("Impusle Response from Productivity Shock")
plt.xlabel("Periods")
plt.ylabel("Capital")
plt.show()
