import os
import sys
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt

from value_fn_iteration import shooting_method, plot_trajectory
from inputs import F, F_valid, beta, f_prime, f, gamma, marginal_utility
from iteration import compute_steady_state, run_iteration

max_iter = 1000


def g(x, y):
    y_next = f(y) - ((beta * f_prime(y)) ** (1 / gamma)) * (f(x) - y)
    if y_next < 1e-12:
        return 1e-12
    if y_next > f(y) - 1e-12:
        return f(y) - 1e-12
    return y_next


####################
### Question 1.8 ###
####################


k_steady = compute_steady_state()
assert g(k_steady, k_steady) == k_steady
k_0 = 0.25 * k_steady
F_x = lambda x, y: (
    marginal_utility(f(x) - y) if x > 1e-12 and x < f(x) - 1e-12 else np.inf
)
y, shooting_fn, score_fn = shooting_method(
    x_0=0.25 * k_steady, y_0=3, g=g, beta=beta, F_x=F_x, max_iter=max_iter
)


path = shooting_fn(y, full_path=True)
path_low = shooting_fn(y - 1e-2, full_path=True)
path_high = shooting_fn(y + 1e-2, full_path=True)

_, ax1 = plt.subplots(1, 1)
_, ax2 = plt.subplots(1, 1)
plot_trajectory(path, score_fn=score_fn, ax1=ax1, ax2=ax2, label="Optimal")
plot_trajectory(path_high, score_fn=score_fn, ax1=ax1, ax2=ax2, label="High")
plot_trajectory(path_low, score_fn=score_fn, ax1=ax1, ax2=ax2, label="Low")
ax1.set_xlim(0, 15)
ax1.set_ylim(0, 15)
ax2.set_xlim(0, 20)
ax2.set_ylim(0, 5)
ax1.legend()
ax2.legend()
ax1.legend()
ax1.set_title("Trajectories")
ax2.set_title("Trajectory Score")
plt.show()

####################
### Question 1.9 ###
####################

_, decision_rule, x = run_iteration(plot=False)

smooth_decision = scipy.interpolate.interp1d(x, decision_rule)
iterations = len(path)
value_path = [y]
for i in range(iterations - 2):
    x_next = smooth_decision(value_path[-1])
    value_path.append(x_next)

value_path = np.array(value_path)

steps = np.arange(iterations - 1)
plt.plot(steps, path[1:], label="Shooting Method")
plt.plot(steps, value_path, label="Value Function Iteration")
plt.xlabel("Steps")
plt.ylabel("Capital")
plt.legend()
plt.title("Comparing Methods")
plt.show()
