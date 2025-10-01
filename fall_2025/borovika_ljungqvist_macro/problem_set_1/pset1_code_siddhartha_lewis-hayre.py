import scipy
import numpy as np
import matplotlib.pyplot as plt

################
### settings ###
################

J = 100

##############
### inputs ###
##############

T = 50
beta = 0.97
B = 1
L = 0
c = 0.3


def f(w, alpha=2, beta=2):
    return (w ** (alpha - 1) * (1 - w) ** (beta - 1)) / (
        scipy.special.beta(alpha, beta)
    )


def F(w, alpha=2, beta=2):
    return scipy.stats.beta.cdf(w, alpha, beta)


############
### grid ###
############


def discretize_distribution(F, L=L, B=B, J=J):
    w_grid = np.linspace(L, B, J)
    f_grid = F(
        (w_grid + np.roll(np.concatenate([w_grid, [B]]), shift=-1)[:-1]) / 2
    ) - F((w_grid + np.roll(np.concatenate([[L], w_grid]), shift=1)[1:]) / 2)
    assert f_grid.sum() == 1
    return w_grid, f_grid


################
### plotting ###
################


def plot_reserve_wage(w_bar, label=None):
    plt.plot(np.arange(T), w_bar, label=label)
    plt.xlabel("Time Period")
    plt.ylabel("Reserve Wage")
    plt.title("Reserve Wage at each Time Period")


#############################################
### Excercise 3.9: iterate value function ###
#############################################

w_grid, f_grid = discretize_distribution(F)


def iterate_value_function_aux(V, t, f_grid):
    Q = np.dot(V, f_grid)
    V_new = np.maximum(w_grid * (1 - beta ** (T - t + 1)) / (1 - beta), c + beta * Q)
    return V_new


def iterate_value_function(f_grid=f_grid):
    V_0 = np.zeros(J)
    Vs = [V_0]
    for t in range(T):
        V_new = iterate_value_function_aux(Vs[-1], t=T - t, f_grid=f_grid)
        Vs.append(V_new)
    Vs = np.array(Vs[::-1])
    return Vs


Vs = iterate_value_function()


def calc_w_bar_from_Vs(Vs, f_grid=f_grid):
    w_bar = (c + beta * np.dot(Vs[1:], f_grid)) / (
        (1 - beta ** np.arange(T, 0, -1)) / (1 - beta)
    )
    return w_bar


w_bar = calc_w_bar_from_Vs(Vs)
Vs = Vs[:-1]


plot_reserve_wage(w_bar, label="$\\beta$=2, $\\alpha$=2")

##############################################
### Excercise 3.11: mean preserving spread ###
##############################################

f_grid_mps = discretize_distribution(lambda w: F(w=w, alpha=1.5, beta=1.5))[1]
Vs_mps = iterate_value_function(f_grid=f_grid_mps)
w_bar_mps = calc_w_bar_from_Vs(Vs_mps, f_grid=f_grid_mps)

plot_reserve_wage(w_bar_mps, label="$\\beta=1.5$, $\\alpha$=1.5")
plt.legend()
plt.show()


######################################################
### Excercise 3.10 direct reserve wage computation ###
######################################################


def iterate_reserve_wage(w, t):
    return c + beta * (((1 - beta ** (T - t)) / (1 - beta ** (T - t + 1)))) * (
        np.dot(w_grid, f_grid) - c + scipy.integrate.quad(F, a=0, b=w)[0]
    )


w_bar_0 = 0
w_bars = [w_bar_0]
for t in range(T):
    w_bar_new = iterate_reserve_wage(w=w_bars[-1], t=T - t)
    w_bars.append(w_bar_new)

w_bars = w_bars[1:][::-1]
plot_reserve_wage(w_bars)
plt.title("Direct Recursive Reserve Wage Calculation")
plt.show()

#######################################
### Excercise 3.12 infinite horizon ###
#######################################

g = (
    lambda w: w
    - c
    - beta / (1 - beta) * np.dot((w_grid[w_gridO > w] - w), f_grid[w_grid > w])
)
w_bar_infinite = scipy.optimize.root(g, x0=0).x
