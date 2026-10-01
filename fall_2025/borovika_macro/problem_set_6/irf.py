import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("/home/sidlh/Documents/reusable_code")
from plotting_utils.plot_additions import add_full_line


################
### settings ###
################

t = np.arange(100)

####################
### IRF Equation ###
####################


def irf_b(t, R, rho, beta):
    return -R * (1 - rho) / (1 - beta * rho) * (1 - rho**t) / (1 - rho)


def irf_c(t, R, rho, beta):
    return np.repeat((1 - beta) * (1 - beta * rho), np.array(t).shape)


def irb_partial_info(t, R, rho, beta, fn):
    if fn == "c":
        fn = irf_c
    elif fn == "b":
        fn = irf_b
    else:
        raise ValueError("Wrong fn.")
    rho = np.array(rho)
    return (
        (rho / rho.shape[0])
        * np.array([fn(t, R, this_rho, beta) for this_rho in rho]).T
    ).sum(axis=1)


####################
### Question 1.1 ###
####################

rho = [0.5, 0]
beta = 0.95
R = beta

plt.plot(t, irf_b(t, R=R, rho=rho[0], beta=beta), label="Debt")
plt.plot(
    t,
    irb_partial_info(t, R=R, rho=rho, beta=beta, fn="b"),
    label="Debt (Partial Info)",
)
plt.plot(t, irf_c(t, R=R, rho=rho[0], beta=beta), label="Consumption")
plt.plot(
    t,
    irb_partial_info(t, R=R, rho=rho, beta=beta, fn="c"),
    label="Consumption (Partial Info)",
)
add_full_line(0, vertical=False, linestyle="dotted", color="r")
plt.legend()
plt.title("Impulse Response Functions")
plt.xlabel("Value")
plt.show()

####################
### Question 1.2 ###
####################

rho = [0.99, 0.5]
beta = 0.95
R = beta
for i in range(len(rho)):
    plt.plot(t, irf_b(t, R=R, rho=rho[i], beta=beta), label=f"Debt (Shock {i+1})")
plt.plot(
    t,
    irb_partial_info(t, R=R, rho=rho, beta=beta, fn="b"),
    label="Debt (Partial Info)",
)
add_full_line(0, vertical=False, linestyle="dotted", color="r")
plt.legend()
plt.title("Impulse Response Functions for Debt")
plt.xlabel("Value")
plt.show()

for i in range(len(rho)):
    plt.plot(
        t, irf_c(t, R=R, rho=rho[i], beta=beta), label=f"Consumption (Shock {i+1})"
    )
plt.plot(
    t,
    irb_partial_info(t, R=R, rho=rho, beta=beta, fn="c"),
    label="Consumption (Partial Info)",
)
add_full_line(0, vertical=False, linestyle="dotted", color="r")
plt.legend()
plt.title("Impulse Response Functions for Consumption")
plt.xlabel("Value")
plt.show()
