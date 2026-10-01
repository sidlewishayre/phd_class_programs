import numpy as np

from howard_improvement import howard_improvement

##############
### inputs ###
##############

beta = 0.95
r = 0.95 ** (-1) - 1  # since (1 + r) = 0.95**(-1)
b = 30
gamma = 1
rho1 = 1.2
rho2 = -0.3

################
### matrices ###
################

A = np.array([[1 - r, -1, 0, 0], [0, rho1, rho2, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
B = np.array([[-1, 0, 0, 0]]).T
R = gamma * np.array(
    [[1 - r, 1 - r, 0, 0], [1 - r, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
)
Q = np.sqrt(1 + gamma) * np.array([[1]])
H = np.array([[2 * (1 - r), 0, 0, -2 * b]])

#########################
### perform iteration ###
#########################

P, F = howard_improvement(R, A, B, H, Q, beta)
print(F)
