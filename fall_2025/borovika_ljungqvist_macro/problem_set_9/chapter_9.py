import numpy as np

# Question 9.6

lambda_1 = 0.97
lambda_2 = 1.03
gamma = 2
beta = 0.95

u_prime = lambda c: c ** (-gamma)


# part b
beta**5 * u_prime(lambda_1**3 * lambda_2**2) * (0.8 * 0.8 * 0.2 * 0.1 * 0.2)

# part c
beta**5 * u_prime(lambda_1**1 * lambda_2**4) * (0.2 * 0.9 * 0.9 * 0.9 * 0.1)


# part i
(
    (lambda_1 * lambda_2) ** (-gamma) * (0.8 * 0.2 + 0.2 * 0.1)
    + (lambda_1**2) ** (-gamma) * (0.8**2)
    + (lambda_2**2) ** (-gamma) * (0.2 * 0.9)
)

(
    (lambda_1 * lambda_2) ** (-gamma) * (0.1 * 0.2 + 0.9 * 0.1)
    + (lambda_1**2) ** (-gamma) * (0.1 * 0.8)
    + (lambda_2**2) ** (-gamma) * (0.9 * 0.9)
)


# Question 9.7

P = np.array([[0.8, 0.2], [0.3, 0.7]])
arrow_prices = np.array([1 / 2, 1])

# part j
(beta**2) * arrow_prices @ np.linalg.matrix_power(P.T, 2)

# part k
beta**5 * np.linalg.matrix_power(P.T, 5)
