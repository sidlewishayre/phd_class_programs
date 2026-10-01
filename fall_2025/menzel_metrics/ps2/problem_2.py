import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# settings
replications = 200
bootstrap_replications = 200
N = 50
percentile = 0.95

# parameters
beta_0 = 1
beta_1 = 1

beta = np.array([beta_0, beta_1])


def generate_data():
    # generate data
    X = np.random.uniform(0, 2, N)
    # add constant term
    X = np.column_stack((np.ones(N), X))
    E = np.random.uniform(-1, 1, N)
    Y = np.dot(X, beta) + E
    return X, Y


X, Y = generate_data()

#############
## part a ###
#############


# LS estimate
def least_squares(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y


ls_estimate = least_squares(X, Y)


# LS residuals
def get_residuals(Y, X, ls_estimate):
    return Y - X @ ls_estimate


residuals = get_residuals(Y, X, ls_estimate)


# compute t-statistic
def get_t_stat(residuals, ls_estimate):
    se = np.mean(residuals**2)
    return (ls_estimate[1] - beta_1) / se


t = get_t_stat(residuals, ls_estimate)

##############
### part b ###
##############


# sample estimated residuals
def get_boostrap_data(bootstrap_vec):
    rand_vec = np.random.randint(0, N, N)
    if type(bootstrap_vec) is list:
        return [vec[rand_vec] for vec in bootstrap_vec]
    return bootstrap_vec[np.random.randint(0, N, N)]


sample_residuals = get_boostrap_data(residuals)


def run_bootstrap(ls_estimate=None, residuals=None, residual=False):
    t_stats = []
    ls_estimates = []
    for i in range(replications):
        bootstrap_vec = get_boostrap_data(
            bootstrap_vec=residuals if residual else [Y, X]
        )
        this_Y = X @ ls_estimate + bootstrap_vec if residual else bootstrap_vec[0]
        this_X = X if residual else bootstrap_vec[1]
        this_ls_estimate = least_squares(this_X, this_Y)
        ls_estimates.append(this_ls_estimate[1])
        this_resid = get_residuals(this_Y, this_X, this_ls_estimate)
        t = get_t_stat(this_resid, this_ls_estimate)
        t_stats.append(t)
    return t_stats, ls_estimates


def repeat_bootstrap(residual=False):
    X, Y = generate_data()
    ls_estimate = least_squares(X, Y)
    residuals = get_residuals(Y, X, ls_estimate)
    t_stat_list = []
    ls_estimate_list = []
    for i in tqdm(range(bootstrap_replications)):
        t_stats, ls_estimates = run_bootstrap(
            ls_estimate=ls_estimate, residuals=residuals, residual=residual
        )
        t_stat_list.append(np.quantile(t_stats, percentile))
        ls_estimate_list.append(np.quantile(ls_estimates, percentile))
    return t_stat_list, ls_estimate_list


def plot(t_stats, ls_estimates, title):
    plt.hist(t_stats)
    plt.title(f"{title} - t-statistics")
    plt.show()
    plt.hist(ls_estimates)
    plt.title(f"{title} - LS estimates")
    plt.show()


##############
### part c ###
##############

t_stats, ls_estimates = repeat_bootstrap(residual=True)
plot(t_stats, ls_estimates, title="Residual Bootstrap")

##############
### part d ###
##############

t_stats, ls_estimates = repeat_bootstrap(residual=False)
plot(t_stats, ls_estimates, title="Bootstrap")
