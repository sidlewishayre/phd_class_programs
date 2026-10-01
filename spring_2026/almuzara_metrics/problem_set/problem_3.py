import os
import sys
import pandas as pd
import numpy as np

CWD = os.path.abspath(os.path.join(__file__, os.path.pardir))

sys.path.append(CWD)
from problem_2 import get_data
from problem_1 import iv_reg, first_difference_transform

df = get_data()


def make_panel(x):
    x.name = "y"
    x = x.astype(float)
    return (
        x.sort_index().reset_index().pivot(index="i", columns="t", values=x.name).values
    )


y = df["income"]
x = pd.Series(df.index.get_level_values("t"), df.index)
z = df["age"] ** 2

x = make_panel(x)
y = make_panel(y)
z = make_panel(z)


y_fd = first_difference_transform(np.expand_dims(y, axis=0))[0]
x_fd = first_difference_transform(np.expand_dims(x, axis=0))[0]
z_fd = first_difference_transform(np.expand_dims(z, axis=0))[0]


def covariance(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    return np.cov(x, y)[0, 1]


# solve for beta using 2SLS first differences
beta_fd = covariance(y_fd, z_fd) / covariance(z_fd, x_fd)

# TODO generalized least squares: cannot run becuase of singular covariance matrix
