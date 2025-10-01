import os
import sys
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def add_diagonal_line(ax=None, **line_kwargs):
    if ax is None:
        ax = plt.gca()
    y_axis_lim = ax.get_ylim()
    x_axis_lim = ax.get_xlim()
    axis_min = max(x_axis_lim[0], y_axis_lim[0])
    axis_max = min(x_axis_lim[1], y_axis_lim[1])
    ax.plot(
        *([[axis_min, axis_max]] * 2),
        **line_kwargs,
    )
    ax.set_ylim(y_axis_lim)
    ax.set_xlim(x_axis_lim)


def outer_product(x, y=None):
    if y is None:
        y = x
    x = np.array(x)
    y = np.array(y)
    x_tile = np.tile(x, (len(y), 1)).T
    y_tile = np.tile(y, (len(x), 1))
    return (x_tile, y_tile)


def __iter(V, F_mat, beta, x=None, argmin=False):
    compute = F_mat + (beta * V).T
    if argmin:
        if x is None:
            raise ValueError("Must pass x if argmin=True")
        return x[np.argmin(compute, axis=1)]
    return np.min(compute, axis=1)


def fixed_point(fn, V_0, tol=1e-6, max_iter=1000):
    V = V_0
    count = 0
    while count < max_iter:
        V_next = fn(V)
        if np.abs(V_next - V).sum() < tol:
            return V
        count += 1
        V = V_next
        if count == max_iter:
            raise ValueError("Maximum iterations reached.")


def plot_value_fn(V, x, g):
    plt.plot(x, V)
    plt.ylabel("Value Function")
    plt.xlabel("Decision Rule")
    plt.show()
    _, ax = plt.subplots(1, 1)
    ax.plot(x, g)
    add_diagonal_line(ax=ax, c="r", linestyle="dashed", linewidth=1)
    plt.ylabel("State Variable")
    plt.xlabel("Decision Rule")
    plt.show()


def value_fn_iteration(
    F,
    x,
    beta,
    F_valid=None,
    V_0=None,
    tol=1e-6,
    max_iter=1000,
    plot=False,
):
    x = np.array(x)
    if V_0 is None:
        V_0 = np.zeros(x.shape)

    # matrixize cost function
    if callable(F):
        X = outer_product(x)
        fn_args = X[0], X[1]
        F_mat = F(*fn_args)
        if F_valid is not None:
            valid_bool = F_valid(*fn_args)
            F_mat[~valid_bool] = 9e9
    else:
        F_mat = np.array(F)

    iter_fn = lambda V, argmin: __iter(V=V, F_mat=F_mat, beta=beta, argmin=argmin, x=x)
    V_f = fixed_point(
        fn=lambda V: iter_fn(V, argmin=False),
        V_0=V_0,
        tol=tol,
        max_iter=max_iter,
    )

    g_f = iter_fn(V_f, argmin=True)

    if plot:
        plot_value_fn(V=V_f, x=x, g=g_f)

    return V_f, g_f


def shooting_aux(x, y, g, max_iter=1000, full_path=False, score_fn=None):
    if full_path:
        path = [x, y]
    count = 0
    while True:
        y_next = g(x, y)
        if full_path:
            path.append(y_next)
        x = y
        y = y_next
        count += 1
        if count == max_iter:
            break
        score = score_fn(y, x) if score_fn is not None else x - y
        if score == np.inf:
            if full_path:
                break
            return 1 / count
    if full_path:
        return np.array(path)
    return score


def plot_trajectory(path, score_fn=None, ax1=None, ax2=None, label=None):
    if ax1 is None:
        fig, ax = plt.subplots(1, 2 if score_fn is not None else 1)
        ax1 = ax[0]
        if score_fn is not None:
            ax2 = ax[1]
    x = path[:-1]
    y = path[1:]
    ax1.plot(x, y, label=label)
    ax1.set_xlabel("State Variable")
    ax1.set_ylabel("Decision Rule")
    if score_fn is not None:
        ax2.plot(
            x[1:],
            [score_fn(this_x, x[i + 1], T=i) for i, this_x in enumerate(x[:-1])],
            label=label,
        )
        ax2.set_ylabel("Transversality Score")
        ax2.set_xlabel("Iteration Number")


def shooting_method(
    x_0,
    g=None,
    F_x_inv=None,
    F_y=None,
    beta=None,
    F_x=None,
    max_iter=1000,
    y_0=None,
    plot=False,
    **kwargs,
):
    if y_0 is None:
        y_0 = x_0
    if g is None and (F_y is None or F_x_inv is None):
        raise ValueError("Must pass g or both F_x_inv and F_y")
    if g is None:
        if beta is None:
            raise ValueError("Must pass beta if F is passed")
        g = lambda x, y: F_x_inv(y, 1 / beta * F_y(x, y))
    if F_x is not None:
        if beta is None:
            raise ValueError("Must pass beta to get transversaility condition.")
        score_fn = lambda y, x, T=max_iter: F_x(x, y) * x * beta**T
    else:
        score_fn = None
    shooting_fn = lambda y, full_path=False: shooting_aux(
        x=x_0,
        y=y,
        g=g,
        max_iter=max_iter,
        score_fn=score_fn,
        full_path=full_path,
    )
    result = scipy.optimize.minimize(
        shooting_fn,
        x0=y_0,
        method="Nelder-Mead",
        **kwargs,
    )
    y = result.x[0]
    if plot:
        path = shooting_fn(y, full_path=True)
        plot_trajectory(path, score_fn=score_fn)
        plt.show()
    if F_x is not None:
        return y, shooting_fn, score_fn
    return y, shooting_fn
