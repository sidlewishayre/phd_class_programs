import sys
import numpy as np

sys.path.append("/home/sidlh/Documents/reusable_code")
from macro.value_fn_iteration import fixed_point


def riccati_iterate(P, F, R, A, B, H, Q, beta):
    return (
        R
        + F.T @ Q @ F
        - (F.T @ H + H.T @ F)
        + beta * (A - B @ F).T @ P @ (A - B @ F)
        # R
        # + beta * A.T @ P @ A
        # - (beta * A.T @ P @ B + H.T)
        # @ np.linalg.inv(Q + beta * B.T @ P @ B)
        # @ (beta * B.T @ P @ A + H)
    )


def solve_riccati(F, R, A, B, H, Q, beta):
    P_0 = np.zeros(R.shape)
    fn = lambda P: riccati_iterate(P=P, F=F, R=R, A=A, B=B, H=H, Q=Q, beta=beta)
    return fixed_point(fn=fn, V_0=P_0)


def f_iterate(F, R, A, B, H, Q, beta):
    P = solve_riccati(F=F, R=R, A=A, B=B, H=H, Q=Q, beta=beta)
    F_new = np.linalg.inv(Q + beta * B.T @ P @ B) @ (H + beta * B.T @ P @ A)
    return P, F_new


def howard_improvement(R, A, B, H, Q, beta):
    F_0 = np.zeros((B.shape[1], A.shape[1]))
    fn = lambda F: f_iterate(F=F, R=R, A=A, B=B, H=H, Q=Q, beta=beta)
    F_final = fixed_point(fn=lambda F: fn(F)[1], V_0=F_0)
    return fn(F_final)
