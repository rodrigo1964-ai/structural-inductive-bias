"""HAM (Homotopy Analysis Method) series computation for the nonlinear pendulum.

System: u'' + sin(u) = 0,  u(0) = pi/3,  u'(0) = 0
Auxiliary operator: L[u] = u'' + u
Initial approximation: u0(t) = (pi/3)*cos(t)
Convergence parameter: hbar = -1

The HAM recurrence for the mth-order deformation is:
  L[u_m] = hbar * R_m(u_{m-1})
where R_m is the residual from the (m-1)th partial sum applied to the original equation.

For simplicity and accuracy, we compute the HAM terms numerically using the
discrete recurrence on a dense grid.
"""

import numpy as np
from scipy.integrate import solve_ivp


def sin_series_residual(u_partial, t):
    """Compute N(u) = u'' + sin(u) numerically on grid t.

    Uses second-order finite differences for u''.
    """
    dt = t[1] - t[0]
    n = len(t)
    upp = np.zeros(n)
    upp[1:-1] = (u_partial[2:] - 2 * u_partial[1:-1] + u_partial[:-2]) / dt**2
    upp[0] = upp[1]
    upp[-1] = upp[-2]
    return upp + np.sin(u_partial)


def solve_linear_aux(rhs, t, u0=0.0, udot0=0.0):
    """Solve L[u] = u'' + u = rhs(t) with given ICs.

    Uses scipy solve_ivp for accuracy.
    """
    from scipy.interpolate import interp1d
    rhs_interp = interp1d(t, rhs, kind='cubic', fill_value='extrapolate')

    def ode_sys(s, y):
        return [y[1], rhs_interp(s) - y[0]]

    sol = solve_ivp(ode_sys, [t[0], t[-1]], [u0, udot0],
                    t_eval=t, method='RK45', rtol=1e-12, atol=1e-14)
    return sol.y[0]


def compute_ham_terms(K_max, t):
    """Compute HAM terms u_0, u_1, ..., u_{K_max} for the nonlinear pendulum.

    Parameters
    ----------
    K_max : int
        Maximum HAM order.
    t : np.ndarray
        Time grid (must be dense enough for accurate finite differences).

    Returns
    -------
    terms : list of np.ndarray
        terms[k] = u_k(t), the k-th HAM correction.
    partial_sums : list of np.ndarray
        partial_sums[k] = S_k(t) = sum_{j=0}^{k} u_j(t).
    """
    hbar = -1.0
    a0 = np.pi / 3

    # u_0(t) = a0 * cos(t)
    u0 = a0 * np.cos(t)
    terms = [u0]
    partial_sums = [u0.copy()]

    for m in range(1, K_max + 1):
        S_prev = partial_sums[m - 1]

        # Residual of previous partial sum: N[S_{m-1}] = S''_{m-1} + sin(S_{m-1})
        R = sin_series_residual(S_prev, t)

        # m-th order deformation: L[u_m] = hbar * R
        # u_m(0) = 0, u_m'(0) = 0 for m >= 1
        u_m = solve_linear_aux(hbar * R, t, u0=0.0, udot0=0.0)
        terms.append(u_m)
        partial_sums.append(S_prev + u_m)

    return terms, partial_sums
