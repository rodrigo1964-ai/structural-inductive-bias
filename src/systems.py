"""Dynamical systems for 15Paper experiments."""

import numpy as np
from scipy.integrate import solve_ivp


# ─── Experiment 1: Damped oscillator ───

LAMBDA = 0.5
T_MAX_EXP1 = 10.0

def g_modulation(t):
    """Slowly varying modulation g(t) = 2 + 0.3*sin(2t) + 0.1*cos(5t)."""
    return 2.0 + 0.3 * np.sin(2 * t) + 0.1 * np.cos(5 * t)

def f_star_exp1(t):
    """Ground truth f*(t) = g(t) * exp(-lambda*t)."""
    return g_modulation(t) * np.exp(-LAMBDA * t)


# ─── Experiment 2: Exponentially transformed smooth function ───

T_MAX_EXP2 = 2 * np.pi

def g_inner_exp2(t):
    """Inner function g(t) = sin(t) + 0.5*cos(2t)."""
    return np.sin(t) + 0.5 * np.cos(2 * t)

def f_star_exp2(t):
    """Ground truth f*(t) = exp(g(t))."""
    return np.exp(g_inner_exp2(t))


# ─── Experiment 3: Nonlinear pendulum ───

T_MAX_EXP3 = 10.0
U0_PENDULUM = np.pi / 3
UDOT0_PENDULUM = 0.0

def pendulum_rhs(t, state):
    """RHS for u'' + sin(u) = 0."""
    u, v = state
    return [v, -np.sin(u)]

def pendulum_reference(t_eval):
    """Compute reference pendulum solution via RK45 (rtol=1e-12)."""
    sol = solve_ivp(
        pendulum_rhs,
        [0, t_eval[-1]],
        [U0_PENDULUM, UDOT0_PENDULUM],
        t_eval=t_eval,
        method='RK45',
        rtol=1e-12,
        atol=1e-14,
    )
    return sol.y[0]
