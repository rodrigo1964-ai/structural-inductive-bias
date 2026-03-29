"""
continuous/__init__.py — HAM continuo (paradigma Liao)

Resuelve N[u(t)] = 0 mediante series homotópicas con convergencia controlada.

Author: Rodolfo H. Rodrigo — UNSJ / INAUT
"""

from .ham_series import ham_solve, ham_solve_system
from .convergence import hbar_curve, optimal_hbar
from .pade import pade_approximant, pade_eval
from .operators import L_derivative, L_second, L_damped, L_harmonic

__all__ = [
    'ham_solve', 'ham_solve_system',
    'hbar_curve', 'optimal_hbar',
    'pade_approximant', 'pade_eval',
    'L_derivative', 'L_second', 'L_damped', 'L_harmonic',
]
