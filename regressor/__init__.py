"""
homotopy_regressors — Unified HAM library for nonlinear ODE systems.

Two paradigms, one library:

  DISCRETE (Rodrigo): Step-by-step numerical solver using BDF discrete
  derivatives + homotopy corrections. For real-time control on embedded
  systems (ESP32, microcontrollers).

  CONTINUOUS (Liao): Analytic series solutions with convergence-control
  parameter hbar, auxiliary linear operator L, and Padé acceleration.
  For qualitative analysis and closed-form approximations.

Author: Rodolfo H. Rodrigo — UNSJ / INAUT
"""

# ==============================================================
# Discrete paradigm — numerical step-by-step solvers
# ==============================================================

# Scalar solvers (1st and 2nd order)
from .solver import solve_order1, solve_order2, solve_order1_numeric

# System solver (2-3 coupled ODEs)
from .solver_system import solve_system, solve_system_numeric

# Symbolic regressor builders (scalar)
from .regressor import (
    build_regressor_order1,
    build_regressor_order2,
    build_inverse_regressor,
)

# Symbolic regressor builder (systems)
from .regressor_system import build_system_regressor

# Step-by-step API with 3pt/4pt variants
from .ode_solver import (
    ode1_step_3pt,
    ode1_step_4pt,
    solve_ode1,
    build_ode1_regressors,
)

# ==============================================================
# Continuous paradigm — analytic HAM series
# ==============================================================

from .continuous import (
    ham_solve,
    ham_solve_system,
    hbar_curve,
    optimal_hbar,
    pade_approximant,
    pade_eval,
    L_derivative,
    L_second,
    L_damped,
    L_harmonic,
)

# ==============================================================
# Tools & utilities
# ==============================================================

# Discrete derivative formulas
from .derivatives import discrete_derivatives

# ODE text parser
from .parser import parse_ode, parse_and_build, show

# Parameter identification (LIP and Non-LIP)
from .identify_parameters import (
    check_lip,
    identify_lip,
    identify_nonlip,
    build_parametric_regressor,
)

# Verification framework
from .verify_regressor import (
    verify_regressor_vs_rk45,
    run_suite,
    print_report,
)

# ==============================================================
# Package metadata
# ==============================================================

__version__ = "0.2.0"
__author__ = "Rodolfo H. Rodrigo"
__affiliation__ = "UNSJ / INAUT-CONICET"

__all__ = [
    # Discrete - scalar
    'solve_order1', 'solve_order2', 'solve_order1_numeric',
    # Discrete - systems
    'solve_system', 'solve_system_numeric',
    # Discrete - symbolic builders
    'build_regressor_order1', 'build_regressor_order2',
    'build_inverse_regressor', 'build_system_regressor',
    # Discrete - step-by-step
    'ode1_step_3pt', 'ode1_step_4pt', 'solve_ode1', 'build_ode1_regressors',
    # Continuous - HAM series
    'ham_solve', 'ham_solve_system',
    # Continuous - convergence
    'hbar_curve', 'optimal_hbar',
    # Continuous - Padé
    'pade_approximant', 'pade_eval',
    # Continuous - operators
    'L_derivative', 'L_second', 'L_damped', 'L_harmonic',
    # Tools
    'discrete_derivatives',
    'parse_ode', 'parse_and_build', 'show',
    'check_lip', 'identify_lip', 'identify_nonlip', 'build_parametric_regressor',
    'verify_regressor_vs_rk45', 'run_suite', 'print_report',
]
