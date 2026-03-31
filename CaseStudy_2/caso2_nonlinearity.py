"""
caso2_nonlinearity.py — Case Study 2: Nonlinearity absorption (β > 1).

§X.2: f*(t) = exp(g(t)), β = exp(1.5) ≈ 4.48.
Demonstrates that large β can negate the structural advantage.

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from experiment2 import run_experiment

if __name__ == "__main__":
    run_experiment()
