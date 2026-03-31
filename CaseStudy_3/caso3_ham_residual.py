"""
caso3_ham_residual.py — Case Study 3: HAM residual learning.

§X.3: Nonlinear pendulum u''+sin(u)=0 with HAM partial sums S_K(t).
Implicit model learns residual h*_K(t) = u*(t) - S_K(t).
MSE decays exponentially with K, achieving 7× improvement at K=5.

Also runs Barron norm analysis (Fig. barron_residuals, barron_terms).

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from experiment3 import run_experiment

if __name__ == "__main__":
    run_experiment()
