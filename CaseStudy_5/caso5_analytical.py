"""
caso5_analytical.py — Case Study 5: Closed-form analytical verification.

§X.5: f*(t) = (1+0.5 sin 3t)·exp(-t), β=1.
Barron norms computed via FFT: S(F) = C_{f*}/C_{h*} = 2.21.
Theoretical ratio vs empirical ratio comparison.

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from experiment5_analytical import run_experiment

if __name__ == "__main__":
    run_experiment()
