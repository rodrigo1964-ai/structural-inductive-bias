"""
caso4_counterexample.py — Case Study 4: Counterexample (S(F)=1, trivial F).

§X.4: Same system as Exp 1 but with F(t,y,z) = y - z (identity).
No structural content → no advantage → validates CE 1.

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from experiment4 import run_experiment

if __name__ == "__main__":
    run_experiment()
