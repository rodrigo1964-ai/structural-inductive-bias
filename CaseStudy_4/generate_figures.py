"""
generate_figures.py — Figures for Case Study 4 (counterexample).

Output: figures/fig_exp4_trivial.pdf
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import RESULTS_DIR

if __name__ == "__main__":
    data_file = os.path.join(RESULTS_DIR, 'exp4', 'experiment4.npz')
    if os.path.exists(data_file):
        print("Found saved data — regenerating figure...")
        import numpy as np
        d = np.load(data_file)
        from experiment4 import plot_trivial
        plot_trivial(d['mse_explicit'], d['mse_implicit'])
        print("Done.")
    else:
        print("No saved data — running full experiment...")
        from experiment4 import run_experiment
        run_experiment()
