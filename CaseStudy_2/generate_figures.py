"""
generate_figures.py — Figures for Case Study 2 (nonlinearity absorption).

Output:
  - figures/fig_exp2_learning_curves.pdf
  - figures/fig_exp2_beta_effect.pdf
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import RESULTS_DIR

if __name__ == "__main__":
    data_file = os.path.join(RESULTS_DIR, 'exp2', 'experiment2.npz')

    if os.path.exists(data_file):
        print("Found saved data — regenerating figures from .npz...")
        import numpy as np
        d = np.load(data_file)
        from experiment2 import plot_learning_curves, plot_beta_effect
        plot_learning_curves(d['mse_explicit'], d['mse_implicit'])
        plot_beta_effect(d['mse_explicit'], d['mse_implicit'])
        print("Done.")
    else:
        print("No saved data — running full experiment...")
        from experiment2 import run_experiment
        run_experiment()
