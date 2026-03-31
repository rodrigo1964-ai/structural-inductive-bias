"""
generate_figures.py — Generate figures for Case Study 1 (envelope absorption).

If results/exp1/experiment1.npz exists, regenerates from saved data (fast).
Otherwise, runs the full experiment first (slow, ~20 min with 20 seeds).

Output:
  - figures/fig_exp1_learning_curves.pdf
  - figures/fig_exp1_predictions.pdf
  - figures/fig_exp1_ratio.pdf

Author: Rodolfo H. Rodrigo — UNSJ — 2026
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import RESULTS_DIR

if __name__ == "__main__":
    data_file = os.path.join(RESULTS_DIR, 'exp1', 'experiment1.npz')

    if os.path.exists(data_file):
        print("Found saved data — regenerating figures from .npz...")
        from regenerate_figures import fig_exp1_learning_curves, fig_exp1_predictions
        fig_exp1_learning_curves()
        fig_exp1_predictions()
        # ratio plot uses same data
        from experiment1 import plot_ratio
        import numpy as np
        d = np.load(data_file)
        plot_ratio(d['mse_explicit'], d['mse_implicit'])
        print("Done.")
    else:
        print("No saved data — running full experiment...")
        from experiment1 import run_experiment
        run_experiment()
