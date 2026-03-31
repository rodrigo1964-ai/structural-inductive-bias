"""
generate_figures.py — Figures for Case Study 5 (analytical verification).

Output: figures/fig_exp5_analytical.pdf
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import RESULTS_DIR

if __name__ == "__main__":
    data_file = os.path.join(RESULTS_DIR, 'exp5', 'experiment5.npz')
    if os.path.exists(data_file):
        print("Found saved data — regenerating figure...")
        import numpy as np
        d = np.load(data_file)
        from experiment5_analytical import plot_learning_curves, compute_empirical_ratio
        S_F = float(d['S_F'])
        theo = float(d['theoretical_ratio'])
        emp = compute_empirical_ratio(d['mse_explicit'], d['mse_implicit'])
        plot_learning_curves(d['mse_explicit'], d['mse_implicit'], S_F, theo, emp)
        print(f"S(F) = {S_F:.2f}, theoretical ratio = {theo:.4f}, empirical = {emp:.4f}")
        print("Done.")
    else:
        print("No saved data — running full experiment...")
        from experiment5_analytical import run_experiment
        run_experiment()
