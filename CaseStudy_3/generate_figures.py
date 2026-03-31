"""
generate_figures.py — Figures for Case Study 3 (HAM residual + Barron analysis).

Output:
  - figures/fig_exp3_mse_vs_K.pdf
  - figures/fig_exp3_residuals.pdf
  - figures/fig_exp3_predictions.pdf
  - figures/fig_barron_terms.pdf
  - figures/fig_barron_residuals.pdf
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import RESULTS_DIR

if __name__ == "__main__":
    data_file = os.path.join(RESULTS_DIR, 'exp3', 'experiment3.npz')

    if os.path.exists(data_file):
        print("Found saved data — regenerating figures from .npz...")
        from regenerate_figures import (fig_exp3_mse_vs_K, fig_exp3_residuals,
                                        fig_exp3_predictions)
        fig_exp3_mse_vs_K()
        fig_exp3_residuals()
        fig_exp3_predictions()

        print("\nRunning Barron norm analysis...")
        from barron_analysis import main as barron_main
        barron_main()
        print("Done.")
    else:
        print("No saved data — running full experiment...")
        from experiment3 import run_experiment
        run_experiment()
        print("\nRunning Barron norm analysis...")
        from barron_analysis import main as barron_main
        barron_main()
