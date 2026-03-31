"""
caso1_envelope.py — Case Study 1: Exponential envelope absorption (β=1).

Corresponds to §X.1 of the paper. Demonstrates that implicit model
with known exponential structure achieves ~3× lower test MSE.

System: f*(t) = g(t)·exp(-λt), λ=0.5, t∈[0,10]
Explicit: MLP learns f*(t) directly
Implicit: MLP learns h*(t) = g(t), F(t,y,z) = y·exp(λt) - z

Protocol: 12 sample sizes × 20 seeds, 3-layer MLP (64 neurons, Tanh, Adam).

Generates:
  - fig_exp1_learning_curves.pdf  (Fig. in paper)
  - fig_exp1_predictions.pdf
  - fig_exp1_ratio.pdf
  - results/exp1/experiment1.npz

Author: Rodolfo H. Rodrigo — UNSJ — 2026
Paper: 15Paper — Neurocomputing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from experiment1 import run_experiment

if __name__ == "__main__":
    run_experiment()
