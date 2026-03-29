"""Experiment 4: Counterexample Verification.

System: Same as Experiment 1 but with F(t,y,z) = y - z (trivial, no structural content).
S(F) = 1 => no advantage expected.
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import MLP, train_model, evaluate_model
from systems import f_star_exp1, T_MAX_EXP1
from utils import (set_seed, set_figure_style, save_figure, save_results,
                   single_column_fig, BLUE, ORANGE)

# ─── Hyperparameters ───
SAMPLE_SIZES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
N_SEEDS = 20
N_TEST = 10000
SIGMA = 0.01
EPOCHS = 5000
LR = 1e-3
PATIENCE = 500
HIDDEN_DIM = 64
NUM_LAYERS = 3


def explicit_loss(pred, y_target, t_input):
    return torch.mean((pred - y_target) ** 2)


def trivial_implicit_loss(pred, y_target, t_input):
    """F(t,y,z) = y - z. Target h*(t) = f*(t). Identical to explicit."""
    return torch.mean((pred - y_target) ** 2)


def run_experiment():
    print("=" * 60)
    print("Experiment 4: Counterexample (S(F)=1, trivial F)")
    print("=" * 60)

    t_test = np.linspace(0, T_MAX_EXP1, N_TEST)
    f_true = f_star_exp1(t_test)

    mse_explicit = np.zeros((len(SAMPLE_SIZES), N_SEEDS))
    mse_implicit = np.zeros((len(SAMPLE_SIZES), N_SEEDS))

    for i, N in enumerate(SAMPLE_SIZES):
        print(f"\n  N = {N}")
        for s in range(N_SEEDS):
            set_seed(s)
            t_train = np.sort(np.random.uniform(0, T_MAX_EXP1, N))
            y_train = f_star_exp1(t_train) + np.random.randn(N) * SIGMA

            lr = 1e-4 if N < 30 else LR
            epochs = 8000 if N < 30 else EPOCHS

            # Explicit
            set_seed(s)
            model_exp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
            train_model(model_exp, t_train, y_train, explicit_loss,
                        epochs=epochs, lr=lr, patience=PATIENCE)
            mse_e, _ = evaluate_model(model_exp, t_test, f_true)
            mse_explicit[i, s] = mse_e

            # Trivial implicit (same loss, so should be identical up to seed noise)
            set_seed(s)
            model_imp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
            train_model(model_imp, t_train, y_train, trivial_implicit_loss,
                        epochs=epochs, lr=lr, patience=PATIENCE)
            mse_i, _ = evaluate_model(model_imp, t_test, f_true)
            mse_implicit[i, s] = mse_i

            if (s + 1) % 5 == 0:
                print(f"    seed {s+1}/{N_SEEDS}: exp={mse_e:.2e}, imp={mse_i:.2e}")

    save_results('experiment4.npz', 'exp4',
                 sample_sizes=np.array(SAMPLE_SIZES),
                 mse_explicit=mse_explicit,
                 mse_implicit=mse_implicit)

    plot_trivial(mse_explicit, mse_implicit)
    print("\nExperiment 4 complete.")


def plot_trivial(mse_explicit, mse_implicit):
    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    N = np.array(SAMPLE_SIZES)
    mean_exp = mse_explicit.mean(axis=1)
    std_exp = mse_explicit.std(axis=1)
    mean_imp = mse_implicit.mean(axis=1)
    std_imp = mse_implicit.std(axis=1)

    ax.loglog(N, mean_exp, 'o-', color=BLUE, label='Explicit', linewidth=1.5, markersize=4)
    ax.fill_between(N, mean_exp - std_exp, mean_exp + std_exp, color=BLUE, alpha=0.2)
    ax.loglog(N, mean_imp, 's--', color=ORANGE, label='Implicit ($\\mathcal{S}=1$)',
              linewidth=1.5, markersize=4)
    ax.fill_between(N, mean_imp - std_imp, mean_imp + std_imp, color=ORANGE, alpha=0.2)

    ax.set_xlabel('Training samples $N$')
    ax.set_ylabel('Test MSE')
    ax.legend()
    save_figure(fig, 'fig_exp4_trivial.pdf')


if __name__ == '__main__':
    run_experiment()
