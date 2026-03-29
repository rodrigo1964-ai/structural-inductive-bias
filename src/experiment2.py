"""Experiment 2: Nonlinearity Absorption (beta > 1).

System: f*(t) = exp(g(t)), g(t) = sin(t) + 0.5*cos(2t)
Explicit: MLP learns f*(t) directly
Implicit: F(t,y,z) = exp(z) - y, so h*(t) = g(t) = ln(f*(t))
  Loss: (1/N) sum (exp(MLP(t_i)) - y_i)^2
  beta = sup|exp(z)| = exp(sup|g|) = exp(1.5) ~ 4.48
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import MLP, train_model, evaluate_model
from systems import f_star_exp2, g_inner_exp2, T_MAX_EXP2
from utils import (set_seed, set_figure_style, save_figure, save_results,
                   single_column_fig, BLUE, ORANGE, GREEN)

# ─── Hyperparameters ───
SAMPLE_SIZES = [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]
N_SEEDS = 20
N_TEST = 10000
SIGMA = 0.05
EPOCHS = 5000
LR = 1e-3
PATIENCE = 500
HIDDEN_DIM = 64
NUM_LAYERS = 3

BETA = np.exp(1.5)  # ~4.48


def explicit_loss(pred, y_target, t_input):
    return torch.mean((pred - y_target) ** 2)


def implicit_loss(pred, y_target, t_input):
    """Loss: (exp(MLP(t)) - y)^2."""
    return torch.mean((torch.exp(pred) - y_target) ** 2)


def implicit_reconstruct(mlp_output, t_test):
    """f_hat(t) = exp(MLP(t))."""
    return np.exp(mlp_output)


def run_experiment():
    print("=" * 60)
    print("Experiment 2: Nonlinearity Absorption (beta > 1)")
    print("=" * 60)

    t_test = np.linspace(0, T_MAX_EXP2, N_TEST)
    f_true = f_star_exp2(t_test)

    mse_explicit = np.zeros((len(SAMPLE_SIZES), N_SEEDS))
    mse_implicit = np.zeros((len(SAMPLE_SIZES), N_SEEDS))

    for i, N in enumerate(SAMPLE_SIZES):
        print(f"\n  N = {N}")
        for s in range(N_SEEDS):
            set_seed(s)
            t_train = np.sort(np.random.uniform(0, T_MAX_EXP2, N))
            y_train = f_star_exp2(t_train) + np.random.randn(N) * SIGMA

            lr = 1e-4 if N < 30 else LR
            epochs = 8000 if N < 30 else EPOCHS

            # Explicit
            set_seed(s)
            model_exp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
            train_model(model_exp, t_train, y_train, explicit_loss,
                        epochs=epochs, lr=lr, patience=PATIENCE)
            mse_e, _ = evaluate_model(model_exp, t_test, f_true)
            mse_explicit[i, s] = mse_e

            # Implicit
            set_seed(s)
            model_imp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
            train_model(model_imp, t_train, y_train, implicit_loss,
                        epochs=epochs, lr=lr, patience=PATIENCE)
            mse_i, _ = evaluate_model(model_imp, t_test, f_true, implicit_reconstruct)
            mse_implicit[i, s] = mse_i

            if (s + 1) % 5 == 0:
                print(f"    seed {s+1}/{N_SEEDS}: exp={mse_e:.2e}, imp={mse_i:.2e}")

    save_results('experiment2.npz', 'exp2',
                 sample_sizes=np.array(SAMPLE_SIZES),
                 mse_explicit=mse_explicit,
                 mse_implicit=mse_implicit,
                 beta=np.array([BETA]))

    plot_learning_curves(mse_explicit, mse_implicit)
    plot_beta_effect(mse_explicit, mse_implicit)
    print("\nExperiment 2 complete.")


def plot_learning_curves(mse_explicit, mse_implicit):
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
    ax.loglog(N, mean_imp, 's--', color=ORANGE, label='Implicit ($\\beta={:.1f}$)'.format(BETA),
              linewidth=1.5, markersize=4)
    ax.fill_between(N, mean_imp - std_imp, mean_imp + std_imp, color=ORANGE, alpha=0.2)

    ax.set_xlabel('Training samples $N$')
    ax.set_ylabel('Test MSE')
    ax.legend()
    save_figure(fig, 'fig_exp2_learning_curves.pdf')


def plot_beta_effect(mse_explicit, mse_implicit):
    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    N = np.array(SAMPLE_SIZES)
    mean_exp = mse_explicit.mean(axis=1)
    mean_imp = mse_implicit.mean(axis=1)

    # Advantage ratio (explicit MSE / implicit MSE)
    advantage = mean_exp / mean_imp

    ax.semilogx(N, advantage, 'o-', color=GREEN, label='Empirical advantage', linewidth=1.5, markersize=4)
    ax.axhline(y=1.0, color='gray', linestyle=':', label='No advantage')

    # Theoretical: advantage should be bounded by S(F)^2 / beta^2
    # For this system, the theoretical advantage depends on complexity reduction
    ax.set_xlabel('Training samples $N$')
    ax.set_ylabel('MSE$_{\\rm exp}$ / MSE$_{\\rm imp}$')
    ax.legend()
    save_figure(fig, 'fig_exp2_beta_effect.pdf')


if __name__ == '__main__':
    run_experiment()
