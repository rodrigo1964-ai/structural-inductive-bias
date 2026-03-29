"""Experiment 1: Exponential Envelope Absorption (beta = 1).

System: f*(t) = g(t) * exp(-lambda*t)
Explicit: MLP learns f*(t)
Implicit: MLP learns h*(t) = g(t), with F(t,y,z) = y*exp(lambda*t) - z
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import MLP, train_model, evaluate_model
from systems import f_star_exp1, g_modulation, LAMBDA, T_MAX_EXP1
from utils import (set_seed, set_figure_style, save_figure, save_results,
                   single_column_fig, double_column_fig,
                   BLUE, ORANGE, GREEN, FIGURES_DIR, RESULTS_DIR)

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
    """MSE between MLP output and y."""
    return torch.mean((pred - y_target) ** 2)


def implicit_loss(pred, y_target, t_input):
    """Loss in observation space: (MLP(t)*exp(-lambda*t) - y)^2.

    The MLP learns h*(t) = g(t), but the loss is computed in the original
    observation space to avoid amplifying noise by exp(lambda*t).
    """
    f_hat = pred * torch.exp(-LAMBDA * t_input)
    return torch.mean((f_hat - y_target) ** 2)


def implicit_reconstruct(mlp_output, t_test):
    """Reconstruct f_hat(t) = MLP(t) * exp(-lambda*t)."""
    return mlp_output * np.exp(-LAMBDA * t_test)


def run_experiment():
    print("=" * 60)
    print("Experiment 1: Exponential Envelope Absorption (beta=1)")
    print("=" * 60)

    # Test grid
    t_test = np.linspace(0, T_MAX_EXP1, N_TEST)
    f_true = f_star_exp1(t_test)

    # Storage
    mse_explicit = np.zeros((len(SAMPLE_SIZES), N_SEEDS))
    mse_implicit = np.zeros((len(SAMPLE_SIZES), N_SEEDS))

    for i, N in enumerate(SAMPLE_SIZES):
        print(f"\n  N = {N}")
        for s in range(N_SEEDS):
            set_seed(s)

            # Sample training data
            t_train = np.sort(np.random.uniform(0, T_MAX_EXP1, N))
            y_train = f_star_exp1(t_train) + np.random.randn(N) * SIGMA

            # Use smaller lr for very small N
            lr = 1e-4 if N < 30 else LR
            epochs = 8000 if N < 30 else EPOCHS

            # Explicit model
            set_seed(s)
            model_exp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
            train_model(model_exp, t_train, y_train, explicit_loss,
                        epochs=epochs, lr=lr, patience=PATIENCE)
            mse_e, _ = evaluate_model(model_exp, t_test, f_true)
            mse_explicit[i, s] = mse_e

            # Implicit model
            set_seed(s)
            model_imp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
            train_model(model_imp, t_train, y_train, implicit_loss,
                        epochs=epochs, lr=lr, patience=PATIENCE)
            mse_i, _ = evaluate_model(model_imp, t_test, f_true, implicit_reconstruct)
            mse_implicit[i, s] = mse_i

            if (s + 1) % 5 == 0:
                print(f"    seed {s+1}/{N_SEEDS}: exp={mse_e:.2e}, imp={mse_i:.2e}")

    # Save raw results
    save_results('experiment1.npz', 'exp1',
                 sample_sizes=np.array(SAMPLE_SIZES),
                 mse_explicit=mse_explicit,
                 mse_implicit=mse_implicit)

    # Generate figures
    plot_learning_curves(mse_explicit, mse_implicit)
    plot_predictions(t_test, f_true)
    plot_ratio(mse_explicit, mse_implicit)

    print("\nExperiment 1 complete.")


def plot_learning_curves(mse_explicit, mse_implicit):
    """Fig 1: Log-log MSE vs N."""
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
    ax.loglog(N, mean_imp, 's--', color=ORANGE, label='Implicit', linewidth=1.5, markersize=4)
    ax.fill_between(N, mean_imp - std_imp, mean_imp + std_imp, color=ORANGE, alpha=0.2)

    ax.set_xlabel('Training samples $N$')
    ax.set_ylabel('Test MSE')
    ax.legend()
    save_figure(fig, 'fig_exp1_learning_curves.pdf')


def plot_predictions(t_test, f_true, N_plot=50):
    """Fig: Predictions at N=50."""
    set_figure_style()
    fig = single_column_fig(0.75)
    ax = fig.add_subplot(111)

    set_seed(0)
    t_train = np.sort(np.random.uniform(0, T_MAX_EXP1, N_plot))
    y_train = f_star_exp1(t_train) + np.random.randn(N_plot) * SIGMA

    # Explicit
    set_seed(0)
    model_exp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    train_model(model_exp, t_train, y_train, explicit_loss, epochs=EPOCHS, lr=LR, patience=PATIENCE)
    _, pred_exp = evaluate_model(model_exp, t_test, f_true)

    # Implicit
    set_seed(0)
    model_imp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    train_model(model_imp, t_train, y_train, implicit_loss, epochs=EPOCHS, lr=LR, patience=PATIENCE)
    _, pred_imp = evaluate_model(model_imp, t_test, f_true, implicit_reconstruct)

    ax.plot(t_test, f_true, '-', color='black', label='$f^*(t)$', linewidth=1.5)
    ax.plot(t_test, pred_exp, '-', color=BLUE, label='Explicit', linewidth=1.0, alpha=0.8)
    ax.plot(t_test, pred_imp, '--', color=ORANGE, label='Implicit', linewidth=1.0, alpha=0.8)
    ax.plot(t_train, y_train, '.', color='gray', markersize=3, label='Training data', zorder=0)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$f(t)$')
    ax.legend(loc='upper right')
    save_figure(fig, 'fig_exp1_predictions.pdf')


def plot_ratio(mse_explicit, mse_implicit):
    """Fig: Empirical sample complexity ratio vs theoretical."""
    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    N = np.array(SAMPLE_SIZES)
    mean_exp = mse_explicit.mean(axis=1)
    mean_imp = mse_implicit.mean(axis=1)

    # For each MSE threshold (from implicit curve), find N_exp needed
    ratios = []
    thresholds = []
    for i in range(len(N)):
        target_mse = mean_imp[i]
        # Find where explicit crosses this threshold via interpolation
        if target_mse < mean_exp[-1]:
            # Explicit never reaches this low
            continue
        if target_mse > mean_exp[0]:
            continue
        # Log-linear interpolation
        log_mse_exp = np.log(mean_exp)
        log_N = np.log(N.astype(float))
        log_target = np.log(target_mse)
        # Find interval
        for j in range(len(N) - 1):
            if log_mse_exp[j] >= log_target >= log_mse_exp[j + 1]:
                alpha = (log_target - log_mse_exp[j]) / (log_mse_exp[j + 1] - log_mse_exp[j])
                N_exp_interp = np.exp(log_N[j] + alpha * (log_N[j + 1] - log_N[j]))
                ratios.append(N[i] / N_exp_interp)
                thresholds.append(target_mse)
                break

    if ratios:
        ax.plot(range(len(ratios)), ratios, 'o-', color=BLUE, label='Empirical ratio $N_{\\rm imp}/N_{\\rm exp}$')
        # Theoretical: for beta=1, ratio = B_imp^2 / B_exp^2
        # S(F) = B_exp / B_imp, so ratio = 1/S(F)^2
        # We estimate S(F) from the data
        # For the exponential envelope, S(F) should be >> 1
        ax.axhline(y=np.mean(ratios), color=ORANGE, linestyle='--', label=f'Mean = {np.mean(ratios):.3f}')
        ax.set_xlabel('Threshold index')
        ax.set_ylabel('$N_{\\rm imp} / N_{\\rm exp}$')
        ax.legend()

    save_figure(fig, 'fig_exp1_ratio.pdf')


if __name__ == '__main__':
    run_experiment()
