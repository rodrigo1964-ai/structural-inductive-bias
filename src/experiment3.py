"""Experiment 3: HAM Residual Learning (Exponential Reduction with K).

System: Nonlinear pendulum u'' + sin(u) = 0, u(0)=pi/3, u'(0)=0.
For each HAM truncation order K=0..6:
  - Compute S_K(t) = sum_{k=0}^K u_k(t)
  - Implicit model learns residual h*_K(t) = u*(t) - S_K(t)
  - Reconstruct: u_hat(t) = S_K(t) + MLP(t)
  - Compare with explicit model learning u*(t) directly
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models import MLP, train_model, evaluate_model
from systems import pendulum_reference, T_MAX_EXP3
from ham import compute_ham_terms
from utils import (set_seed, set_figure_style, save_figure, save_results,
                   single_column_fig, double_column_fig,
                   BLUE, ORANGE, GREEN, CYAN, PURPLE, YELLOW, RED)

# ─── Hyperparameters ───
K_VALUES = [0, 1, 2, 3, 4, 5, 6]
N_TRAIN = 100
N_SEEDS = 20
N_TEST = 10000
SIGMA = 0.01
EPOCHS = 5000
LR = 1e-3
PATIENCE = 500
HIDDEN_DIM = 64
NUM_LAYERS = 3

# Dense grid for HAM computation (needs ~10000 pts for finite diff accuracy)
N_HAM_GRID = 20000


def explicit_loss(pred, y_target, t_input):
    return torch.mean((pred - y_target) ** 2)


def make_implicit_loss(S_K_interp_fn):
    """Create implicit loss for a given HAM partial sum S_K."""
    def loss_fn(pred, y_target, t_input):
        # Target for MLP: h*(t) = y - S_K(t)
        t_np = t_input.detach().squeeze().numpy()
        S_K_vals = torch.tensor(S_K_interp_fn(t_np), dtype=torch.float32).unsqueeze(1)
        target_h = y_target - S_K_vals
        return torch.mean((pred - target_h) ** 2)
    return loss_fn


def make_reconstruct(S_K_interp_fn):
    """Create reconstruction function for implicit model."""
    def reconstruct(mlp_output, t_test):
        return mlp_output + S_K_interp_fn(t_test)
    return reconstruct


def run_experiment():
    print("=" * 60)
    print("Experiment 3: HAM Residual Learning")
    print("=" * 60)

    # Test grid
    t_test = np.linspace(0, T_MAX_EXP3, N_TEST)
    u_ref_test = pendulum_reference(t_test)

    # Compute HAM terms on dense grid
    print("\n  Computing HAM terms on dense grid...")
    t_ham = np.linspace(0, T_MAX_EXP3, N_HAM_GRID)
    terms, partial_sums = compute_ham_terms(max(K_VALUES), t_ham)

    # Interpolators for partial sums
    from scipy.interpolate import interp1d
    S_K_interps = {}
    for K in K_VALUES:
        S_K_interps[K] = interp1d(t_ham, partial_sums[K], kind='cubic', fill_value='extrapolate')

    # Storage
    mse_explicit = np.zeros(N_SEEDS)
    mse_implicit = np.zeros((len(K_VALUES), N_SEEDS))

    # Run explicit model once (doesn't depend on K)
    print("\n  Training explicit models (N=100)...")
    for s in range(N_SEEDS):
        set_seed(s)
        t_train = np.sort(np.random.uniform(0, T_MAX_EXP3, N_TRAIN))
        y_train = pendulum_reference(t_train) + np.random.randn(N_TRAIN) * SIGMA

        set_seed(s)
        model_exp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
        train_model(model_exp, t_train, y_train, explicit_loss,
                    epochs=EPOCHS, lr=LR, patience=PATIENCE)
        mse_e, _ = evaluate_model(model_exp, t_test, u_ref_test)
        mse_explicit[s] = mse_e

    print(f"    Explicit mean MSE: {mse_explicit.mean():.2e} +/- {mse_explicit.std():.2e}")

    # Implicit models for each K
    for ki, K in enumerate(K_VALUES):
        print(f"\n  K = {K}")
        S_K_fn = S_K_interps[K]
        loss_fn = make_implicit_loss(S_K_fn)
        recon_fn = make_reconstruct(S_K_fn)

        for s in range(N_SEEDS):
            set_seed(s)
            t_train = np.sort(np.random.uniform(0, T_MAX_EXP3, N_TRAIN))
            y_train = pendulum_reference(t_train) + np.random.randn(N_TRAIN) * SIGMA

            set_seed(s)
            model_imp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
            train_model(model_imp, t_train, y_train, loss_fn,
                        epochs=EPOCHS, lr=LR, patience=PATIENCE)
            mse_i, _ = evaluate_model(model_imp, t_test, u_ref_test, recon_fn)
            mse_implicit[ki, s] = mse_i

        print(f"    Implicit K={K} mean MSE: {mse_implicit[ki].mean():.2e} +/- {mse_implicit[ki].std():.2e}")

    # Save results
    S_K_on_test = np.array([S_K_interps[K](t_test) for K in K_VALUES])
    save_results('experiment3.npz', 'exp3',
                 K_values=np.array(K_VALUES),
                 mse_explicit=mse_explicit,
                 mse_implicit=mse_implicit,
                 t_test=t_test,
                 u_ref_test=u_ref_test,
                 S_K_on_test=S_K_on_test)

    # Figures
    plot_mse_vs_K(mse_explicit, mse_implicit)
    plot_residuals(t_test, u_ref_test, S_K_interps)
    plot_predictions(t_test, u_ref_test, S_K_interps)
    print("\nExperiment 3 complete.")


def plot_mse_vs_K(mse_explicit, mse_implicit):
    """THE money figure: semilog MSE vs K."""
    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    K = np.array(K_VALUES)
    mean_imp = mse_implicit.mean(axis=1)
    std_imp = mse_implicit.std(axis=1)
    mean_exp = mse_explicit.mean()
    std_exp = mse_explicit.std()

    # Explicit as horizontal band
    ax.axhline(y=mean_exp, color=BLUE, linewidth=1.5, label='Explicit')
    ax.axhspan(mean_exp - std_exp, mean_exp + std_exp, color=BLUE, alpha=0.15)

    # Implicit as decreasing curve
    ax.semilogy(K, mean_imp, 's--', color=ORANGE, label='Implicit (HAM residual)',
                linewidth=1.5, markersize=5)
    ax.fill_between(K, mean_imp - std_imp, mean_imp + std_imp, color=ORANGE, alpha=0.2)

    ax.set_xlabel('HAM order $K$')
    ax.set_ylabel('Test MSE')
    ax.set_xticks(K)
    ax.legend()
    save_figure(fig, 'fig_exp3_mse_vs_K.pdf')


def plot_residuals(t_test, u_ref_test, S_K_interps):
    """Show how residual h*_K(t) shrinks with K."""
    set_figure_style()
    fig = single_column_fig(0.85)
    ax = fig.add_subplot(111)

    colors = [BLUE, GREEN, ORANGE, PURPLE]
    for idx, K in enumerate([0, 2, 4, 6]):
        residual = u_ref_test - S_K_interps[K](t_test)
        ax.plot(t_test, residual, color=colors[idx], label=f'$K={K}$',
                linewidth=1.0, alpha=0.8)

    ax.set_xlabel('$t$')
    ax.set_ylabel('Residual $h^*_K(t)$')
    ax.legend()
    save_figure(fig, 'fig_exp3_residuals.pdf')


def plot_predictions(t_test, u_ref_test, S_K_interps, K_show=3):
    """Predictions at K=3."""
    set_figure_style()
    fig = single_column_fig(0.75)
    ax = fig.add_subplot(111)

    set_seed(0)
    t_train = np.sort(np.random.uniform(0, T_MAX_EXP3, N_TRAIN))
    y_train = pendulum_reference(t_train) + np.random.randn(N_TRAIN) * SIGMA

    # Explicit
    set_seed(0)
    model_exp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    train_model(model_exp, t_train, y_train, explicit_loss, epochs=EPOCHS, lr=LR, patience=PATIENCE)
    _, pred_exp = evaluate_model(model_exp, t_test, u_ref_test)

    # Implicit K=3
    S_K_fn = S_K_interps[K_show]
    loss_fn = make_implicit_loss(S_K_fn)
    recon_fn = make_reconstruct(S_K_fn)

    set_seed(0)
    model_imp = MLP(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS)
    train_model(model_imp, t_train, y_train, loss_fn, epochs=EPOCHS, lr=LR, patience=PATIENCE)
    _, pred_imp = evaluate_model(model_imp, t_test, u_ref_test, recon_fn)

    ax.plot(t_test, u_ref_test, '-', color='black', label='Reference', linewidth=1.5)
    ax.plot(t_test, pred_exp, '-', color=BLUE, label='Explicit', linewidth=1.0, alpha=0.8)
    ax.plot(t_test, pred_imp, '--', color=ORANGE, label=f'Implicit ($K={K_show}$)',
            linewidth=1.0, alpha=0.8)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$u(t)$')
    ax.legend()
    save_figure(fig, 'fig_exp3_predictions.pdf')


if __name__ == '__main__':
    run_experiment()
